from __future__ import annotations
import torch
from torch import optim
from ..utils.metrics import mse_loss, slope_budget, slope_entropy, slope_deviation, effective_rank, gate_stats, compute_path_metrics
from .pruning import prune_identity_like

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    L=A=n=0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        L += torch.mean((yhat - yb)**2).item() * xb.size(0)
        A += torch.sign(yhat).eq(yb).float().mean().item() * xb.size(0)
        n += xb.size(0)
    model.train(False)
    return A/n, L/n

@torch.no_grad()
def dataset_masks(model, loader, device):
    """Collect all sign masks z_l over the whole dataset (list of [P, d_l] bool tensors)."""
    model.eval()
    masks = None
    for xb, _ in loader:
        xb = xb.to(device)
        _, cache = model(xb, return_cache=True)
        batch_masks = [z.bool() for z in cache["z"]]  # Keep on GPU
        if masks is None:
            masks = [bm.clone() for bm in batch_masks]
        else:
            for l in range(len(masks)):
                masks[l] = torch.cat([masks[l], batch_masks[l]], dim=0)
        break  # OPTIMIZATION: Only use first batch for churn approximation to save time
    # Move to CPU only at the end
    return [m.cpu() for m in masks] if masks is not None else None

@torch.no_grad()
def compute_effective_ranks(model, loader, device):
    """Compute effective rank for each layer's hidden activations."""
    model.eval()
    activations_list = None
    for xb, _ in loader:
        xb = xb.to(device)
        _, cache = model(xb, return_cache=True)
        batch_activations = [h for h in cache["h"]]  # Keep on GPU
        if activations_list is None:
            activations_list = [act.clone() for act in batch_activations]
        else:
            for l in range(len(activations_list)):
                activations_list[l] = torch.cat([activations_list[l], batch_activations[l]], dim=0)
        break  # Use first batch for efficiency
    if activations_list is None:
        return []
    # Move to CPU only for effective_rank computation (which uses SVD)
    return [effective_rank(act.cpu()) for act in activations_list]

def mask_churn(prev_masks, cur_masks):
    if prev_masks is None: return [0.0]*len(cur_masks)
    churn = []
    P = cur_masks[0].shape[0]
    for l in range(len(cur_masks)):
        changed = (prev_masks[l] != cur_masks[l]).float().mean().item()
        churn.append(changed)
    return churn

def train_alt_em_sgd(model, train_loader, val_loader, config, test_loader=None):
    # Store test_loader for early stopping
    train_alt_em_sgd._test_loader = test_loader
    device  = config["device"]
    t, r, p = config["training"], config["regularization"], config["pruning"]
    cycles, lr_w, lr_a = int(t["cycles"]), float(t["lr_w"]), float(t["lr_a"])
    k_w, k_a = int(t["steps_w_per_cycle"]), int(t["steps_a_per_cycle"])
    lam = float(r["lambda_identity"]) if r["identity_reg"] else 0.0
    
    # Computation frequency controls (for speed optimization)
    # NOTE: Path metrics are ALWAYS computed every cycle for alternating EM (only 100 cycles, need full detail)
    logging_cfg = config.get("logging", {})
    effective_rank_freq = int(logging_cfg.get("effective_rank_every_n_cycles", 1))

    model.to(device)
    history = []
    prev_masks = None
    prev_path_hashes = None  # Track path hashes for confident churn
    
    import time as time_module
    cycle_start_time = time_module.time()
    try:
        sample_batch = next(iter(train_loader))
        batch_size = len(sample_batch[0]) if isinstance(sample_batch, tuple) else len(sample_batch)
    except:
        batch_size = "unknown"
    print(f"[alt_em_sgd] Starting training: {cycles} cycles, batch_size={batch_size}")

    # OPTIMIZATION: Create persistent iterator to avoid recreating it
    train_iter = iter(train_loader)
    
    for cyc in range(1, cycles+1):
        if cyc % 10 == 0 or cyc == 1:
            elapsed = time_module.time() - cycle_start_time
            print(f"[alt_em_sgd] Cycle {cyc}/{cycles} (elapsed: {elapsed:.1f}s, avg: {elapsed/cyc:.2f}s/cycle)")
        # ---- W-phase --------------------------------------------------------
        model.set_weights_requires_grad(True)
        model.set_gates_requires_grad(False)
        opt_w = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_w)
        for _ in range(k_w):
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)  # Reset iterator if exhausted
                xb, yb = next(train_iter)
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            loss = mse_loss(yhat, yb)
            opt_w.zero_grad(); loss.backward(); opt_w.step()

        # ---- A-phase --------------------------------------------------------
        if getattr(model, "use_gates", False):
            model.set_weights_requires_grad(False)
            model.set_gates_requires_grad(True)
            opt_a = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_a)
            for _ in range(k_a):
                try:
                    xb, yb = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)  # Reset iterator if exhausted
                    xb, yb = next(train_iter)
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                loss = mse_loss(yhat, yb)
                if lam > 0:
                    for a_p, a_m in model.layer_slopes():
                        loss = loss + lam*(torch.sum((a_p-1.0)**2) + torch.sum((a_m-1.0)**2))
                opt_a.zero_grad(); loss.backward(); opt_a.step()

        # ---- Pruning + metrics ---------------------------------------------
        current_masks = dataset_masks(model, train_loader, device)
        churn_layers = mask_churn(prev_masks, current_masks)
        prev_masks = current_masks

        pruning_stats = []
        if getattr(model, "use_gates", False) and p["enabled"]:
            pruning_stats = prune_identity_like(model, tau=float(p["tau"]))

        tr_acc, tr_loss = eval_loader(model, train_loader, device)
        va_acc, va_loss = eval_loader(model, val_loader, device)

        if getattr(model, "use_gates", False):
            slopes = model.layer_slopes()
            B_total, B_layers = slope_budget(slopes)
            H_total, H_layers = slope_entropy(slopes)
            deltas = slope_deviation(slopes)
            gate_stats_dict = gate_stats(slopes)
            # Effective rank uses SVD - compute less frequently for speed
            if (cyc % effective_rank_freq == 0) or (cyc == 1) or (cyc == cycles):
                eff_ranks = compute_effective_ranks(model, train_loader, device)
            else:
                eff_ranks = None  # Skip expensive SVD computation
            
            # Compute path metrics (using full_train_loader for complete dataset)
            # NOTE: For alternating EM, we ALWAYS compute full path metrics every cycle (only 100 cycles, need full detail)
            # Create a non-shuffled loader for path metrics - cache it to avoid recreating
            # BUT: Check if cached loader matches current dataset to avoid cross-run contamination
            from torch.utils.data import DataLoader
            current_dataset = train_loader.dataset
            current_dataset_size = len(current_dataset)
            
            # Check if we need to recreate the cache
            need_new_cache = True
            if hasattr(train_alt_em_sgd, '_path_loader_cache'):
                cached_loader = train_alt_em_sgd._path_loader_cache
                # Check if cached loader's dataset is the same object and size
                if (hasattr(cached_loader, 'dataset') and 
                    cached_loader.dataset is current_dataset and
                    len(cached_loader.dataset) == current_dataset_size):
                    need_new_cache = False
            
            if need_new_cache:
                train_alt_em_sgd._path_loader_cache = DataLoader(
                    current_dataset, batch_size=current_dataset_size, shuffle=False
                )
            path_loader = train_alt_em_sgd._path_loader_cache
            
            # Extract group_ids if available (for SEI computation)
            group_ids = None
            n_groups = None
            if hasattr(train_loader.dataset, 'get_group_ids'):
                group_ids = train_loader.dataset.get_group_ids()
                if group_ids is not None:
                    # Get n_groups from dataset or config
                    if hasattr(train_loader.dataset, 'n_groups'):
                        n_groups = train_loader.dataset.n_groups
                    else:
                        # Try to infer from group_ids
                        n_groups = int(group_ids.max() + 1) if len(group_ids) > 0 else None
            
            path_metrics = compute_path_metrics(
                model, path_loader, device=device, 
                prev_masks=prev_masks, 
                prev_path_hashes=prev_path_hashes,
                return_masks=False,
                return_path_hashes=True,  # Need path hashes for next cycle
                group_ids=group_ids, n_groups=n_groups
            )
            # Update prev_path_hashes for next cycle
            if path_metrics is not None and "cur_path_hashes" in path_metrics:
                prev_path_hashes = path_metrics["cur_path_hashes"]
        else:
            B_total=B_layers=H_total=H_layers=deltas=None
            gate_stats_dict = None
            # Effective rank uses SVD - compute less frequently for speed
            if (cyc % effective_rank_freq == 0) or (cyc == 1) or (cyc == cycles):
                eff_ranks = compute_effective_ranks(model, train_loader, device)
            else:
                eff_ranks = None
            path_metrics = None

        # Early stopping check
        early_stopped = False
        test_loss = None
        test_acc = None
        if hasattr(train_alt_em_sgd, '_test_loader') and train_alt_em_sgd._test_loader is not None:
            test_acc, test_loss = eval_loader(model, train_alt_em_sgd._test_loader, device)
            if test_loss < 0.01:
                print(f"Early stopping: test loss {test_loss:.6f} < 0.01")
                early_stopped = True
        
        if va_loss < 0.01:
            print(f"Early stopping: validation loss {va_loss:.6f} < 0.01")
            early_stopped = True

        hist_entry = {
            "cycle": cyc,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss,   "val_acc": va_acc,
            "slope_budget_total": B_total,  "slope_budget_layers": B_layers,
            "slope_entropy_total": H_total, "slope_entropy_layers": H_layers,
            "slope_deviation_layers": deltas,
            "mask_churn_layers": churn_layers,
            "pruning": pruning_stats,
            "effective_rank_layers": eff_ranks,
            "gate_stats": gate_stats_dict
        }
        # Add path metrics if computed
        if path_metrics is not None:
            hist_entry.update({
                "path_pressure_layers": path_metrics.get("path_pressure_layers"),
                "path_entropy_layers": path_metrics.get("path_entropy_layers"),
                "active_path_complexity": path_metrics.get("active_path_complexity"),
                "snr_max_layers": path_metrics.get("snr_max_layers"),
                "snr_p95_layers": path_metrics.get("snr_p95_layers"),
                "churn_active_layers": path_metrics.get("churn_active_layers"),
                "sei_layers": path_metrics.get("sei_layers"),
                # New path-centric metrics
                "H_path": path_metrics.get("H_path"),
                "H_gain": path_metrics.get("H_gain"),
                "I_layers": path_metrics.get("I_layers"),
                "confident_churn_layers": path_metrics.get("confident_churn_layers"),
                "path_snr_count_above_threshold": path_metrics.get("path_snr_count_above_threshold"),
                "path_snr_threshold": path_metrics.get("path_snr_threshold"),
                "path_snr_num_paths": path_metrics.get("path_snr_num_paths"),
                # Path SNR components: c_gamma (label correlation) - mean, median, std
                "path_snr_c_gamma_mean": path_metrics.get("path_snr_c_gamma_mean"),
                "path_snr_c_gamma_median": path_metrics.get("path_snr_c_gamma_median"),
                "path_snr_c_gamma_std": path_metrics.get("path_snr_c_gamma_std"),
                # Path SNR components: N_gamma (sample support) - mean, median, std
                "path_snr_N_gamma_mean": path_metrics.get("path_snr_N_gamma_mean"),
                "path_snr_N_gamma_median": path_metrics.get("path_snr_N_gamma_median"),
                "path_snr_N_gamma_std": path_metrics.get("path_snr_N_gamma_std"),
                "path_snr_N_gamma_total": path_metrics.get("path_snr_N_gamma_total"),
                # Path SNR components: SNR_gamma (signal-to-noise ratio) - mean, median, std
                "path_snr_SNR_gamma_mean": path_metrics.get("path_snr_SNR_gamma_mean"),
                "path_snr_SNR_gamma_median": path_metrics.get("path_snr_SNR_gamma_median"),
                "path_snr_SNR_gamma_std": path_metrics.get("path_snr_SNR_gamma_std"),
                "nri": path_metrics.get("nri"),
            })
        history.append(hist_entry)
        if test_loss is not None:
            history[-1]["test_loss"] = test_loss
            history[-1]["test_acc"] = test_acc
        
        if early_stopped:
            break
    return history
