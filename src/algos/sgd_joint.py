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
    # SVD works on GPU, so keep on GPU
    return [effective_rank(act) for act in activations_list]

def mask_churn(prev_masks, cur_masks):
    if prev_masks is None: return [0.0]*len(cur_masks)
    churn = []
    P = cur_masks[0].shape[0]
    for l in range(len(cur_masks)):
        changed = (prev_masks[l] != cur_masks[l]).float().mean().item()
        churn.append(changed)
    return churn

def train_sgd_joint(model, train_loader, val_loader, config, test_loader=None):
    """
    Train weights and slopes simultaneously using SGD.
    Both weights and gate slopes are updated in the same optimization step.
    """
    device  = config["device"]
    t, r, p = config["training"], config["regularization"], config["pruning"]
    epochs = int(t["epochs"])
    lr_w = float(t["lr_w"])
    lr_a = float(t.get("lr_a", t["lr_w"]))  # fallback to lr_w if lr_a not specified
    lam = float(r["lambda_identity"]) if r.get("identity_reg", False) else 0.0

    model.to(device)
    
    # Enable gradients for both weights and gates
    if getattr(model, "use_gates", False):
        model.set_weights_requires_grad(True)
        model.set_gates_requires_grad(True)
        # Create optimizer with both weight and gate parameters
        all_params = list(model.parameters())
        opt = optim.AdamW(all_params, lr=lr_w)
    else:
        # If no gates, just train weights
        model.set_weights_requires_grad(True)
        opt = optim.AdamW(model.parameters(), lr=lr_w)
    
    # Computation frequency controls (for speed optimization)
    # NOTE: Path metrics frequency only applies to epoch-based training (sgd_joint has many epochs)
    logging_cfg = config.get("logging", {})
    path_metrics_freq = int(logging_cfg.get("path_metrics_every_n_epochs", 1))
    effective_rank_freq = int(logging_cfg.get("effective_rank_every_n_cycles", 1))

    history = []
    prev_masks = None
    prev_path_hashes = None  # Track path hashes for confident churn
    
    import time as time_module
    epoch_start_time = time_module.time()
    try:
        sample_batch = next(iter(train_loader))
        batch_size = len(sample_batch[0]) if isinstance(sample_batch, tuple) else len(sample_batch)
    except:
        batch_size = "unknown"
    print(f"[sgd_joint] Starting training: {epochs} epochs, batch_size={batch_size}")

    for ep in range(1, epochs+1):
        if ep % 10 == 0 or ep == 1:
            elapsed = time_module.time() - epoch_start_time
            print(f"[sgd_joint] Epoch {ep}/{epochs} (elapsed: {elapsed:.1f}s, avg: {elapsed/ep:.2f}s/epoch)")
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            loss = mse_loss(yhat, yb)
            
            # Add identity regularization if enabled
            if lam > 0 and getattr(model, "use_gates", False):
                for a_p, a_m in model.layer_slopes():
                    loss = loss + lam*(torch.sum((a_p-1.0)**2) + torch.sum((a_m-1.0)**2))
            
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Calculate metrics BEFORE pruning (to see actual slope values)
        if getattr(model, "use_gates", False):
            slopes = model.layer_slopes()
            B_total, B_layers = slope_budget(slopes)
            H_total, H_layers = slope_entropy(slopes)
            deltas = slope_deviation(slopes)
            gate_stats_dict = gate_stats(slopes)
            # Effective rank uses SVD - compute less frequently for speed
            if (ep % effective_rank_freq == 0) or (ep == 1) or (ep == epochs):
                eff_ranks = compute_effective_ranks(model, train_loader, device)
            else:
                eff_ranks = None  # Skip expensive SVD computation
            
            # Compute path metrics - only compute full metrics every N epochs for speed
            compute_full_path_metrics = (ep % path_metrics_freq == 0) or (ep == 1) or (ep == epochs)
            
            if compute_full_path_metrics:
                # Cache loader to avoid recreating
                # BUT: Check if cached loader matches current dataset to avoid cross-run contamination
                from torch.utils.data import DataLoader
                current_dataset = train_loader.dataset
                current_dataset_size = len(current_dataset)
                
                # Check if we need to recreate the cache
                need_new_cache = True
                if hasattr(train_sgd_joint, '_path_loader_cache'):
                    cached_loader = train_sgd_joint._path_loader_cache
                    # Check if cached loader's dataset is the same object and size
                    if (hasattr(cached_loader, 'dataset') and 
                        cached_loader.dataset is current_dataset and
                        len(cached_loader.dataset) == current_dataset_size):
                        need_new_cache = False
                
                if need_new_cache:
                    train_sgd_joint._path_loader_cache = DataLoader(
                        current_dataset, batch_size=current_dataset_size, shuffle=False
                    )
                path_loader = train_sgd_joint._path_loader_cache
                
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
                # Still need path hashes for confident churn, but skip expensive metrics
                from torch.utils.data import DataLoader
                current_dataset = train_loader.dataset
                current_dataset_size = len(current_dataset)
                
                # Check if we need to recreate the cache
                need_new_cache = True
                if hasattr(train_sgd_joint, '_path_loader_cache'):
                    cached_loader = train_sgd_joint._path_loader_cache
                    # Check if cached loader's dataset is the same object and size
                    if (hasattr(cached_loader, 'dataset') and 
                        cached_loader.dataset is current_dataset and
                        len(cached_loader.dataset) == current_dataset_size):
                        need_new_cache = False
                
                if need_new_cache:
                    train_sgd_joint._path_loader_cache = DataLoader(
                        current_dataset, batch_size=current_dataset_size, shuffle=False
                    )
                path_loader = train_sgd_joint._path_loader_cache
                
                # Quick path hash computation only
                model.eval()
                all_z_list = None
                all_margins = None
                for xb, yb in path_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    yhat, cache = model(xb, return_cache=True)
                    zs = [z.detach().cpu() for z in cache["z"]]
                    if all_z_list is None:
                        all_z_list = zs
                    else:
                        for l in range(len(zs)):
                            all_z_list[l] = torch.cat([all_z_list[l], zs[l]], dim=0)
                    
                    yhat_flat = yhat.squeeze() if yhat.dim() > 1 else yhat
                    margins_batch = yhat_flat.abs().cpu()
                    if all_margins is None:
                        all_margins = margins_batch
                    else:
                        all_margins = torch.cat([all_margins, margins_batch], dim=0)
                
                from ..utils.paths import hash_mask_list, compute_confident_churn
                L = len(all_z_list)
                cur_path_hashes = []
                for l in range(L):
                    partial_z_list = all_z_list[:l+1]
                    partial_hashes = hash_mask_list(partial_z_list)
                    cur_path_hashes.append(partial_hashes)
                
                if prev_path_hashes is not None and len(prev_path_hashes) == L:
                    confident_churn_layers = compute_confident_churn(
                        prev_path_hashes, cur_path_hashes, all_margins, tau=0.0
                    )
                else:
                    confident_churn_layers = None
                
                prev_path_hashes = cur_path_hashes
                path_metrics = {
                    "confident_churn_layers": confident_churn_layers,
                    "cur_path_hashes": cur_path_hashes,
                }
        else:
            B_total=B_layers=H_total=H_layers=deltas=None
            gate_stats_dict = None
            # Effective rank uses SVD - compute less frequently for speed
            if (ep % effective_rank_freq == 0) or (ep == 1) or (ep == epochs):
                eff_ranks = compute_effective_ranks(model, train_loader, device)
            else:
                eff_ranks = None
            path_metrics = None

        # Pruning + mask tracking
        if getattr(model, "use_gates", False):
            current_masks = dataset_masks(model, train_loader, device)
            churn_layers = mask_churn(prev_masks, current_masks)
            prev_masks = current_masks

            pruning_stats = []
            if p.get("enabled", False):
                pruning_stats = prune_identity_like(model, tau=float(p["tau"]))
        else:
            churn_layers = None
            pruning_stats = []

        tr_acc, tr_loss = eval_loader(model, train_loader, device)
        va_acc, va_loss = eval_loader(model, val_loader, device)

        # Early stopping check
        early_stopped = False
        test_loss = None
        test_acc = None
        if test_loader is not None:
            test_acc, test_loss = eval_loader(model, test_loader, device)
            if test_loss < 0.01:
                print(f"Early stopping: test loss {test_loss:.6f} < 0.01")
                early_stopped = True
        
        if va_loss < 0.01:
            print(f"Early stopping: validation loss {va_loss:.6f} < 0.01")
            early_stopped = True

        hist_entry = {
            "epoch": ep,
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

