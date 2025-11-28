from __future__ import annotations
import torch
from torch import optim
from .pruning import prune_identity_like
# Removed: from ..utils.metrics import compute_path_metrics

# Local utility function
def mse_loss(yhat, y):
    """Mean squared error loss."""
    return torch.mean((yhat - y) ** 2)

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
    logging_cfg = config.get("logging", {})
    path_kernel_metrics_freq = int(logging_cfg.get("path_kernel_metrics_every_n_cycles", 1))  # Compute path kernel metrics every cycle by default
    path_analysis_freq = path_kernel_metrics_freq  # Use same frequency as path kernel metrics
    path_analysis_out_dir = config.get("path_analysis_out_dir", None)  # Output directory for path analysis plots

    model.to(device)
    history = []
    prev_masks = None  # For regular churn (from train_loader, first batch only)
    prev_masks_path_metrics = None  # For path metrics (from path_loader, full dataset)
    prev_path_hashes = None  # Track path hashes for confident churn
    # Checkpoint embeddings for lineage/centroid metrics
    checkpoint_embeddings = []  # List of (cycle, embedding_tensor) tuples
    checkpoint_metrics_history = []  # List of checkpoint metrics dicts
    
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

        # Compute path kernel metrics (effective rank, variance explained, etc.)
        path_kernel_metrics = {}
        if (cyc % path_kernel_metrics_freq == 0) or (cyc == 1) or (cyc == cycles):
            try:
                from ..analysis.path_analysis import compute_path_kernel_metrics
                # Use test_loader if available, otherwise val_loader
                test_loader_for_metrics = test_loader if test_loader is not None else val_loader
                path_kernel_metrics = compute_path_kernel_metrics(
                    model,
                    train_loader,
                    test_loader_for_metrics,
                    mode="routing_gain",
                    k=48,
                    max_samples=5000,
                    device=device,
                )
            except Exception as e:
                print(f"  [path_kernel_metrics] Warning: Failed at cycle {cyc}: {e}")

        # Path metrics removed - no longer computing standard path metrics

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

        # Run path analysis at intervals (start, end, and every N cycles)
        checkpoint_metrics = {}
        if path_analysis_out_dir is not None and ((cyc % path_analysis_freq == 0) or (cyc == 1) or (cyc == cycles)):
            try:
                from ..analysis.path_analysis import (
                    run_full_analysis_at_checkpoint,
                    path_embedding,
                    compute_path_shapley_metrics,
                    compute_centroid_drift_metrics,
                    compute_lineage_sankey_metrics,
                )
                from ..analysis.path_kernel import compute_path_kernel_eigs
                
                run_full_analysis_at_checkpoint(
                    model=model,
                    val_loader=val_loader,
                    out_dir=path_analysis_out_dir,
                    step_tag=f"cycle_{cyc:04d}",
                    kernel_k=48,
                    kernel_mode="routing_gain",
                    include_input_in_kernel=True,
                    block_size=1024,
                    max_samples_kernel=5000,  # Limit samples for speed
                    max_samples_embed=5000,
                )
                
                # Collect checkpoint embeddings for lineage/centroid metrics
                try:
                    Epack = path_embedding(
                        model, val_loader, device=device, mode="routing_gain",
                        normalize=True, max_samples=5000
                    )
                    E_tensor = Epack["E"]
                    # Safely get y_data - avoid boolean evaluation of tensors
                    y_data = Epack.get("labels")
                    if y_data is None:
                        y_data = Epack.get("y")
                    
                    checkpoint_embeddings.append((cyc, E_tensor.detach().cpu()))
                    
                    # Compute Path-Shapley if we have kernel eigenvectors
                    if y_data is not None and (not isinstance(y_data, torch.Tensor) or y_data.numel() > 0):
                        try:
                            kern = compute_path_kernel_eigs(
                                model, val_loader, device=device, mode="routing_gain",
                                include_input=True, k=24, n_iter=30, block_size=1024,
                                max_samples=5000, verbose=False
                            )
                            if "evecs" in kern:
                                evecs = kern["evecs"].detach().cpu().numpy()
                                y_np = y_data.numpy() if isinstance(y_data, torch.Tensor) else y_data
                                top_m = min(24, evecs.shape[1])
                                scores = evecs[:, :top_m]
                                shapley_metrics = compute_path_shapley_metrics(scores, y_np)
                                checkpoint_metrics.update(shapley_metrics)
                        except Exception as e:
                            print(f"  [checkpoint_metrics] Path-Shapley failed: {e}")
                except Exception as e:
                    print(f"  [checkpoint_metrics] Embedding collection failed: {e}")
                
                print(f"  [path_analysis] Completed for cycle {cyc}")
            except Exception as e:
                print(f"  [path_analysis] Warning: Failed at cycle {cyc}: {e}")
        
        # Compute centroid drift and lineage metrics from collected embeddings
        if len(checkpoint_embeddings) >= 2:
            try:
                from ..analysis.path_analysis import (
                    compute_centroid_drift_metrics,
                    compute_lineage_sankey_metrics,
                )
                E_time = [E for _, E in checkpoint_embeddings]
                drift_metrics = compute_centroid_drift_metrics(E_time, k=8)
                lineage_metrics = compute_lineage_sankey_metrics(E_time, k=8)
                checkpoint_metrics.update(drift_metrics)
                checkpoint_metrics.update(lineage_metrics)
            except Exception as e:
                print(f"  [checkpoint_metrics] Drift/Lineage computation failed: {e}")
        
        if checkpoint_metrics:
            checkpoint_metrics_history.append({
                "cycle": cyc,
                **checkpoint_metrics
            })

        hist_entry = {
            "cycle": cyc,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss,   "val_acc": va_acc,
            "mask_churn_layers": churn_layers,
            "pruning": pruning_stats,
        }
        # Add path kernel metrics
        if path_kernel_metrics:
            hist_entry.update(path_kernel_metrics)
        history.append(hist_entry)
        if test_loss is not None:
            history[-1]["test_loss"] = test_loss
            history[-1]["test_acc"] = test_acc
        
        if early_stopped:
            break
    
    # Return checkpoint metrics history for saving
    return history, checkpoint_metrics_history
