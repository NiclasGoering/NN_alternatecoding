from __future__ import annotations
import torch
from torch import optim
# Local utility functions (no longer importing from utils.metrics)

def mse_loss(yhat, y):
    """Mean squared error loss."""
    return torch.mean((yhat - y) ** 2)

def effective_rank(act):
    """Compute effective rank of activations using SVD."""
    if act.numel() == 0:
        return 0.0
    act_2d = act.view(act.shape[0], -1)  # Flatten to (batch, features)
    if act_2d.shape[1] == 0:
        return 0.0
    try:
        U, s, _ = torch.linalg.svd(act_2d, full_matrices=False)
        s = s[s > 1e-8]  # Filter near-zero singular values
        if len(s) == 0:
            return 0.0
        p = s / s.sum()
        p = p[p > 1e-12]  # Avoid log(0)
        entropy = -(p * torch.log(p)).sum()
        return torch.exp(entropy).item()
    except:
        return 0.0

def slope_budget(slopes):
    """Compute slope budget (sum of slopes)."""
    if not slopes:
        return 0.0, []
    # Handle case where slopes might be a tuple or have unexpected structure
    if isinstance(slopes, tuple):
        slopes = list(slopes)
    total = 0.0
    per_layer = []
    for s in slopes:
        if isinstance(s, torch.Tensor):
            total += s.sum().item()
            per_layer.append(s.sum().item())
        else:
            # If it's not a tensor, try to convert or skip
            per_layer.append(0.0)
    return total, per_layer

def slope_entropy(slopes):
    """Compute slope entropy."""
    if not slopes:
        return 0.0, []
    # Handle case where slopes might be a tuple or have unexpected structure
    if isinstance(slopes, tuple):
        slopes = list(slopes)
    total_entropy = 0.0
    per_layer = []
    for s in slopes:
        if not isinstance(s, torch.Tensor):
            per_layer.append(0.0)
            continue
        s_flat = s.flatten()
        s_flat = s_flat[s_flat > 1e-12]
        if len(s_flat) == 0:
            per_layer.append(0.0)
            continue
        p = s_flat / s_flat.sum()
        p = p[p > 1e-12]
        if len(p) == 0:
            per_layer.append(0.0)
            continue
        entropy = -(p * torch.log(p)).sum().item()
        total_entropy += entropy
        per_layer.append(entropy)
    return total_entropy, per_layer

def slope_deviation(slopes):
    """Compute slope deviation (std of slopes)."""
    if not slopes:
        return []
    # Handle case where slopes might be a tuple or have unexpected structure
    if isinstance(slopes, tuple):
        slopes = list(slopes)
    per_layer = []
    for s in slopes:
        if isinstance(s, torch.Tensor):
            per_layer.append(s.std().item())
        else:
            per_layer.append(0.0)
    return per_layer

def gate_stats(slopes):
    """Compute gate statistics."""
    if not slopes:
        return {}
    # Handle case where slopes might be a tuple or have unexpected structure
    if isinstance(slopes, tuple):
        slopes = list(slopes)
    stats = {}
    tensor_slopes = [s.flatten() for s in slopes if isinstance(s, torch.Tensor)]
    if tensor_slopes:
        all_slopes = torch.cat(tensor_slopes)
        if len(all_slopes) > 0:
            stats["mean"] = all_slopes.mean().item()
            stats["std"] = all_slopes.std().item()
            stats["min"] = all_slopes.min().item()
            stats["max"] = all_slopes.max().item()
    return stats
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
    path_metrics_freq = int(logging_cfg.get("path_metrics_every_n_epochs", 1))  # Compute path metrics every N epochs (set to 1 for every epoch)
    effective_rank_freq = int(logging_cfg.get("effective_rank_every_n_cycles", 1))
    path_kernel_metrics_freq = int(logging_cfg.get("path_kernel_metrics_every_n_epochs", 1))  # Compute path kernel metrics every N epochs
    path_analysis_freq = path_kernel_metrics_freq  # Use same frequency as path kernel metrics
    path_analysis_out_dir = config.get("path_analysis_out_dir", None)  # Output directory for path analysis plots

    history = []
    prev_masks = None  # For regular churn (from train_loader, first batch only)
    prev_masks_path_metrics = None  # For path metrics (from path_loader, full dataset)
    prev_path_hashes = None  # Track path hashes for confident churn
    # Checkpoint embeddings for lineage/centroid metrics
    checkpoint_embeddings = []  # List of (epoch, embedding_tensor) tuples
    checkpoint_metrics_history = []  # List of checkpoint metrics dicts
    
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
            
            # Compute path kernel metrics (effective rank, variance explained, etc.)
            path_kernel_metrics = {}
            if (ep % path_kernel_metrics_freq == 0) or (ep == 1) or (ep == epochs):
                try:
                    from ..analysis.path_analysis import compute_path_kernel_metrics
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
                    print(f"  [path_kernel_metrics] Warning: Failed at epoch {ep}: {e}")
            
            # Path metrics removed - no longer computing standard path metrics
            path_metrics = None
        else:
            B_total=B_layers=H_total=H_layers=deltas=None
            gate_stats_dict = None
            # Effective rank uses SVD - compute less frequently for speed
            if (ep % effective_rank_freq == 0) or (ep == 1) or (ep == epochs):
                eff_ranks = compute_effective_ranks(model, train_loader, device)
            else:
                eff_ranks = None
            
            # Compute path kernel metrics (effective rank, variance explained, etc.)
            path_kernel_metrics = {}
            if (ep % path_kernel_metrics_freq == 0) or (ep == 1) or (ep == epochs):
                try:
                    from ..analysis.path_analysis import compute_path_kernel_metrics
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
                    print(f"  [path_kernel_metrics] Warning: Failed at epoch {ep}: {e}")
            
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

        # Run path analysis at intervals (start, end, and every N epochs)
        checkpoint_metrics = {}
        if path_analysis_out_dir is not None and ((ep % path_analysis_freq == 0) or (ep == 1) or (ep == epochs)):
            try:
                from ..analysis.path_analysis import (
                    run_full_analysis_at_checkpoint,
                    path_embedding,
                    compute_path_shapley_metrics,
                    compute_centroid_drift_metrics,
                    compute_lineage_sankey_metrics,
                )
                from ..analysis.path_kernel import compute_path_kernel_eigs
                
                # Use routing_gain mode if gates are enabled, otherwise routing mode
                kernel_mode = "routing_gain" if getattr(model, "use_gates", False) else "routing"
                run_full_analysis_at_checkpoint(
                    model=model,
                    val_loader=val_loader,
                    out_dir=path_analysis_out_dir,
                    step_tag=f"epoch_{ep:04d}",
                    kernel_k=48,
                    kernel_mode=kernel_mode,
                    include_input_in_kernel=True,
                    block_size=1024,
                    max_samples_kernel=5000,  # Limit samples for speed
                    max_samples_embed=5000,
                )
                
                # Collect checkpoint embeddings for lineage/centroid metrics
                try:
                    Epack = path_embedding(
                        model, val_loader, device=device, mode=kernel_mode,
                        normalize=True, max_samples=5000
                    )
                    E_tensor = Epack["E"]
                    # Safely get y_data - avoid boolean evaluation of tensors
                    y_data = Epack.get("labels")
                    if y_data is None:
                        y_data = Epack.get("y")
                    
                    checkpoint_embeddings.append((ep, E_tensor.detach().cpu()))
                    
                    # Compute Path-Shapley if we have kernel eigenvectors
                    if y_data is not None and (not isinstance(y_data, torch.Tensor) or y_data.numel() > 0):
                        try:
                            kern = compute_path_kernel_eigs(
                                model, val_loader, device=device, mode=kernel_mode,
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
                
                print(f"  [path_analysis] Completed for epoch {ep}")
            except Exception as e:
                print(f"  [path_analysis] Warning: Failed at epoch {ep}: {e}")
        
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
                "epoch": ep,
                **checkpoint_metrics
            })

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

