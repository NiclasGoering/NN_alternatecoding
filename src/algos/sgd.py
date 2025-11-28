from __future__ import annotations
import torch
from torch import optim

# Local utility functions (no longer importing from utils.metrics)
def mse_loss(yhat, y):
    """Mean squared error loss."""
    return torch.mean((yhat - y) ** 2)

def accuracy_from_logits(yhat, y):
    """Compute accuracy from logits."""
    return torch.sign(yhat).eq(y).float().mean()

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

def train_sgd_relu(model, train_loader, val_loader, config, test_loader=None):
    device = config["device"]
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["lr_w"])
    
    # Computation frequency controls (for speed optimization)
    logging_cfg = config.get("logging", {})
    effective_rank_freq = int(logging_cfg.get("effective_rank_every_n_cycles", 1))
    path_metrics_freq = int(logging_cfg.get("path_metrics_every_n_epochs", 1))  # Compute path metrics every N epochs (set to 1 for every epoch)
    path_kernel_metrics_freq = int(logging_cfg.get("path_kernel_metrics_every_n_epochs", 1))  # Compute path kernel metrics every N epochs
    path_analysis_freq = path_kernel_metrics_freq  # Use same frequency as path kernel metrics
    path_analysis_out_dir = config.get("path_analysis_out_dir", None)  # Output directory for path analysis plots
    path_kernel_metrics_freq = int(logging_cfg.get("path_kernel_metrics_every_n_epochs", 1))  # Compute path kernel metrics every N epochs

    # Force plain ReLU
    model.use_gates = False
    model.gates = None

    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    history = []
    prev_masks_path_metrics = None  # For path metrics (from path_loader, full dataset)
    prev_path_hashes = None  # Track path hashes for confident churn
    
    import time as time_module
    epoch_start_time = time_module.time()
    print(f"[sgd_relu] Starting training: {epochs} epochs")

    for ep in range(1, epochs+1):
        if ep % 10 == 0 or ep == 1:
            elapsed = time_module.time() - epoch_start_time
            print(f"[sgd_relu] Epoch {ep}/{epochs} (elapsed: {elapsed:.1f}s, avg: {elapsed/ep:.2f}s/epoch)")
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            loss = mse_loss(yhat, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        tr_acc, tr_loss = _eval(model, train_loader, device)
        va_acc, va_loss = _eval(model, val_loader, device)
        
        # Compute effective ranks - SVD is expensive, compute less frequently
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
                    mode="routing",  # ReLU uses routing mode
                    k=48,
                    max_samples=5000,
                    device=device,
                )
            except Exception as e:
                print(f"  [path_kernel_metrics] Warning: Failed at epoch {ep}: {e}")
        
        # Path metrics removed - no longer computing standard path metrics
        
        # Early stopping check
        early_stopped = False
        test_loss = None
        test_acc = None
        if test_loader is not None:
            test_acc, test_loss = _eval(model, test_loader, device)
            if test_loss < 0.01:
                print(f"Early stopping: test loss {test_loss:.6f} < 0.01")
                early_stopped = True
        
        if va_loss < 0.01:
            print(f"Early stopping: validation loss {va_loss:.6f} < 0.01")
            early_stopped = True

        # Run path analysis at intervals (start, end, and every N epochs)
        # Note: sgd_relu doesn't use gates, so path analysis will use routing mode (binary)
        if path_analysis_out_dir is not None and ((ep % path_analysis_freq == 0) or (ep == 1) or (ep == epochs)):
            try:
                from ..analysis.path_analysis import run_full_analysis_at_checkpoint
                run_full_analysis_at_checkpoint(
                    model=model,
                    val_loader=val_loader,
                    out_dir=path_analysis_out_dir,
                    step_tag=f"epoch_{ep:04d}",
                    kernel_k=48,
                    kernel_mode="routing",  # sgd_relu doesn't have gates, so use routing mode
                    include_input_in_kernel=True,
                    block_size=1024,
                    max_samples_kernel=5000,  # Limit samples for speed
                    max_samples_embed=5000,
                )
                print(f"  [path_analysis] Completed for epoch {ep}")
            except Exception as e:
                print(f"  [path_analysis] Warning: Failed at epoch {ep}: {e}")

        hist_entry = {
            "epoch": ep, 
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
            "effective_rank_layers": eff_ranks
        }
        if test_loss is not None:
            hist_entry.update({"test_loss": test_loss, "test_acc": test_acc})
        
        # Add path kernel metrics
        if path_kernel_metrics:
            hist_entry.update(path_kernel_metrics)
        history.append(hist_entry)
        
        if early_stopped:
            break
    return history

@torch.no_grad()
def _eval(model, loader, device):
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
def compute_effective_ranks(model, loader, device):
    """Compute effective rank for each layer's hidden activations."""
    model.eval()
    activations_list = None
    for xb, _ in loader:
        xb = xb.to(device)
        _, cache = model(xb, return_cache=True)
        batch_activations = [h.cpu() for h in cache["h"]]
        if activations_list is None:
            activations_list = [act.clone() for act in batch_activations]
        else:
            for l in range(len(activations_list)):
                activations_list[l] = torch.cat([activations_list[l], batch_activations[l]], dim=0)
        break  # Use first batch for efficiency
    if activations_list is None:
        return []
    return [effective_rank(act) for act in activations_list]
