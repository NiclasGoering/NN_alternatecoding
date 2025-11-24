from __future__ import annotations
import torch
from torch import optim
from ..utils.metrics import mse_loss, accuracy_from_logits, effective_rank

def train_sgd_relu(model, train_loader, val_loader, config, test_loader=None):
    device = config["device"]
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["lr_w"])
    
    # Computation frequency controls (for speed optimization)
    logging_cfg = config.get("logging", {})
    effective_rank_freq = int(logging_cfg.get("effective_rank_every_n_cycles", 1))

    # Force plain ReLU
    model.use_gates = False
    model.gates = None

    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    history = []
    
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

        if test_loss is not None:
            history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                            "val_loss": va_loss, "val_acc": va_acc, "test_loss": test_loss, "test_acc": test_acc,
                            "effective_rank_layers": eff_ranks})
        else:
            history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                            "val_loss": va_loss, "val_acc": va_acc,
                            "effective_rank_layers": eff_ranks})
        
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
