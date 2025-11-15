from __future__ import annotations
import torch
from torch import optim
from ..utils.metrics import mse_loss, slope_budget, slope_entropy, slope_deviation
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
        batch_masks = [z.bool().cpu() for z in cache["z"]]
        if masks is None:
            masks = [bm.clone() for bm in batch_masks]
        else:
            for l in range(len(masks)):
                masks[l] = torch.cat([masks[l], batch_masks[l]], dim=0)
    return masks

def mask_churn(prev_masks, cur_masks):
    if prev_masks is None: return [0.0]*len(cur_masks)
    churn = []
    P = cur_masks[0].shape[0]
    for l in range(len(cur_masks)):
        changed = (prev_masks[l] != cur_masks[l]).float().mean().item()
        churn.append(changed)
    return churn

def train_alt_em_sgd(model, train_loader, val_loader, config):
    device  = config["device"]
    t, r, p = config["training"], config["regularization"], config["pruning"]
    cycles, lr_w, lr_a = t["cycles"], t["lr_w"], t["lr_a"]
    k_w, k_a = t["steps_w_per_cycle"], t["steps_a_per_cycle"]
    lam = r["lambda_identity"] if r["identity_reg"] else 0.0

    model.to(device)
    history = []
    prev_masks = None

    for cyc in range(1, cycles+1):
        # ---- W-phase --------------------------------------------------------
        model.set_weights_requires_grad(True)
        model.set_gates_requires_grad(False)
        opt_w = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_w)
        for _ in range(k_w):
            xb, yb = next(iter(train_loader))
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
                xb, yb = next(iter(train_loader))
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
            pruning_stats = prune_identity_like(model, tau=p["tau"])

        tr_acc, tr_loss = eval_loader(model, train_loader, device)
        va_acc, va_loss = eval_loader(model, val_loader, device)

        if getattr(model, "use_gates", False):
            B_total, B_layers = slope_budget(model.layer_slopes())
            H_total, H_layers = slope_entropy(model.layer_slopes())
            deltas = slope_deviation(model.layer_slopes())
        else:
            B_total=B_layers=H_total=H_layers=deltas=None

        history.append({
            "cycle": cyc,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss,   "val_acc": va_acc,
            "slope_budget_total": B_total,  "slope_budget_layers": B_layers,
            "slope_entropy_total": H_total, "slope_entropy_layers": H_layers,
            "slope_deviation_layers": deltas,
            "mask_churn_layers": churn_layers,
            "pruning": pruning_stats
        })
    return history
