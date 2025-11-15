from __future__ import annotations
import torch
from torch import optim
from ..utils.metrics import mse_loss, accuracy_from_logits

def train_sgd_relu(model, train_loader, val_loader, config):
    device = config["device"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr_w"]

    # Force plain ReLU
    model.use_gates = False
    model.gates = None

    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    history = []
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            loss = mse_loss(yhat, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        tr_acc, tr_loss = _eval(model, train_loader, device)
        va_acc, va_loss = _eval(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
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
