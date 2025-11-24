# src/algos/alt_em_closed_form.py
from __future__ import annotations
import torch
from .pruning import prune_identity_like
from ..utils.metrics import slope_budget, slope_entropy, slope_deviation
from ..utils.lsq import stable_lstsq

# ---------------------- utilities ----------------------

@torch.no_grad()
def _forward_cache(model, X):
    yhat, cache = model(X, return_cache=True)
    return yhat, cache["u"], cache["z"], cache["h"], cache["h_last"]

@torch.no_grad()
def _batch_mse(model, X, y):
    yhat = model(X)
    return torch.mean((yhat - y)**2).item()

@torch.no_grad()
def _compute_backprop_rows(model, us, zs, hs):
    P = hs[0].shape[0]; L = len(hs)
    w_out = model.readout.weight.detach().clone().t()           # (d_L,1)
    B = w_out.squeeze(1).unsqueeze(0).expand(P, -1)             # (P,d_L)
    B_list = [None]*L
    B_list[-1] = B.clone()
    for l in reversed(range(L-1)):
        W_next = model.linears[l+1].weight.detach().clone()     # (d_{l+1}, d_l)
        a_p = model.gates[l+1].a_plus.detach()
        a_m = model.gates[l+1].a_minus.detach()
        u_next = us[l+1]                                        # (P, d_{l+1})
        s = torch.where(u_next >= 0, a_p.unsqueeze(0), a_m.unsqueeze(0))
        B = (B * s) @ W_next                                    # (P, d_l)
        B_list[l] = B.clone()
    return B_list

@torch.no_grad()
def _W_readout_ls(model, H_last, y, ridge_out=1e-3, beta_out=1.0, eps=1e-8):
    """ Ridge LS readout fit with damping. """
    device = H_last.device
    P, dL = H_last.shape
    H = H_last.to(torch.float64)
    y64 = y.to(torch.float64)
    K = H.t() @ H + (ridge_out + eps) * torch.eye(dL, device=device, dtype=torch.float64)
    b = H.t() @ y64
    w_ls = torch.linalg.solve(K, b).t().to(torch.float32)       # (1,dL)
    w_old = model.readout.weight.detach()
    w_new = (1.0 - beta_out) * w_old + beta_out * w_ls
    model.readout.weight.copy_(w_new)

# ---------------------- W-step (Sylvester) ----------------------

@torch.no_grad()
def _solve_weight_block_sylvester(H, G, y, ridge_w, eps=1e-8):
    """ Solve for W in min || y - G W H^T ||^2 + ridge ||W||_F^2 """
    device = H.device
    dtype  = torch.float64
    K_in  = (H.t() @ H).to(dtype) + eps * torch.eye(H.shape[1], device=device, dtype=dtype)
    K_out = (G.t() @ G).to(dtype) + eps * torch.eye(G.shape[1], device=device, dtype=dtype)
    YH = (y.unsqueeze(1) * H).to(dtype)                          # (P,d_in)
    S  = (G.t().to(dtype) @ YH)                                  # (d_out,d_in)
    e_out, Uo = torch.linalg.eigh(K_out)
    e_in,  Ui = torch.linalg.eigh(K_in)
    St = Uo.t() @ S @ Ui
    denom = (e_out.unsqueeze(1) * e_in.unsqueeze(0)) + ridge_w
    Wt = St / denom
    return (Uo @ Wt @ Ui.t()).to(H.dtype)

# ---------------------- A-step per-layer with backtracking ----------------------

@torch.no_grad()
def _A_step_layer_backtrack(
    model, l, X, y, lam_id, ridge_a, beta_init, beta_min, slope_clamp,
    improve_tol, max_backtracks, ridge_boost=10.0
):
    """
    Update slopes on one layer with backtracking to ensure loss decreases.
    """
    # cache and design for this layer
    yhat, us, zs, hs, _ = _forward_cache(model, X)
    base_loss = torch.mean((yhat - y)**2).item()
    B_list = _compute_backprop_rows(model, us, zs, hs)
    u = us[l]; B = B_list[l]
    a_p = model.gates[l].a_plus
    a_m = model.gates[l].a_minus
    a_cur = torch.cat([a_p, a_m], dim=0)

    u_plus  = torch.relu(u)
    u_minus = torch.relu(-u)
    Phi = torch.cat([B * u_plus, -(B * u_minus)], dim=1)        # (P,2d_l)
    rhs = (y - yhat).squeeze(1) + (Phi @ a_cur)                 # (P,)

    # Use stable_lstsq which handles augmentation internally
    # Pass original Phi and rhs, let it handle ridge and id_pull
    a_ls = stable_lstsq(Phi, rhs, ridge=ridge_a, id_pull=lam_id if lam_id > 0.0 else None)
    a_ls = a_ls.to(a_cur.dtype)

    # backtracking on beta
    beta = beta_init
    best_loss = base_loss
    accepted = False
    ridge_here = ridge_a

    for _ in range(max_backtracks):
        a_try = (1.0 - beta) * a_cur + beta * a_ls
        a_try = torch.clamp(a_try, slope_clamp[0], slope_clamp[1])

        # apply tentative slopes
        d = u.shape[1]
        a_p_old = a_p.detach().clone()
        a_m_old = a_m.detach().clone()
        a_p.copy_(a_try[:d]); a_m.copy_(a_try[d:])

        # refit readout to be fair, then measure loss
        yhat2, *_, hL2 = _forward_cache(model, X)
        _W_readout_ls(model, hL2, y, ridge_out=1e-3, beta_out=1.0)
        loss_try = _batch_mse(model, X, y)

        if loss_try <= best_loss * (1.0 - improve_tol):
            best_loss = loss_try
            accepted = True
            break
        else:
            # revert and shrink step; for early cycles also boost ridge to tame solves
            a_p.copy_(a_p_old); a_m.copy_(a_m_old)
            beta = max(beta / 2.0, beta_min)
            ridge_here *= ridge_boost

    # if not accepted, keep old slopes (already restored)
    return accepted, best_loss

# ---------------------- W-step per-layer with LM backtracking ----------------------

@torch.no_grad()
def _W_step_layer_backtrack(
    model, l, X, y, ridge_w, beta_init, beta_min, improve_tol, max_backtracks, lm_up=10.0, lm_down=0.5
):
    """
    Update W_l with Sylvester closed form, LM-style ridge adaptation, and backtracking.
    """
    # cache for this layer
    yhat, us, zs, hs, _ = _forward_cache(model, X)
    base_loss = torch.mean((yhat - y)**2).item()

    H_prev = X if l == 0 else hs[l-1]
    a_p = model.gates[l].a_plus.detach()
    a_m = model.gates[l].a_minus.detach()
    z   = (us[l] >= 0)
    m   = torch.where(z, a_p.unsqueeze(0), a_m.unsqueeze(0))     # (P,d_out)
    B_list = _compute_backprop_rows(model, us, zs, hs)
    G = B_list[l] * m
    H = H_prev
    yv = y.squeeze(1)

    W_old = model.linears[l].weight.detach().clone()
    beta = beta_init
    best_loss = base_loss
    lam = ridge_w
    accepted = False

    for _ in range(max_backtracks):
        # closed-form solve with current LM ridge
        W_solved = _solve_weight_block_sylvester(H, G, yv, ridge_w=lam)
        W_try = (1.0 - beta) * W_old + beta * W_solved
        model.linears[l].weight.copy_(W_try)

        # refit readout and evaluate
        yhat2, *_, hL2 = _forward_cache(model, X)
        _W_readout_ls(model, hL2, y, ridge_out=lam, beta_out=1.0)
        loss_try = _batch_mse(model, X, y)

        if loss_try <= best_loss * (1.0 - improve_tol):
            best_loss = loss_try
            accepted = True
            # be a bit bolder next time (reduce LM ridge)
            lam = max(lam * lm_down, 1e-6)
            break
        else:
            # backtrack: shrink step and increase LM ridge
            model.linears[l].weight.copy_(W_old)
            beta = max(beta / 2.0, beta_min)
            lam *= lm_up

    # if not accepted, keep old W (already restored)
    return accepted, best_loss

# ---------------------- masks & churn ----------------------

@torch.no_grad()
def dataset_masks(model, loader, device):
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
    if prev_masks is None:
        return [0.0]*len(cur_masks)
    return [(prev_masks[l] != cur_masks[l]).float().mean().item() for l in range(len(cur_masks))]

# ---------------------- main loop ----------------------

def train_alt_em_closed_form(model, full_train_loader, val_loader, config, test_loader=None):
    """
    Monotone alternating EM with per-layer backtracking and readout refits:
      For each cycle:
        - A-step per layer (1..L) with backtracking; refit readout after each layer.
        - W-step per layer (L..1) with LM backtracking; refit readout after each layer.
        - Final readout refit; prune/eval/log.
    """
    device = config["device"]
    t, r, p = config["training"], config["regularization"], config["pruning"]

    cycles        = int(t.get("cycles", 50))
    ridge_a       = float(t.get("ridge_a", 1e-2))
    ridge_w       = float(t.get("ridge_w", 1e-3))
    beta_a_init   = float(t.get("beta_a", 0.3))
    beta_w_init   = float(t.get("beta_w", 0.25))
    beta_min      = float(t.get("beta_min", 0.05))
    improve_tol   = float(t.get("improve_tol", 1e-4))     # relative improvement required
    max_back      = int(t.get("max_backtracks", 6))
    slope_clamp   = tuple(t.get("slope_clamp", [0.7, 1.4]))
    colnorm       = bool(t.get("colnorm", True))
    adapt_ridge   = bool(t.get("adapt_ridge", True))
    lm_up         = float(t.get("lm_up", 10.0))
    lm_down       = float(t.get("lm_down", 0.5))
    ridge_out     = float(t.get("ridge_out", ridge_w))
    lam_id        = float(r.get("lambda_identity", 0.0)) if r.get("identity_reg", False) else 0.0

    model.to(device).eval()
    Xfull, yfull = next(iter(full_train_loader))
    Xfull, yfull = Xfull.to(device), yfull.to(device)

    history = []
    prev_masks = None

    for cyc in range(1, cycles+1):
        # --------- A-step per layer (bottom-up) ----------
        if getattr(model, "use_gates", False):
            L = len(model.linears)
            for l in range(L):
                _A_step_layer_backtrack(
                    model, l, Xfull, yfull,
                    lam_id=lam_id, ridge_a=ridge_a,
                    beta_init=beta_a_init, beta_min=beta_min,
                    slope_clamp=slope_clamp,
                    improve_tol=improve_tol, max_backtracks=max_back,
                    ridge_boost=10.0
                )
                # readout refit after each layer update
                _, _, _, hs, _ = _forward_cache(model, Xfull)
                _W_readout_ls(model, hs[-1], yfull, ridge_out=ridge_out, beta_out=1.0)

        # --------- W-step per layer (top-down) ----------
        L = len(model.linears)
        for l in reversed(range(L)):
            _W_step_layer_backtrack(
                model, l, Xfull, yfull,
                ridge_w=ridge_w,
                beta_init=beta_w_init, beta_min=beta_min,
                improve_tol=improve_tol, max_backtracks=max_back,
                lm_up=lm_up, lm_down=lm_down
            )
            # readout refit after each layer
            _, _, _, hs, _ = _forward_cache(model, Xfull)
            _W_readout_ls(model, hs[-1], yfull, ridge_out=ridge_out, beta_out=1.0)

        # final readout polish this cycle
        _, _, _, hs, _ = _forward_cache(model, Xfull)
        _W_readout_ls(model, hs[-1], yfull, ridge_out=ridge_out, beta_out=1.0)

        # churn & optional pruning
        cur_masks = dataset_masks(model, full_train_loader, device)
        churn_layers = mask_churn(prev_masks, cur_masks)
        prev_masks = cur_masks

        pruning_stats = []
        if getattr(model, "use_gates", False) and p.get("enabled", False) and cyc >= int(p.get("apply_after_cycles", 3)):
            pruning_stats = prune_identity_like(model, tau=float(p.get("tau", 0.01)))

        # eval
        tr_acc, tr_loss = _eval(model, full_train_loader, device)
        va_acc, va_loss = _eval(model, val_loader, device)

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
        if test_loss is not None:
            history[-1]["test_loss"] = test_loss
            history[-1]["test_acc"] = test_acc
        
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
