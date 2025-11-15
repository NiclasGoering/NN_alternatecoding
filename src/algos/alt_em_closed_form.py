import torch
from .pruning import prune_identity_like
from ..utils.metrics import slope_budget, slope_entropy, slope_deviation

@torch.no_grad()
def _forward_cache(model, X):
    yhat, cache = model(X, return_cache=True)
    return yhat, cache["u"], cache["z"], cache["h"], cache["h_last"]


# utils/lsq.py
import torch

@torch.no_grad()
def stable_lstsq(Phi, y, ridge=0.0, id_pull=None, a0=None, alpha=1.0, max_backtracks=5, churn_ok=None, churn_fn=None, clamp=None):
    """
    Solve (Phi^T Phi + ridge I + lambda_id I) a = Phi^T y + lambda_id * 1
    using QR/SVD-based least squares, with:
      - adaptive ridge (if QR warns)
      - optional identity-pull (id_pull = lambda_id)
      - interpolated update: a <- (1-alpha) a0 + alpha a_ls
      - optional mask-churn backtracking via churn_fn(new_a) -> churn_value
      - optional clamping of a to [clamp_min, clamp_max]
    """
    device = Phi.device
    P, D = Phi.shape
    y = y.reshape(P, -1)  # (P,1) for 1D output

    # Column normalize Phi to improve conditioning; keep scale vector to re-scale back
    col_norm = torch.linalg.norm(Phi, dim=0).clamp_min(1e-8)
    Phi_n = Phi / col_norm

    # Form augmented system for ridge + identity-pull in LS form
    # Minimize ||Phi_n a' - y||^2 + ridge ||a'||^2 + lambda_id ||a' - 1'||^2
    I = torch.eye(D, device=device)
    aug_A = Phi_n
    aug_b = y
    if ridge > 0.0:
        aug_A = torch.cat([aug_A, (ridge**0.5) * I], dim=0)
        aug_b = torch.cat([aug_b, torch.zeros(D, 1, device=device)], dim=0)
    if id_pull is not None and id_pull > 0.0:
        aug_A = torch.cat([aug_A, (id_pull**0.5) * I], dim=0)
        ones = torch.ones(D, 1, device=device)
        aug_b = torch.cat([aug_b, (id_pull**0.5) * ones], dim=0)

    # Solve via lstsq (QR) in float64 for numerical stability
    sol = torch.linalg.lstsq(aug_A.to(torch.float64), aug_b.to(torch.float64), rcond=None)
    a_ls_scaled = sol.solution.squeeze(-1).to(torch.float32)  # (D,)

    # Unscale back
    a_ls = a_ls_scaled / col_norm

    # Interpolate with trust region
    if a0 is not None:
        a_candidate = a0 + alpha * (a_ls - a0)
    else:
        a_candidate = a_ls

    # Clamp slopes to a safe range (optional)
    if clamp is not None:
        lo, hi = clamp
        a_candidate = a_candidate.clamp(min=lo, max=hi)

    # Optional backtracking if churn is too high
    if churn_ok is not None and churn_fn is not None and a0 is not None:
        bt = 0
        a_try = a_candidate
        while bt < max_backtracks:
            churn = churn_fn(a_try)
            if churn <= churn_ok:
                break
            # shrink step
            a_try = a0 + 0.5 * (a_try - a0)
            bt += 1
        a_candidate = a_try

    return a_candidate




@torch.no_grad()
def _compute_backprop_rows(model, us, zs, hs):
    P = hs[0].shape[0]; L = len(hs)
    w_out = model.readout.weight.detach().clone().t()          # (d_L, 1)
    B = w_out.squeeze(1).unsqueeze(0).expand(P, -1)            # (P, d_L)
    B_list = [None]*L
    B_list[-1] = B.clone()
    for l in reversed(range(L-1)):
        W_next = model.linears[l+1].weight.detach().clone()    # (d_{l+1}, d_l)
        a_p = model.gates[l+1].a_plus.detach()
        a_m = model.gates[l+1].a_minus.detach()
        u_next = us[l+1]                                       # (P, d_{l+1})
        s = torch.where(u_next >= 0, a_p.unsqueeze(0), a_m.unsqueeze(0))
        dudu = B * s                                           # (P, d_{l+1})
        B = torch.matmul(dudu, W_next)                         # (P, d_l)
        B_list[l] = B.clone()
    return B_list


# in your training file, replace _A_step_closed_form with:

from ..utils.lsq import stable_lstsq
@torch.no_grad()
def _A_step_closed_form(model, X, y, us, zs, hs, lam_id, ridge_a, alpha=0.25, clamp=(0.05, 5.0)):
    """
    Solve per-layer slopes with proper ridge + identity pull on LHS,
    column scaling and damping. Uses the 'r + Φ a_cur' trick for the RHS.
    """
    device = X.device
    yhat, *_ = _forward_cache(model, X)
    r = (y - yhat).squeeze(1)                                  # (P,)
    B_list = _compute_backprop_rows(model, us, zs, hs)

    for l in range(len(hs)):
        a_p, a_m = model.gates[l].a_plus, model.gates[l].a_minus
        u = us[l]; B = B_list[l]
        u_plus  = torch.relu(u)
        u_minus = torch.relu(-u)
        Phi = torch.cat([B*u_plus, -(B*u_minus)], dim=1).to(torch.float64)  # (P, 2 d_l)

        # Column scaling for conditioning
        col_norm = Phi.norm(dim=0) + 1e-12
        Phi_n = Phi / col_norm

        a_cur = torch.cat([a_p, a_m], dim=0).to(torch.float64)
        rhs = (r + (Phi @ a_cur)).to(torch.float64)            # equals (y - constant)

        # IMPORTANT: put identity-pull (lam_id) on the LHS too
        A = Phi_n.T @ Phi_n + (ridge_a + lam_id) * torch.eye(Phi_n.size(1), device=device, dtype=torch.float64)
        b = Phi_n.T @ rhs + lam_id * (torch.ones_like(a_cur, dtype=torch.float64) / col_norm)

        a_new_scaled = torch.linalg.solve(A, b)
        a_new = a_new_scaled / col_norm

        # Damp & clamp
        a_upd = (1.0 - alpha) * a_cur + alpha * a_new
        a_upd = torch.clamp(a_upd, clamp[0], clamp[1]).to(a_p.dtype)

        d = u.shape[1]
        a_p.copy_(a_upd[:d])
        a_m.copy_(a_upd[d:])




@torch.no_grad()
def _W_step_closed_form(model, X, y, us, zs, hs, ridge_w):
    L = len(hs)
    # diag of “local derivative” m = a_plus if u>=0 else a_minus
    Mdiag = []
    for l in range(L):
        a_p = model.gates[l].a_plus.detach()
        a_m = model.gates[l].a_minus.detach()
        u   = us[l]
        m = torch.where(u >= 0, a_p.unsqueeze(0), a_m.unsqueeze(0))
        Mdiag.append(m)

    B_list = _compute_backprop_rows(model, us, zs, hs)

    for l in range(L):
        H_prev = X if l == 0 else hs[l-1]                      # (P, d_in)
        d_in = H_prev.shape[1]; d_out = hs[l].shape[1]
        B = B_list[l]; m = Mdiag[l]
        G = B * m                                              # (P, d_out)
        H = H_prev                                             # (P, d_in)
        yv = y.squeeze(1)                                      # (P,)

        # Normal equations via eigendecomposition (toy scale)
        K_in  = H.t() @ H + 1e-8*torch.eye(d_in,  device=X.device)
        K_out = G.t() @ G + 1e-8*torch.eye(d_out, device=X.device)
        S = G.t() @ (yv.unsqueeze(1) * H)                      # (d_out x d_in)

        e_out, Uo = torch.linalg.eigh(K_out)
        e_in,  Ui = torch.linalg.eigh(K_in)
        St = Uo.T @ S @ Ui                                     # basis change
        denom = (e_out.unsqueeze(1) * e_in.unsqueeze(0)) + ridge_w
        Wt = St / denom
        Wnew = Uo @ Wt @ Ui.T
        model.linears[l].weight.copy_(Wnew)

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
    if prev_masks is None: return [0.0]*len(cur_masks)
    return [(prev_masks[l] != cur_masks[l]).float().mean().item() for l in range(len(cur_masks))]


def train_alt_em_closed_form(model, full_train_loader, val_loader, config):
    device = config["device"]
    t, r, p = config["training"], config["regularization"], config["pruning"]

    cycles     = int(t.get("cycles", 50))
    ridge_a    = float(t.get("ridge_a", 1e-2))
    ridge_w    = float(t.get("ridge_w", 1e-3))
    alpha      = float(t.get("alpha", 0.2))           # trust region for slope updates
    churn_tgt  = float(t.get("churn_target", 0.01))
    lam_id     = float(r.get("lambda_identity", 0.0)) if r.get("identity_reg", False) else 0.0

    model.to(device).eval()
    Xfull, yfull = next(iter(full_train_loader))
    Xfull, yfull = Xfull.to(device), yfull.to(device)

    history = []
    prev_masks = None

    for cyc in range(1, cycles+1):
        # cache with current params
        yhat, us, zs, hs, _ = _forward_cache(model, Xfull)

        # A-step (stable closed form with trust region)
        if getattr(model, "use_gates", False):
            _A_step_closed_form_stable(model, Xfull, yfull, us, zs, hs,
                                       lam_id=lam_id, ridge_a=ridge_a,
                                       alpha=alpha, churn_target=churn_tgt)

        # W-step: last layer LS, earlier layers (optional) a couple of small SGD steps
        _W_last_layer_ls(model, Xfull, yfull, ridge_w=ridge_w)

        # recompute masks and churn
        cur_masks = dataset_masks(model, full_train_loader, device)
        churn_layers = mask_churn(prev_masks, cur_masks)
        prev_masks = cur_masks

        # pruning after masks settle a bit
        pruning_stats = []
        if getattr(model, "use_gates", False) and p.get("enabled", False) and cyc >= int(p.get("apply_after_cycles", 2)):
            pruning_stats = prune_identity_like(model, tau=float(p.get("tau", 0.01)))

        # evaluate
        tr_acc, tr_loss = _eval(model, full_train_loader, device)
        va_acc, va_loss = _eval(model, val_loader, device)

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
