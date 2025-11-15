import torch
from .pruning import prune_identity_like
from ..utils.metrics import slope_budget, slope_entropy, slope_deviation

@torch.no_grad()
def _forward_cache(model, X):
    yhat, cache = model(X, return_cache=True)
    return yhat, cache["u"], cache["z"], cache["h"], cache["h_last"]

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

@torch.no_grad()
def _A_step_closed_form(model, X, y, us, zs, hs, lam_id, ridge_a):
    yhat, *_ = _forward_cache(model, X)
    r = (y - yhat).squeeze(1)                                  # (P,)
    B_list = _compute_backprop_rows(model, us, zs, hs)

    for l in range(len(hs)):
        a_p, a_m = model.gates[l].a_plus, model.gates[l].a_minus
        u = us[l]; B = B_list[l]
        u_plus  = torch.relu(u)                                # (P, d_l)
        u_minus = torch.relu(-u)                               # (P, d_l)
        Phi = torch.cat([B*u_plus, -(B*u_minus)], dim=1)       # (P, 2 d_l)
        a_cur = torch.cat([a_p, a_m], dim=0)                   # (2 d_l,)
        rhs = r + (Phi @ a_cur)                                # (P,)
        A = Phi.T @ Phi + ridge_a * torch.eye(2*u.shape[1], device=X.device)
        b = Phi.T @ rhs + lam_id * torch.ones(2*u.shape[1], device=X.device)
        a_new = torch.linalg.solve(A, b)
        a_p.copy_(a_new[:u.shape[1]])
        a_m.copy_(a_new[u.shape[1]:])

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
    cycles, ridge_a, ridge_w = t["cycles"], t["ridge_a"], t["ridge_w"]
    lam = r["lambda_identity"] if r["identity_reg"] else 0.0

    model.to(device); model.eval()
    Xfull, yfull = next(iter(full_train_loader))
    Xfull, yfull = Xfull.to(device), yfull.to(device)

    history = []
    prev_masks = None

    for cyc in range(1, cycles+1):
        # freeze masks at current params
        yhat, us, zs, hs, _ = _forward_cache(model, Xfull)

        # A-step (slopes)
        if getattr(model, "use_gates", False):
            _A_step_closed_form(model, Xfull, yfull, us, zs, hs, lam, ridge_a)

        # W-step (weights)
        _W_step_closed_form(model, Xfull, yfull, us, zs, hs, ridge_w)

        # pruning & churn
        cur_masks = dataset_masks(model, full_train_loader, device)
        churn_layers = mask_churn(prev_masks, cur_masks)
        prev_masks = cur_masks

        pruning_stats = []
        if getattr(model, "use_gates", False) and p["enabled"]:
            pruning_stats = prune_identity_like(model, tau=p["tau"])

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
