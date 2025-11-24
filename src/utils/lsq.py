# src/utils/lsq.py
import torch

@torch.no_grad()
def stable_lstsq(
    Phi,                 # (P, D)
    y,                   # (P,) or (P,1)
    ridge=0.0,
    id_pull=None,        # lambda for ||a-1||^2, or None
    a0=None,             # previous a for trust region
    alpha=1.0,           # step fraction in (0,1]
    max_backtracks=5,
    churn_ok=None,
    churn_fn=None,
    clamp=None,          # (lo, hi)
):
    """
    Solve (Phi^T Phi + ridge I + lambda_id I) a = Phi^T y + lambda_id * 1
    with a numerically-stable augmented least-squares solve.

    - Column normalization for conditioning.
    - Ridge and identity-pull (quadratic prior around 1).
    - Trust region damping toward a0.
    - Optional backtracking on a user-supplied churn metric.
    - Optional clamping of the solution.

    Returns: a_candidate (D,)
    """
    device = Phi.device
    P, D = Phi.shape
    y = y.reshape(P, 1)

    # Column normalize for conditioning
    col_norm = torch.linalg.norm(Phi, dim=0).clamp_min(1e-8)
    Phi_n = Phi / col_norm

    # Augmented system for ridge + identity pull
    I = torch.eye(D, device=device)
    A = Phi_n
    b = y
    if ridge and ridge > 0.0:
        A = torch.cat([A, (ridge**0.5) * I], dim=0)
        b = torch.cat([b, torch.zeros(D, 1, device=device)], dim=0)
    if id_pull and id_pull > 0.0:
        A = torch.cat([A, (id_pull**0.5) * I], dim=0)
        b = torch.cat([b, (id_pull**0.5) * torch.ones(D, 1, device=device)], dim=0)

    # Solve in float64 for stability
    sol = torch.linalg.lstsq(A.to(torch.float64), b.to(torch.float64), rcond=None)
    a_scaled = sol.solution.squeeze(-1).to(torch.float32)   # (D,)
    a_ls = a_scaled / col_norm                              # unscale

    # Trust region
    a_candidate = a_ls if a0 is None else (a0 + alpha * (a_ls - a0))

    # Clamp
    if clamp is not None:
        lo, hi = clamp
        a_candidate = a_candidate.clamp(min=lo, max=hi)

    # Optional backtracking on churn
    if churn_ok is not None and churn_fn is not None and a0 is not None:
        bt = 0
        a_try = a_candidate
        while bt < max_backtracks:
            if churn_fn(a_try) <= churn_ok:
                break
            a_try = a0 + 0.5 * (a_try - a0)
            bt += 1
        a_candidate = a_try

    return a_candidate
