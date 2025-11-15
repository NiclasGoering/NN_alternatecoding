from __future__ import annotations
import torch

def prune_identity_like(model, tau: float):
    """
    Mark units as identity-like if |a_plus-1|<=tau and |a_minus-1|<=tau.
    We *clamp* such slopes to 1 (don’t resize the Linear layer in-place—it keeps code simple and analyzable).
    """
    if not getattr(model, "use_gates", False):
        return []
    stats = []
    for gate in model.gates:
        a_p, a_m = gate.a_plus, gate.a_minus
        keep = (torch.abs(a_p - 1.0) > tau) | (torch.abs(a_m - 1.0) > tau)
        n_total = a_p.numel()
        n_pruned = int((~keep).sum().item())
        with torch.no_grad():
            a_p[~keep] = 1.0
            a_m[~keep] = 1.0
        stats.append((n_pruned, int(n_total)))
    return stats
