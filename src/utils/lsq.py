# src/utils/lsq.py
import torch

def stable_lstsq(A: torch.Tensor, b: torch.Tensor, ridge: float = 0.0, rcond: float | None = None) -> torch.Tensor:
    """
    Numerically stable least-squares with optional ridge.
    A: (N, D), b: (N,) or (N, K)
    Returns x: (D,) or (D, K)
    """
    dtype = torch.float64
    A64 = A.to(dtype)
    b64 = b.to(dtype)

    if ridge > 0.0:
        D = A64.shape[1]
        AtA = A64.T @ A64 + ridge * torch.eye(D, device=A.device, dtype=dtype)
        Atb = A64.T @ b64
        x64 = torch.linalg.solve(AtA, Atb)
    else:
        # lstsq is more robust than solve when A is rank-deficient
        x64 = torch.linalg.lstsq(A64, b64, rcond=rcond).solution

    return x64.to(A.dtype)
