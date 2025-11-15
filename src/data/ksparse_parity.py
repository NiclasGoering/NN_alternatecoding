from __future__ import annotations
import numpy as np
import torch

def gen_ksparse_parity(d: int, k: int, n: int, x_dist: str = "pm1",
                       label_noise: float = 0.0, seed: int | None = None):
    """Generate a k-sparse parity dataset with labels in {-1, +1}."""
    rng = np.random.default_rng(seed)
    S = rng.choice(d, size=k, replace=False)

    if x_dist == "pm1":
        X = rng.choice([-1.0, 1.0], size=(n, d), replace=True).astype(np.float32)
        bits = (X[:, S] < 0).sum(axis=1) % 2
    elif x_dist == "bernoulli01":
        X = rng.integers(0, 2, size=(n, d), dtype=np.int64).astype(np.float32)
        bits = (X[:, S].sum(axis=1) % 2)
    else:
        raise ValueError(f"Unknown x_dist={x_dist}")

    y = (2.0 * bits - 1.0).astype(np.float32)
    if label_noise > 0.0:
        flips = rng.random(size=n) < label_noise
        y[flips] *= -1.0
    return X, y, S

class ParityDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]
