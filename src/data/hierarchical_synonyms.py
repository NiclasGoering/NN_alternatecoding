# src/data/hierarchical_synonyms.py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "SynonymTreeDataset",
    "build_synonym_tree_datasets",
    "gen_synonym_tree_split"
]

class SynonymTreeDataset(Dataset):
    """
    Vectorized hierarchy / RHM toy: one active leaf-synonym per sample.
    Input:  one-hot over leaves * synonyms (plus optional distractors).
    Target: y in {-1, +1}, tied to the ROOT child (half children -> +1, half -> -1).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, "X must be (N, d)"
        assert y.ndim == 2 and y.shape[1] == 1, "y must be (N,1) with +/-1"
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def gen_synonym_tree_split(
    L: int,
    m: int,
    s: int,
    N: int,
    root_pos_children=None,
    distractor_dims: int = 0,
    distractor_prob: float = 0.0,
    label_noise: float = 0.0,
    seed: int = 0,
):
    """
    Generate N samples from a depth-L, branching-m hierarchy with s synonyms per leaf.

    - Root child determines label y in {-1,+1}.
    - Path: [root_child, child_l2, ..., child_lL]; leaf index g is base-m index of this path.
    - We then choose one synonym among s and set that single feature to 1.
    - Optionally append 'distractor_dims' features sampled Bernoulli(distractor_prob).

    Returns:
        X: (N, d) float32
        y: (N, 1) float32 with +/-1 labels
        meta: dict with structure info, including total input dim 'd'
    """
    rng = np.random.default_rng(seed)

    n_groups = m ** L          # number of leaf groups (pre-synonym)
    d_leaf   = n_groups * s    # features from leaf synonyms
    d        = d_leaf + int(distractor_dims)

    if root_pos_children is None:
        # First half of root children -> +1 class; second half -> -1 class
        half = max(1, m // 2)
        root_pos_children = set(range(half))
    root_pos_children = set(root_pos_children)
    root_all          = set(range(m))
    root_neg_children = list(root_all.difference(root_pos_children))
    root_pos_children = list(sorted(root_pos_children))

    X = np.zeros((N, d), dtype=np.float32)
    y = np.empty((N, 1), dtype=np.float32)

    for n in range(N):
        # class prior 0.5
        y_n = 1.0 if rng.random() < 0.5 else -1.0
        # choose root child consistent with the label
        if y_n > 0:
            rc = rng.choice(root_pos_children)
        else:
            rc = rng.choice(root_neg_children) if len(root_neg_children) > 0 else rng.integers(0, m)

        # choose remaining children uniformly
        child_idxs = [rc] + [int(rng.integers(0, m)) for _ in range(L - 1)]

        # base-m leaf index
        g = 0
        for idx in child_idxs:
            g = g * m + idx

        # choose one synonym among s
        syn = int(rng.integers(0, s))
        leaf_dim = g * s + syn

        # set active synonym
        X[n, leaf_dim] = 1.0

        # optional distractors
        if distractor_dims > 0 and distractor_prob > 0.0:
            X[n, d_leaf:] = (rng.random(distractor_dims) < distractor_prob).astype(np.float32)

        # label noise
        if rng.random() < label_noise:
            y_n = -y_n
        y[n, 0] = y_n

    meta = {
        "name": "synonym_tree",
        "L": int(L),
        "m": int(m),
        "s": int(s),
        "n_groups": int(n_groups),
        "d_leaf": int(d_leaf),
        "d": int(d),
        "distractor_dims": int(distractor_dims),
        "distractor_prob": float(distractor_prob),
        "root_pos_children": list(root_pos_children),
        "root_neg_children": root_neg_children,
        # mapping from leaf-group g to (start,end) feature indices for synonyms
        "group_to_slice": [[g * s, (g + 1) * s] for g in range(n_groups)],
    }
    return X, y, meta


def build_synonym_tree_datasets(cfg):
    """
    Builds train/val/test splits for the hierarchy dataset using cfg['dataset'] keys:
      - L, m, s
      - n_train, n_val, n_test
      - distractor_dims (optional), distractor_prob (optional)
      - label_noise (optional)
    """
    ds = cfg["dataset"]
    L  = int(ds.get("L", 3))
    m  = int(ds.get("m", 4))
    s  = int(ds.get("s", 6))
    ntr, nva, nte = int(ds["n_train"]), int(ds["n_val"]), int(ds["n_test"])
    label_noise        = float(ds.get("label_noise", 0.0))
    distractor_dims    = int(ds.get("distractor_dims", 0))
    distractor_prob    = float(ds.get("distractor_prob", 0.0))
    seed               = int(cfg.get("seed", 0))

    Xtr, ytr, meta = gen_synonym_tree_split(
        L, m, s, ntr,
        distractor_dims=distractor_dims,
        distractor_prob=distractor_prob,
        label_noise=label_noise,
        seed=seed,
    )
    Xva, yva, _ = gen_synonym_tree_split(
        L, m, s, nva,
        distractor_dims=distractor_dims,
        distractor_prob=distractor_prob,
        label_noise=label_noise,
        seed=seed + 1,
    )
    Xte, yte, _ = gen_synonym_tree_split(
        L, m, s, nte,
        distractor_dims=distractor_dims,
        distractor_prob=distractor_prob,
        label_noise=label_noise,
        seed=seed + 2,
    )
    return Xtr, ytr, Xva, yva, Xte, yte, meta
