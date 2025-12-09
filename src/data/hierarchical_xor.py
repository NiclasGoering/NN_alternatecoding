# src/data/hierarchical_xor.py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "HierarchicalXORDataset",
    "build_hierarchical_xor_datasets",
    "gen_xor_tree_split"
]

class HierarchicalXORDataset(Dataset):
    """
    Hierarchical XOR:
    1. Hierarchy determines which 'pair' of features is active.
    2. Label is the XOR of that pair.
    
    This destroys linear separability. A linear classifier gets 50% acc.
    A deep network must learn the path to find the active pair.
    """
    def __init__(self, X, y, groups=None, n_groups=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.groups = groups  # Optional: group IDs for SEI computation
        self.n_groups = n_groups  # Optional: number of groups

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_group_ids(self):
        """Return group IDs as numpy array if available."""
        if self.groups is not None:
            return self.groups
        return None

def gen_xor_tree_split(L, m, s, N, distractor_dims=0, seed=0):
    """
    L: Depth of tree
    m: Branching factor
    s: Synonyms (Number of valid pairs per leaf group)
    """
    rng = np.random.default_rng(seed)
    
    # Total Leaf Groups
    n_groups = m**L
    
    # Each synonym is now a PAIR of features.
    # We allocate 2 features per synonym.
    # d_leaf = n_groups * s * 2
    features_per_synonym = 2
    d_leaf = n_groups * s * features_per_synonym
    d_total = d_leaf + distractor_dims
    
    X = np.zeros((N, d_total), dtype=np.float32)
    y = np.zeros((N, 1), dtype=np.float32)
    
    # 1. Generate Random Paths (Context)
    # Choose a leaf group uniformly
    groups = rng.integers(0, n_groups, size=N)
    
    # 2. Choose a Synonym (Which specific pair to use)
    synonyms = rng.integers(0, s, size=N)
    
    # 3. Calculate Indices
    # The base index for this group
    group_base = groups * (s * features_per_synonym)
    # The offset for this synonym pair
    syn_offset = synonyms * features_per_synonym
    
    # The two active feature indices
    idx_1 = group_base + syn_offset
    idx_2 = group_base + syn_offset + 1
    
    # 4. Set Feature Values (Bernoulli 0.5)
    # Unlike the previous dataset, these values are random!
    val_1 = rng.integers(0, 2, size=N).astype(np.float32)
    val_2 = rng.integers(0, 2, size=N).astype(np.float32)
    
    # 5. Fill X
    rows = np.arange(N)
    X[rows, idx_1] = val_1
    X[rows, idx_2] = val_2
    
    # 6. Set Label y = XOR(val_1, val_2)
    # Map {0,1} to {-1, +1} for regression stability usually, 
    # but for XOR logic: 0^0=0, 1^1=0, 1^0=1, 0^1=1
    # Let's use: y = +1 if XOR is 1, y = -1 if XOR is 0
    xor_result = (val_1 != val_2) # Logical XOR
    y[:, 0] = np.where(xor_result, 1.0, -1.0)
    
    # 7. Distractors (Bernoulli 0.5 - High Noise)
    # Distractors are random 0/1, indistinguishable from signal features
    # except they are uncorrelated with y.
    if distractor_dims > 0:
        X[:, d_leaf:] = rng.integers(0, 2, size=(N, distractor_dims)).astype(np.float32)
        
    meta = {
        "name": "hierarchical_xor",
        "L": int(L),
        "m": int(m),
        "s": int(s),
        "n_groups": int(n_groups),
        "d_leaf": int(d_leaf),
        "d": int(d_total),
        "distractor_dims": int(distractor_dims),
        "features_per_synonym": int(features_per_synonym),
    }
    return X, y, groups, meta


def build_hierarchical_xor_datasets(cfg):
    """
    Builds train/val/test splits for the hierarchical XOR dataset using cfg['dataset'] keys:
      - L, m, s
      - n_train, n_val, n_test
      - distractor_dims (optional)
      - random_labels (optional): If True, randomly assign labels to break input-target correlation
    """
    ds = cfg["dataset"]
    L  = int(ds.get("L", 3))
    m  = int(ds.get("m", 4))
    s  = int(ds.get("s", 6))
    ntr, nva, nte = int(ds["n_train"]), int(ds["n_val"]), int(ds["n_test"])
    distractor_dims = int(ds.get("distractor_dims", 0))
    random_labels = ds.get("random_labels", False)
    seed = int(cfg.get("seed", 0))

    Xtr, ytr, groups_tr, meta = gen_xor_tree_split(
        L, m, s, ntr,
        distractor_dims=distractor_dims,
        seed=seed,
    )
    Xva, yva, groups_va, _ = gen_xor_tree_split(
        L, m, s, nva,
        distractor_dims=distractor_dims,
        seed=seed + 1,
    )
    Xte, yte, groups_te, _ = gen_xor_tree_split(
        L, m, s, nte,
        distractor_dims=distractor_dims,
        seed=seed + 2,
    )
    
    # Randomly assign labels if requested (breaks the relationship between inputs and labels)
    if random_labels:
        print(f"[hierarchical_xor] WARNING: Random labels enabled - randomly assigning labels to break input-target relationship")
        rng = np.random.default_rng(seed)
        # Randomly assign +1 or -1 to each sample independently
        # This means the same input pattern can have different labels in different samples
        ytr = rng.choice([1.0, -1.0], size=(len(ytr), 1)).astype(np.float32)
        yva = rng.choice([1.0, -1.0], size=(len(yva), 1)).astype(np.float32)
        yte = rng.choice([1.0, -1.0], size=(len(yte), 1)).astype(np.float32)
        print(f"[hierarchical_xor] Labels randomly assigned: each sample gets a random label (+1 or -1) independent of its input")
    
    # Apply alpha scaling to labels
    alpha = float(ds.get("alpha", 1.0))
    if alpha != 1.0:
        print(f"[hierarchical_xor] Applying alpha scaling: {alpha} (labels will be multiplied by {alpha})")
        ytr = (ytr * alpha).astype(np.float32)
        yva = (yva * alpha).astype(np.float32)
        yte = (yte * alpha).astype(np.float32)
    
    # Add alpha to meta
    meta["alpha"] = alpha
    
    return Xtr, ytr, Xva, yva, Xte, yte, meta, groups_tr, groups_va, groups_te

