# src/data/mnist.py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

__all__ = [
    "MNISTDataset",
    "build_mnist_datasets",
]

class MNISTDataset(Dataset):
    """
    MNIST dataset wrapper.
    Converts MNIST to binary classification task (even vs odd digits by default).
    Labels are in {-1, +1} format.
    """
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_mnist_datasets(cfg):
    """
    Build MNIST train/val/test datasets.
    
    Args:
        cfg: Config dict with dataset settings:
            - n_train: Number of training samples (or list for sweeps)
            - n_val: Number of validation samples
            - n_test: Number of test samples
            - task_type: "multiclass" (10 classes) or "binary" (default: "multiclass")
            - binary_task: Type of binary classification (only if task_type="binary"):
                - "even_odd": Even digits (0,2,4,6,8) vs Odd digits (1,3,5,7,9)
                - "low_high": Low digits (0,1,2,3,4) vs High digits (5,6,7,8,9)
                - "specific": Specific digit pair (requires digit_pair config)
            - digit_pair: Tuple of (digit1, digit2) for specific binary task (optional)
            - random_labels: Randomly shuffle labels (breaks digit-label relationship)
            - seed: Random seed for reproducibility
    
    Returns:
        Xtr, ytr, Xva, yva, Xte, yte, meta
    """
    ds_cfg = cfg["dataset"]
    seed = cfg.get("seed", 42)
    
    # Get task type (multiclass or binary)
    task_type = ds_cfg.get("task_type", "multiclass")
    binary_task = ds_cfg.get("binary_task", "even_odd")
    digit_pair = ds_cfg.get("digit_pair", None)
    random_labels = ds_cfg.get("random_labels", False)
    
    # Set random seed
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    # Load full MNIST datasets (without transform to get raw data)
    # Download and load training set
    train_full = datasets.MNIST(
        root='./data', train=True, download=True, transform=None
    )
    
    # Download and load test set
    test_full = datasets.MNIST(
        root='./data', train=False, download=True, transform=None
    )
    
    # Extract data and labels (normalize to [0, 1] and then apply MNIST normalization)
    X_train_full = train_full.data.numpy().reshape(len(train_full), -1).astype(np.float32) / 255.0
    y_train_full = train_full.targets.numpy()
    
    X_test_full = test_full.data.numpy().reshape(len(test_full), -1).astype(np.float32) / 255.0
    y_test_full = test_full.targets.numpy()
    
    # Normalize using MNIST mean and std
    X_train_full = (X_train_full - 0.1307) / 0.3081
    X_test_full = (X_test_full - 0.1307) / 0.3081
    
    # Handle task type: multiclass (10 classes) or binary
    if task_type == "multiclass":
        # Keep original labels (0-9) for 10-class classification
        y_train_labels = y_train_full.astype(np.int64)
        y_test_labels = y_test_full.astype(np.int64)
        
        # Randomly shuffle labels if requested
        if random_labels:
            print(f"[mnist] WARNING: Random labels enabled - randomly assigning class labels")
            y_train_labels = rng.integers(0, 10, size=len(y_train_labels))
            y_test_labels = rng.integers(0, 10, size=len(y_test_labels))
            print(f"[mnist] Labels randomly assigned: each sample gets a random class (0-9) independent of its digit")
    else:
        # Binary classification
        if binary_task == "even_odd":
            # Even digits (0,2,4,6,8) -> +1, Odd digits (1,3,5,7,9) -> -1
            y_train_labels = np.where(y_train_full % 2 == 0, 1.0, -1.0)
            y_test_labels = np.where(y_test_full % 2 == 0, 1.0, -1.0)
        elif binary_task == "low_high":
            # Low digits (0,1,2,3,4) -> +1, High digits (5,6,7,8,9) -> -1
            y_train_labels = np.where(y_train_full < 5, 1.0, -1.0)
            y_test_labels = np.where(y_test_full < 5, 1.0, -1.0)
        elif binary_task == "specific":
            if digit_pair is None:
                raise ValueError("binary_task='specific' requires 'digit_pair' config (e.g., [0, 1])")
            d1, d2 = digit_pair
            # Keep only samples with digit d1 or d2
            train_mask = (y_train_full == d1) | (y_train_full == d2)
            test_mask = (y_test_full == d1) | (y_test_full == d2)
            X_train_full = X_train_full[train_mask]
            y_train_full = y_train_full[train_mask]
            X_test_full = X_test_full[test_mask]
            y_test_full = y_test_full[test_mask]
            # d1 -> +1, d2 -> -1
            y_train_labels = np.where(y_train_full == d1, 1.0, -1.0)
            y_test_labels = np.where(y_test_full == d1, 1.0, -1.0)
        else:
            raise ValueError(f"Unknown binary_task: {binary_task}. Options: 'even_odd', 'low_high', 'specific'")
        
        # Randomly assign labels if requested (breaks the relationship between digits and labels)
        if random_labels:
            print(f"[mnist] WARNING: Random labels enabled - randomly assigning labels to break digit-label relationship")
            # Randomly assign +1 or -1 to each sample independently
            y_train_labels = rng.choice([1.0, -1.0], size=len(y_train_labels))
            y_test_labels = rng.choice([1.0, -1.0], size=len(y_test_labels))
            print(f"[mnist] Labels randomly assigned: each sample gets a random label (+1 or -1) independent of its digit")
    
    # Get sizes
    n_train_cfg = ds_cfg.get("n_train", 50000)
    n_val = ds_cfg.get("n_val", 10000)
    n_test = ds_cfg.get("n_test", 10000)
    
    # Handle n_train as list (take first value for dataset building)
    if isinstance(n_train_cfg, list):
        n_train = n_train_cfg[0]  # Use first value for initial dataset
    else:
        n_train = n_train_cfg
    
    # Limit sizes to available data
    n_train = min(n_train, len(X_train_full))
    n_val = min(n_val, len(X_train_full) - n_train)
    n_test = min(n_test, len(X_test_full))
    
    # Shuffle training data
    train_indices = rng.permutation(len(X_train_full))
    X_train_full = X_train_full[train_indices]
    y_train_labels = y_train_labels[train_indices]
    
    # Get alpha parameter (multiplies labels - only for binary classification)
    alpha = float(ds_cfg.get("alpha", 1.0))
    
    # Split train/val
    Xtr = X_train_full[:n_train]
    if task_type == "multiclass":
        # For multiclass with MSE: multiply labels by alpha (e.g., 0,1,2...9 become 0,10,20...90 with alpha=10)
        # Labels will be used for one-hot encoding scaled by alpha
        ytr = (y_train_labels[:n_train].astype(np.float32) * alpha)
    else:
        # For binary, multiply by alpha
        ytr = (y_train_labels[:n_train] * alpha).reshape(-1, 1)  # Binary: multiply and reshape to (N, 1)
    
    Xva = X_train_full[n_train:n_train + n_val]
    if task_type == "multiclass":
        # For multiclass with MSE: multiply labels by alpha
        yva = (y_train_labels[n_train:n_train + n_val].astype(np.float32) * alpha)
    else:
        # For binary, multiply by alpha
        yva = (y_train_labels[n_train:n_train + n_val] * alpha).reshape(-1, 1)  # Binary: multiply and reshape to (N, 1)
    
    # Test set
    test_indices = rng.permutation(len(X_test_full))[:n_test]
    Xte = X_test_full[test_indices]
    if task_type == "multiclass":
        # For multiclass with MSE: multiply labels by alpha
        yte = (y_test_labels[test_indices].astype(np.float32) * alpha)
    else:
        # For binary, multiply by alpha
        yte = (y_test_labels[test_indices] * alpha).reshape(-1, 1)  # Binary: multiply and reshape to (N, 1)
    
    # Meta information
    n_classes = 10 if task_type == "multiclass" else 1
    meta = {
        "name": "mnist",
        "d": int(Xtr.shape[1]),  # Input dimension (784 for MNIST)
        "n_classes": n_classes,
        "task_type": task_type,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "alpha": alpha,  # Alpha applies to both binary and multiclass (for MSE regression)
    }
    if task_type == "binary":
        meta["binary_task"] = binary_task
        if digit_pair is not None:
            meta["digit_pair"] = digit_pair
    
    return Xtr, ytr, Xva, yva, Xte, yte, meta

