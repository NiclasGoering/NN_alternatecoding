# src/data/cifar10.py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

__all__ = [
    "CIFAR10Dataset",
    "build_cifar10_datasets",
]

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset wrapper.
    Converts CIFAR-10 to binary classification task (even vs odd classes by default).
    Labels are in {-1, +1} format.
    """
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_cifar10_datasets(cfg):
    """
    Build CIFAR-10 train/val/test datasets.
    
    Args:
        cfg: Config dict with dataset settings:
            - n_train: Number of training samples (or list for sweeps)
            - n_val: Number of validation samples
            - n_test: Number of test samples
            - binary_task: Type of binary classification:
                - "even_odd": Even classes (0,2,4,6,8) vs Odd classes (1,3,5,7,9)
                - "low_high": Low classes (0,1,2,3,4) vs High classes (5,6,7,8,9)
                - "specific": Specific class pair (requires class_pair config)
            - class_pair: Tuple of (class1, class2) for specific binary task (optional)
            - seed: Random seed for reproducibility
    
    Returns:
        Xtr, ytr, Xva, yva, Xte, yte, meta
    """
    ds_cfg = cfg["dataset"]
    seed = cfg.get("seed", 42)
    
    # Get binary task type
    binary_task = ds_cfg.get("binary_task", "even_odd")
    class_pair = ds_cfg.get("class_pair", None)
    random_labels = ds_cfg.get("random_labels", False)
    
    # Set random seed
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    # Load full CIFAR-10 datasets (without transform to get raw data)
    # Download and load training set
    train_full = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    
    # Download and load test set
    test_full = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=None
    )
    
    # Extract data and labels
    # CIFAR-10 data is already a numpy array when transform=None
    # Shape: (N, 32, 32, 3) for RGB images
    X_train_full = np.array(train_full.data, dtype=np.float32)
    y_train_full = np.array(train_full.targets)
    
    X_test_full = np.array(test_full.data, dtype=np.float32)
    y_test_full = np.array(test_full.targets)
    
    # Reshape to (N, 32*32*3) = (N, 3072) and normalize to [0, 1]
    X_train_full = X_train_full.reshape(len(X_train_full), -1) / 255.0
    X_test_full = X_test_full.reshape(len(X_test_full), -1) / 255.0
    
    # Normalize using CIFAR-10 mean and std (computed across all channels)
    # CIFAR-10 normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    # For flattened data, we'll use a simple normalization
    # Alternatively, we can normalize per channel, but for simplicity, use global normalization
    mean = X_train_full.mean()
    std = X_train_full.std()
    X_train_full = (X_train_full - mean) / (std + 1e-7)
    X_test_full = (X_test_full - mean) / (std + 1e-7)
    
    # Convert to binary classification
    if binary_task == "even_odd":
        # Even classes (0,2,4,6,8) -> +1, Odd classes (1,3,5,7,9) -> -1
        y_train_binary = np.where(y_train_full % 2 == 0, 1.0, -1.0)
        y_test_binary = np.where(y_test_full % 2 == 0, 1.0, -1.0)
    elif binary_task == "low_high":
        # Low classes (0,1,2,3,4) -> +1, High classes (5,6,7,8,9) -> -1
        y_train_binary = np.where(y_train_full < 5, 1.0, -1.0)
        y_test_binary = np.where(y_test_full < 5, 1.0, -1.0)
    elif binary_task == "specific":
        if class_pair is None:
            raise ValueError("binary_task='specific' requires 'class_pair' config (e.g., [0, 1])")
        c1, c2 = class_pair
        # Keep only samples with class c1 or c2
        train_mask = (y_train_full == c1) | (y_train_full == c2)
        test_mask = (y_test_full == c1) | (y_test_full == c2)
        X_train_full = X_train_full[train_mask]
        y_train_full = y_train_full[train_mask]
        X_test_full = X_test_full[test_mask]
        y_test_full = y_test_full[test_mask]
        # c1 -> +1, c2 -> -1
        y_train_binary = np.where(y_train_full == c1, 1.0, -1.0)
        y_test_binary = np.where(y_test_full == c1, 1.0, -1.0)
    else:
        raise ValueError(f"Unknown binary_task: {binary_task}. Options: 'even_odd', 'low_high', 'specific'")
    
    # Randomly assign labels if requested (breaks the relationship between classes and labels)
    if random_labels:
        print(f"[cifar10] WARNING: Random labels enabled - randomly assigning labels to break class-label relationship")
        # Randomly assign +1 or -1 to each sample independently
        # This means the same class can have different labels in different samples
        y_train_binary = rng.choice([1.0, -1.0], size=len(y_train_binary))
        y_test_binary = rng.choice([1.0, -1.0], size=len(y_test_binary))
        print(f"[cifar10] Labels randomly assigned: each sample gets a random label (+1 or -1) independent of its class")
    
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
    y_train_binary = y_train_binary[train_indices]
    
    # Split train/val
    Xtr = X_train_full[:n_train]
    ytr = y_train_binary[:n_train].reshape(-1, 1)
    
    Xva = X_train_full[n_train:n_train + n_val]
    yva = y_train_binary[n_train:n_train + n_val].reshape(-1, 1)
    
    # Test set
    test_indices = rng.permutation(len(X_test_full))[:n_test]
    Xte = X_test_full[test_indices]
    yte = y_test_binary[test_indices].reshape(-1, 1)
    
    # Meta information
    meta = {
        "name": "cifar10",
        "d": int(Xtr.shape[1]),  # Input dimension (3072 for CIFAR-10)
        "binary_task": binary_task,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
    if class_pair is not None:
        meta["class_pair"] = class_pair
    
    return Xtr, ytr, Xva, yva, Xte, yte, meta

