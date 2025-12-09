# src/analysis/path_ablation.py
"""
Eigenpath ablation study: Remove top k eigenpaths and measure performance degradation.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt

from .path_kernel import compute_path_kernel_eigs, collect_path_factors
from .hidden_layer_analysis import collect_hidden_activations, compute_gram_kernel_eigs


def _detect_n_classes(model, y_train: Optional[torch.Tensor] = None) -> Optional[int]:
    """
    Detect number of classes from model or labels.
    Returns None for binary classification, or n_classes for multi-class.
    """
    # Check if model has n_classes attribute
    if hasattr(model, 'n_classes'):
        n_classes = model.n_classes
        if n_classes > 1:
            return n_classes
    
    # Check readout layer output size
    if hasattr(model, 'readout'):
        out_features = model.readout.out_features
        if out_features > 1:
            return out_features
        # For binary, readout might output 1 (scalar logit)
        # Check if labels suggest multi-class
        if y_train is not None:
            y_unique = torch.unique(y_train)
            # If labels are integers 0..n-1, it's multi-class
            if len(y_unique) > 2 and torch.allclose(y_unique, torch.arange(len(y_unique), dtype=y_unique.dtype, device=y_unique.device)):
                return len(y_unique)
    
    # Check labels directly
    if y_train is not None:
        y_unique = torch.unique(y_train)
        # If we have integer labels 0..n-1, it's multi-class
        if len(y_unique) > 2:
            # Check if labels are consecutive integers starting from 0
            y_sorted = torch.sort(y_unique)[0]
            if torch.allclose(y_sorted.float(), torch.arange(len(y_unique), dtype=torch.float32, device=y_sorted.device)):
                return len(y_unique)
    
    # Default: binary classification
    return None


def _compute_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, n_classes: Optional[int] = None) -> float:
    """
    Compute accuracy for binary or multi-class classification.
    
    Args:
        y_pred: Predictions (scalar for binary, logit vector for multi-class)
        y_true: True labels (scalar for binary, class index for multi-class)
        n_classes: Number of classes (None for binary)
    
    Returns:
        Accuracy as float between 0 and 1
    """
    if n_classes is None or n_classes == 1:
        # Binary classification: use sign agreement
        # Handle both {-1, +1} and {0, 1} label formats
        y_pred_sign = torch.sign(y_pred - y_pred.mean())
        y_true_sign = torch.sign(y_true - y_true.mean())
        return (y_pred_sign == y_true_sign).float().mean().item()
    else:
        # Multi-class classification: use argmax
        # y_pred should be (N, n_classes) logit vectors
        # y_true should be (N,) class indices
        if y_pred.dim() == 1:
            # If y_pred is 1D, it's already class predictions (argmax was done)
            pred_classes = y_pred.long()
        else:
            pred_classes = y_pred.argmax(dim=1)
        
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        return (pred_classes == y_true.long()).float().mean().item()


@torch.no_grad()
def compute_eigenpath_ablation(
    model,
    train_loader,
    test_loader,
    *,
    mode: str = "routing",
    k: int = 150,
    max_samples: int = 1000,
    device: Optional[str] = None,
    block_size: int = 1024,   # kept for API compatibility, not used now
    power_iters: int = 30,    # kept for API compatibility, not used now
    ridge: float = 1e-4,
    n_classes: Optional[int] = None,  # Number of classes (None for binary)
) -> Dict[str, object]:
    """
    Eigenpath ablation study using a *consistent* kernel:

    - We treat the concatenated path features Phi = [X, E_0, E_1, ...]
      as a feature map and define K = Phi Phi^T.
    - We compute the top-k eigenpairs of this kernel via SVD
      (using compute_gram_kernel_eigs).
    - We then remove the top k eigenpaths one by one and measure
      performance degradation on train and test.

    Returns:
        Dictionary with:
        - k_values: List of k values (number of eigenpaths removed)
        - train_errors: List of train MSEs
        - test_errors: List of test MSEs
        - train_accs: List of train accuracies
        - test_accs: List of test accuracies
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()

    print(f"[eigenpath_ablation] Collecting path factors for train/test...")

    # ----- Collect path factors (single pass over each loader) -----
    factors_train = collect_path_factors(
        model, train_loader, device=dev, mode=mode,
        include_input=True, max_samples=max_samples
    )
    factors_test = collect_path_factors(
        model, test_loader, device=dev, mode=mode,
        include_input=True, max_samples=max_samples
    )

    Phi_train = _build_path_feature_matrix(factors_train).to(dev)  # (P_train, D_paths)
    Phi_test = _build_path_feature_matrix(factors_test).to(dev)    # (P_test, D_paths)

    y_train = factors_train.get("y")
    y_test = factors_test.get("y")
    if y_train is None or y_test is None:
        raise ValueError("Training and test labels must be available in factors for eigenpath ablation.")

    y_train = y_train.to(dev)
    y_test = y_test.to(dev)
    
    # Detect n_classes if not provided
    if n_classes is None:
        n_classes = _detect_n_classes(model, y_train)
    
    # Flatten labels for binary, keep shape for multi-class
    if n_classes is None or n_classes == 1:
        y_train = y_train.view(-1)
        y_test = y_test.view(-1)
    else:
        # For multi-class, ensure labels are class indices (0..n_classes-1)
        if y_train.dim() > 1:
            y_train = y_train.view(-1)
        if y_test.dim() > 1:
            y_test = y_test.view(-1)
        # Convert {-1, +1} to {0, 1} if needed for binary, or ensure 0..n-1 for multi-class
        if y_train.min() < 0:
            # Assume {-1, +1} format, convert to {0, 1} for binary, or keep as-is for multi-class
            if n_classes == 2:
                y_train = ((y_train + 1) / 2).long()
                y_test = ((y_test + 1) / 2).long()

    P_train = Phi_train.shape[0]
    print(f"[eigenpath_ablation] Train path features: {Phi_train.shape}, Test path features: {Phi_test.shape}")
    if n_classes is not None and n_classes > 1:
        print(f"[eigenpath_ablation] Multi-class classification with {n_classes} classes")
    else:
        print(f"[eigenpath_ablation] Binary classification")

    # ----- Compute kernel eigenpairs CONSISTENTLY with Phi -----
    # We re-use the hidden-layer helper: it just does SVD on Phi
    print(f"[eigenpath_ablation] Computing gram kernel eigendecomposition with k={k}...")
    kern_train = compute_gram_kernel_eigs(Phi_train, k=k, device=dev)
    evals = kern_train["evals"].to(dev)   # (k,)
    evecs = kern_train["evecs"].to(dev)   # (k, P_train)

    # Sanity check
    if evecs.shape[1] != P_train:
        raise ValueError(
            f"Eigenvectors expect {evecs.shape[1]} samples, "
            f"but Phi_train has {P_train}."
        )

    print(f"[eigenpath_ablation] Got {evecs.shape[0]} eigenvectors.")

    # ----- Build kernel matrices from the SAME features -----
    K_train = Phi_train @ Phi_train.T          # (P_train, P_train)
    K_test_train = Phi_test @ Phi_train.T      # (P_test, P_train)

    # Normalize eigenvectors (should already be orthonormal, but for safety)
    evecs_normalized = evecs / (torch.norm(evecs, dim=1, keepdim=True) + 1e-8)

    # Pre-compute predictions using all eigenpaths (k_remove = 0 case)
    y_train_pred_orig = _predict_with_eigenpaths(
        K_train, evecs_normalized, evals, y_train, ridge=ridge, n_classes=n_classes
    )
    y_test_pred_orig = _predict_with_eigenpaths(
        K_test_train, evecs_normalized, evals, y_train, ridge=ridge, n_classes=n_classes
    )

    # ----- Ablation loop -----
    k_max = evecs_normalized.shape[0]
    k_values = list(range(0, min(k + 1, k_max)))
    train_errors: List[float] = []
    test_errors: List[float] = []
    train_accs: List[float] = []
    test_accs: List[float] = []

    print(f"[eigenpath_ablation] Running ablation for k in {k_values}...")

    for k_remove in k_values:
        if k_remove == 0:
            # Use full basis
            y_train_pred = y_train_pred_orig
            y_test_pred = y_test_pred_orig
        else:
            # Remove top k_remove eigenpaths (keep from k_remove onward)
            evecs_remaining = evecs_normalized[k_remove:]
            evals_remaining = evals[k_remove:]

            if evecs_remaining.shape[0] == 0:
                # No eigenpaths left: fallback to mean predictor
                if n_classes is None or n_classes == 1:
                    y_mean = y_train.mean()
                    y_train_pred = torch.full_like(y_train, y_mean)
                    y_test_pred = torch.full_like(y_test, y_mean)
                else:
                    # For multi-class, use uniform distribution over classes
                    y_train_pred = torch.zeros(P_train, n_classes, device=dev, dtype=torch.float32)
                    y_test_pred = torch.zeros(Phi_test.shape[0], n_classes, device=dev, dtype=torch.float32)
            else:
                y_train_pred = _predict_with_eigenpaths(
                    K_train, evecs_remaining, evals_remaining, y_train, ridge=ridge, n_classes=n_classes
                )
                y_test_pred = _predict_with_eigenpaths(
                    K_test_train, evecs_remaining, evals_remaining, y_train, ridge=ridge, n_classes=n_classes
                )

        # MSE errors (for binary) or cross-entropy (for multi-class)
        if n_classes is None or n_classes == 1:
            train_error = torch.mean((y_train_pred - y_train) ** 2).item()
            test_error = torch.mean((y_test_pred - y_test) ** 2).item()
        else:
            # For multi-class, compute cross-entropy loss
            import torch.nn.functional as F
            train_error = F.cross_entropy(y_train_pred, y_train.long()).item()
            test_error = F.cross_entropy(y_test_pred, y_test.long()).item()

        # Accuracies (binary: sign agreement, multi-class: argmax)
        train_acc = _compute_accuracy(y_train_pred, y_train, n_classes)
        test_acc = _compute_accuracy(y_test_pred, y_test, n_classes)

        train_errors.append(train_error)
        test_errors.append(test_error)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if k_remove % 10 == 0 or k_remove == k_values[-1]:
            print(
                f"  k={k_remove}: "
                f"train_err={train_error:.4e}, test_err={test_error:.4e}, "
                f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}"
            )

    return {
        "k_values": k_values,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "train_accs": train_accs,
        "test_accs": test_accs,
    }

    



def _build_path_feature_matrix(factors: Dict) -> torch.Tensor:
    """
    Build path feature matrix Phi from collected factors.
    Phi[i, :] is the path feature vector for sample i.
    """
    X = factors.get("X")  # (P, d_in) or None
    E_list = factors["E_list"]  # List of (P, d_l) tensors
    
    # Concatenate all E layers
    E_concat = torch.cat(E_list, dim=1)  # (P, sum d_l)
    
    if X is not None:
        # Include input features
        Phi = torch.cat([X, E_concat], dim=1)  # (P, d_in + sum d_l)
    else:
        Phi = E_concat
    
    return Phi


def _predict_with_eigenpaths(
    K: torch.Tensor,           # (P_query, P_train) or (P_train, P_train)
    evecs: torch.Tensor,       # (k_remaining, P_train)
    evals: torch.Tensor,       # (k_remaining,)
    y_train: torch.Tensor,     # (P_train,) for binary or (P_train,) class indices for multi-class
    ridge: float = 1e-4,
    n_classes: Optional[int] = None,  # Number of classes (None for binary)
) -> torch.Tensor:
    """
    Predict using kernel eigen-decomposition with Kernel Ridge Regression.

    We assume:
        - K_train = V diag(lambda) V^T  (evecs = rows of V^T)
        - K_query is the *same* kernel between query points and training points.

    For binary classification:
        Prediction (centered):
            y_hat = sum_i (1 / (lambda_i + ridge)) *
                        (v_i^T (y - mean)) * (K_query @ v_i) + mean
    
    For multi-class classification:
        We perform one-vs-rest regression for each class, then return logit vectors.
    """
    P_query = K.shape[0]
    P_train = evecs.shape[1]
    
    if n_classes is None or n_classes == 1:
        # Binary classification: original implementation
        # 1. Center labels
        y_mean = y_train.mean()
        y_centered = y_train - y_mean

        # overlaps: alpha_i = v_i^T y_centered
        overlaps = torch.mv(evecs, y_centered)  # (k,)

        # 2. Accumulate contributions from each eigenpath
        y_pred_centered = torch.zeros(P_query, device=K.device, dtype=K.dtype)

        for i in range(evecs.shape[0]):
            lambda_i = evals[i]
            if lambda_i < 1e-9:
                continue  # skip tiny eigenvalues for numerical stability

            v = evecs[i]        # (P_train,)
            alpha = overlaps[i]  # scalar

            # component = K_query @ v_i
            component = K @ v    # (P_query,)

            weight = 1.0 / (lambda_i + ridge)
            y_pred_centered += (weight * alpha) * component

        # 3. Add mean back
        return y_pred_centered + y_mean
    else:
        # Multi-class classification: one-vs-rest regression for each class
        # Convert class indices to one-hot for each class
        y_train_long = y_train.long()
        y_pred_logits = torch.zeros(P_query, n_classes, device=K.device, dtype=K.dtype)
        
        for c in range(n_classes):
            # Create binary labels for class c: 1 if class c, -1 otherwise
            y_binary = torch.where(y_train_long == c, 
                                  torch.ones_like(y_train, dtype=K.dtype),
                                  -torch.ones_like(y_train, dtype=K.dtype))
            
            # Center binary labels
            y_mean = y_binary.mean()
            y_centered = y_binary - y_mean
            
            # Compute overlaps for this class
            overlaps = torch.mv(evecs, y_centered)  # (k,)
            
            # Accumulate contributions
            y_pred_centered = torch.zeros(P_query, device=K.device, dtype=K.dtype)
            
            for i in range(evecs.shape[0]):
                lambda_i = evals[i]
                if lambda_i < 1e-9:
                    continue
                
                v = evecs[i]
                alpha = overlaps[i]
                component = K @ v
                weight = 1.0 / (lambda_i + ridge)
                y_pred_centered += (weight * alpha) * component
            
            # Store logit for this class (centered prediction + mean)
            y_pred_logits[:, c] = y_pred_centered + y_mean
        
        return y_pred_logits



def plot_eigenpath_ablation(
    ablation_results: Dict[str, object],
    out_path: str,
):
    """
    Plot train and test error vs number of eigenpaths removed.
    """
    _ensure_dir(os.path.dirname(out_path))
    
    k_values = ablation_results["k_values"]
    train_errors = ablation_results["train_errors"]
    test_errors = ablation_results["test_errors"]
    train_accs = ablation_results["train_accs"]
    test_accs = ablation_results["test_accs"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Plot 1: Errors
    ax1.plot(k_values, train_errors, marker='o', label='Train Error', 
             color='C0', linewidth=2, markersize=4)
    ax1.plot(k_values, test_errors, marker='s', label='Test Error', 
             color='C1', linewidth=2, markersize=4)
    ax1.set_xlabel('Number of Top Eigenpaths Removed (k)', color='black')
    ax1.set_ylabel('MSE Error', color='black')
    ax1.set_title('Error vs Eigenpaths Removed', color='black')
    ax1.tick_params(colors='black')
    for spine in ax1.spines.values(): spine.set_color('black')
    ax1.legend(framealpha=1.0, facecolor='white', edgecolor='black')
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.set_yscale('log')
    
    # Plot 2: Accuracies
    ax2.plot(k_values, train_accs, marker='o', label='Train Accuracy', 
             color='C0', linewidth=2, markersize=4)
    ax2.plot(k_values, test_accs, marker='s', label='Test Accuracy', 
             color='C1', linewidth=2, markersize=4)
    ax2.set_xlabel('Number of Top Eigenpaths Removed (k)', color='black')
    ax2.set_ylabel('Accuracy', color='black')
    ax2.set_title('Accuracy vs Eigenpaths Removed', color='black')
    ax2.tick_params(colors='black')
    for spine in ax2.spines.values(): spine.set_color('black')
    ax2.legend(framealpha=1.0, facecolor='white', edgecolor='black')
    ax2.grid(True, alpha=0.3, color='gray')
    ax2.set_ylim(bottom=0.0, top=1.0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[eigenpath_ablation] Saved ablation plot -> {out_path}")


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


@torch.no_grad()
def compute_layerwise_path_ablation(
    model,
    train_loader,
    test_loader,
    *,
    mode: str = "routing",
    k: int = 150,
    max_samples: int = 1000,
    device: Optional[str] = None,
    block_size: int = 1024,   # kept for API compatibility, not used now
    power_iters: int = 30,    # kept for API compatibility, not used now
    ridge: float = 1e-4,
    n_classes: Optional[int] = None,  # Number of classes (None for binary)
) -> Dict[int, Dict[str, object]]:
    """
    Layer-wise path ablation using a *consistent* linear kernel on concatenated
    path features.

    For each layer l, we:
      - Build Phi_train^(l) = concat( X, E_0, ..., E_l )
      - Define K_train^(l) = Phi_train^(l) @ Phi_train^(l).T
      - Compute top-k eigenpairs of K_train^(l) via SVD
      - Perform eigenpath ablation as in compute_eigenpath_ablation.

    Returns:
        Dict[layer_idx] -> same ablation dict as compute_eigenpath_ablation.
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()

    depth = model.depth

    print(f"[layerwise_path_ablation] Collecting path factors for all layers...")
    factors_train = collect_path_factors(
        model, train_loader, device=dev, mode=mode,
        include_input=True, max_samples=max_samples
    )
    factors_test = collect_path_factors(
        model, test_loader, device=dev, mode=mode,
        include_input=True, max_samples=max_samples
    )

    X_train = factors_train.get("X")
    E_list_train = factors_train["E_list"]  # list of (P, d_l)
    y_train = factors_train.get("y")

    X_test = factors_test.get("X")
    E_list_test = factors_test["E_list"]
    y_test = factors_test.get("y")

    if y_train is None or y_test is None:
        raise ValueError("Labels not available for layer-wise path ablation")

    y_train = y_train.to(dev)
    y_test = y_test.to(dev)
    
    # Detect n_classes if not provided
    if n_classes is None:
        n_classes = _detect_n_classes(model, y_train)
    
    # Flatten labels for binary, keep shape for multi-class
    if n_classes is None or n_classes == 1:
        y_train = y_train.view(-1)
        y_test = y_test.view(-1)
    else:
        # For multi-class, ensure labels are class indices (0..n_classes-1)
        if y_train.dim() > 1:
            y_train = y_train.view(-1)
        if y_test.dim() > 1:
            y_test = y_test.view(-1)
        # Convert {-1, +1} to {0, 1} if needed for binary, or ensure 0..n-1 for multi-class
        if y_train.min() < 0:
            # Assume {-1, +1} format, convert to {0, 1} for binary, or keep as-is for multi-class
            if n_classes == 2:
                y_train = ((y_train + 1) / 2).long()
                y_test = ((y_test + 1) / 2).long()

    results: Dict[int, Dict[str, object]] = {}

    # ----- For each layer, build Phi and run ablation -----
    for layer_idx in range(depth):
        print(f"\n[layerwise_path_ablation] Processing layer {layer_idx}...")

        # Build list of factors up to this layer
        factors_l_train: List[torch.Tensor] = []
        factors_l_test: List[torch.Tensor] = []

        if X_train is not None:
            factors_l_train.append(X_train.to(dev))
            factors_l_test.append(X_test.to(dev))

        for l in range(layer_idx + 1):
            factors_l_train.append(E_list_train[l].to(dev))
            factors_l_test.append(E_list_test[l].to(dev))

        # Build Phi matrices
        Phi_train_l = _build_path_feature_matrix_layerwise(factors_l_train).to(dev)
        Phi_test_l = _build_path_feature_matrix_layerwise(factors_l_test).to(dev)

        print(f"  Phi_train layer {layer_idx}: {Phi_train_l.shape}, "
              f"Phi_test layer {layer_idx}: {Phi_test_l.shape}")

        # Compute eigenpairs of K_train^(l) = Phi Phi^T
        kern_train = compute_gram_kernel_eigs(Phi_train_l, k=k, device=dev)
        evals = kern_train["evals"].to(dev)
        evecs = kern_train["evecs"].to(dev)  # (k, P_train)

        # Kernel matrices for this layer
        K_train_l = Phi_train_l @ Phi_train_l.T
        K_test_train_l = Phi_test_l @ Phi_train_l.T

        # Normalize eigenvectors
        evecs_normalized = evecs / (torch.norm(evecs, dim=1, keepdim=True) + 1e-8)

        # Ablation
        k_max = evecs_normalized.shape[0]
        k_values = list(range(0, min(k + 1, k_max)))
        train_errors: List[float] = []
        test_errors: List[float] = []
        train_accs: List[float] = []
        test_accs: List[float] = []

        for k_remove in k_values:
            if k_remove == 0:
                evecs_remaining = evecs_normalized
                evals_remaining = evals
            else:
                evecs_remaining = evecs_normalized[k_remove:]
                evals_remaining = evals[k_remove:]

            if evecs_remaining.shape[0] == 0:
                # All eigenpaths removed: mean predictor
                if n_classes is None or n_classes == 1:
                    y_mean = y_train.mean()
                    y_train_pred = torch.full_like(y_train, y_mean)
                    y_test_pred = torch.full_like(y_test, y_mean)
                else:
                    # For multi-class, use uniform distribution over classes
                    P_train_l = Phi_train_l.shape[0]
                    P_test_l = Phi_test_l.shape[0]
                    y_train_pred = torch.zeros(P_train_l, n_classes, device=dev, dtype=torch.float32)
                    y_test_pred = torch.zeros(P_test_l, n_classes, device=dev, dtype=torch.float32)
            else:
                y_train_pred = _predict_with_eigenpaths(
                    K_train_l, evecs_remaining, evals_remaining, y_train, ridge=ridge, n_classes=n_classes
                )
                y_test_pred = _predict_with_eigenpaths(
                    K_test_train_l, evecs_remaining, evals_remaining, y_train, ridge=ridge, n_classes=n_classes
                )

            # MSE errors (for binary) or cross-entropy (for multi-class)
            if n_classes is None or n_classes == 1:
                train_error = torch.mean((y_train_pred - y_train) ** 2).item()
                test_error = torch.mean((y_test_pred - y_test) ** 2).item()
            else:
                # For multi-class, compute cross-entropy loss
                import torch.nn.functional as F
                train_error = F.cross_entropy(y_train_pred, y_train.long()).item()
                test_error = F.cross_entropy(y_test_pred, y_test.long()).item()

            train_acc = _compute_accuracy(y_train_pred, y_train, n_classes)
            test_acc = _compute_accuracy(y_test_pred, y_test, n_classes)

            train_errors.append(train_error)
            test_errors.append(test_error)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        results[layer_idx] = {
            "k_values": k_values,
            "train_errors": train_errors,
            "test_errors": test_errors,
            "train_accs": train_accs,
            "test_accs": test_accs,
        }

    return results



def _build_path_feature_matrix_layerwise(factors: List[torch.Tensor]) -> torch.Tensor:
    """
    Build path feature matrix from list of factor tensors (X, E_0, E_1, ...).

    Phi[i, :] is the concatenated feature vector for sample i.
    """
    if len(factors) == 0:
        raise ValueError("At least one factor required for path features")

    return torch.cat(factors, dim=1)



@torch.no_grad()
def compute_layerwise_hidden_ablation(
    model,
    train_loader,
    test_loader,
    *,
    k: int = 150,
    max_samples: int = 1000,
    device: Optional[str] = None,
    ridge: float = 1e-4,
    n_classes: Optional[int] = None,  # Number of classes (None for binary)
) -> Dict[int, Dict[str, object]]:
    """
    Compute layer-wise hidden kernel ablation: for each layer l, compute hidden kernel h_l(x)^T h_l(x)
    and perform eigenvector ablation.
    
    Returns:
        Dictionary mapping layer index to ablation results
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    depth = model.depth
    
    print(f"[layerwise_hidden_ablation] Collecting hidden activations for all layers...")
    
    # Collect hidden activations for all layers from train and test
    H_list_train = []
    H_list_test = []
    y_train = None
    y_test = None
    
    # Train
    seen = 0
    for xb, yb in train_loader:
        if max_samples is not None and seen >= max_samples:
            break
        
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        
        bsz_original = xb.shape[0]
        if max_samples is not None and (seen + bsz_original > max_samples):
            take = max_samples - seen
            xb = xb[:take]
            yb = yb[:take]
            bsz = take
        else:
            bsz = bsz_original
        
        _, cache = model(xb, return_cache=True)
        h_list = cache["h"]  # List of (bsz, d_l) tensors
        
        if len(H_list_train) == 0:
            # Initialize
            H_list_train = [[] for _ in range(depth)]
            y_list_train = []
        
        for l in range(depth):
            H_list_train[l].append(h_list[l])
        
        y_list_train.append(yb.view(-1))
        seen += bsz
    
    # Concatenate train activations
    H_train_by_layer = [torch.cat(H_list_train[l], dim=0) for l in range(depth)]
    y_train = torch.cat(y_list_train, dim=0).to(dev)
    
    # Detect n_classes if not provided
    if n_classes is None:
        n_classes = _detect_n_classes(model, y_train)
    
    # Flatten labels
    if y_train.dim() > 1:
        y_train = y_train.view(-1)
    if n_classes is not None and n_classes > 1:
        # For multi-class, ensure labels are class indices (0..n_classes-1)
        if y_train.min() < 0 and n_classes == 2:
            y_train = ((y_train + 1) / 2).long()
    
    # Test
    seen = 0
    for xb, yb in test_loader:
        if max_samples is not None and seen >= max_samples:
            break
        
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        
        bsz_original = xb.shape[0]
        if max_samples is not None and (seen + bsz_original > max_samples):
            take = max_samples - seen
            xb = xb[:take]
            yb = yb[:take]
            bsz = take
        else:
            bsz = bsz_original
        
        _, cache = model(xb, return_cache=True)
        h_list = cache["h"]
        
        if len(H_list_test) == 0:
            H_list_test = [[] for _ in range(depth)]
            y_list_test = []
        
        for l in range(depth):
            H_list_test[l].append(h_list[l])
        
        y_list_test.append(yb.view(-1))
        seen += bsz
    
    # Concatenate test activations
    H_test_by_layer = [torch.cat(H_list_test[l], dim=0) for l in range(depth)]
    y_test = torch.cat(y_list_test, dim=0).to(dev)
    if y_test.dim() > 1:
        y_test = y_test.view(-1)
    if n_classes is not None and n_classes > 1:
        # For multi-class, ensure labels are class indices (0..n_classes-1)
        if y_test.min() < 0 and n_classes == 2:
            y_test = ((y_test + 1) / 2).long()
    
    results = {}
    
    # For each layer, compute hidden kernel and perform ablation
    for layer_idx in range(depth):
        print(f"\n[layerwise_hidden_ablation] Processing layer {layer_idx}...")
        
        H_train = H_train_by_layer[layer_idx]  # (P_train, d_l)
        H_test = H_test_by_layer[layer_idx]    # (P_test, d_l)
        
        print(f"  Train hidden: {H_train.shape}, Test hidden: {H_test.shape}")
        
        # Compute gram kernel eigendecomposition
        kern_train = compute_gram_kernel_eigs(H_train, k=k, device=dev)
        evecs = kern_train["evecs"].to(dev)  # (k, P_train)
        evals = kern_train["evals"].to(dev)  # (k,)
        
        # Compute kernel matrices
        K_train = H_train @ H_train.T
        K_test_train = H_test @ H_train.T
        
        # Normalize eigenvectors
        evecs_normalized = evecs / (torch.norm(evecs, dim=1, keepdim=True) + 1e-8)
        
        # Perform ablation using hidden kernel prediction
        k_values = list(range(0, min(k + 1, len(evecs))))
        train_errors = []
        test_errors = []
        train_accs = []
        test_accs = []
        
        for k_remove in k_values:
            if k_remove == 0:
                evecs_remaining = evecs_normalized
                evals_remaining = evals
            else:
                evecs_remaining = evecs_normalized[k_remove:]
                evals_remaining = evals[k_remove:]
            
            if len(evecs_remaining) == 0:
                if n_classes is None or n_classes == 1:
                    y_train_mean = y_train.mean()
                    y_train_pred = torch.full_like(y_train, y_train_mean)
                    y_test_pred = torch.full_like(y_test, y_train_mean)
                else:
                    # For multi-class, use uniform distribution over classes
                    P_train_h = H_train.shape[0]
                    P_test_h = H_test.shape[0]
                    y_train_pred = torch.zeros(P_train_h, n_classes, device=dev, dtype=torch.float32)
                    y_test_pred = torch.zeros(P_test_h, n_classes, device=dev, dtype=torch.float32)
            else:
                y_train_pred = _predict_with_hidden_eigenvectors(
                    K_train, evecs_remaining, evals_remaining, y_train,
                    H_train=H_train, H_query=H_train, ridge=ridge, n_classes=n_classes
                )
                y_test_pred = _predict_with_hidden_eigenvectors(
                    K_test_train, evecs_remaining, evals_remaining, y_train,
                    H_train=H_train, H_query=H_test, ridge=ridge, n_classes=n_classes
                )
            
            # MSE errors (for binary) or cross-entropy (for multi-class)
            if n_classes is None or n_classes == 1:
                train_error = torch.mean((y_train_pred - y_train) ** 2).item()
                test_error = torch.mean((y_test_pred - y_test) ** 2).item()
            else:
                # For multi-class, compute cross-entropy loss
                import torch.nn.functional as F
                train_error = F.cross_entropy(y_train_pred, y_train.long()).item()
                test_error = F.cross_entropy(y_test_pred, y_test.long()).item()
            
            train_acc = _compute_accuracy(y_train_pred, y_train, n_classes)
            test_acc = _compute_accuracy(y_test_pred, y_test, n_classes)
            
            train_errors.append(train_error)
            test_errors.append(test_error)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        
        results[layer_idx] = {
            "k_values": k_values,
            "train_errors": train_errors,
            "test_errors": test_errors,
            "train_accs": train_accs,
            "test_accs": test_accs,
        }
    
    return results


def _predict_with_hidden_eigenvectors(
    K: torch.Tensor,
    evecs: torch.Tensor,
    evals: torch.Tensor,
    y_train: torch.Tensor,
    H_train: Optional[torch.Tensor] = None,
    H_query: Optional[torch.Tensor] = None,
    ridge: float = 1e-4,
    n_classes: Optional[int] = None,  # Number of classes (None for binary)
) -> torch.Tensor:
    """
    Predict using hidden kernel eigenvector decomposition.
    Similar to _predict_with_eigenpaths but for hidden kernels.
    """
    P_query = K.shape[0]
    
    if n_classes is None or n_classes == 1:
        # Binary classification: original implementation
        y_mean = y_train.mean()
        y_train_centered = y_train - y_mean
        
        y_pred_centered = torch.zeros(P_query, device=K.device, dtype=K.dtype)
        
        overlaps = torch.mv(evecs, y_train_centered)
        
        if H_train is not None and H_query is not None:
            for i in range(len(evecs)):
                lambda_i = evals[i]
                if lambda_i < 1e-9:
                    continue
                
                v = evecs[i]
                alpha = overlaps[i]
                
                # H_train^T @ v gives feature space representation
                H_train_T_v = H_train.T @ v
                component = H_query @ H_train_T_v
                
                weight = 1.0 / (lambda_i + ridge)
                y_pred_centered += (weight * alpha) * component
        else:
            for i in range(len(evecs)):
                lambda_i = evals[i]
                if lambda_i < 1e-9:
                    continue
                
                v = evecs[i]
                alpha = overlaps[i]
                component = K @ v
                weight = 1.0 / (lambda_i + ridge)
                y_pred_centered += (weight * alpha) * component
        
        return y_pred_centered + y_mean
    else:
        # Multi-class classification: one-vs-rest regression for each class
        y_train_long = y_train.long()
        y_pred_logits = torch.zeros(P_query, n_classes, device=K.device, dtype=K.dtype)
        
        for c in range(n_classes):
            # Create binary labels for class c: 1 if class c, -1 otherwise
            y_binary = torch.where(y_train_long == c,
                                  torch.ones_like(y_train, dtype=K.dtype),
                                  -torch.ones_like(y_train, dtype=K.dtype))
            
            # Center binary labels
            y_mean = y_binary.mean()
            y_train_centered = y_binary - y_mean
            
            # Compute overlaps for this class
            overlaps = torch.mv(evecs, y_train_centered)
            
            # Accumulate contributions
            y_pred_centered = torch.zeros(P_query, device=K.device, dtype=K.dtype)
            
            if H_train is not None and H_query is not None:
                for i in range(len(evecs)):
                    lambda_i = evals[i]
                    if lambda_i < 1e-9:
                        continue
                    
                    v = evecs[i]
                    alpha = overlaps[i]
                    H_train_T_v = H_train.T @ v
                    component = H_query @ H_train_T_v
                    weight = 1.0 / (lambda_i + ridge)
                    y_pred_centered += (weight * alpha) * component
            else:
                for i in range(len(evecs)):
                    lambda_i = evals[i]
                    if lambda_i < 1e-9:
                        continue
                    
                    v = evecs[i]
                    alpha = overlaps[i]
                    component = K @ v
                    weight = 1.0 / (lambda_i + ridge)
                    y_pred_centered += (weight * alpha) * component
            
            # Store logit for this class
            y_pred_logits[:, c] = y_pred_centered + y_mean
        
        return y_pred_logits


def plot_layerwise_path_ablation(
    layer_results: Dict[int, Dict[str, object]],
    out_path: str,
    plot_type: str = "error",  # "error" or "accuracy"
):
    """
    Plot layer-wise path ablation results: one plot showing all layers together.
    
    Args:
        layer_results: Dictionary mapping layer index to ablation results
        out_path: Output path for the plot
        plot_type: "error" or "accuracy" - what to plot
    """
    _ensure_dir(os.path.dirname(out_path))
    
    layers = sorted(layer_results.keys())
    n_layers = len(layers)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    
    for layer_idx, color in zip(layers, colors):
        results = layer_results[layer_idx]
        k_values = results["k_values"]
        
        if plot_type == "error":
            train_vals = results["train_errors"]
            label = f'Layer {layer_idx} (Train)'
            ax.plot(k_values, train_vals, marker='o', label=label, 
                   color=color, linewidth=2, markersize=3, alpha=0.8)
        elif plot_type == "accuracy":
            train_vals = results["train_accs"]
            label = f'Layer {layer_idx} (Train)'
            ax.plot(k_values, train_vals, marker='o', label=label, 
                   color=color, linewidth=2, markersize=3, alpha=0.8)
    
    if plot_type == "error":
        ax.set_ylabel('MSE Error', color='black')
        ax.set_title('Path Ablation: Error vs Eigenpaths Removed (Train, All Layers)', color='black')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('Accuracy', color='black')
        ax.set_title('Path Ablation: Accuracy vs Eigenpaths Removed (Train, All Layers)', color='black')
        ax.set_ylim(bottom=0.0, top=1.0)
    
    ax.set_xlabel('Number of Top Eigenpaths Removed (k)', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(framealpha=1.0, facecolor='white', edgecolor='black', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[layerwise_path_ablation] Saved plot -> {out_path}")


def plot_layerwise_path_ablation_test(
    layer_results: Dict[int, Dict[str, object]],
    out_path: str,
    plot_type: str = "error",
):
    """
    Plot layer-wise path ablation results for test set: one plot showing all layers together.
    """
    _ensure_dir(os.path.dirname(out_path))
    
    layers = sorted(layer_results.keys())
    n_layers = len(layers)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    
    for layer_idx, color in zip(layers, colors):
        results = layer_results[layer_idx]
        k_values = results["k_values"]
        
        if plot_type == "error":
            test_vals = results["test_errors"]
            label = f'Layer {layer_idx} (Test)'
            ax.plot(k_values, test_vals, marker='s', label=label, 
                   color=color, linewidth=2, markersize=3, alpha=0.8)
        elif plot_type == "accuracy":
            test_vals = results["test_accs"]
            label = f'Layer {layer_idx} (Test)'
            ax.plot(k_values, test_vals, marker='s', label=label, 
                   color=color, linewidth=2, markersize=3, alpha=0.8)
    
    if plot_type == "error":
        ax.set_ylabel('MSE Error', color='black')
        ax.set_title('Path Ablation: Error vs Eigenpaths Removed (Test, All Layers)', color='black')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('Accuracy', color='black')
        ax.set_title('Path Ablation: Accuracy vs Eigenpaths Removed (Test, All Layers)', color='black')
        ax.set_ylim(bottom=0.0, top=1.0)
    
    ax.set_xlabel('Number of Top Eigenpaths Removed (k)', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(framealpha=1.0, facecolor='white', edgecolor='black', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[layerwise_path_ablation] Saved plot -> {out_path}")


def plot_layerwise_hidden_ablation(
    layer_results: Dict[int, Dict[str, object]],
    out_path: str,
    plot_type: str = "error",
):
    """
    Plot layer-wise hidden kernel ablation results: one plot showing all layers together.
    
    Args:
        layer_results: Dictionary mapping layer index to ablation results
        out_path: Output path for the plot
        plot_type: "error" or "accuracy" - what to plot
    """
    _ensure_dir(os.path.dirname(out_path))
    
    layers = sorted(layer_results.keys())
    n_layers = len(layers)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    
    for layer_idx, color in zip(layers, colors):
        results = layer_results[layer_idx]
        k_values = results["k_values"]
        
        if plot_type == "error":
            train_vals = results["train_errors"]
            label = f'Layer {layer_idx} (Train)'
            ax.plot(k_values, train_vals, marker='o', label=label, 
                   color=color, linewidth=2, markersize=3, alpha=0.8)
        elif plot_type == "accuracy":
            train_vals = results["train_accs"]
            label = f'Layer {layer_idx} (Train)'
            ax.plot(k_values, train_vals, marker='o', label=label, 
                   color=color, linewidth=2, markersize=3, alpha=0.8)
    
    if plot_type == "error":
        ax.set_ylabel('MSE Error', color='black')
        ax.set_title('Hidden Kernel Ablation: Error vs Eigenvectors Removed (Train, All Layers)', color='black')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('Accuracy', color='black')
        ax.set_title('Hidden Kernel Ablation: Accuracy vs Eigenvectors Removed (Train, All Layers)', color='black')
        ax.set_ylim(bottom=0.0, top=1.0)
    
    ax.set_xlabel('Number of Top Eigenvectors Removed (k)', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(framealpha=1.0, facecolor='white', edgecolor='black', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[layerwise_hidden_ablation] Saved plot -> {out_path}")


def plot_layerwise_hidden_ablation_test(
    layer_results: Dict[int, Dict[str, object]],
    out_path: str,
    plot_type: str = "error",
):
    """
    Plot layer-wise hidden kernel ablation results for test set: one plot showing all layers together.
    """
    _ensure_dir(os.path.dirname(out_path))
    
    layers = sorted(layer_results.keys())
    n_layers = len(layers)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    
    for layer_idx, color in zip(layers, colors):
        results = layer_results[layer_idx]
        k_values = results["k_values"]
        
        if plot_type == "error":
            test_vals = results["test_errors"]
            label = f'Layer {layer_idx} (Test)'
            ax.plot(k_values, test_vals, marker='s', label=label, 
                   color=color, linewidth=2, markersize=3, alpha=0.8)
        elif plot_type == "accuracy":
            test_vals = results["test_accs"]
            label = f'Layer {layer_idx} (Test)'
            ax.plot(k_values, test_vals, marker='s', label=label, 
                   color=color, linewidth=2, markersize=3, alpha=0.8)
    
    if plot_type == "error":
        ax.set_ylabel('MSE Error', color='black')
        ax.set_title('Hidden Kernel Ablation: Error vs Eigenvectors Removed (Test, All Layers)', color='black')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('Accuracy', color='black')
        ax.set_title('Hidden Kernel Ablation: Accuracy vs Eigenvectors Removed (Test, All Layers)', color='black')
        ax.set_ylim(bottom=0.0, top=1.0)
    
    ax.set_xlabel('Number of Top Eigenvectors Removed (k)', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(framealpha=1.0, facecolor='white', edgecolor='black', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, color='gray')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[layerwise_hidden_ablation] Saved plot -> {out_path}")
