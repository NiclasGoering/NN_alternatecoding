"""
Kernel Tracking Module

Provides functions for tracking kernel metrics during training:
- Path kernel rank and eigenvalues
- Hidden layer kernel (h^l h^T) rank and eigenvalues
- CKA (Centered Kernel Alignment) between initial and current kernels
- Wasserstein distance between eigenvalue distributions
- Gradient eigenvalue tracking per layer
"""
from __future__ import annotations
import threading
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from scipy.stats import wasserstein_distance
from scipy.linalg import eigvalsh

# Thread lock for safe linalg operations in multi-GPU environments
_LINALG_LOCK = threading.Lock()


def compute_kernel_matrix(features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Compute kernel matrix K = F @ F^T.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        normalize: If True, normalize by number of features
        
    Returns:
        Kernel matrix of shape (n_samples, n_samples)
    """
    # Handle NaN/Inf in features
    if torch.isnan(features).any() or torch.isinf(features).any():
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if normalize and features.shape[1] > 0:
        K = features @ features.T / features.shape[1]
    else:
        K = features @ features.T
    return K


def compute_path_kernel_matrix(
    X: torch.Tensor,
    E_list: List[torch.Tensor],
    device: torch.device,
    block_size: int = 2048
) -> torch.Tensor:
    """
    Compute path kernel K = (X @ X^T) ⊙ ∏_l (E_l @ E_l^T).
    
    Uses block-wise computation for memory efficiency.
    
    Args:
        X: Input features of shape (n_samples, d_in)
        E_list: List of routing matrices, each of shape (n_samples, width)
        device: Device to compute on
        block_size: Block size for memory-efficient computation
        
    Returns:
        Path kernel matrix of shape (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    
    # Handle NaN/Inf in input tensors
    if torch.isnan(X).any() or torch.isinf(X).any():
        X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    E_list_clean = []
    for E in E_list:
        if torch.isnan(E).any() or torch.isinf(E).any():
            E = torch.nan_to_num(E, nan=0.0, posinf=1e6, neginf=-1e6)
        E_list_clean.append(E)
    E_list = E_list_clean
    
    # For small matrices, compute directly
    if n_samples <= block_size:
        # Start with input kernel
        K = X @ X.T
        
        # Hadamard product with each routing kernel
        for E in E_list:
            K_E = E @ E.T
            K = K * K_E
        
        # Clamp to avoid extreme values
        K = torch.clamp(K, min=-1e10, max=1e10)
        
        return K
    
    # For large matrices, use block-wise computation
    K = torch.zeros(n_samples, n_samples, device=device, dtype=X.dtype)
    
    for i in range(0, n_samples, block_size):
        i_end = min(i + block_size, n_samples)
        for j in range(0, n_samples, block_size):
            j_end = min(j + block_size, n_samples)
            
            # Input kernel block
            K_block = X[i:i_end] @ X[j:j_end].T
            
            # Hadamard product with each routing kernel
            for E in E_list:
                K_E_block = E[i:i_end] @ E[j:j_end].T
                K_block = K_block * K_E_block
            
            K[i:i_end, j:j_end] = K_block
    
    # Clamp to avoid extreme values
    K = torch.clamp(K, min=-1e10, max=1e10)
    
    return K


def compute_top_eigenvalues(
    K: torch.Tensor,
    topk: int = 50,
    use_cpu: bool = True
) -> np.ndarray:
    """
    Compute top-k eigenvalues of a kernel matrix.
    
    Args:
        K: Kernel matrix of shape (n, n)
        topk: Number of top eigenvalues to return
        use_cpu: If True, move to CPU for numerical stability
        
    Returns:
        Array of top-k eigenvalues in descending order
    """
    if use_cpu:
        K_np = K.cpu().numpy().astype(np.float64)
    else:
        K_np = K.detach().cpu().numpy().astype(np.float64)
    
    # Check for NaN/Inf and handle them
    if np.any(np.isnan(K_np)) or np.any(np.isinf(K_np)):
        # Replace NaN with 0 and clip Inf
        K_np = np.nan_to_num(K_np, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Make symmetric (for numerical stability)
    K_np = (K_np + K_np.T) / 2
    
    # Additional numerical stability: ensure positive semi-definite by clipping small negative eigenvalues
    # Add small regularization to diagonal
    n = K_np.shape[0]
    K_np = K_np + np.eye(n) * 1e-10
    
    # Compute all eigenvalues (eigvalsh is for symmetric matrices, returns ascending order)
    try:
        eigenvalues = eigvalsh(K_np)
        # Reverse to get descending order
        eigenvalues = eigenvalues[::-1]
        # Take top-k
        topk_eigs = eigenvalues[:min(topk, len(eigenvalues))]
        return topk_eigs
    except Exception as e:
        print(f"Warning: eigvalsh failed after cleanup: {e}")
        return np.array([np.nan] * topk)


def compute_effective_rank(
    eigenvalues: np.ndarray,
    threshold: float = 1e-10
) -> float:
    """
    Compute effective rank from eigenvalues.
    
    Effective rank = exp(entropy of normalized eigenvalue distribution)
    
    Args:
        eigenvalues: Array of eigenvalues
        threshold: Threshold for filtering out near-zero eigenvalues
        
    Returns:
        Effective rank (float)
    """
    # Filter positive eigenvalues
    eigs = eigenvalues[eigenvalues > threshold]
    
    if len(eigs) == 0:
        return 0.0
    
    # Normalize to get probability distribution
    eigs_sum = np.sum(eigs)
    if eigs_sum <= 0:
        return 0.0
    
    p = eigs / eigs_sum
    
    # Compute entropy
    entropy = -np.sum(p * np.log(p + 1e-12))
    
    # Effective rank = exp(entropy)
    return np.exp(entropy)


def compute_numerical_rank(
    eigenvalues: np.ndarray,
    threshold_ratio: float = 1e-6
) -> int:
    """
    Compute numerical rank (number of eigenvalues above threshold).
    
    Args:
        eigenvalues: Array of eigenvalues
        threshold_ratio: Ratio of max eigenvalue to use as threshold
        
    Returns:
        Numerical rank (int)
    """
    if len(eigenvalues) == 0:
        return 0
    
    max_eig = np.max(np.abs(eigenvalues))
    if max_eig <= 0:
        return 0
    
    threshold = max_eig * threshold_ratio
    return int(np.sum(np.abs(eigenvalues) > threshold))


def compute_cka(K1: torch.Tensor, K2: torch.Tensor) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two kernel matrices.
    
    CKA(K1, K2) = HSIC(K1, K2) / sqrt(HSIC(K1, K1) * HSIC(K2, K2))
    
    where HSIC is the Hilbert-Schmidt Independence Criterion.
    
    Args:
        K1: First kernel matrix of shape (n, n)
        K2: Second kernel matrix of shape (n, n)
        
    Returns:
        CKA value in [0, 1]
    """
    n = K1.shape[0]
    
    # Center the kernels
    H = torch.eye(n, device=K1.device, dtype=K1.dtype) - torch.ones(n, n, device=K1.device, dtype=K1.dtype) / n
    K1_c = H @ K1 @ H
    K2_c = H @ K2 @ H
    
    # HSIC(K1, K2) = trace(K1_c @ K2_c) / (n-1)^2
    hsic_12 = torch.trace(K1_c @ K2_c) / ((n - 1) ** 2)
    hsic_11 = torch.trace(K1_c @ K1_c) / ((n - 1) ** 2)
    hsic_22 = torch.trace(K2_c @ K2_c) / ((n - 1) ** 2)
    
    # CKA
    denom = torch.sqrt(hsic_11 * hsic_22)
    if denom < 1e-12:
        return 0.0
    
    cka = (hsic_12 / denom).item()
    return max(0.0, min(1.0, cka))  # Clamp to [0, 1]


def compute_wasserstein_distance(
    eigs1: np.ndarray,
    eigs2: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Compute 1-Wasserstein distance between two eigenvalue distributions.
    
    Args:
        eigs1: First array of eigenvalues
        eigs2: Second array of eigenvalues
        normalize: If True, normalize eigenvalues to sum to 1
        
    Returns:
        Wasserstein distance
    """
    # Filter out NaN and negative values
    eigs1 = eigs1[~np.isnan(eigs1)]
    eigs2 = eigs2[~np.isnan(eigs2)]
    eigs1 = np.maximum(eigs1, 0)
    eigs2 = np.maximum(eigs2, 0)
    
    if len(eigs1) == 0 or len(eigs2) == 0:
        return float('nan')
    
    if normalize:
        sum1 = np.sum(eigs1)
        sum2 = np.sum(eigs2)
        if sum1 > 0:
            eigs1 = eigs1 / sum1
        if sum2 > 0:
            eigs2 = eigs2 / sum2
    
    return wasserstein_distance(eigs1, eigs2)


@torch.no_grad()
def collect_hidden_layer_features(
    model,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int = -1,
    max_samples: int = 8192
) -> torch.Tensor:
    """
    Collect hidden layer activations h^l(x) for computing kernel.
    
    Args:
        model: Neural network model
        loader: DataLoader
        device: Device to compute on
        layer_idx: Which hidden layer to extract (-1 for last)
        max_samples: Maximum number of samples to collect
        
    Returns:
        Hidden layer features of shape (n_samples, width)
    """
    model.eval()
    features_list = []
    seen = 0
    
    for xb, yb in loader:
        if seen >= max_samples:
            break
        
        xb = xb.to(device)
        take = min(xb.shape[0], max_samples - seen)
        xb = xb[:take]
        
        # Forward pass with cache
        _, cache = model(xb, return_cache=True)
        
        # Get hidden layer activations
        if "h" in cache and len(cache["h"]) > 0:
            h_list = cache["h"]
            # Get the specified layer (default: last hidden layer)
            if layer_idx == -1 or layer_idx >= len(h_list):
                h = h_list[-1]
            else:
                h = h_list[layer_idx]
            features_list.append(h.detach())
        elif "h_last" in cache:
            features_list.append(cache["h_last"].detach())
        
        seen += take
    
    if len(features_list) == 0:
        return torch.zeros(0, 0, device=device)
    
    return torch.cat(features_list, dim=0)


def compute_gradient_eigenvalues_per_layer(
    model,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    loss_fn,
    device: torch.device,
    n_classes: int = 1,
    alpha: float = 1.0,
    topk: int = 50
) -> Dict[int, np.ndarray]:
    """
    Compute eigenvalues of gradient outer products for each layer.
    
    For each layer l, compute eigenvalues of G_l @ G_l^T where G_l is the 
    gradient of the layer's weights.
    
    Args:
        model: Neural network model
        x_batch: Input batch
        y_batch: Target batch
        loss_fn: Loss function
        device: Device to compute on
        n_classes: Number of output classes
        alpha: Label scaling factor
        topk: Number of top eigenvalues to return
        
    Returns:
        Dict mapping layer index -> top-k eigenvalues
    """
    model.train()
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    model.zero_grad()
    yhat = model(x_batch)
    
    # Compute loss
    if n_classes == 1:
        loss = loss_fn(yhat, y_batch)
    else:
        if y_batch.dim() > 1:
            y_batch = y_batch.view(-1)
        yb_class = (y_batch / alpha).long()
        yb_class = torch.clamp(yb_class, 0, n_classes - 1)
        yb_onehot = torch.zeros_like(yhat)
        src_values = torch.ones_like(y_batch.unsqueeze(1)) * alpha
        yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values)
        loss = loss_fn(yhat, yb_onehot)
    
    loss.backward()
    
    gradient_eigenvalues = {}
    
    # Collect gradients per layer
    from src.models.lazarus import LazarusMLP
    
    def _compute_grad_eigs(G, topk):
        """Helper to compute eigenvalues of gradient matrix with NaN/Inf handling."""
        G = G.astype(np.float64)
        # Handle NaN/Inf
        if np.any(np.isnan(G)) or np.any(np.isinf(G)):
            G = np.nan_to_num(G, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Compute eigenvalues of G @ G^T (or G^T @ G for efficiency)
        if G.shape[0] <= G.shape[1]:
            GGT = G @ G.T
        else:
            GGT = G.T @ G
        
        # Add small regularization for numerical stability
        GGT = GGT + np.eye(GGT.shape[0]) * 1e-10
        
        try:
            eigs = eigvalsh(GGT)
            eigs = eigs[::-1]  # Descending order
            return eigs[:min(topk, len(eigs))]
        except Exception:
            return np.array([np.nan] * topk)
    
    if isinstance(model, LazarusMLP):
        # LazarusMLP: track by block
        for l, block in enumerate(model.blocks):
            first_linear = block[0]
            if first_linear.weight.grad is not None:
                G = first_linear.weight.grad.detach().cpu().numpy()
                gradient_eigenvalues[l] = _compute_grad_eigs(G, topk)
    else:
        # Standard MLP
        if hasattr(model, 'linears'):
            for l, linear in enumerate(model.linears):
                if linear.weight.grad is not None:
                    G = linear.weight.grad.detach().cpu().numpy()
                    gradient_eigenvalues[l] = _compute_grad_eigs(G, topk)
    
    return gradient_eigenvalues


def compute_all_kernel_metrics(
    model,
    loader: DataLoader,
    device: torch.device,
    config: Dict,
    initial_path_eigs: Optional[np.ndarray] = None,
    initial_hidden_eigs: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute all kernel-related metrics for tracking.
    
    Uses numerically stable methods:
    - Path kernel: HadamardGramOperator + power iteration (avoids materializing full kernel)
    - Hidden kernel: SVD-based eigenvalue computation
    
    Args:
        model: Neural network model
        loader: DataLoader for training data
        device: Device to compute on
        config: Configuration dict (contains kernel_max_samples, topk_spectrum)
        initial_path_eigs: Initial path kernel eigenvalues for Wasserstein/CKA
        initial_hidden_eigs: Initial hidden kernel eigenvalues for Wasserstein/CKA
        
    Returns:
        Dict with all kernel metrics:
        - path_kernel_rank: Effective rank of path kernel
        - path_kernel_eigs: Top-k eigenvalues
        - path_kernel_cka: CKA with initial (eigenvalue-based approx)
        - path_kernel_wasserstein: Wasserstein distance from initial
        - hidden_kernel_rank: Effective rank of hidden kernel
        - hidden_kernel_eigs: Top-k eigenvalues
        - hidden_kernel_cka: CKA with initial (eigenvalue-based approx)
        - hidden_kernel_wasserstein: Wasserstein distance from initial
    """
    from src.analysis.path_kernel import collect_path_factors
    
    max_samples = int(config.get("logging", {}).get("kernel_max_samples", 8192))
    topk = int(config.get("logging", {}).get("topk_spectrum", 50))
    
    results = {
        "path_kernel_rank": float('nan'),
        "path_kernel_numerical_rank": 0,
        "path_kernel_eigs": None,
        "path_kernel_cka": float('nan'),
        "path_kernel_wasserstein": float('nan'),
        "hidden_kernel_rank": float('nan'),
        "hidden_kernel_numerical_rank": 0,
        "hidden_kernel_eigs": None,
        "hidden_kernel_cka": float('nan'),
        "hidden_kernel_wasserstein": float('nan'),
    }
    
    # Ensure model is in eval mode
    was_training = model.training
    model.eval()
    
    try:
        # Use existing infrastructure from path_kernel.py for numerically stable computation
        from src.analysis.path_kernel import HadamardGramOperator, top_eigenpairs_block_power
        
        # Ensure device is a string for HadamardGramOperator
        device_str = str(device) if not isinstance(device, str) else device
        
        # Collect path factors with error handling
        try:
            pack = collect_path_factors(
                model, loader, device_str,
                mode="routing_gain",
                include_input=True,
                max_samples=max_samples
            )
        except Exception as collect_err:
            print(f"Warning: collect_path_factors failed: {collect_err}")
            pack = {"X": None, "E_list": []}
        
        X = pack.get("X")
        E_list = pack.get("E_list", [])
        
        if X is not None and len(E_list) > 0:
            # Build factors list for HadamardGramOperator
            factors = [X] + E_list
            P = X.shape[0]
            
            # Use HadamardGramOperator for implicit kernel (avoids materializing full matrix)
            op = HadamardGramOperator(
                factors, device=device_str, dtype=torch.float32, 
                block_size=min(2048, P), use_tf32=True
            )
            
            # Compute top eigenvalues using power iteration (numerically stable)
            with torch.no_grad():
                evals, evecs = top_eigenpairs_block_power(
                    op, k=topk, n_iter=30, tol=1e-6, seed=123, verbose=False
                )
            
            # Convert to numpy for storage
            path_eigs = evals.cpu().numpy()
            results["path_kernel_eigs"] = path_eigs
            
            # Debug: print top-5 eigenvalues for path kernel
            if len(path_eigs) >= 5:
                top5 = path_eigs[:5]
                total = np.sum(path_eigs[path_eigs > 1e-10])
                if total > 0:
                    top5_pct = 100 * np.sum(top5) / total
                    print(f"  [DEBUG] Path kernel top-5 eigs: {top5[:5]}, cover {top5_pct:.1f}% of spectrum")
            
            # Compute ranks from eigenvalues
            if len(path_eigs) > 0 and not np.all(np.isnan(path_eigs)):
                # Use torch-based effective rank computation for consistency
                evals_t = evals[evals > 0]
                if len(evals_t) > 0:
                    total = evals_t.sum()
                    if total > 0:
                        p = evals_t / total
                        p = p[p > 1e-12]
                        entropy = -(p * torch.log(p)).sum()
                        results["path_kernel_rank"] = torch.exp(entropy).item()
                results["path_kernel_numerical_rank"] = compute_numerical_rank(path_eigs)
            
            # Wasserstein distance (eigenvalue-based, doesn't need full kernel)
            if initial_path_eigs is not None:
                results["path_kernel_wasserstein"] = compute_wasserstein_distance(
                    initial_path_eigs, path_eigs
                )
            
            # CKA: use eigenvalue-based approximation when we don't have full kernels
            # CKA ≈ (sum of squared eigenvalue overlaps) / sqrt(sum_i λ_i² * sum_j μ_j²)
            if initial_path_eigs is not None:
                # Simple eigenvalue-based CKA approximation
                eigs1 = np.maximum(initial_path_eigs, 0)
                eigs2 = np.maximum(path_eigs, 0)
                if len(eigs1) > 0 and len(eigs2) > 0:
                    norm1 = np.sqrt(np.sum(eigs1**2))
                    norm2 = np.sqrt(np.sum(eigs2**2))
                    if norm1 > 0 and norm2 > 0:
                        # Use overlap of normalized spectra
                        min_len = min(len(eigs1), len(eigs2))
                        overlap = np.sum(eigs1[:min_len] * eigs2[:min_len])
                        results["path_kernel_cka"] = overlap / (norm1 * norm2)
                        
    except Exception as e:
        import traceback
        print(f"Warning: Failed to compute path kernel metrics: {e}")
        print(traceback.format_exc())
    
    try:
        # Collect hidden layer features (last hidden layer)
        h_features = collect_hidden_layer_features(
            model, loader, device, layer_idx=-1, max_samples=max_samples
        )
        
        if h_features.shape[0] > 0 and h_features.shape[1] > 0:
            # Use SVD for numerically stable eigenvalue computation
            # For K = H @ H^T, eigenvalues of K are singular values squared of H
            # This is more stable than computing K and then its eigendecomposition
            
            # Normalize features - center and scale for numerical stability
            H = h_features.cpu().float()
            
            # Center the features (subtract mean)
            H_mean = H.mean(dim=0, keepdim=True)
            H_centered = H - H_mean
            
            # Scale by Frobenius norm to prevent overflow
            frob_norm = torch.norm(H_centered, p='fro')
            if frob_norm > 1e-10:
                H_normalized = H_centered / frob_norm
            else:
                H_normalized = H_centered
            
            # Use torch SVD for numerical stability (thread-safe)
            with _LINALG_LOCK:
                try:
                    # Compute singular values (eigenvalues of K = H @ H^T are σ²)
                    U, S, Vh = torch.linalg.svd(H_normalized, full_matrices=False)
                    # Scale back: true eigenvalues = (frob_norm^2) * (S^2 / sum(S^2))
                    # But for effective rank, we only need relative eigenvalues
                    hidden_eigs = (S**2).numpy()[:topk]
                except Exception as svd_e:
                    print(f"SVD failed, falling back to eigvalsh: {svd_e}")
                    K_hidden = compute_kernel_matrix(h_features, normalize=True)
                    hidden_eigs = compute_top_eigenvalues(K_hidden, topk=topk)
            
            results["hidden_kernel_eigs"] = hidden_eigs
            
            # Debug: print top-5 eigenvalues for hidden kernel
            if len(hidden_eigs) >= 5:
                top5 = hidden_eigs[:5]
                total = np.sum(hidden_eigs[hidden_eigs > 1e-10])
                if total > 0:
                    top5_pct = 100 * np.sum(top5) / total
                    print(f"  [DEBUG] Hidden kernel top-5 eigs: {top5[:5]}, cover {top5_pct:.1f}% of spectrum")
            
            # Compute ranks from eigenvalues
            if len(hidden_eigs) > 0 and not np.all(np.isnan(hidden_eigs)):
                results["hidden_kernel_rank"] = compute_effective_rank(hidden_eigs)
                results["hidden_kernel_numerical_rank"] = compute_numerical_rank(hidden_eigs)
            
            # Wasserstein distance
            if initial_hidden_eigs is not None:
                results["hidden_kernel_wasserstein"] = compute_wasserstein_distance(
                    initial_hidden_eigs, hidden_eigs
                )
            
            # CKA: use eigenvalue-based approximation
            if initial_hidden_eigs is not None:
                eigs1 = np.maximum(initial_hidden_eigs, 0)
                eigs2 = np.maximum(hidden_eigs, 0)
                if len(eigs1) > 0 and len(eigs2) > 0:
                    norm1 = np.sqrt(np.sum(eigs1**2))
                    norm2 = np.sqrt(np.sum(eigs2**2))
                    if norm1 > 0 and norm2 > 0:
                        min_len = min(len(eigs1), len(eigs2))
                        overlap = np.sum(eigs1[:min_len] * eigs2[:min_len])
                        results["hidden_kernel_cka"] = overlap / (norm1 * norm2)
                        
    except Exception as e:
        import traceback
        print(f"Warning: Failed to compute hidden kernel metrics: {e}")
        print(traceback.format_exc())
    finally:
        # Restore model training state
        if was_training:
            model.train()
    
    return results


def concatenate_gradient_eigenvalues(
    grad_eigs_per_layer: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Concatenate gradient eigenvalues from all layers into a single array.
    
    Args:
        grad_eigs_per_layer: Dict mapping layer index -> eigenvalues
        
    Returns:
        Concatenated array of all gradient eigenvalues
    """
    if not grad_eigs_per_layer:
        return np.array([])
    
    # Sort by layer index and concatenate
    sorted_layers = sorted(grad_eigs_per_layer.keys())
    all_eigs = []
    for l in sorted_layers:
        eigs = grad_eigs_per_layer[l]
        if eigs is not None and len(eigs) > 0:
            all_eigs.extend(eigs.tolist())
    
    return np.array(all_eigs)

