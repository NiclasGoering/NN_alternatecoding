# src/analysis/hidden_layer_analysis.py
"""
Hidden layer gram kernel analysis: Compute gram kernel on last hidden layer activations
and analyze variance explained and perform eigenvector ablation.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def collect_hidden_activations(
    model,
    loader,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect last hidden layer activations and labels from a dataloader.
    
    Returns:
        H: (P, d_hidden) tensor of hidden activations
        y: (P,) tensor of labels
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    H_list = []
    y_list = []
    seen = 0
    
    for xb, yb in loader:
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
        
        # Forward pass to get last hidden layer
        _, cache = model(xb, return_cache=True)
        h_last = cache["h_last"]  # (bsz, d_hidden)
        
        H_list.append(h_last)
        y_list.append(yb)
        seen += bsz
    
    H = torch.cat(H_list, dim=0)  # (P, d_hidden)
    y = torch.cat(y_list, dim=0)  # (P,)
    
    return H, y


@torch.no_grad()
def compute_gram_kernel_eigs(
    H: torch.Tensor,  # (P, d_hidden)
    k: int = 150,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute gram kernel K = H @ H^T and its top k eigenvalues/eigenvectors.
    
    Returns:
        Dictionary with:
        - evals: (k,) eigenvalues (sorted descending)
        - evecs: (k, P) eigenvectors (each row is an eigenvector)
    """
    dev = device or H.device
    H = H.to(dev)
    P = H.shape[0]
    
    # Compute gram kernel: K = H @ H^T
    # For large P, we can use SVD on H directly: H = U @ S @ V^T
    # Then K = H @ H^T = U @ S^2 @ U^T, so eigenvectors of K are columns of U
    # and eigenvalues are S^2
    
    # Use SVD for numerical stability
    U, s, _ = torch.linalg.svd(H, full_matrices=False)
    # s is (min(P, d_hidden),) - singular values
    # U is (P, min(P, d_hidden)) - left singular vectors (eigenvectors of K)
    
    # Get top k
    k_actual = min(k, len(s))
    evals = s[:k_actual] ** 2  # Eigenvalues are squares of singular values
    evecs = U[:, :k_actual].T  # (k, P) - each row is an eigenvector
    
    # Pad if needed
    if k_actual < k:
        evals_padded = torch.zeros(k, device=dev, dtype=evals.dtype)
        evals_padded[:k_actual] = evals
        evecs_padded = torch.zeros(k, P, device=dev, dtype=evecs.dtype)
        evecs_padded[:k_actual] = evecs
        evals = evals_padded
        evecs = evecs_padded
    
    return {
        "evals": evals,
        "evecs": evecs,
    }


@torch.no_grad()
def compute_hidden_variance_explained(
    model,
    train_loader,
    test_loader,
    *,
    k: int = 150,
    max_samples: int = 1000,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Compute variance explained by top k eigenvectors of gram kernel on last hidden layer.
    
    Returns:
        Dictionary with:
        - train_variance_explained_per_component: List of variance explained per component
        - test_variance_explained_per_component: List of variance explained per component
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    print(f"[hidden_layer_analysis] Collecting hidden activations...")
    
    # Collect train activations
    H_train, y_train = collect_hidden_activations(
        model, train_loader, device=dev, max_samples=max_samples
    )
    print(f"[hidden_layer_analysis] Train hidden activations: {H_train.shape}")
    
    # Compute gram kernel and eigendecomposition for train
    kern_train = compute_gram_kernel_eigs(H_train, k=k, device=dev)
    evals_train = kern_train["evals"]
    evecs_train = kern_train["evecs"]  # (k, P_train)
    
    print(f"[hidden_layer_analysis] Computed {len(evals_train[evals_train > 1e-10])} non-zero eigenvalues")
    
    # Compute variance explained for train
    y_train_centered = y_train - y_train.mean()
    y_train_var = y_train_centered.var().item()
    
    train_variance_explained = []
    if y_train_var > 1e-12:
        y_train_norm = y_train_centered / (torch.norm(y_train_centered) + 1e-8)
        alignments = (y_train_norm @ evecs_train.T) ** 2  # (k,)
        train_variance_explained = alignments.cpu().numpy().tolist()
    else:
        train_variance_explained = [0.0] * len(evals_train)
    
    # Collect test activations and compute variance explained
    test_variance_explained = []
    if test_loader is not None:
        H_test, y_test = collect_hidden_activations(
            model, test_loader, device=dev, max_samples=max_samples
        )
        print(f"[hidden_layer_analysis] Test hidden activations: {H_test.shape}")
        
        # For test, we project test activations onto train eigenvectors
        # K_test_train = H_test @ H_train^T
        K_test_train = H_test @ H_train.T  # (P_test, P_train)
        
        # Project test labels onto train eigenvectors
        y_test_centered = y_test - y_test.mean()
        y_test_var = y_test_centered.var().item()
        
        if y_test_var > 1e-12:
            y_test_norm = y_test_centered / (torch.norm(y_test_centered) + 1e-8)
            # For each train eigenvector v, compute alignment with test labels
            # We need to project test data onto train eigenvectors
            # alignment = (y_test_norm @ (K_test_train @ v)) / ||K_test_train @ v||
            # But simpler: use the fact that eigenvectors span the space
            # We can compute: alignment = (y_test_norm @ (K_test_train @ v))^2 / (||K_test_train @ v||^2 * ||y_test_norm||^2)
            # Actually, for variance explained, we want: (y_test_norm @ (K_test_train @ v))^2
            alignments_test = []
            for i in range(len(evecs_train)):
                v = evecs_train[i]  # (P_train,)
                K_v = K_test_train @ v  # (P_test,)
                # Normalize K_v
                K_v_norm = K_v / (torch.norm(K_v) + 1e-8)
                alignment = (y_test_norm @ K_v_norm) ** 2
                alignments_test.append(alignment.item())
            test_variance_explained = alignments_test
        else:
            test_variance_explained = [0.0] * len(evals_train)
    
    return {
        "train_variance_explained_per_component": train_variance_explained,
        "test_variance_explained_per_component": test_variance_explained,
    }


def plot_hidden_variance_explained_by_k(
    variance_train: List[float],
    variance_test: List[float],
    out_path_train: str,
    out_path_test: str,
    max_k: int = 150,
):
    """
    Plot variance explained by k for train and test sets.
    Similar to plot_variance_explained_by_k_vs_epoch but for a single snapshot.
    """
    _ensure_dir(os.path.dirname(out_path_train))
    _ensure_dir(os.path.dirname(out_path_test))
    
    # Compute cumulative variance explained
    k_values = list(range(1, min(len(variance_train) + 1, max_k + 1)))
    cumsum_train = []
    cumsum_test = []
    
    for k in k_values:
        cumsum_train.append(sum(variance_train[:k]))
        if len(variance_test) > 0:
            cumsum_test.append(sum(variance_test[:k]))
    
    # Plot train
    if len(cumsum_train) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        ax.plot(k_values, cumsum_train, marker='o', color='C0', 
                linewidth=2, markersize=4, label='Train')
        
        ax.set_xlabel('k (Number of Components)', color='black')
        ax.set_ylabel('R² (Variance Explained)', color='black')
        ax.set_title('Hidden Layer Gram Kernel Variance Explained by k (Train)', color='black')
        ax.tick_params(colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.legend(framealpha=1.0, facecolor='white', edgecolor='black')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_ylim(bottom=0.0, top=1.0)
        
        plt.tight_layout()
        plt.savefig(out_path_train, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f"[hidden_layer_analysis] Saved variance explained plot (train) -> {out_path_train}")
    
    # Plot test
    if len(cumsum_test) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        ax.plot(k_values, cumsum_test, marker='s', color='C1', 
                linewidth=2, markersize=4, label='Test')
        
        ax.set_xlabel('k (Number of Components)', color='black')
        ax.set_ylabel('R² (Variance Explained)', color='black')
        ax.set_title('Hidden Layer Gram Kernel Variance Explained by k (Test)', color='black')
        ax.tick_params(colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.legend(framealpha=1.0, facecolor='white', edgecolor='black')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_ylim(bottom=0.0, top=1.0)
        
        plt.tight_layout()
        plt.savefig(out_path_test, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f"[hidden_layer_analysis] Saved variance explained plot (test) -> {out_path_test}")


@torch.no_grad()
def compute_eigenvector_ablation(
    model,
    train_loader,
    test_loader,
    *,
    k: int = 150,
    max_samples: int = 1000,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Compute ablation study: remove top k eigenvectors one by one and measure error.
    
    CORRECTION APPLIED: 
    Data (H) and Labels (y) are now centered before SVD/Regression. 
    The mean is added back to predictions before calculating Error/Accuracy.
    This prevents the "Top-1 Eigenvector = Bias" issue.
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    print(f"[hidden_layer_analysis] Computing gram kernel with k={k}...")
    
    # Step 1: Collect hidden activations (RAW)
    H_train_raw, y_train_raw = collect_hidden_activations(
        model, train_loader, device=dev, max_samples=max_samples
    )
    H_test_raw, y_test_raw = collect_hidden_activations(
        model, test_loader, device=dev, max_samples=max_samples
    )
    
    print(f"[hidden_layer_analysis] Train hidden: {H_train_raw.shape}, Test hidden: {H_test_raw.shape}")

    # --- FIX START: CENTER THE DATA ---
    # Center Hidden Activations (using Train Mean)
    H_mean = H_train_raw.mean(dim=0, keepdim=True)
    H_train = H_train_raw - H_mean
    H_test = H_test_raw - H_mean

    # Center Labels (using Train Mean)
    y_mean = y_train_raw.mean()
    y_train = y_train_raw - y_mean
    # We leave y_test_raw as is for final error comparison, 
    # but we will need y_test_raw for accuracy calculation.
    y_train_raw = y_train_raw.to(dev)
    y_test_raw = y_test_raw.to(dev)
    # --- FIX END ---
    
    # Step 2: Compute gram kernel and eigendecomposition on CENTERED data
    kern_train = compute_gram_kernel_eigs(H_train, k=k, device=dev)
    evecs = kern_train["evecs"].to(dev)  # (k, P_train)
    evals = kern_train["evals"].to(dev)  # (k,)
    
    y_train = y_train.to(dev)
    # y_test is not strictly needed for regression, we compare against y_test_raw later
    
    print(f"[hidden_layer_analysis] Computed {len(evecs)} eigenvectors (centered)")
    
    # Step 3: Compute gram kernels on CENTERED data
    K_train = H_train @ H_train.T  # (P_train, P_train)
    K_test_train = H_test @ H_train.T  # (P_test, P_train)
    
    # Normalize eigenvectors
    evecs_normalized = evecs / (torch.norm(evecs, dim=1, keepdim=True) + 1e-8)  # (k, P_train)
    
    # Step 4: Original predictions using all eigenvectors (on CENTERED data)
    # Note: We pass H_train (centered) and H_test (centered) for reconstruction
    y_train_pred_centered_orig = _predict_with_eigenvectors(
        K_train, evecs_normalized, evals, y_train, 
        H_train=H_train, H_query=H_train, ridge=0.0
    )
    y_test_pred_centered_orig = _predict_with_eigenvectors(
        K_test_train, evecs_normalized, evals, y_train, 
        H_train=H_train, H_query=H_test, ridge=0.0
    )
    
    # Step 5: Ablation - remove top k eigenvectors one by one
    k_values = list(range(0, min(k + 1, len(evecs))))
    train_errors = []
    test_errors = []
    train_accs = []
    test_accs = []
    
    print(f"[hidden_layer_analysis] Running ablation for k in {k_values}...")
    
    for k_remove in k_values:
        if k_remove == 0:
            y_train_pred_centered = y_train_pred_centered_orig
            y_test_pred_centered = y_test_pred_centered_orig
        else:
            evecs_remaining = evecs_normalized[k_remove:]
            evals_remaining = evals[k_remove:]
            
            if len(evecs_remaining) == 0:
                y_train_pred_centered = torch.zeros_like(y_train_pred_centered_orig)
                y_test_pred_centered = torch.zeros_like(y_test_pred_centered_orig)
            else:
                y_train_pred_centered = _predict_with_eigenvectors(
                    K_train, evecs_remaining, evals_remaining, y_train, 
                    H_train=H_train, H_query=H_train, ridge=0.0
                )
                y_test_pred_centered = _predict_with_eigenvectors(
                    K_test_train, evecs_remaining, evals_remaining, y_train, 
                    H_train=H_train, H_query=H_test, ridge=0.0
                )
        
        # --- FIX START: RESTORE MEAN ---
        # Add the mean back to get final predictions in original space
        y_train_pred_final = y_train_pred_centered + y_mean
        y_test_pred_final = y_test_pred_centered + y_mean
        # --- FIX END ---

        # Compute errors (Compare against RAW labels)
        train_error = torch.mean((y_train_pred_final - y_train_raw) ** 2).item()
        test_error = torch.mean((y_test_pred_final - y_test_raw) ** 2).item()
        
        # Compute accuracies (Compare signs of RAW labels)
        # We assume classification is based on sign relative to 0 in original space
        train_acc = (torch.sign(y_train_pred_final) == torch.sign(y_train_raw)).float().mean().item()
        test_acc = (torch.sign(y_test_pred_final) == torch.sign(y_test_raw)).float().mean().item()
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if k_remove % 10 == 0 or k_remove == len(k_values) - 1:
            print(f"  k={k_remove}: train_err={train_error:.4f}, test_err={test_error:.4f}, "
                  f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    return {
        "k_values": k_values,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "train_accs": train_accs,
        "test_accs": test_accs,
    }

def _predict_with_eigenvectors(
    K: torch.Tensor,  # Kernel matrix (P_query, P_train) or (P_train, P_train)
    evecs: torch.Tensor,  # Eigenvectors (k_remaining, P_train)
    evals: torch.Tensor,  # Eigenvalues (k_remaining,)
    y_train: torch.Tensor,  # Training labels (P_train,)
    H_train: Optional[torch.Tensor] = None,  # (P_train, d_hidden) - needed for test predictions
    H_query: Optional[torch.Tensor] = None,  # (P_query, d_hidden) - needed for test predictions
    ridge: float = 0.0,
) -> torch.Tensor:
    """
    Predict using eigenvector decomposition.
    Reconstruct kernel from remaining eigenvectors and use for prediction.
    
    For train: y_hat = sum_i (lambda_i / (lambda_i + ridge)) * (v_i^T @ y_train) * v_i
    For test: reconstruct K_test_train_recon = sum_j lambda_j * (H_query @ v_j) * (H_train^T @ v_j)^T
             then y_hat = K_test_train_recon @ (K_train_recon + ridge*I)^(-1) @ y_train
    """
    y_pred = torch.zeros(K.shape[0], device=K.device, dtype=K.dtype)
    
    # For train queries (K is square), use simple formula
    if K.shape[0] == K.shape[1]:
        for i in range(len(evecs)):
            v = evecs[i]  # (P_train,)
            lambda_i = evals[i]
            
            if lambda_i > 1e-10:
                # K_recon @ v = lambda_i * v
                K_v = lambda_i * v
                v_y = torch.dot(v, y_train)
                
                if ridge > 0:
                    weight = lambda_i / (lambda_i + ridge)
                else:
                    weight = 1.0
                
                y_pred += weight * K_v * v_y
    else:
        # For test queries, reconstruct kernel from remaining eigenvectors
        # K_test_train_recon = sum_j lambda_j * (H_query @ (H_train^T @ v_j)) @ (H_train @ (H_train^T @ v_j))^T
        if H_train is not None and H_query is not None:
            # Reconstruct kernel: K_recon = sum_j lambda_j * v_j @ v_j^T
            # For test: K_test_train_recon = H_query @ H_train^T (reconstructed)
            # Since v_j are eigenvectors of H_train @ H_train^T, we have:
            # H_train^T @ v_j gives feature space coefficients
            # So: K_test_train_recon = sum_j lambda_j * (H_query @ (H_train^T @ v_j)) @ (H_train @ (H_train^T @ v_j))^T
            K_recon = torch.zeros(K.shape[0], K.shape[1], device=K.device, dtype=K.dtype)
            for j in range(len(evecs)):
                v_j = evecs[j]  # (P_train,)
                lambda_j = evals[j]
                if lambda_j > 1e-10:
                    # H_train^T @ v_j gives feature space representation
                    H_train_T_vj = H_train.T @ v_j  # (d_hidden,)
                    # Compute outer product: (H_query @ H_train_T_vj) @ (H_train @ H_train_T_vj)^T
                    H_query_vj = H_query @ H_train_T_vj  # (P_query,)
                    H_train_vj = H_train @ H_train_T_vj  # (P_train,)
                    K_recon += lambda_j * H_query_vj.unsqueeze(1) @ H_train_vj.unsqueeze(0)
            
            # Use reconstructed kernel for prediction
            # y_hat = sum_i (lambda_i / (lambda_i + ridge)) * (v_i^T @ y_train) * (K_recon @ v_i)
            for i in range(len(evecs)):
                v = evecs[i]
                lambda_i = evals[i]
                if lambda_i > 1e-10:
                    K_v = K_recon @ v
                    v_y = torch.dot(v, y_train)
                    
                    if ridge > 0:
                        weight = lambda_i / (lambda_i + ridge)
                    else:
                        weight = 1.0
                    
                    y_pred += weight * K_v * v_y
        else:
            # Fallback: use full kernel (approximation - not ideal but works)
            for i in range(len(evecs)):
                v = evecs[i]
                lambda_i = evals[i]
                if lambda_i > 1e-10:
                    K_v = K @ v
                    v_y = torch.dot(v, y_train)
                    
                    if ridge > 0:
                        weight = lambda_i / (lambda_i + ridge)
                    else:
                        weight = 1.0
                    
                    y_pred += weight * K_v * v_y
    
    return y_pred


def plot_eigenvector_ablation(
    ablation_results: Dict[str, object],
    out_path: str,
):
    """
    Plot train and test error vs number of eigenvectors removed.
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
    ax1.set_xlabel('Number of Top Eigenvectors Removed (k)', color='black')
    ax1.set_ylabel('MSE Error', color='black')
    ax1.set_title('Error vs Eigenvectors Removed', color='black')
    ax1.tick_params(colors='black')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.legend(framealpha=1.0, facecolor='white', edgecolor='black')
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.set_yscale('log')
    
    # Plot 2: Accuracies
    ax2.plot(k_values, train_accs, marker='o', label='Train Accuracy', 
             color='C0', linewidth=2, markersize=4)
    ax2.plot(k_values, test_accs, marker='s', label='Test Accuracy', 
             color='C1', linewidth=2, markersize=4)
    ax2.set_xlabel('Number of Top Eigenvectors Removed (k)', color='black')
    ax2.set_ylabel('Accuracy', color='black')
    ax2.set_title('Accuracy vs Eigenvectors Removed', color='black')
    ax2.tick_params(colors='black')
    ax2.spines['bottom'].set_color('black')
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.legend(framealpha=1.0, facecolor='white', edgecolor='black')
    ax2.grid(True, alpha=0.3, color='gray')
    ax2.set_ylim(bottom=0.0, top=1.0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[hidden_layer_analysis] Saved eigenvector ablation plot -> {out_path}")


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

