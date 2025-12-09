# src/analysis/dream_algorithm.py
"""
Dream algorithm: Find images that maximally excite top eigenpaths.
For MNIST, after training, identify top k eigenpaths that explain the target function,
then optimize images to maximally activate these paths.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim

from .path_kernel import compute_path_kernel_eigs, collect_path_factors


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


@torch.no_grad()
def get_top_eigenpaths(
    model,
    train_loader,
    *,
    k: int = 15,
    mode: str = "routing",
    max_samples: int = 1000,
    device: Optional[str] = None,
    block_size: int = 1024,
    power_iters: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get top k eigenpaths that explain the target function.
    
    Returns:
        top_evecs: (k, P_train) - top k eigenvectors in sample space
        top_evals: (k,) - top k eigenvalues
        Phi_train: (P_train, D_paths) - path feature matrix for training data
        variance_explained: (k,) - variance explained by each eigenpath
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    print(f"[dream] Computing path kernel to identify top {k} eigenpaths...")
    
    # Compute path kernel eigenvectors
    kern_train = compute_path_kernel_eigs(
        model, train_loader, device=dev, mode=mode, include_input=True,
        k=k, n_iter=power_iters, block_size=block_size, max_samples=max_samples, verbose=False
    )
    
    evecs = kern_train["evecs"].to(dev)  # (P_train, k) - eigenvectors are columns
    evals = kern_train["evals"].to(dev)  # (k,)
    y_train = kern_train.get("y")
    
    if y_train is None:
        raise ValueError("Training labels not available for identifying top eigenpaths")
    
    y_train = y_train.to(dev)
    # Ensure y_train is (P_train,) shape
    if y_train.dim() > 1:
        y_train = y_train.squeeze()
    
    # Collect path factors to build Phi_train
    factors_train = collect_path_factors(
        model, train_loader, device=dev, mode=mode,
        include_input=True, max_samples=max_samples
    )
    Phi_train = _build_path_feature_matrix(factors_train).to(dev)  # (P_train, D_paths)
    
    # Compute variance explained by each eigenpath
    y_train_centered = y_train - y_train.mean()
    y_train_var = y_train_centered.var().item()
    
    if y_train_var > 1e-12:
        y_train_norm = y_train_centered / (torch.norm(y_train_centered) + 1e-8)  # (P_train,)
        # Alignment: (y_train_norm @ evecs) gives (k,) - dot product with each eigenvector
        alignments = (y_train_norm @ evecs) ** 2  # (k,) - variance explained per component
        variance_explained = alignments
    else:
        variance_explained = torch.zeros(k, device=dev)
    
    # Sort by variance explained (descending)
    sorted_indices = torch.argsort(variance_explained, descending=True)
    # evecs is (P_train, k), so we need to index columns, then transpose to get (k, P_train)
    top_evecs = evecs[:, sorted_indices[:k]].T  # (k, P_train) - transpose to get (k, P_train)
    top_evals = evals[sorted_indices[:k]]  # (k,)
    top_variance = variance_explained[sorted_indices[:k]]  # (k,)
    
    print(f"[dream] Identified top {k} eigenpaths with variance explained: {top_variance.cpu().numpy()}")
    
    return top_evecs, top_evals, Phi_train, top_variance


def dream_algorithm_with_variance(
    model,
    train_loader,
    *,
    k: int = 15,
    mode: str = "routing",
    max_samples: int = 1000,
    device: Optional[str] = None,
    block_size: int = 1024,
    power_iters: int = 30,
    lr: float = 0.1,
    n_iter: int = 500,
    image_shape: Optional[Tuple[int, ...]] = None,
    init_method: str = "random",
    regularization: float = 0.01,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Dream algorithm that returns both images and variance explained.
    Wrapper around dream_algorithm for convenience.
    """
    # Get top eigenpaths first to get variance explained
    top_evecs, top_evals, Phi_train, variance_explained = get_top_eigenpaths(
        model, train_loader, k=k, mode=mode, max_samples=max_samples,
        device=device or next(iter(model.parameters())).device,
        block_size=block_size, power_iters=power_iters
    )
    
    # Generate dream images (pass pre-computed eigenpaths to avoid duplicate computation)
    dream_images = dream_algorithm(
        model, train_loader, k=k, mode=mode, max_samples=max_samples,
        device=device, block_size=block_size, power_iters=power_iters,
        lr=lr, n_iter=n_iter, image_shape=image_shape,
        init_method=init_method, regularization=regularization,
        top_evecs=top_evecs, Phi_train=Phi_train, variance_explained=variance_explained
    )
    
    return dream_images, variance_explained


def compute_path_activation(
    model,
    x: torch.Tensor,  # (1, d_in) - single input
    top_evecs: torch.Tensor,  # (k, P_train) - top k eigenvectors
    Phi_train: torch.Tensor,  # (P_train, D_paths) - training path features
    *,
    mode: str = "routing",
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute activation of top eigenpaths for input x.
    This function allows gradients to flow through x.
    
    Returns:
        activations: (k,) - activation of each top eigenpath
    """
    dev = device or x.device
    # Keep model in eval mode but allow gradients
    # CRITICAL: We need to ensure model parameters allow gradient computation
    # even though we're in eval mode (we don't want to update params, just track gradients through them)
    model.eval()
    # Ensure all parameters are set to allow gradient computation (even if we won't update them)
    for param in model.parameters():
        if param.requires_grad is False:
            param.requires_grad = True  # Temporarily enable to allow gradient flow
    
    # CRITICAL: Ensure x has gradients
    if not x.requires_grad:
        raise RuntimeError(f"x does not require grad! x.requires_grad={x.requires_grad}, x.shape={x.shape}")
    
    # Forward pass to get activations (with gradients)
    # We need to recompute activations without detaching to allow gradients
    # Build path features manually to allow gradients through x
    L = len(model.linears)
    E_list = []
    
    # Get gate parameters if available
    if hasattr(model, "gates") and model.gates is not None:
        ap_am = [(g.a_plus, g.a_minus) for g in model.gates]
    else:
        ap_am = [(None, None) for _ in range(L)]
    
    # Forward pass through model to get activations with gradients
    # Ensure we're using x directly (not a copy) to preserve gradient connection
    h = x  # (1, d_in) - this requires grad
    
    # Debug: Verify h has gradients at start
    if not h.requires_grad:
        raise RuntimeError(f"h (initialized from x) does not require grad! x.requires_grad={x.requires_grad}")
    
    for l in range(L):
        # Linear layer: should preserve gradients from h
        # Use functional linear to ensure gradients flow properly
        linear_layer = model.linears[l]
        # Compute linear transformation: u = h @ W^T + b
        u = torch.nn.functional.linear(h, linear_layer.weight, linear_layer.bias)  # (1, d_l)
        
        # Debug: Check if u has gradients
        if not u.requires_grad:
            raise RuntimeError(
                f"u (layer {l}) does not require grad! "
                f"h.requires_grad={h.requires_grad}, h.shape={h.shape}, "
                f"linear.weight.requires_grad={linear_layer.weight.requires_grad}"
            )
        
        # Activation function: should preserve gradients from u
        # Get the activation function directly to ensure gradients flow
        if model.activation_name == "relu":
            z = torch.relu(u)  # Preserves gradients
        elif model.activation_name == "gelu":
            z = torch.nn.functional.gelu(u)  # Preserves gradients
        elif model.activation_name == "tanh":
            z = torch.tanh(u)  # Preserves gradients
        elif model.activation_name == "sigmoid":
            z = torch.sigmoid(u)  # Preserves gradients
        elif model.activation_name == "elu":
            z = torch.nn.functional.elu(u)  # Preserves gradients
        else:
            # Fallback to model.activation if it's a custom function
            z = model.activation(u)
        
        # Debug: Check if z has gradients
        if not z.requires_grad:
            raise RuntimeError(f"z (layer {l}) does not require grad! u.requires_grad={u.requires_grad}, activation={model.activation_name}")
        
        h = z  # for next layer - preserve gradient connection
        
        # Build E based on mode
        a_plus, a_minus = ap_am[l]
        if mode == "routing":
            E = z  # (1, d_l)
        elif mode == "routing_posdev":
            if a_plus is None:
                E = z
            else:
                ap = (a_plus - 1.0).clamp_min(0.0)
                E = z * ap.unsqueeze(0)
        else:  # "routing_gain"
            if a_plus is None:
                E = z
            else:
                E = z * a_plus.unsqueeze(0) + (1.0 - z) * a_minus.unsqueeze(0)
        E_list.append(E)
    
    # Concatenate E layers
    E_concat = torch.cat(E_list, dim=1)  # (1, sum d_l) - should have gradients
    
    # Debug: Check if E_concat has gradients
    if not E_concat.requires_grad:
        # Check which E tensors have gradients
        grad_status = [f"E[{i}].requires_grad={E.requires_grad}" for i, E in enumerate(E_list)]
        raise RuntimeError(
            f"E_concat does not require grad! "
            f"x.requires_grad={x.requires_grad}, "
            f"E_list gradient status: {', '.join(grad_status)}"
        )
    
    # Include input
    Phi_x = torch.cat([x, E_concat], dim=1)  # (1, d_in + sum d_l) - should have gradients
    
    # Compute kernel between x and training data: K(x, train) = Phi_x @ Phi_train^T
    # Phi_train is detached (from no_grad computation), but gradients should flow through Phi_x
    # We need to ensure Phi_x has gradients
    if not Phi_x.requires_grad:
        raise RuntimeError(
            f"Phi_x does not require grad! "
            f"x.requires_grad={x.requires_grad}, "
            f"E_concat.requires_grad={E_concat.requires_grad}"
        )
    
    # Matrix multiplication: even if Phi_train is detached, gradients should flow through Phi_x
    # Use explicit matmul to ensure gradient tracking
    Phi_train_T = Phi_train.T.contiguous()  # (D_paths, P_train) - detached but contiguous
    K_x_train = torch.matmul(Phi_x, Phi_train_T)  # (1, P_train)
    
    # Verify K_x_train has gradients
    if not K_x_train.requires_grad:
        raise RuntimeError(f"K_x_train does not require grad! Phi_x.requires_grad={Phi_x.requires_grad}")
    
    # Activation of eigenpath v_i on x is: K(x, train) @ v_i
    # Even though top_evecs is detached, gradients should flow through K_x_train
    # top_evecs is (1, P_train) for a single eigenpath, so top_evecs.T is (P_train, 1)
    top_evecs_T = top_evecs.T.contiguous()  # (P_train, 1) - detached but contiguous
    activations = torch.matmul(K_x_train, top_evecs_T)  # (1, 1) for single eigenpath
    
    # Verify activations has gradients
    if not activations.requires_grad:
        raise RuntimeError(f"activations does not require grad! K_x_train.requires_grad={K_x_train.requires_grad}")
    
    # Squeeze to get scalar or 1D tensor
    activations = activations.squeeze()  # () or (1,) - preserves gradients
    
    # If activations is a scalar (0D), we need to ensure it's 1D for indexing later
    if activations.dim() == 0:
        activations = activations.unsqueeze(0)  # (1,)
    
    return activations


def dream_algorithm(
    model,
    train_loader,
    *,
    k: int = 15,
    mode: str = "routing",
    max_samples: int = 1000,
    device: Optional[str] = None,
    block_size: int = 1024,
    power_iters: int = 30,
    lr: float = 0.1,
    n_iter: int = 500,
    image_shape: Optional[Tuple[int, ...]] = None,  # e.g., (28, 28) for MNIST
    init_method: str = "random",  # "random" or "zero" or "mean"
    regularization: float = 0.01,  # L2 regularization on image
    top_evecs: Optional[torch.Tensor] = None,  # Optional pre-computed eigenvectors
    Phi_train: Optional[torch.Tensor] = None,  # Optional pre-computed path features
    variance_explained: Optional[torch.Tensor] = None,  # Optional pre-computed variance
) -> List[torch.Tensor]:
    """
    Dream algorithm: Find images that maximally excite each of the top k eigenpaths.
    Generates one image per eigenpath, where each image maximizes activation of that specific path.
    
    Args:
        model: Trained model
        train_loader: Training data loader
        k: Number of top eigenpaths (will generate k images, one per path)
        mode: Path kernel mode
        max_samples: Max samples for path kernel computation
        device: Device to use
        block_size: Block size for path kernel
        power_iters: Power iterations for path kernel
        lr: Learning rate for optimization
        n_iter: Number of optimization iterations
        image_shape: Shape of image (for visualization), e.g., (28, 28) for MNIST
        init_method: Initialization method ("random", "zero", "mean")
        regularization: L2 regularization strength
        top_evecs: Optional pre-computed eigenvectors (k, P_train)
        Phi_train: Optional pre-computed path features (P_train, D_paths)
        variance_explained: Optional pre-computed variance explained (k,)
    
    Returns:
        List of k dream images (each is (d_in,) tensor), one per eigenpath
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    print(f"[dream] Starting dream algorithm for top {k} eigenpaths (generating {k} images, one per path)...")
    
    # Get top eigenpaths if not provided
    if top_evecs is None or Phi_train is None or variance_explained is None:
        top_evecs, top_evals, Phi_train, variance_explained = get_top_eigenpaths(
            model, train_loader, k=k, mode=mode, max_samples=max_samples,
            device=dev, block_size=block_size, power_iters=power_iters
        )
    
    # Get input dimension
    d_in = next(iter(train_loader))[0].shape[1]
    
    # Generate one dream image per eigenpath
    dream_images = []
    
    for path_idx in range(k):
        print(f"[dream] Generating dream image for eigenpath {path_idx + 1}/{k} "
              f"(variance explained: {variance_explained[path_idx].item():.4f})...")
        
        # Initialize image
        # CRITICAL: x must be a leaf tensor (not a computation result) for the optimizer
        if init_method == "random":
            # Random initialization with small values
            # Create tensor first, then scale to ensure it's a leaf tensor
            x = torch.randn(1, d_in, device=dev, requires_grad=True)
            with torch.no_grad():
                x.mul_(0.1)  # In-place multiplication to keep it as a leaf tensor
        elif init_method == "zero":
            x = torch.zeros(1, d_in, device=dev, requires_grad=True)
        elif init_method == "mean":
            # Initialize with mean of training data
            x_mean = torch.stack([xb.mean(dim=0) for xb, _ in train_loader]).mean(dim=0)
            # Create a new leaf tensor from the mean
            x = x_mean.unsqueeze(0).clone().detach().requires_grad_(True).to(dev)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        # Verify x is a leaf tensor
        if not x.is_leaf:
            # If x is not a leaf, create a new leaf tensor from it
            x = x.detach().clone().requires_grad_(True)
        
        # Optimizer - requires leaf tensors
        optimizer = optim.Adam([x], lr=lr)
        
        # Get the specific eigenpath for this image
        target_evec = top_evecs[path_idx:path_idx+1]  # (1, P_train) - keep dim for compatibility
        
        # Optimization loop
        for iter_idx in range(n_iter):
            # Ensure x still requires grad (it might have been modified)
            if not x.requires_grad:
                x.requires_grad_(True)
            
            optimizer.zero_grad()
            
            # Compute path activations
            activations = compute_path_activation(
                model, x, target_evec, Phi_train, mode=mode, device=dev
            )
            
            # Debug: Check if activations has gradients
            if not activations.requires_grad:
                raise RuntimeError(
                    f"activations does not require grad! "
                    f"activations.shape={activations.shape}, "
                    f"x.requires_grad={x.requires_grad}, "
                    f"x.grad_fn={x.grad_fn}"
                )
            
            # Objective: maximize activation of this specific eigenpath
            # activations should be (1,) for a single eigenpath
            if activations.dim() == 0:
                objective = activations  # Already a scalar
            else:
                objective = activations[0]  # Get first (and only) element
            
            # Debug: Check if objective has gradients
            if not objective.requires_grad:
                raise RuntimeError(
                    f"objective does not require grad! "
                    f"activations.requires_grad={activations.requires_grad}, "
                    f"activations.shape={activations.shape}, "
                    f"objective.shape={objective.shape}"
                )
            
            # Add L2 regularization to keep image reasonable
            reg_loss = regularization * (x ** 2).sum()
            
            # Total loss (negative because we maximize)
            loss = -(objective - reg_loss)
            
            # Debug: Check if loss has gradients
            if not loss.requires_grad:
                raise RuntimeError(
                    f"loss does not require grad! "
                    f"objective.requires_grad={objective.requires_grad}, "
                    f"reg_loss.requires_grad={reg_loss.requires_grad}, "
                    f"loss.shape={loss.shape}, "
                    f"loss.grad_fn={loss.grad_fn}"
                )
            
            loss.backward()
            optimizer.step()
            
            # Clamp to reasonable range (for MNIST, roughly [-3, 3] after normalization)
            with torch.no_grad():
                x.clamp_(-5.0, 5.0)
            
            if (iter_idx + 1) % 100 == 0:
                print(f"  Iter {iter_idx + 1}/{n_iter}: activation={objective.item():.4f}, "
                      f"reg_loss={reg_loss.item():.4f}")
        
        dream_images.append(x.detach().squeeze(0))  # (d_in,)
        print(f"[dream] Completed dream image for eigenpath {path_idx + 1}")
    
    return dream_images


def plot_dream_images(
    dream_images: List[torch.Tensor],
    out_path: str,
    image_shape: Optional[Tuple[int, ...]] = None,
    variance_explained: Optional[torch.Tensor] = None,
    n_cols: int = 5,
):
    """
    Plot dream images in a grid.
    
    Args:
        dream_images: List of dream images, each (d_in,), one per eigenpath
        out_path: Path to save plot
        image_shape: Shape to reshape images to, e.g., (28, 28) for MNIST
        variance_explained: Optional tensor of variance explained per path for titles
        n_cols: Number of columns in grid
    """
    _ensure_dir(os.path.dirname(out_path))
    
    n_images = len(dream_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), facecolor='white')
    fig.patch.set_facecolor('white')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_images > 1 else [axes]
    if n_cols == 1:
        axes = axes.reshape(-1, 1) if n_images > 1 else [[ax] for ax in axes]
    
    for idx, dream_img in enumerate(dream_images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Convert to numpy and reshape if needed
        img_data = dream_img.cpu().numpy()
        if image_shape is not None:
            img_data = img_data.reshape(image_shape)
        
        # Denormalize for MNIST (if needed)
        # MNIST normalization: (x - 0.1307) / 0.3081
        # To denormalize: x * 0.3081 + 0.1307
        # But we'll just show the normalized version
        
        im = ax.imshow(img_data, cmap='gray', vmin=img_data.min(), vmax=img_data.max())
        
        # Title with path index and variance explained if available
        if variance_explained is not None and idx < len(variance_explained):
            var_exp = variance_explained[idx].item()
            ax.set_title(f'Path {idx + 1}\n(var={var_exp:.4f})', color='black', fontsize=9)
        else:
            ax.set_title(f'Path {idx + 1}', color='black', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[dream] Saved dream images plot -> {out_path}")


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

