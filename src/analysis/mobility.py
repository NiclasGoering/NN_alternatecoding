"""
Gate Mobility Analysis Functions

This module contains functions for computing gate mobility metrics:
- Gate Mobility Number (M_g)
- Distance to flip (d_f)
- Gradient norms
- LR computation from target mobility
- Path deformation capacity and covariance entropy
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from src.models.lazarus import LazarusMLP


def compute_distance_to_flip_lazarus(
    model, x_batch: torch.Tensor, device: torch.device, 
    return_dead_neurons: bool = False
) -> Dict:
    """
    Compute distance to flip for LazarusMLP architecture.
    Adapts the structure to work with the standard compute_distance_to_flip interface.
    
    Args:
        model: LazarusMLP model
        x_batch: Input batch
        device: Device to compute on
        return_dead_neurons: If True, also return dead neuron fractions
        
    Returns:
        Dict with "distances" and optionally "dead_neuron_fractions"
    """
    model.eval()
    x_batch = x_batch.to(device, non_blocking=True)
    
    distances = {}
    dead_neuron_fractions = {} if return_dead_neurons else None
    h = x_batch
    
    DEAD_THRESHOLD = 1e-6
    
    # Input projection (skip for now, we only track blocks)
    h = model.input_proj(h)
    
    # Process each residual block
    for l, block in enumerate(model.blocks):
        # For d_f, we compute it on the first linear in the branch
        first_linear = block[0]
        u = first_linear(h)  # Pre-activation of first linear
        
        x_norm = torch.norm(h, dim=1, keepdim=True)
        if return_dead_neurons:
            dead_samples = (x_norm.squeeze() < DEAD_THRESHOLD).float()
            dead_fraction = dead_samples.mean().item()
            dead_neuron_fractions[l] = dead_fraction
        
        x_norm = torch.clamp(x_norm, min=1e-8)
        d_f = torch.abs(u) / x_norm
        distances[l] = d_f
        
        # Forward through block: h = h + Branch(h)
        branch_out = block(h)
        h = h + branch_out
    
    result = {"distances": distances}
    if return_dead_neurons:
        result["dead_neuron_fractions"] = dead_neuron_fractions
    return result


def compute_gradient_norms_lazarus(
    model, x_batch: torch.Tensor, y_batch: torch.Tensor, 
    loss_fn, device: torch.device, n_classes: int = 1, alpha: float = 1.0
) -> Dict[int, float]:
    """
    Compute gradient norms for LazarusMLP architecture.
    Returns gradient norms for each block (using first linear in branch).
    
    Args:
        model: LazarusMLP model
        x_batch: Input batch
        y_batch: Target batch
        loss_fn: Loss function
        device: Device to compute on
        n_classes: Number of output classes
        alpha: Label scaling factor
        
    Returns:
        Dict mapping block index -> gradient norm
    """
    model.train()
    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)
    
    model.zero_grad()
    yhat = model(x_batch)
    
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
    
    gradient_norms = {}
    for l, block in enumerate(model.blocks):
        first_linear = block[0]
        if first_linear.weight.grad is not None:
            grad_norm = torch.norm(first_linear.weight.grad).item()
            gradient_norms[l] = grad_norm
        else:
            gradient_norms[l] = 0.0
    
    return gradient_norms


def compute_gate_mobility_lazarus(
    model,
    train_loader: DataLoader,
    lr: float,
    device: torch.device,
    n_batches: int = 10,
    n_classes: int = 1,
    alpha: float = 1.0,
    return_distributions: bool = False
) -> Dict:
    """
    Compute Gate Mobility Number M_g for LazarusMLP architecture.
    Adapted version that works with the block structure.
    
    M_g = (lr * E[||∇w||]) / E[d_f]
    
    Args:
        model: LazarusMLP model
        train_loader: DataLoader for training data
        lr: Learning rate
        device: Device to compute on
        n_batches: Number of batches to average over
        n_classes: Number of output classes
        alpha: Label scaling factor
        return_distributions: If True, also return d_f distributions
        
    Returns:
        Dict with "M_g", "dead_neuron_fractions", and optionally "d_f_distributions"
    """
    model = model.to(device)
    loss_fn = nn.MSELoss()
    
    # Use torch tensors for accumulation (GPU-friendly)
    grad_norm_sums = torch.zeros(model.depth, device=device)
    distance_sums = torch.zeros(model.depth, device=device)
    dead_neuron_sums = torch.zeros(model.depth, device=device)
    
    # Store distributions if requested
    d_f_distributions = {l: [] for l in range(model.depth)} if return_distributions else None
    
    batch_count = 0
    
    for x_batch, y_batch in train_loader:
        if batch_count >= n_batches:
            break
        
        # Compute distances to flip
        result_dict = compute_distance_to_flip_lazarus(model, x_batch, device, return_dead_neurons=True)
        distances = result_dict["distances"]
        dead_fractions = result_dict["dead_neuron_fractions"]
        
        # Compute gradient norms
        grad_norms = compute_gradient_norms_lazarus(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        
        # Accumulate on GPU
        for l in range(model.depth):
            mean_dist = distances[l].mean()
            distance_sums[l] += mean_dist
            dead_neuron_sums[l] += dead_fractions.get(l, 0.0)
            
            if return_distributions:
                d_f_flat = distances[l].detach().flatten().cpu().numpy()
                d_f_distributions[l].extend(d_f_flat)
            
            grad_norm_sums[l] += grad_norms.get(l, 0.0)
        
        batch_count += 1
    
    # Compute M_g for each layer
    M_g = {}
    dead_neuron_fractions = {}
    for l in range(model.depth):
        E_grad_norm = (grad_norm_sums[l] / batch_count).item() if batch_count > 0 else 0.0
        E_d_f = (distance_sums[l] / batch_count).item() if batch_count > 0 else 0.0
        avg_dead_fraction = (dead_neuron_sums[l] / batch_count).item() if batch_count > 0 else 0.0
        
        dead_neuron_fractions[l] = avg_dead_fraction
        
        if E_d_f > 0:
            M_g[l] = (lr * E_grad_norm) / E_d_f
        else:
            M_g[l] = float('inf')
    
    result = {
        "M_g": M_g,
        "dead_neuron_fractions": dead_neuron_fractions
    }
    if return_distributions:
        result["d_f_distributions"] = d_f_distributions
    
    return result


def compute_initial_lr_from_target_mobility(
    model,
    train_loader: DataLoader,
    device: torch.device,
    n_classes: int,
    alpha: float,
    eta_0: float,
    target_mobility: float = 0.3,
    n_batches: int = 1
) -> Dict[int, float]:
    """
    Compute initial LR per layer based on target gate mobility M*.
    
    Formula: M* ≈ η* · E[||∇W||] / E[d_f]  ⇒  η* ≈ η₀ · M* / M̄_g
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        device: Device to compute on
        n_classes: Number of output classes
        alpha: Label scaling factor
        eta_0: Base learning rate (η₀)
        target_mobility: Target gate mobility M* (default: 0.3)
        n_batches: Number of batches to average over
        
    Returns:
        Dict mapping layer index -> initial LR
    """
    # Import standard compute_gate_mobility if not LazarusMLP
    from src.analysis.mobility import compute_gate_mobility_lazarus
    
    model.train()
    loss_fn = nn.MSELoss()
    
    if isinstance(model, LazarusMLP):
        m_g_result = compute_gate_mobility_lazarus(
            model, train_loader, lr=1.0, device=device,
            n_batches=n_batches, n_classes=n_classes, alpha=alpha,
            return_distributions=False
        )
        M_g_dict = m_g_result["M_g"]
        
        initial_lrs = {}
        
        # For input projection, use layer index -1 and first block's M_g as proxy
        INPUT_PROJ_LAYER = -1
        if 0 in M_g_dict:
            M_g_input = M_g_dict[0]
            if np.isinf(M_g_input) or np.isnan(M_g_input) or M_g_input <= 0:
                initial_lrs[INPUT_PROJ_LAYER] = eta_0
                print(f"  Warning: Input projection M_g={M_g_input} invalid, using fallback LR={eta_0:.6e}")
            else:
                initial_lr = eta_0 * target_mobility / M_g_input
                initial_lrs[INPUT_PROJ_LAYER] = initial_lr
        else:
            initial_lrs[INPUT_PROJ_LAYER] = eta_0
        
        # For blocks (layers 0 to depth-1)
        for layer_idx, M_g_val in M_g_dict.items():
            if np.isinf(M_g_val) or np.isnan(M_g_val) or M_g_val <= 0:
                initial_lrs[layer_idx] = eta_0
                print(f"  Warning: Block {layer_idx} has invalid M_g={M_g_val}, using fallback LR={eta_0:.6e}")
            else:
                initial_lr = eta_0 * target_mobility / M_g_val
                initial_lrs[layer_idx] = initial_lr
        
        print(f"  Computed LRs for {len(initial_lrs)} layers (input_proj + {len(M_g_dict)} blocks)")
    else:
        # Use standard function from gate_velocity_with_capacity
        try:
            from outputs.gates.gate_velocity_with_capacity import compute_gate_mobility
            m_g_result = compute_gate_mobility(
                model, train_loader, lr=1.0, device=device,
                n_batches=n_batches, n_classes=n_classes, alpha=alpha,
                return_distributions=False
            )
            M_g_dict = m_g_result["M_g"]
        except ImportError:
            # Fallback: compute simple M_g
            M_g_dict = {}
        
        initial_lrs = {}
        for layer_idx, M_g_val in M_g_dict.items():
            if np.isinf(M_g_val) or np.isnan(M_g_val) or M_g_val <= 0:
                initial_lrs[layer_idx] = eta_0
            else:
                initial_lr = eta_0 * target_mobility / M_g_val
                initial_lrs[layer_idx] = initial_lr
    
    if len(initial_lrs) == 0:
        raise ValueError("No valid LRs computed from target mobility - all M_g values were invalid")
    
    return initial_lrs


def compute_optimal_lr_path(
    model,
    train_loader: DataLoader,
    device: torch.device,
    n_classes: int,
    alpha: float,
    n_batches: int = 1,
    return_per_layer: bool = False
) -> float | Dict[int, float]:
    """
    Compute optimal learning rate using path parameterization recipe:
    η_optimal ≈ median(d_f) / E[||∇w||]
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        device: Device to compute on
        n_classes: Number of output classes
        alpha: Label scaling factor
        n_batches: Number of batches to average over
        return_per_layer: If True, return dict of per-layer LRs
        
    Returns:
        Optimal learning rate (scalar) or dict mapping layer index -> optimal LR
    """
    model.train()
    loss_fn = nn.MSELoss()
    
    all_d_f = {l: [] for l in range(model.depth)}
    grad_norm_sums = {l: 0.0 for l in range(model.depth)}
    batch_count = 0
    
    for x_batch, y_batch in train_loader:
        if batch_count >= n_batches:
            break
        
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        # Compute d_f
        if isinstance(model, LazarusMLP):
            result_dict = compute_distance_to_flip_lazarus(model, x_batch, device, return_dead_neurons=False)
        else:
            try:
                from outputs.gates.gate_velocity_with_capacity import compute_distance_to_flip
                result_dict = compute_distance_to_flip(model, x_batch, device, return_dead_neurons=False)
            except ImportError:
                result_dict = {"distances": {}}
        distances = result_dict["distances"]
        
        for l in range(model.depth):
            if l in distances:
                d_f_flat = distances[l].detach().flatten().cpu().numpy()
                all_d_f[l].extend(d_f_flat)
        
        # Check for NaN
        has_nan = any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters())
        if has_nan:
            continue
        
        # Compute gradient norms
        if isinstance(model, LazarusMLP):
            grad_norms = compute_gradient_norms_lazarus(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        else:
            try:
                from outputs.gates.gate_velocity_with_capacity import compute_gradient_norms
                grad_norms = compute_gradient_norms(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
            except ImportError:
                grad_norms = {}
        
        # Validate gradients
        valid_grads = all(
            not (np.isnan(grad_norms.get(l, 0)) or np.isinf(grad_norms.get(l, 0)) or grad_norms.get(l, 0) < 0)
            for l in range(model.depth) if l in grad_norms
        )
        
        if valid_grads:
            for l in range(model.depth):
                if l in grad_norms:
                    grad_norm_sums[l] += grad_norms[l]
            batch_count += 1
    
    if batch_count == 0:
        if return_per_layer:
            return {l: 1e-4 for l in range(model.depth)}
        return 1e-4
    
    # Compute median d_f and average gradient norm for each layer
    median_d_f_per_layer = {}
    avg_grad_norm_per_layer = {}
    
    for l in range(model.depth):
        if len(all_d_f[l]) > 0:
            median_d_f_per_layer[l] = np.median(all_d_f[l])
        else:
            median_d_f_per_layer[l] = 0.0
        
        if batch_count > 0:
            avg_grad_norm_per_layer[l] = grad_norm_sums[l] / batch_count
        else:
            avg_grad_norm_per_layer[l] = 0.0
    
    # Compute optimal LR per layer
    optimal_lrs_per_layer = {}
    optimal_lrs = []
    for l in range(model.depth):
        if avg_grad_norm_per_layer[l] > 1e-8 and median_d_f_per_layer[l] > 0:
            optimal_lr = median_d_f_per_layer[l] / avg_grad_norm_per_layer[l]
            if not np.isnan(optimal_lr) and not np.isinf(optimal_lr) and optimal_lr > 0:
                optimal_lrs_per_layer[l] = optimal_lr
                optimal_lrs.append(optimal_lr)
            else:
                optimal_lrs_per_layer[l] = None
        else:
            optimal_lrs_per_layer[l] = None
    
    if return_per_layer:
        if len(optimal_lrs) > 0:
            median_fallback = float(np.median(optimal_lrs))
        else:
            median_fallback = 1e-4
        
        for l in range(model.depth):
            if optimal_lrs_per_layer[l] is None:
                optimal_lrs_per_layer[l] = median_fallback
        
        return optimal_lrs_per_layer
    
    if len(optimal_lrs) > 0:
        return float(np.median(optimal_lrs))
    return 1e-4


def compute_gate_mobility_per_layer_path(
    model,
    train_loader: DataLoader,
    current_lrs: Dict[int, float],
    device: torch.device,
    n_classes: int,
    alpha: float,
    n_batches: int = 1
) -> Dict[int, float]:
    """
    Compute gate mobility M_g,l for each layer using current learning rates.
    
    M_g,l = (η_l * E[||∇w_l||]) / E[d_f_l]
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        current_lrs: Dict mapping layer index -> current learning rate
        device: Device to compute on
        n_classes: Number of output classes
        alpha: Label scaling factor
        n_batches: Number of batches to average over
        
    Returns:
        Dict mapping layer index -> M_g,l value
    """
    model.train()
    loss_fn = nn.MSELoss()
    
    all_d_f = {l: [] for l in range(model.depth)}
    grad_norm_sums = {l: 0.0 for l in range(model.depth)}
    batch_count = 0
    
    for x_batch, y_batch in train_loader:
        if batch_count >= n_batches:
            break
        
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        # Compute d_f
        if isinstance(model, LazarusMLP):
            result_dict = compute_distance_to_flip_lazarus(model, x_batch, device, return_dead_neurons=False)
        else:
            try:
                from outputs.gates.gate_velocity_with_capacity import compute_distance_to_flip
                result_dict = compute_distance_to_flip(model, x_batch, device, return_dead_neurons=False)
            except ImportError:
                result_dict = {"distances": {}}
        distances = result_dict["distances"]
        
        for l in range(model.depth):
            if l in distances:
                d_f_flat = distances[l].detach().flatten().cpu().numpy()
                all_d_f[l].extend(d_f_flat)
        
        # Check for NaN
        has_nan = any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters())
        if has_nan:
            continue
        
        # Compute gradient norms
        if isinstance(model, LazarusMLP):
            grad_norms = compute_gradient_norms_lazarus(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        else:
            try:
                from outputs.gates.gate_velocity_with_capacity import compute_gradient_norms
                grad_norms = compute_gradient_norms(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
            except ImportError:
                grad_norms = {}
        
        # Validate gradients
        valid_grads = all(
            not (np.isnan(grad_norms.get(l, 0)) or np.isinf(grad_norms.get(l, 0)) or grad_norms.get(l, 0) < 0)
            for l in range(model.depth) if l in grad_norms
        )
        
        if valid_grads:
            for l in range(model.depth):
                if l in grad_norms:
                    grad_norm_sums[l] += grad_norms[l]
            batch_count += 1
    
    if batch_count == 0:
        return {l: None for l in range(model.depth)}
    
    # Compute M_g,l
    mean_d_f_per_layer = {}
    avg_grad_norm_per_layer = {}
    
    for l in range(model.depth):
        if len(all_d_f[l]) > 0:
            mean_d_f_per_layer[l] = np.mean(all_d_f[l])
        else:
            mean_d_f_per_layer[l] = 0.0
        
        if batch_count > 0:
            avg_grad_norm_per_layer[l] = grad_norm_sums[l] / batch_count
        else:
            avg_grad_norm_per_layer[l] = 0.0
    
    M_g_per_layer = {}
    
    if isinstance(model, LazarusMLP):
        # Handle input_proj (layer -1)
        INPUT_PROJ_LAYER = -1
        if INPUT_PROJ_LAYER in current_lrs and 0 in mean_d_f_per_layer and 0 in avg_grad_norm_per_layer:
            if mean_d_f_per_layer[0] > 1e-8 and avg_grad_norm_per_layer[0] > 1e-8:
                eta_l = current_lrs[INPUT_PROJ_LAYER]
                M_g_l = (eta_l * avg_grad_norm_per_layer[0]) / mean_d_f_per_layer[0]
                if not np.isnan(M_g_l) and not np.isinf(M_g_l) and M_g_l > 0:
                    M_g_per_layer[INPUT_PROJ_LAYER] = M_g_l
                else:
                    M_g_per_layer[INPUT_PROJ_LAYER] = None
            else:
                M_g_per_layer[INPUT_PROJ_LAYER] = None
        else:
            M_g_per_layer[INPUT_PROJ_LAYER] = None
        
        # Handle blocks (layers 0 to depth-1)
        for l in range(model.depth):
            if l in current_lrs and mean_d_f_per_layer[l] > 1e-8 and avg_grad_norm_per_layer[l] > 1e-8:
                eta_l = current_lrs[l]
                M_g_l = (eta_l * avg_grad_norm_per_layer[l]) / mean_d_f_per_layer[l]
                if not np.isnan(M_g_l) and not np.isinf(M_g_l) and M_g_l > 0:
                    M_g_per_layer[l] = M_g_l
                else:
                    M_g_per_layer[l] = None
            else:
                M_g_per_layer[l] = None
    else:
        # Standard MLP
        for l in range(model.depth):
            if l in current_lrs and mean_d_f_per_layer[l] > 1e-8 and avg_grad_norm_per_layer[l] > 1e-8:
                eta_l = current_lrs[l]
                M_g_l = (eta_l * avg_grad_norm_per_layer[l]) / mean_d_f_per_layer[l]
                if not np.isnan(M_g_l) and not np.isinf(M_g_l) and M_g_l > 0:
                    M_g_per_layer[l] = M_g_l
                else:
                    M_g_per_layer[l] = None
            else:
                M_g_per_layer[l] = None
    
    return M_g_per_layer


def update_lr_damped_mobility(
    current_lrs: Dict[int, float],
    M_g_per_layer: Dict[int, float],
    target_mobility: float,
    alpha: float,
    eps: float,
    min_scale: float,
    max_scale: float,
    lr_max: float = 1.0
) -> Dict[int, float]:
    """
    Update learning rates using damped mobility-based rule.
    
    η_l(t+1) = clamp(η_l(t) * clamp((M* / (M_g,l(t) + eps))^alpha, min_scale, max_scale), 0, lr_max)
    
    Args:
        current_lrs: Dict mapping layer index -> current learning rate
        M_g_per_layer: Dict mapping layer index -> measured mobility
        target_mobility: Target mobility M*
        alpha: Smoothing exponent α ∈ (0,1]
        eps: Small epsilon to avoid division by zero
        min_scale: Minimum multiplicative change per step
        max_scale: Maximum multiplicative change per step
        lr_max: Maximum learning rate cap
        
    Returns:
        Dict mapping layer index -> updated learning rate
    """
    updated_lrs = {}
    
    for l in current_lrs:
        if l in M_g_per_layer and M_g_per_layer[l] is not None:
            M_g_l = M_g_per_layer[l]
            ratio = target_mobility / (M_g_l + eps)
            scale = ratio ** alpha
            scale = max(min_scale, min(max_scale, scale))
            updated_lrs[l] = min(current_lrs[l] * scale, lr_max)
        else:
            # If M_g computation failed, reduce LR by min_scale (conservative fallback)
            updated_lrs[l] = min(current_lrs[l] * min_scale, lr_max)
    
    return updated_lrs


def compute_gradient_norms_per_layer(
    model, x_batch: torch.Tensor, y_batch: torch.Tensor,
    loss_fn, device: torch.device, n_classes: int = 1, alpha: float = 1.0
) -> Dict[int, float]:
    """
    Compute gradient norms per layer (before clipping) for any architecture.
    Returns dict mapping layer index -> gradient norm.
    """
    model.train()
    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)
    
    model.zero_grad()
    yhat = model(x_batch)
    
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
    
    gradient_norms = {}
    
    if isinstance(model, LazarusMLP):
        for l, block in enumerate(model.blocks):
            first_linear = block[0]
            if first_linear.weight.grad is not None:
                grad_norm = torch.norm(first_linear.weight.grad).item()
                gradient_norms[l] = grad_norm
            else:
                gradient_norms[l] = 0.0
    else:
        if hasattr(model, 'linears'):
            for l, linear in enumerate(model.linears):
                if linear.weight.grad is not None:
                    grad_norm = torch.norm(linear.weight.grad).item()
                    gradient_norms[l] = grad_norm
                else:
                    gradient_norms[l] = 0.0
        else:
            layer_idx = 0
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    if m.weight.grad is not None:
                        grad_norm = torch.norm(m.weight.grad).item()
                        gradient_norms[layer_idx] = grad_norm
                    else:
                        gradient_norms[layer_idx] = 0.0
                    layer_idx += 1
    
    return gradient_norms


# Re-export path kernel functions from existing module
def compute_path_deformation_capacity(X, E_list, device, block_size=2048, dtype=torch.float32, use_tf32=True):
    """Compute path deformation capacity. Wrapper around existing implementation."""
    try:
        from outputs.gates.gate_velocity_with_capacity import compute_path_deformation_capacity as _compute
        return _compute(X, E_list, device, block_size=block_size, dtype=dtype, use_tf32=use_tf32)
    except ImportError:
        return float('nan')


def compute_path_covariance_entropy(E_list, device, n_bins=100, block_size=2048, dtype=torch.float32, 
                                     use_tf32=True, normalize_factors=False, epsilon=1e-12, use_float64=True):
    """Compute path covariance entropy. Wrapper around existing implementation."""
    try:
        from outputs.gates.gate_velocity_with_capacity import compute_path_covariance_entropy as _compute
        return _compute(E_list, device, n_bins=n_bins, block_size=block_size, dtype=dtype,
                       use_tf32=use_tf32, normalize_factors=normalize_factors, epsilon=epsilon, use_float64=use_float64)
    except ImportError:
        return float('nan')


def compute_path_kernel_matrix(factors, device, block_size=2048, dtype=torch.float32, use_tf32=True):
    """Compute path kernel matrix. Wrapper around existing implementation."""
    try:
        from outputs.gates.gate_velocity_with_capacity import compute_path_kernel_matrix as _compute
        return _compute(factors, device, block_size=block_size, dtype=dtype, use_tf32=use_tf32)
    except ImportError:
        return None


def compute_gate_mobility(model, train_loader, lr, device, n_batches=10, n_classes=1, alpha=1.0, return_distributions=False):
    """Compute gate mobility. Wrapper around existing implementation."""
    try:
        from outputs.gates.gate_velocity_with_capacity import compute_gate_mobility as _compute
        return _compute(model, train_loader, lr, device, n_batches=n_batches, n_classes=n_classes, 
                       alpha=alpha, return_distributions=return_distributions)
    except ImportError:
        return {"M_g": {}, "dead_neuron_fractions": {}}

