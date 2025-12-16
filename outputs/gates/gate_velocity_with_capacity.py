"""
Gate Velocity / Gate Mobility Number Experiment with Path Kernel Capacity Metrics

This experiment computes:
1. Gate Mobility Number (M_g): measures the energy cost required to flip a gate
2. Path Deformation Capacity (C_def): measures how much the path kernel H differs from input kernel Σ
3. Path Covariance Entropy (H_Λ): entropy of the path overlap matrix Λ

M_g = (η * E[||∇w||]) / E[d_f]
C_def = ||H_norm - Σ_norm||_F  (Frobenius norm of normalized kernel difference)
H_Λ = -∑_{i,j} P(Λ_ij) log P(Λ_ij)  (entropy of path overlap distribution)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import json
import copy
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy

import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.data.models.ffnn import MLP
from src.data.mnist import build_mnist_datasets
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.save_io import ensure_dir
from src.analysis.path_kernel import collect_path_factors, HadamardGramOperator

# Import model variants from local files (same directory)
# Add script directory to path for local imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from mlp_resnet import MLPResNet
from mlp_batchnorm import MLPBatchNorm
from mlp_skip import MLPSkip


@torch.no_grad()
def compute_distance_to_flip(
    model, x_batch: torch.Tensor, device: torch.device, 
    return_dead_neurons: bool = False
) -> Dict:
    """
    Compute distance to sign flip d_f(i) = |h_i(x)| / ||x|| for each neuron in each layer.
    GPU-optimized version.
    
    Args:
        model: The neural network model
        x_batch: Input batch (batch_size, d_in)
        device: Device to run on
        return_dead_neurons: If True, also return dead neuron fractions
    
    Returns:
        Dictionary with:
        - "distances": Dict mapping layer index to tensor of distances (batch_size, width_l)
        - "dead_neuron_fractions": Dict mapping layer index to dead neuron fraction (if requested)
    """
    model.eval()
    x_batch = x_batch.to(device, non_blocking=True)
    
    distances = {}
    dead_neuron_fractions = {} if return_dead_neurons else None
    h = x_batch  # Start with input
    
    # Threshold for considering a neuron "dead" (very small activation)
    DEAD_THRESHOLD = 1e-6
    
    for l in range(model.depth):
        # Get pre-activation: h_i(x) = w_i · x + b_i
        linear = model.linears[l]
        u = linear(h)  # (batch_size, width_l)
        
        # Compute ||x|| for each sample in batch
        # x is h (the input to this layer)
        x_norm = torch.norm(h, dim=1, keepdim=True)  # (batch_size, 1)
        
        # Track dead neurons (where x_norm is essentially zero)
        if return_dead_neurons:
            # A neuron is "dead" if the input norm is below threshold
            # Count samples where x_norm < DEAD_THRESHOLD
            dead_samples = (x_norm.squeeze() < DEAD_THRESHOLD).float()  # (batch_size,)
            dead_fraction = dead_samples.mean().item()
            dead_neuron_fractions[l] = dead_fraction
        
        # Avoid division by zero
        x_norm = torch.clamp(x_norm, min=1e-8)
        
        # Distance to flip: d_f(i) = |h_i(x)| / ||x||
        d_f = torch.abs(u) / x_norm  # (batch_size, width_l)
        
        distances[l] = d_f
        
        # Update h for next layer (after activation)
        h = model.activation(u)
    
    result = {"distances": distances}
    if return_dead_neurons:
        result["dead_neuron_fractions"] = dead_neuron_fractions
    
    return result


def compute_gradient_norms(model, x_batch: torch.Tensor, y_batch: torch.Tensor, 
                           loss_fn, device: torch.device, n_classes: int = 1, alpha: float = 1.0) -> Dict[int, float]:
    """
    Compute gradient norms for each layer. GPU-optimized.
    
    CRITICAL: This function assumes loss_fn uses MEAN reduction (default for MSELoss).
    The gradient norm computed here is the norm of the mean gradient over the batch,
    which is the correct metric for measuring effective step size in SGD.
    
    Args:
        model: The neural network model
        x_batch: Input batch
        y_batch: Target batch
        loss_fn: Loss function (MUST use mean reduction, e.g., nn.MSELoss())
        device: Device to run on
        n_classes: Number of classes (1 for binary, >1 for multiclass)
        alpha: Alpha scaling factor for labels
    
    Returns:
        Dictionary mapping layer index to gradient norm (scalar)
    """
    model.train()
    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)
    
    # Zero gradients
    model.zero_grad()
    
    # Forward pass
    yhat = model(x_batch)
    
    # Handle loss computation based on number of classes
    if n_classes == 1:
        # Binary classification: MSE loss (uses mean reduction by default)
        loss = loss_fn(yhat, y_batch)
    else:
        # Multi-class classification: MSE loss with one-hot targets scaled by alpha
        # y_batch contains scaled class indices (e.g., 0, 10, 20, ..., 90 for alpha=10)
        if y_batch.dim() > 1:
            y_batch = y_batch.view(-1)  # Flatten to 1D
        yb_class = (y_batch / alpha).long()  # Get original class index
        # Clamp to valid range [0, n_classes-1] to avoid CUDA assert errors
        yb_class = torch.clamp(yb_class, 0, n_classes - 1)
        yb_onehot = torch.zeros_like(yhat)
        src_values = torch.ones_like(y_batch.unsqueeze(1)) * alpha
        yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values)  # Fill with scaled alpha value
        loss = loss_fn(yhat, yb_onehot)  # Uses mean reduction by default
    
    # Backward pass
    # With mean reduction, loss.backward() computes mean gradients over batch
    # weight.grad contains the mean gradient, so ||weight.grad|| is the correct metric
    loss.backward()
    
    gradient_norms = {}
    for l in range(model.depth):
        weight = model.linears[l].weight
        if weight.grad is not None:
            # This is the norm of the mean gradient (correct for measuring step size)
            grad_norm = torch.norm(weight.grad).item()
            gradient_norms[l] = grad_norm
        else:
            gradient_norms[l] = 0.0
    
    return gradient_norms


@torch.no_grad()
def compute_path_kernel_matrix(
    factors: List[torch.Tensor],
    device: torch.device,
    block_size: int = 1024,
    dtype: torch.dtype = torch.float32,
    use_tf32: bool = True,
) -> torch.Tensor:
    """
    Compute the full path kernel matrix H = ∘_f (F_f F_f^T).
    
    Args:
        factors: List of factor matrices [X, E_1, ..., E_L] where each is (P, d_f)
        device: Device to run on
        block_size: Block size for computation
        dtype: Data type
        use_tf32: Whether to use TF32
    
    Returns:
        H: (P, P) path kernel matrix
    """
    if len(factors) == 0:
        raise ValueError("At least one factor required")
    
    P = factors[0].shape[0]
    
    # Use HadamardGramOperator to compute H @ I efficiently
    op = HadamardGramOperator(
        factors, device=device, dtype=dtype, block_size=block_size, use_tf32=use_tf32
    )
    
    # Compute H by applying to identity matrix in blocks
    H = torch.zeros((P, P), device=device, dtype=dtype)
    
    # Process in blocks to manage memory
    for i0 in range(0, P, block_size):
        i1 = min(P, i0 + block_size)
        # Create identity block: columns i0 to i1
        I_block = torch.eye(P, device=device, dtype=dtype)[:, i0:i1]  # (P, block_size)
        H[:, i0:i1] = op.mm(I_block)  # (P, block_size)
    
    return H


@torch.no_grad()
def compute_path_deformation_capacity(
    X: torch.Tensor,
    E_list: List[torch.Tensor],
    device: torch.device,
    block_size: int = 1024,
    dtype: torch.dtype = torch.float32,
    use_tf32: bool = True,
) -> float:
    """
    Compute Path Deformation Capacity: C_def = ||H_norm - Σ_norm||_F
    
    Measures how much the path kernel H differs from the input kernel Σ.
    High value = high non-linear capacity.
    
    Args:
        X: Input matrix (P, d_in)
        E_list: List of path factor matrices [E_1, ..., E_L] where each is (P, d_l)
        device: Device to run on
        block_size: Block size for kernel computation
        dtype: Data type
        use_tf32: Whether to use TF32
    
    Returns:
        C_def: Path deformation capacity (scalar)
    """
    # Compute input kernel: Σ = X @ X.T
    Sigma = X @ X.T  # (P, P)
    
    # Normalize Σ: trace = 1
    trace_Sigma = torch.trace(Sigma)
    if trace_Sigma > 0:
        Sigma_norm = Sigma / trace_Sigma
    else:
        Sigma_norm = Sigma
    
    # Compute path kernel: H = (X X^T) ∘ ∏_l (E_l E_l^T)
    factors = [X] + E_list
    H = compute_path_kernel_matrix(factors, device, block_size, dtype, use_tf32)
    
    # Normalize H: trace = 1
    trace_H = torch.trace(H)
    if trace_H > 0:
        H_norm = H / trace_H
    else:
        H_norm = H
    
    # Compute Frobenius norm of difference
    diff = H_norm - Sigma_norm
    C_def = torch.norm(diff, p='fro').item()
    
    return C_def


@torch.no_grad()
def compute_path_covariance_entropy(
    E_list: List[torch.Tensor],
    device: torch.device,
    n_bins: int = 100,
    block_size: int = 1024,
    dtype: torch.dtype = torch.float32,
    use_tf32: bool = True,
    normalize_factors: bool = True,
    epsilon: float = 1e-12,
    use_float64: bool = True,
) -> float:
    """
    Compute Path Covariance Entropy: H_Λ = -∑_{i,j} P(Λ_ij) log P(Λ_ij)
    
    Measures the entropy of the path overlap matrix Λ.
    Λ_ij = product over layers of (E_l[i] @ E_l[j])
    
    Args:
        E_list: List of path factor matrices [E_1, ..., E_L] where each is (P, d_l)
        device: Device to run on
        n_bins: Number of bins for histogram
        block_size: Block size for computation
        dtype: Data type
        use_tf32: Whether to use TF32
    
    Returns:
        H_Λ: Path covariance entropy (scalar)
    """
    if len(E_list) == 0:
        return 0.0
    
    P = E_list[0].shape[0]
    
    # Optional normalization to avoid overflow/underflow in the Hadamard product.
    # Using Frobenius scaling preserves the normalized kernel (same logic as C_def).
    proc_E_list = []
    for E in E_list:
        if normalize_factors:
            fro = torch.norm(E, p="fro")
            if fro > 1e-8:
                proc_E_list.append(E / fro)
            else:
                proc_E_list.append(E)
        else:
            proc_E_list.append(E)
    
    # Use higher precision by default to reduce overflow risk
    entropy_dtype = torch.float64 if use_float64 else dtype

    # Compute Λ matrix: Λ = ∏_l (E_l E_l^T) (path overlap without input)
    # We can compute this efficiently using HadamardGramOperator
    op = HadamardGramOperator(
        proc_E_list, device=device, dtype=entropy_dtype, block_size=block_size, use_tf32=use_tf32
    )
    
    # Compute Λ by applying to identity matrix in blocks
    # But we only need the values, not the full matrix
    # For entropy, we can sample or compute full matrix
    # For memory efficiency, let's compute in blocks and accumulate histogram
    
    # Compute Λ in blocks and accumulate values
    Lambda_values = []
    block_size_hist = min(block_size, 512)  # Smaller blocks for histogram accumulation
    
    for i0 in range(0, P, block_size_hist):
        i1 = min(P, i0 + block_size_hist)
        # Create identity block
        I_block = torch.eye(P, device=device, dtype=entropy_dtype)[:, i0:i1]  # (P, block_size)
        Lambda_block = op.mm(I_block)  # (P, block_size)
        
        # Clamp to avoid exact zeros (which collapse entropy) and strip inf/nan
        Lambda_block = torch.clamp(Lambda_block, min=epsilon)
        Lambda_block = torch.where(torch.isfinite(Lambda_block), Lambda_block, torch.zeros_like(Lambda_block))
        
        # Extract values and move to CPU
        Lambda_values.append(Lambda_block.cpu().numpy().flatten())
    
    # Concatenate all values
    Lambda_flat = np.concatenate(Lambda_values)
    
    # Compute histogram
    # Filter out any NaN or inf values
    Lambda_flat = Lambda_flat[np.isfinite(Lambda_flat)]
    
    if len(Lambda_flat) == 0:
        return 0.0
    
    # Check if all values are identical (this causes entropy = 0)
    Lambda_min = Lambda_flat.min()
    Lambda_max = Lambda_flat.max()
    Lambda_std = Lambda_flat.std()
    Lambda_unique = len(np.unique(Lambda_flat))
    
    # If all values are identical (or very close), entropy will be 0
    if Lambda_max <= Lambda_min + 1e-10 or Lambda_unique <= 1:
        # All Lambda values are identical - this means routing is completely uniform
        # Return 0.0 (genuine zero entropy, not an error)
        return 0.0
    
    # Normalize to [0, 1] for consistent binning
    if Lambda_max > Lambda_min:
        Lambda_normalized = (Lambda_flat - Lambda_min) / (Lambda_max - Lambda_min)
    else:
        Lambda_normalized = Lambda_flat * 0.0  # All zeros (shouldn't reach here due to check above)
    
    # Compute histogram
    hist, bin_edges = np.histogram(Lambda_normalized, bins=n_bins, range=(0.0, 1.0))
    
    # Normalize to get probabilities
    hist = hist.astype(float)
    hist_sum = hist.sum()
    if hist_sum > 0:
        probs = hist / hist_sum
        # Filter out zero probabilities for entropy calculation
        probs = probs[probs > 0]
        H_Lambda = entropy(probs, base=2)  # Use base 2 for bits
    else:
        H_Lambda = 0.0
    
    return float(H_Lambda)


def compute_gate_mobility(
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
    Compute Gate Mobility Number M_g for each layer.
    GPU-optimized with batched operations.
    
    M_g = (η * E[||∇w||]) / E[d_f]
    
    Args:
        model: The neural network model
        train_loader: Data loader for training data
        lr: Learning rate η
        device: Device to run on
        n_batches: Number of batches to average over
        n_classes: Number of classes (1 for binary, >1 for multiclass)
        alpha: Alpha scaling factor for labels
        return_distributions: If True, also return d_f distributions
    
    Returns:
        Dictionary with M_g values, dead_neuron_fractions, and optionally d_f distributions
    """
    model = model.to(device)
    
    # Loss function (MSE) - CRITICAL: uses mean reduction by default
    # This ensures weight.grad contains mean gradients over batch
    loss_fn = nn.MSELoss()  # reduction='mean' is default
    
    # Use torch tensors for accumulation (GPU-friendly)
    grad_norm_sums = torch.zeros(model.depth, device=device)
    distance_sums = torch.zeros(model.depth, device=device)
    
    # Track dead neuron fractions (accumulate over batches)
    dead_neuron_sums = torch.zeros(model.depth, device=device)
    
    # Store distributions if requested
    d_f_distributions = {l: [] for l in range(model.depth)} if return_distributions else None
    
    batch_count = 0
    
    for x_batch, y_batch in train_loader:
        if batch_count >= n_batches:
            break
        
        # Compute distances to flip (now returns dict with "distances" and "dead_neuron_fractions")
        result_dict = compute_distance_to_flip(model, x_batch, device, return_dead_neurons=True)
        distances = result_dict["distances"]
        dead_fractions = result_dict["dead_neuron_fractions"]
        
        # Compute gradient norms (uses mean reduction, so ||weight.grad|| is correct)
        grad_norms = compute_gradient_norms(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        
        # Accumulate on GPU
        for l in range(model.depth):
            # Average distance over batch and neurons (keep on GPU)
            mean_dist = distances[l].mean()
            distance_sums[l] += mean_dist
            
            # Accumulate dead neuron fractions
            dead_neuron_sums[l] += dead_fractions[l]
            
            # Store distribution if requested (flatten and move to CPU)
            if return_distributions:
                d_f_flat = distances[l].flatten().cpu().numpy()
                d_f_distributions[l].extend(d_f_flat)
            
            # Gradient norm (convert to tensor)
            grad_norm_sums[l] += grad_norms[l]
        
        batch_count += 1
    
    # Compute M_g for each layer
    M_g = {}
    dead_neuron_fractions = {}
    for l in range(model.depth):
        E_grad_norm = (grad_norm_sums[l] / batch_count).item()
        E_d_f = (distance_sums[l] / batch_count).item()
        avg_dead_fraction = (dead_neuron_sums[l] / batch_count).item()
        
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


def run_experiment(
    depths: List[int],
    widths: List[int],
    device: torch.device,
    config_path: str = "configs/config_mnist.yaml",
    n_batches: int = 10,
    batch_size: int = 1024,
    architecture: str = "standard",  # "standard", "batchnorm", "skip", "resnet"
    max_samples_for_kernel: Optional[int] = 5000,  # Limit samples for kernel computation
) -> Dict:
    """
    Run the gate velocity experiment with capacity metrics for all depth/width combinations.
    
    Args:
        depths: List of depths (number of hidden layers)
        widths: List of widths (neurons per layer)
        device: Device to run on
        config_path: Path to config file
        n_batches: Number of batches to average over for M_g
        batch_size: Batch size for data loading
        architecture: Architecture type
        max_samples_for_kernel: Maximum number of samples to use for kernel computation
    
    Returns:
        Dictionary with results: {depth: {width: {M_g, C_def, H_Lambda, ...}}}
    """
    # Load config
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    
    # Fix config to handle list values (for alpha and n_train)
    cfg_fixed = copy.deepcopy(cfg)
    if isinstance(cfg_fixed["dataset"].get("alpha"), list):
        cfg_fixed["dataset"]["alpha"] = cfg_fixed["dataset"]["alpha"][0]
    if isinstance(cfg_fixed["dataset"].get("n_train"), list):
        cfg_fixed["dataset"]["n_train"] = cfg_fixed["dataset"]["n_train"][0]
    
    # Build dataset
    Xtr, ytr, Xva, yva, Xte, yte, meta = build_mnist_datasets(cfg_fixed)
    input_dim = meta["d"]
    n_classes = meta["n_classes"]
    alpha = meta.get("alpha", 1.0)
    
    # Create data loaders (pin memory for faster GPU transfer)
    train_dataset = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), 
                                  torch.tensor(ytr, dtype=torch.float32))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Create a separate loader for kernel computation (with max_samples limit)
    train_loader_kernel = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Don't shuffle for consistent sampling
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )
    
    # Get learning rate (ensure it's a float, not a list)
    lr_cfg = cfg["training"]["lr_w"]
    if isinstance(lr_cfg, list):
        lr = float(lr_cfg[0])
    else:
        lr = float(lr_cfg)
    
    results = {}
    
    # Select model class based on architecture
    if architecture == "standard":
        model_class = MLP
        arch_name = "standard"
    elif architecture == "batchnorm":
        model_class = MLPBatchNorm
        arch_name = "batchnorm"
    elif architecture == "skip":
        model_class = MLPSkip
        arch_name = "skip"
    elif architecture == "resnet":
        model_class = MLPResNet
        arch_name = "resnet"
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Options: standard, batchnorm, skip, resnet")
    
    for depth in tqdm(depths, desc=f"Depths ({arch_name})"):
        results[depth] = {}
        for width in tqdm(widths, desc=f"Widths (L={depth}, {arch_name})", leave=False):
            # Create model with specified depth and width
            model_widths = [width] * depth
            model = model_class(
                d_in=input_dim,
                widths=model_widths,
                bias=cfg["model"].get("bias", True),
                activation=cfg["model"].get("activation", "relu"),
                n_classes=n_classes
            ).to(device)
            
            # Initialize model (Xavier/Kaiming initialization)
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            
            # Compute M_g and d_f distributions
            result_dict = compute_gate_mobility(model, train_loader, lr, device, n_batches, n_classes, alpha, return_distributions=True)
            M_g_dict = result_dict["M_g"]
            d_f_distributions = result_dict["d_f_distributions"]
            dead_neuron_fractions = result_dict["dead_neuron_fractions"]
            
            # Compute path kernel capacity metrics
            print(f"  Computing path kernel metrics for depth={depth}, width={width}...")
            try:
                # Collect path factors
                pack = collect_path_factors(
                    model, train_loader_kernel, device, 
                    mode="routing_gain", 
                    include_input=True, 
                    max_samples=max_samples_for_kernel
                )
                X = pack["X"]
                E_list = pack["E_list"]
                
                if X is not None and len(E_list) > 0:
                    # Compute Path Deformation Capacity
                    # H100-optimized: larger block_size for better GPU utilization
                    C_def = compute_path_deformation_capacity(
                        X, E_list, device, block_size=2048, dtype=torch.float32, use_tf32=True
                    )
                    
                    # Compute Path Covariance Entropy
                    # H100-optimized: larger block_size for better GPU utilization
                    H_Lambda = compute_path_covariance_entropy(
                        E_list, device, n_bins=100, block_size=2048, dtype=torch.float32, use_tf32=True
                    )
                else:
                    C_def = float('nan')
                    H_Lambda = float('nan')
                    print(f"    Warning: Could not collect path factors")
            except Exception as e:
                print(f"    Error computing path kernel metrics: {e}")
                C_def = float('nan')
                H_Lambda = float('nan')
            
            # Store results (average M_g across layers)
            avg_M_g = np.mean(list(M_g_dict.values()))
            avg_dead_fraction = np.mean(list(dead_neuron_fractions.values()))
            
            # Convert numpy arrays to Python lists for JSON serialization
            d_f_distributions_serializable = {}
            for l in range(depth):
                # Convert numpy array to list of native Python floats
                d_f_list = d_f_distributions[l]
                if isinstance(d_f_list, np.ndarray):
                    d_f_distributions_serializable[str(l)] = [float(x) for x in d_f_list]
                elif isinstance(d_f_list, list):
                    d_f_distributions_serializable[str(l)] = [float(x) for x in d_f_list]
                else:
                    d_f_distributions_serializable[str(l)] = []
            
            results[depth][width] = {
                "M_g_avg": float(avg_M_g),
                "M_g_by_layer": {str(l): float(M_g_dict[l]) for l in range(depth)},
                "dead_neuron_fraction_avg": float(avg_dead_fraction),
                "dead_neuron_fractions_by_layer": {str(l): float(dead_neuron_fractions[l]) for l in range(depth)},
                "C_def": float(C_def),
                "H_Lambda": float(H_Lambda),
                "d_f_distributions": d_f_distributions_serializable,
                "depth": depth,
                "width": width
            }
            
            # Warn if dead neuron fraction is high (indicates broken/frozen network)
            if avg_dead_fraction > 0.5:
                print(f"    WARNING: High dead neuron fraction ({avg_dead_fraction:.2%}) - M_g may be unreliable!")
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


def plot_d_f_distributions(results: Dict, out_dir: str, suffix: str = ""):
    """
    Plot distributions of d_f(i) for each layer, grouped by depth.
    Each depth gets its own plot with layers stacked vertically.
    Different widths are color-coded.
    
    Args:
        results: Results dictionary from run_experiment
        out_dir: Output directory for plots
    """
    ensure_dir(out_dir)
    
    depths = sorted(results.keys())
    widths = sorted(set(w for d in results.values() for w in d.keys()))
    
    # Color map for widths
    colors = plt.cm.tab10(np.linspace(0, 1, len(widths)))
    width_to_color = {w: colors[i] for i, w in enumerate(widths)}
    
    for depth in depths:
        if depth not in results:
            continue
        
        # Create figure with subplots (one per layer)
        fig, axes = plt.subplots(depth, 1, figsize=(12, 3 * depth))
        if depth == 1:
            axes = [axes]  # Make it a list for consistency
        
        # Determine title suffix based on architecture
        if "_resnet" in suffix:
            title_suffix = " (ResNet: Skip + BatchNorm)"
        elif "_batchnorm" in suffix:
            title_suffix = " (BatchNorm Only)"
        elif "_skip" in suffix:
            title_suffix = " (Skip Connections Only)"
        else:
            title_suffix = " (Standard)"
        fig.suptitle(f'Distribution of d_f(i) by Layer (Depth = {depth}){title_suffix}', fontsize=16, y=0.995)
        
        for layer_idx in range(depth):
            ax = axes[layer_idx]
            
            # Plot distributions for each width
            for width in widths:
                if width not in results[depth]:
                    continue
                
                d_f_data = results[depth][width]["d_f_distributions"].get(str(layer_idx), [])
                if len(d_f_data) == 0:
                    continue
                
                # Convert to numpy array and filter out extreme values for better visualization
                d_f_array = np.array(d_f_data)
                # Filter to reasonable range (remove outliers beyond 99th percentile)
                p99 = np.percentile(d_f_array, 99)
                d_f_filtered = d_f_array[d_f_array <= p99]
                
                # Plot histogram/KDE
                ax.hist(d_f_filtered, bins=50, alpha=0.6, label=f'W={width}', 
                       color=width_to_color[width], density=True)
            
            ax.set_xlabel(f'Distance to Flip d_f(i)', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'Layer {layer_idx}', fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'd_f_distributions_depth_{depth}{suffix}.png'
        plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved d_f distribution plot for depth {depth}{suffix}")


def plot_heatmaps(results: Dict, out_dir: str, suffix: str = "", metric: str = "M_g"):
    """
    Create heatmaps of a metric vs depth and width.
    
    Args:
        results: Results dictionary from run_experiment
        out_dir: Output directory for plots
        suffix: Suffix for filename
        metric: Metric to plot ("M_g", "C_def", "H_Lambda")
    """
    ensure_dir(out_dir)
    
    # Extract data for heatmap
    depths = sorted(results.keys())
    widths = sorted(set(w for d in results.values() for w in d.keys()))
    
    # Create matrix for the metric
    metric_matrix = np.zeros((len(depths), len(widths)))
    
    metric_key_map = {
        "M_g": "M_g_avg",
        "C_def": "C_def",
        "H_Lambda": "H_Lambda"
    }
    
    metric_key = metric_key_map.get(metric, metric)
    metric_label_map = {
        "M_g": "M_g (Gate Mobility Number)",
        "C_def": "C_def (Path Deformation Capacity)",
        "H_Lambda": "H_Λ (Path Covariance Entropy)"
    }
    metric_label = metric_label_map.get(metric, metric)
    
    for i, depth in enumerate(depths):
        for j, width in enumerate(widths):
            if width in results[depth]:
                value = results[depth][width].get(metric_key, np.nan)
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    metric_matrix[i, j] = value
                else:
                    metric_matrix[i, j] = np.nan
            else:
                metric_matrix[i, j] = np.nan
    
    # Determine title suffix based on architecture
    if "_resnet" in suffix:
        title_suffix = " (ResNet: Skip + BatchNorm)"
    elif "_batchnorm" in suffix:
        title_suffix = " (BatchNorm Only)"
    elif "_skip" in suffix:
        title_suffix = " (Skip Connections Only)"
    else:
        title_suffix = " (Standard)"
    
    # Normal scale heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(metric_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label=metric_label)
    plt.xticks(range(len(widths)), widths)
    plt.yticks(range(len(depths)), depths)
    plt.xlabel('Width (neurons per layer)')
    plt.ylabel('Depth (number of layers)')
    plt.title(f'{metric_label} (Normal Scale){title_suffix}')
    
    # Add text annotations
    for i in range(len(depths)):
        for j in range(len(widths)):
            if not np.isnan(metric_matrix[i, j]):
                text = plt.text(j, i, f'{metric_matrix[i, j]:.3f}',
                              ha="center", va="center", 
                              color="white" if metric_matrix[i, j] > np.nanmean(metric_matrix) else "black",
                              fontsize=8)
    
    plt.tight_layout()
    filename = f'{metric}_heatmap_normal{suffix}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()
    
    # Log scale heatmap (if values are positive)
    metric_matrix_log = np.copy(metric_matrix)
    metric_matrix_log[metric_matrix_log <= 0] = np.nan
    if np.any(metric_matrix_log > 0):
        metric_matrix_log = np.log10(metric_matrix_log)
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(metric_matrix_log, cmap='viridis', aspect='auto')
        plt.colorbar(im, label=f'log10({metric_label})')
        plt.xticks(range(len(widths)), widths)
        plt.yticks(range(len(depths)), depths)
        plt.xlabel('Width (neurons per layer)')
        plt.ylabel('Depth (number of layers)')
        plt.title(f'{metric_label} (Log Scale){title_suffix}')
        
        # Add text annotations
        for i in range(len(depths)):
            for j in range(len(widths)):
                if not np.isnan(metric_matrix_log[i, j]):
                    text = plt.text(j, i, f'{metric_matrix_log[i, j]:.3f}',
                                  ha="center", va="center", 
                                  color="white" if metric_matrix_log[i, j] > np.nanmean(metric_matrix_log) else "black",
                                  fontsize=8)
        
        plt.tight_layout()
        filename = f'{metric}_heatmap_log{suffix}.png'
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()
    
    print(f"Saved {metric} heatmaps{suffix} to {out_dir}")


def main():
    # Configuration
    depths =  [1,2,3, 4, 7,12]
    widths = [16,32, 64,128,256,512, 1024,2048,5124]
    
    # Flag to control JSON saving (set to True to save results)
    save_json_results = False
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Output directory - user can modify this
    # Default: outputs/gates/capacity_experiment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out_dir = os.path.join(script_dir, "capacity_experiment")
    
    # Allow user to specify output directory
    # You can modify this line to set your desired directory
    # Examples:
    #   out_dir = default_out_dir  # Use default (outputs/gates/capacity_experiment)
    #   out_dir = "/path/to/my/output"  # Use absolute path
    #   out_dir = os.path.join(project_root, "my_custom_output")  # Use path relative to project root
    out_dir = "/home/goring/NN_alternatecoding/outputs/gates/results/10_12"  # CHANGE THIS LINE to your desired output directory
    
    ensure_dir(out_dir)
    print(f"Output directory: {out_dir}")
    
    # Config path (relative to project root)
    config_path = os.path.join(project_root, "configs", "config_mnist.yaml")
    
    # Run experiments for all architectures
    print("Starting Gate Velocity + Capacity Experiment...")
    print(f"Depths: {depths}")
    print(f"Widths: {widths}")
    print(f"Output directory: {out_dir}")
    
    # Number of samples to use: n_batches * batch_size
    # H100-optimized settings (80GB VRAM):
    n_batches = 16  # More batches for better M_g statistics
    batch_size = 8192  # H100 can handle large batches efficiently (2-4x faster than 1024)
    max_samples_for_kernel = 8192  # H100 can handle larger kernel matrices (was 5000)
    # Note: With blocked computation, 12k samples = ~144M floats = ~576MB per kernel matrix
    # H100 has 80GB VRAM, so this is very safe
    
    # Experiment 0: Standard MLP (no batch norm, no skip connections)
    print("\n" + "="*60)
    print("Experiment 0: Standard MLP (Baseline)")
    print("="*60)
    results_standard = run_experiment(
        depths=depths,
        widths=widths,
        device=device,
        config_path=config_path,
        n_batches=n_batches,
        batch_size=batch_size,
        architecture="standard",
        max_samples_for_kernel=max_samples_for_kernel
    )
    
    # Save standard results (if flag is enabled)
    if save_json_results:
        results_file_standard = os.path.join(out_dir, "gate_mobility_results_standard.json")
        with open(results_file_standard, 'w') as f:
            json.dump(results_standard, f, indent=2)
        print(f"Saved standard results to {results_file_standard}")
    
    # Create heatmaps for standard
    plot_heatmaps(results_standard, out_dir, suffix="_standard", metric="M_g")
    plot_heatmaps(results_standard, out_dir, suffix="_standard", metric="C_def")
    plot_heatmaps(results_standard, out_dir, suffix="_standard", metric="H_Lambda")
    
    # Create d_f distribution plots for standard
    plot_d_f_distributions(results_standard, out_dir, suffix="_standard")
    
    # Experiment 1: Batch Norm only
    print("\n" + "="*60)
    print("Experiment 1: Batch Norm Only")
    print("="*60)
    results_batchnorm = run_experiment(
        depths=depths,
        widths=widths,
        device=device,
        config_path=config_path,
        n_batches=n_batches,
        batch_size=batch_size,
        architecture="batchnorm",
        max_samples_for_kernel=max_samples_for_kernel
    )
    
    # Save batch norm results (if flag is enabled)
    if save_json_results:
        results_file_batchnorm = os.path.join(out_dir, "gate_mobility_results_batchnorm.json")
        with open(results_file_batchnorm, 'w') as f:
            json.dump(results_batchnorm, f, indent=2)
        print(f"Saved batch norm results to {results_file_batchnorm}")
    
    # Create heatmaps for batch norm
    plot_heatmaps(results_batchnorm, out_dir, suffix="_batchnorm", metric="M_g")
    plot_heatmaps(results_batchnorm, out_dir, suffix="_batchnorm", metric="C_def")
    plot_heatmaps(results_batchnorm, out_dir, suffix="_batchnorm", metric="H_Lambda")
    
    # Create d_f distribution plots for batch norm
    plot_d_f_distributions(results_batchnorm, out_dir, suffix="_batchnorm")
    
    # Experiment 2: Skip Connections only
    print("\n" + "="*60)
    print("Experiment 2: Skip Connections Only")
    print("="*60)
    results_skip = run_experiment(
        depths=depths,
        widths=widths,
        device=device,
        config_path=config_path,
        n_batches=n_batches,
        batch_size=batch_size,
        architecture="skip",
        max_samples_for_kernel=max_samples_for_kernel
    )
    
    # Save skip connections results (if flag is enabled)
    if save_json_results:
        results_file_skip = os.path.join(out_dir, "gate_mobility_results_skip.json")
        with open(results_file_skip, 'w') as f:
            json.dump(results_skip, f, indent=2)
        print(f"Saved skip connections results to {results_file_skip}")
    
    # Create heatmaps for skip connections
    plot_heatmaps(results_skip, out_dir, suffix="_skip", metric="M_g")
    plot_heatmaps(results_skip, out_dir, suffix="_skip", metric="C_def")
    plot_heatmaps(results_skip, out_dir, suffix="_skip", metric="H_Lambda")
    
    # Create d_f distribution plots for skip connections
    plot_d_f_distributions(results_skip, out_dir, suffix="_skip")
    
    # Experiment 3: Both combined (ResNet)
    print("\n" + "="*60)
    print("Experiment 3: Both Combined (Skip Connections + Batch Norm)")
    print("="*60)
    results_resnet = run_experiment(
        depths=depths,
        widths=widths,
        device=device,
        config_path=config_path,
        n_batches=n_batches,
        batch_size=batch_size,
        architecture="resnet",
        max_samples_for_kernel=max_samples_for_kernel
    )
    
    # Save ResNet results (if flag is enabled)
    if save_json_results:
        results_file_resnet = os.path.join(out_dir, "gate_mobility_results_resnet.json")
        with open(results_file_resnet, 'w') as f:
            json.dump(results_resnet, f, indent=2)
        print(f"Saved ResNet results to {results_file_resnet}")
    
    # Create heatmaps for ResNet
    plot_heatmaps(results_resnet, out_dir, suffix="_resnet", metric="M_g")
    plot_heatmaps(results_resnet, out_dir, suffix="_resnet", metric="C_def")
    plot_heatmaps(results_resnet, out_dir, suffix="_resnet", metric="H_Lambda")
    
    # Create d_f distribution plots for ResNet
    plot_d_f_distributions(results_resnet, out_dir, suffix="_resnet")
    
    print("\n" + "="*60)
    print("All Experiments Complete!")
    print("="*60)
    print(f"All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
