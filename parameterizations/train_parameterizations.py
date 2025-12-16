"""
Train FFNN on MNIST with different parameterizations (standard, mup, ntk)
and track metrics: train_loss, test_loss, M_g, C_def, H_Lambda every n epochs.
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
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional
from queue import Queue, Empty
from threading import Thread

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.models.ffnn import MLP
from src.data.mnist import build_mnist_datasets
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.save_io import ensure_dir, save_json
from src.analysis.path_kernel import collect_path_factors, HadamardGramOperator

# Import ResNet architecture
resnet_path = os.path.join(project_root, "outputs", "gates", "mlp_resnet.py")
if os.path.exists(resnet_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mlp_resnet", resnet_path)
    resnet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resnet_module)
    MLPResNet = resnet_module.MLPResNet
else:
    raise FileNotFoundError(f"Could not find mlp_resnet.py at {resnet_path}")

# Import BatchNorm architecture
batchnorm_path = os.path.join(project_root, "outputs", "gates", "mlp_batchnorm.py")
if os.path.exists(batchnorm_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mlp_batchnorm", batchnorm_path)
    batchnorm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batchnorm_module)
    MLPBatchNorm = batchnorm_module.MLPBatchNorm
else:
    raise FileNotFoundError(f"Could not find mlp_batchnorm.py at {batchnorm_path}")

# Import LazarusMLP architecture
lazarus_path = os.path.join(os.path.dirname(__file__), "mlp_lazarus.py")
if os.path.exists(lazarus_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mlp_lazarus", lazarus_path)
    lazarus_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lazarus_module)
    LazarusMLP = lazarus_module.LazarusMLP
else:
    raise FileNotFoundError(f"Could not find mlp_lazarus.py at {lazarus_path}")

# Import functions from gate_velocity_with_capacity.py
gate_velocity_path = os.path.join(project_root, "outputs", "gates", "gate_velocity_with_capacity.py")
if os.path.exists(gate_velocity_path):
    # Import by executing the file and extracting functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("gate_velocity_with_capacity", gate_velocity_path)
    gate_velocity_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate_velocity_module)
    
    compute_gate_mobility = gate_velocity_module.compute_gate_mobility
    compute_path_deformation_capacity = gate_velocity_module.compute_path_deformation_capacity
    compute_path_covariance_entropy = gate_velocity_module.compute_path_covariance_entropy
    compute_distance_to_flip = gate_velocity_module.compute_distance_to_flip
    compute_gradient_norms = gate_velocity_module.compute_gradient_norms
    compute_path_kernel_matrix = gate_velocity_module.compute_path_kernel_matrix
else:
    raise FileNotFoundError(f"Could not find gate_velocity_with_capacity.py at {gate_velocity_path}")


def parse_parameterization_name(param_name: str) -> tuple[str, str | None]:
    """
    Parse parameterization name to extract base parameterization and optimizer.
    
    Examples:
        "standard" -> ("standard", None)
        "standard_adam" -> ("standard", "adam")
        "standard_batchnorm" -> ("standard_batchnorm", None)
        "standard_batchnorm_adam" -> ("standard_batchnorm", "adam")
        "mup_sgd" -> ("mup", "sgd")
    
    Args:
        param_name: Parameterization name (may include optimizer suffix and/or batchnorm)
    
    Returns:
        Tuple of (base_parameterization, optimizer_override or None)
    """
    # Known optimizers that can be suffixes
    known_optimizers = ["adam", "muon", "sgd"]
    
    # Check if name contains an underscore and ends with a known optimizer
    if "_" in param_name:
        parts = param_name.rsplit("_", 1)  # Split from right
        if len(parts) == 2:
            base_param, suffix = parts
            if suffix.lower() in known_optimizers:
                # Found optimizer suffix, return base and optimizer
                return base_param, suffix.lower()
    
    # No optimizer suffix found - return as-is (may contain "batchnorm" or other modifiers)
    return param_name, None


def initialize_parameterization(
    model,
    parameterization: str,
    device: torch.device
):
    """
    Initialize model weights according to parameterization scheme.
    Supports MLP, MLPBatchNorm, and MLPResNet architectures.
    
    Args:
        model: MLP, MLPBatchNorm, or MLPResNet model
        parameterization: "standard", "standard_batchnorm", "mup", or "ntk"
        device: Device to initialize on
    """
    model = model.to(device)
    
    # Check if model has batch_norms (ResNet or BatchNorm architecture)
    has_batch_norm = hasattr(model, 'batch_norms') and len(model.batch_norms) > 0
    has_skip_projections = hasattr(model, 'skip_projections')
    
    # Extract base parameterization (remove "_batchnorm" suffix if present)
    base_param = parameterization.replace("_batchnorm", "") if "_batchnorm" in parameterization else parameterization
    
    if base_param == "standard":
        # Standard initialization: Xavier/Kaiming
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize batch norm if present
        if has_batch_norm:
            for bn in model.batch_norms:
                if isinstance(bn, nn.BatchNorm1d):
                    nn.init.ones_(bn.weight)
                    nn.init.zeros_(bn.bias)
    
    elif base_param == "mup":
        # Maximal Update Parametrization (μP):
        # - Hidden layers: scale by 1/sqrt(width) (standard initialization)
        # - Output layer: scale by 1/sqrt(width) (standard initialization)
        # Note: The "maximal update" property comes from learning rate scaling during training,
        # not from initialization. Here we use standard 1/sqrt(width) initialization.
        for l, linear in enumerate(model.linears):
            # Hidden layers: 1/sqrt(width) scaling
            width = linear.weight.shape[0]  # input dimension
            scale = 1.0 / np.sqrt(width)
            nn.init.normal_(linear.weight, mean=0.0, std=scale)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        
        # Initialize skip projections if present
        if has_skip_projections:
            for proj in model.skip_projections:
                if isinstance(proj, nn.Linear):
                    width = proj.weight.shape[0]  # input dimension
                    scale = 1.0 / np.sqrt(width)
                    nn.init.normal_(proj.weight, mean=0.0, std=scale)
        
        # Output layer: 1/sqrt(width) scaling
        output_width = model.readout.weight.shape[0]  # input dimension to readout
        output_scale = 1.0 / np.sqrt(output_width)
        nn.init.normal_(model.readout.weight, mean=0.0, std=output_scale)
        if model.readout.bias is not None:
            nn.init.zeros_(model.readout.bias)
        
        # Initialize batch norm if present
        if has_batch_norm:
            for bn in model.batch_norms:
                if isinstance(bn, nn.BatchNorm1d):
                    nn.init.ones_(bn.weight)
                    nn.init.zeros_(bn.bias)
    
    elif base_param == "ntk":
        # Neural Tangent Kernel (NTK) parametrization:
        # - All layers: scale by 1/sqrt(width)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                width = m.weight.shape[0]  # input dimension
                scale = 1.0 / np.sqrt(width)
                nn.init.normal_(m.weight, mean=0.0, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize batch norm if present
        if has_batch_norm:
            for bn in model.batch_norms:
                if isinstance(bn, nn.BatchNorm1d):
                    nn.init.ones_(bn.weight)
                    nn.init.zeros_(bn.bias)
    
    elif base_param == "mup_L":
        # Maximal Update Parametrization with Lazarus depth scaling:
        # - Hidden layers: mup initialization (1/sqrt(width))
        # - Output layer: mup initialization (1/sqrt(width))
        # - For LazarusMLP: Apply depth scaling (α = 1/sqrt(2*depth)) to branch outputs
        if isinstance(model, LazarusMLP):
            # For LazarusMLP: mup init + depth scaling
            depth = model.depth
            alpha = 1.0 / np.sqrt(2 * depth)
            
            # Input projection: mup init
            width = model.input_proj.weight.shape[0]
            scale = 1.0 / np.sqrt(width)
            nn.init.normal_(model.input_proj.weight, mean=0.0, std=scale)
            if model.input_proj.bias is not None:
                nn.init.zeros_(model.input_proj.bias)
            
            # Residual blocks: mup init + depth scaling on branch output
            for block in model.blocks:
                # First Linear in branch: mup init
                width = block[0].weight.shape[0]
                scale = 1.0 / np.sqrt(width)
                nn.init.normal_(block[0].weight, mean=0.0, std=scale)
                if block[0].bias is not None:
                    nn.init.zeros_(block[0].bias)
                
                # Second Linear in branch (last in branch): mup init + depth scaling
                width = block[2].weight.shape[0]
                scale = 1.0 / np.sqrt(width)
                nn.init.normal_(block[2].weight, mean=0.0, std=scale)
                block[2].weight.data *= alpha  # Apply depth scaling
                if block[2].bias is not None:
                    nn.init.zeros_(block[2].bias)
            
            # Output readout: mup init
            width = model.readout.weight.shape[0]
            scale = 1.0 / np.sqrt(width)
            nn.init.normal_(model.readout.weight, mean=0.0, std=scale)
            if model.readout.bias is not None:
                nn.init.zeros_(model.readout.bias)
        else:
            # For standard architectures: mup init (same as mup)
            for l, linear in enumerate(model.linears):
                width = linear.weight.shape[0]  # input dimension
                scale = 1.0 / np.sqrt(width)
                nn.init.normal_(linear.weight, mean=0.0, std=scale)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
            
            # Initialize skip projections if present
            if has_skip_projections:
                for proj in model.skip_projections:
                    if isinstance(proj, nn.Linear):
                        width = proj.weight.shape[0]  # input dimension
                        scale = 1.0 / np.sqrt(width)
                        nn.init.normal_(proj.weight, mean=0.0, std=scale)
            
            # Output layer: mup init
            output_width = model.readout.weight.shape[0]  # input dimension to readout
            output_scale = 1.0 / np.sqrt(output_width)
            nn.init.normal_(model.readout.weight, mean=0.0, std=output_scale)
            if model.readout.bias is not None:
                nn.init.zeros_(model.readout.bias)
            
            # Initialize batch norm if present
            if has_batch_norm:
                for bn in model.batch_norms:
                    if isinstance(bn, nn.BatchNorm1d):
                        nn.init.ones_(bn.weight)
                        nn.init.zeros_(bn.bias)
    
    elif parameterization == "path":
        # Path parameterization always uses LazarusMLP architecture
        # Lazarus initialization is handled in LazarusMLP.__init__
        # This should never be called for path parameterization since we create
        # LazarusMLP directly and it initializes itself
        if not isinstance(model, LazarusMLP):
            raise ValueError("Path parameterization requires LazarusMLP architecture. "
                           "This should be handled automatically in model creation.")
        # LazarusMLP initializes itself in __init__, so nothing to do here
    
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}. Options: standard, mup, ntk, mup_L, path")


def _compute_distance_to_flip_lazarus(
    model, x_batch: torch.Tensor, device: torch.device, 
    return_dead_neurons: bool = False
) -> Dict:
    """
    Compute distance to flip for LazarusMLP architecture.
    Adapts the structure to work with the standard compute_distance_to_flip interface.
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
        # Get the first linear layer in the branch
        first_linear = block[0]
        u = first_linear(h)  # Pre-activation of first linear
        
        x_norm = torch.norm(h, dim=1, keepdim=True)
        if return_dead_neurons:
            dead_samples = (x_norm.squeeze() < DEAD_THRESHOLD).float()
            dead_fraction = dead_samples.mean().item()
            dead_neuron_fractions[l] = dead_fraction  # Use l (0-indexed) to match block index
        
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


def _compute_gradient_norms_lazarus(
    model, x_batch: torch.Tensor, y_batch: torch.Tensor, 
    loss_fn, device: torch.device, n_classes: int = 1, alpha: float = 1.0
) -> Dict[int, float]:
    """
    Compute gradient norms for LazarusMLP architecture.
    Returns gradient norms for each block (using first linear in branch).
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
    # Get gradient norm from first linear in each block
    for l, block in enumerate(model.blocks):
        first_linear = block[0]
        if first_linear.weight.grad is not None:
            grad_norm = torch.norm(first_linear.weight.grad).item()
            gradient_norms[l] = grad_norm
        else:
            gradient_norms[l] = 0.0
    
    return gradient_norms


def _compute_gate_mobility_lazarus(
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
        
        # Compute distances to flip using Lazarus-adapted function
        result_dict = _compute_distance_to_flip_lazarus(model, x_batch, device, return_dead_neurons=True)
        distances = result_dict["distances"]
        dead_fractions = result_dict["dead_neuron_fractions"]
        
        # Compute gradient norms using Lazarus-adapted function
        grad_norms = _compute_gradient_norms_lazarus(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        
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
    
    Process:
    1. Compute M_g for each layer with LR=1 (or eta_0)
    2. Calculate initial LR per layer: η* = η₀ · M* / M_g[layer]
    
    Args:
        model: The neural network model
        train_loader: Data loader for training data
        device: Device to run on
        n_classes: Number of classes
        alpha: Alpha scaling factor for labels
        eta_0: Base learning rate (η₀)
        target_mobility: Target gate mobility M* (default: 0.3)
        n_batches: Number of batches to average over
    
    Returns:
        Dict mapping layer index -> initial LR
    """
    model.train()
    loss_fn = nn.MSELoss()
    
    # Compute M_g with LR=1 (or eta_0) to get baseline mobility
    # M_g = (lr * E[||∇w||]) / E[d_f]
    # So with lr=1: M_g = E[||∇w||] / E[d_f]
    
    if isinstance(model, LazarusMLP):
        # Use Lazarus-adapted function
        # Note: _compute_gate_mobility_lazarus computes M_g for blocks only (0 to depth-1)
        # The optimizer groups are: input_proj (layer 0), blocks (layers 0 to depth-1), readout
        m_g_result = _compute_gate_mobility_lazarus(
            model, train_loader, lr=1.0, device=device,
            n_batches=n_batches, n_classes=n_classes, alpha=alpha,
            return_distributions=False
        )
        M_g_dict = m_g_result["M_g"]
        
        # For LazarusMLP, M_g_dict contains values for blocks (0 to depth-1)
        # We need to handle:
        # - Input projection: use layer index -1 (distinct from blocks)
        # - Blocks: use layer indices 0 to depth-1 (already computed)
        # - Readout: use last block's M_g (already handled in optimizer setup)
        
        # Compute initial LR per layer: η* = η₀ · M* / M_g[layer]
        initial_lrs = {}
        
        # For input projection, use layer index -1 and first block's M_g (block 0) as proxy
        INPUT_PROJ_LAYER = -1
        if 0 in M_g_dict:
            M_g_input = M_g_dict[0]
            if np.isinf(M_g_input) or np.isnan(M_g_input) or M_g_input <= 0:
                initial_lrs[INPUT_PROJ_LAYER] = eta_0
                print(f"  Warning: Input projection (using block 0 M_g={M_g_input}) is invalid, using fallback LR={eta_0:.6e}")
            else:
                initial_lr = eta_0 * target_mobility / M_g_input
                initial_lrs[INPUT_PROJ_LAYER] = initial_lr
        else:
            initial_lrs[INPUT_PROJ_LAYER] = eta_0
            print(f"  Warning: No M_g for block 0, using fallback LR={eta_0:.6e} for input projection")
        
        # For blocks (layers 0 to depth-1), use computed M_g
        for layer_idx, M_g_val in M_g_dict.items():
            if np.isinf(M_g_val) or np.isnan(M_g_val) or M_g_val <= 0:
                # Fallback: use eta_0 if M_g is invalid
                initial_lrs[layer_idx] = eta_0
                print(f"  Warning: Block {layer_idx} has invalid M_g={M_g_val}, using fallback LR={eta_0:.6e}")
            else:
                # η* = η₀ · M* / M_g
                initial_lr = eta_0 * target_mobility / M_g_val
                initial_lrs[layer_idx] = initial_lr
        
        print(f"  Computed LRs for {len(initial_lrs)} layers (input_proj + {len(M_g_dict)} blocks)")
    else:
        # Use standard function
        m_g_result = compute_gate_mobility(
            model, train_loader, lr=1.0, device=device,
            n_batches=n_batches, n_classes=n_classes, alpha=alpha,
            return_distributions=False
        )
        M_g_dict = m_g_result["M_g"]
        
        # Compute initial LR per layer: η* = η₀ · M* / M_g[layer]
        initial_lrs = {}
        for layer_idx, M_g_val in M_g_dict.items():
            if np.isinf(M_g_val) or np.isnan(M_g_val) or M_g_val <= 0:
                # Fallback: use eta_0 if M_g is invalid
                initial_lrs[layer_idx] = eta_0
                print(f"  Warning: Layer {layer_idx} has invalid M_g={M_g_val}, using fallback LR={eta_0:.6e}")
            else:
                # η* = η₀ · M* / M_g
                initial_lr = eta_0 * target_mobility / M_g_val
                initial_lrs[layer_idx] = initial_lr
    
    # Validate that we have LRs for all expected layers
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
    
    This is computed cheaply using one forward/backward pass.
    
    Args:
        model: The neural network model
        train_loader: Data loader for training data
        device: Device to run on
        n_classes: Number of classes
        alpha: Alpha scaling factor for labels
        n_batches: Number of batches to average over (default: 1 for cheap computation)
        return_per_layer: If True, return dict of per-layer LRs; if False, return median LR
    
    Returns:
        Optimal learning rate (scalar) or dict mapping layer index -> optimal LR
    """
    model.train()
    loss_fn = nn.MSELoss()
    
    # Collect d_f values and gradient norms
    all_d_f = {l: [] for l in range(model.depth)}
    grad_norm_sums = {l: 0.0 for l in range(model.depth)}
    batch_count = 0
    
    for x_batch, y_batch in train_loader:
        if batch_count >= n_batches:
            break
        
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        # Compute d_f (distance to flip) - adapt for LazarusMLP
        if isinstance(model, LazarusMLP):
            result_dict = _compute_distance_to_flip_lazarus(model, x_batch, device, return_dead_neurons=False)
        else:
            result_dict = compute_distance_to_flip(model, x_batch, device, return_dead_neurons=False)
        distances = result_dict["distances"]
        
        # Store d_f values
        for l in range(model.depth):
            if l in distances:  # Check if layer exists in distances dict
                d_f_flat = distances[l].detach().flatten().cpu().numpy()
                all_d_f[l].extend(d_f_flat)
        
        # Check if model has NaN before computing gradients
        has_nan = False
        for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                has_nan = True
                break
        
        if has_nan:
            # Model has NaN - gradients will be NaN, skip this batch
            continue
        
        # Compute gradient norms - adapt for LazarusMLP
        if isinstance(model, LazarusMLP):
            grad_norms = _compute_gradient_norms_lazarus(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        else:
            grad_norms = compute_gradient_norms(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        
        # Check if gradients are valid before using them
        valid_grads = True
        for l in range(model.depth):
            if l in grad_norms:
                grad_val = grad_norms[l]
                if np.isnan(grad_val) or np.isinf(grad_val) or grad_val < 0:
                    valid_grads = False
                    break
        
        if valid_grads:
            for l in range(model.depth):
                if l in grad_norms:
                    grad_norm_sums[l] += grad_norms[l]
            batch_count += 1
        else:
            # Skip this batch if gradients are invalid
            continue
    
    # If no valid batches were processed, return fallback values
    if batch_count == 0:
        if return_per_layer:
            # Return fallback dict (all 1e-4) to signal failure but allow training to continue
            return {l: 1e-4 for l in range(model.depth)}
        else:
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
    
    # Compute optimal LR per layer: η_l ≈ median(d_f_l) / E[||∇w_l||]
    optimal_lrs_per_layer = {}
    optimal_lrs = []
    for l in range(model.depth):
        if avg_grad_norm_per_layer[l] > 1e-8 and median_d_f_per_layer[l] > 0:  # Avoid division by zero
            optimal_lr = median_d_f_per_layer[l] / avg_grad_norm_per_layer[l]
            if not np.isnan(optimal_lr) and not np.isinf(optimal_lr) and optimal_lr > 0:
                optimal_lrs_per_layer[l] = optimal_lr
                optimal_lrs.append(optimal_lr)
            else:
                optimal_lrs_per_layer[l] = None
        else:
            optimal_lrs_per_layer[l] = None
    
    if return_per_layer:
        # Return per-layer LRs, using median as fallback for layers with invalid values
        if len(optimal_lrs) > 0:
            median_fallback = float(np.median(optimal_lrs))
        else:
            # If all layers failed, use a conservative default
            median_fallback = 1e-4
            # Debug: print why computation failed (only print once per call, not every layer)
            if batch_count == 0:
                pass  # Will be handled by caller
            else:
                # Check what went wrong - summarize across all layers
                low_grad_count = sum(1 for l in range(model.depth) if avg_grad_norm_per_layer[l] <= 1e-8)
                zero_d_f_count = sum(1 for l in range(model.depth) if median_d_f_per_layer[l] <= 0)
                no_data_count = sum(1 for l in range(model.depth) if len(all_d_f[l]) == 0)
                
                if low_grad_count > 0 or zero_d_f_count > 0 or no_data_count > 0:
                    reasons = []
                    if low_grad_count > 0:
                        reasons.append(f"{low_grad_count} layers with low grad_norm")
                    if zero_d_f_count > 0:
                        reasons.append(f"{zero_d_f_count} layers with zero/neg d_f")
                    if no_data_count > 0:
                        reasons.append(f"{no_data_count} layers with no d_f data")
                    # Only print if this is a real issue (not just initial epochs)
                    pass  # Don't spam - the fallback will be used
        
        for l in range(model.depth):
            if optimal_lrs_per_layer[l] is None:
                optimal_lrs_per_layer[l] = median_fallback
        
        return optimal_lrs_per_layer
    
    # Return median across layers (backward compatibility)
    if len(optimal_lrs) > 0:
        optimal_lr = float(np.median(optimal_lrs))
    else:
        # Fallback: use a default small LR
        optimal_lr = 1e-4
    
    return optimal_lr


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
    
    This is the standard gate mobility formula.
    
    Args:
        model: The neural network model
        train_loader: Data loader for training data
        current_lrs: Dict mapping layer index -> current learning rate η_l(t)
        device: Device to run on
        n_classes: Number of classes
        alpha: Alpha scaling factor for labels
        n_batches: Number of batches to average over (default: 1)
    
    Returns:
        Dict mapping layer index -> M_g,l value
    """
    model.train()
    loss_fn = nn.MSELoss()
    
    # Collect d_f values and gradient norms
    all_d_f = {l: [] for l in range(model.depth)}
    grad_norm_sums = {l: 0.0 for l in range(model.depth)}
    batch_count = 0
    
    for x_batch, y_batch in train_loader:
        if batch_count >= n_batches:
            break
        
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        # Compute d_f (distance to flip) - adapt for LazarusMLP
        if isinstance(model, LazarusMLP):
            result_dict = _compute_distance_to_flip_lazarus(model, x_batch, device, return_dead_neurons=False)
        else:
            result_dict = compute_distance_to_flip(model, x_batch, device, return_dead_neurons=False)
        distances = result_dict["distances"]
        
        # Store d_f values
        for l in range(model.depth):
            if l in distances:
                d_f_flat = distances[l].detach().flatten().cpu().numpy()
                all_d_f[l].extend(d_f_flat)
        
        # Check if model has NaN before computing gradients
        has_nan = False
        for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                has_nan = True
                break
        
        if has_nan:
            continue
        
        # Compute gradient norms - adapt for LazarusMLP
        if isinstance(model, LazarusMLP):
            grad_norms = _compute_gradient_norms_lazarus(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        else:
            grad_norms = compute_gradient_norms(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        
        # Check if gradients are valid
        valid_grads = True
        for l in range(model.depth):
            if l in grad_norms:
                grad_val = grad_norms[l]
                if np.isnan(grad_val) or np.isinf(grad_val) or grad_val < 0:
                    valid_grads = False
                    break
        
        if valid_grads:
            for l in range(model.depth):
                if l in grad_norms:
                    grad_norm_sums[l] += grad_norms[l]
            batch_count += 1
        else:
            continue
    
    # If no valid batches, return None values
    if batch_count == 0:
        return {l: None for l in range(model.depth)}
    
    # Compute E[d_f] (mean) and E[||∇w||] (mean) for each layer
    # Standard formula: M_g = (η * E[||∇w||]) / E[d_f]
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
    
    # Compute M_g,l = (η_l * E[||∇w_l||]) / E[d_f_l]
    # This is the standard gate mobility formula
    # For LazarusMLP: input_proj uses layer -1, blocks use 0 to depth-1
    M_g_per_layer = {}
    
    if isinstance(model, LazarusMLP):
        # Handle input_proj (layer -1) - use block 0's d_f and grad_norm as proxy
        INPUT_PROJ_LAYER = -1
        if INPUT_PROJ_LAYER in current_lrs and 0 in mean_d_f_per_layer and 0 in avg_grad_norm_per_layer:
            if mean_d_f_per_layer[0] > 1e-8 and avg_grad_norm_per_layer[0] > 1e-8:
                eta_l = current_lrs[INPUT_PROJ_LAYER]
                # Use block 0's statistics as proxy for input_proj
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
                # M_g,l = (η_l * E[||∇w_l||]) / E[d_f_l]
                M_g_l = (eta_l * avg_grad_norm_per_layer[l]) / mean_d_f_per_layer[l]
                if not np.isnan(M_g_l) and not np.isinf(M_g_l) and M_g_l > 0:
                    M_g_per_layer[l] = M_g_l
                else:
                    M_g_per_layer[l] = None
            else:
                M_g_per_layer[l] = None
    else:
        # Standard MLP: layers 0 to depth-1
        for l in range(model.depth):
            if l in current_lrs and mean_d_f_per_layer[l] > 1e-8 and avg_grad_norm_per_layer[l] > 1e-8:
                eta_l = current_lrs[l]
                # M_g,l = (η_l * E[||∇w_l||]) / E[d_f_l]
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
        current_lrs: Dict mapping layer index -> current learning rate η_l(t)
        M_g_per_layer: Dict mapping layer index -> measured mobility M_g,l(t)
        target_mobility: Target mobility M*
        alpha: Smoothing exponent α ∈ (0,1]
        eps: Small epsilon to avoid division by zero
        min_scale: Minimum multiplicative change per step
        max_scale: Maximum multiplicative change per step
        lr_max: Maximum learning rate cap
    
    Returns:
        Dict mapping layer index -> updated learning rate η_l(t+1)
    """
    updated_lrs = {}
    
    for l in current_lrs:
        if l in M_g_per_layer and M_g_per_layer[l] is not None:
            M_g_l = M_g_per_layer[l]
            # Compute ratio
            ratio = target_mobility / (M_g_l + eps)
            # Apply smoothing exponent
            scale = ratio ** alpha
            # Clamp scale
            scale = max(min_scale, min(max_scale, scale))
            # Update LR and apply cap
            updated_lrs[l] = min(current_lrs[l] * scale, lr_max)
        else:
            # If M_g computation failed, reduce LR by min_scale (conservative fallback)
            updated_lrs[l] = min(current_lrs[l] * min_scale, lr_max)
    
    return updated_lrs


@torch.no_grad()
def evaluate_loss(model, loader, device, n_classes: int, alpha: float) -> float:
    """Evaluate MSE loss on a dataset."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        
        if n_classes == 1:
            # Binary classification
            loss = torch.mean((yhat - yb) ** 2)
        else:
            # Multi-class classification: convert to one-hot
            if yb.dim() > 1:
                yb = yb.view(-1)
            yb_class = (yb / alpha).long()
            yb_class = torch.clamp(yb_class, 0, n_classes - 1)
            yb_onehot = torch.zeros_like(yhat)
            src_values = torch.ones_like(yb.unsqueeze(1)) * alpha
            yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values)
            loss = torch.mean((yhat - yb_onehot) ** 2)
        
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    
    return total_loss / n_samples if n_samples > 0 else 0.0


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
    
    # Handle different architectures
    if isinstance(model, LazarusMLP):
        # LazarusMLP: track by block
        for l, block in enumerate(model.blocks):
            first_linear = block[0]
            if first_linear.weight.grad is not None:
                grad_norm = torch.norm(first_linear.weight.grad).item()
                gradient_norms[l] = grad_norm
            else:
                gradient_norms[l] = 0.0
    else:
        # Standard MLP or MLPResNet: track by linear layer
        if hasattr(model, 'linears'):
            for l, linear in enumerate(model.linears):
                if linear.weight.grad is not None:
                    grad_norm = torch.norm(linear.weight.grad).item()
                    gradient_norms[l] = grad_norm
                else:
                    gradient_norms[l] = 0.0
        else:
            # Fallback: track all linear layers
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


def train_with_parameterization(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict,
    device: torch.device,
    parameterization: str,
    architecture: str,
    out_dir: str,
    optimizer_override: str = None
) -> Dict:
    """
    Train model and track metrics every n epochs.
    
    Returns:
        Dictionary with training history containing:
        - epochs: List of epoch numbers
        - train_loss: List of train losses
        - test_loss: List of test losses
        - M_g_avg: List of average M_g values
        - C_def: List of C_def values
        - H_Lambda: List of H_Lambda values
    """
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["lr_w"])
    batch_size = int(config["training"]["batch_size"])
    # Use optimizer override if provided, otherwise use config
    if optimizer_override is not None:
        optimizer_type = optimizer_override.lower()
    else:
        optimizer_type = config["training"].get("optimizer", "adam").lower()
    if isinstance(optimizer_type, list):
        optimizer_type = optimizer_type[0]
    grad_clip = float(config["training"].get("grad_clip_max_norm", 0.0))
    
    # Get n_classes and alpha
    n_classes = model.n_classes
    alpha = float(config.get("dataset", {}).get("alpha", 1.0))
    if isinstance(alpha, list):
        alpha = alpha[0]
    
    # Metrics tracking frequency
    metrics_freq = int(config.get("logging", {}).get("metrics_every_n_epochs", 100))
    compute_metrics = config.get("logging", {}).get("compute_metrics", True)
    if isinstance(compute_metrics, list):
        compute_metrics = compute_metrics[0]
    compute_metrics = bool(compute_metrics)
    m_g_n_batches = int(config.get("logging", {}).get("m_g_n_batches", 16))
    kernel_max_samples = int(config.get("logging", {}).get("kernel_max_samples", 8192))
    
    # For "path" parameterization, use separate initial LR if specified
    if parameterization == "path":
        initial_lr = float(config["training"].get("lr_w_path", config["training"]["lr_w"]))
    else:
        initial_lr = lr
    
    # Check for automatic LR computation flag
    automatic_lr = config["training"].get("automatic", False)
    if isinstance(automatic_lr, list):
        automatic_lr = automatic_lr[0]
    automatic_lr = bool(automatic_lr)
    
    # Get target mobility (default: 0.3)
    target_mobility = float(config["training"].get("target_mobility", 0.3))
    
    # Get damped LR update hyperparameters for path parameterization
    lr_update_alpha = float(config["training"].get("lr_update_alpha", 0.25))
    lr_update_eps = float(config["training"].get("lr_update_eps", 1e-8))
    lr_update_min_scale = float(config["training"].get("lr_update_min_scale", 0.5))
    lr_update_max_scale = float(config["training"].get("lr_update_max_scale", 1.5))
    lr_max = float(config["training"].get("lr_max", 1.0))  # Maximum LR cap
    
    # Get warmup epochs for path parameterization (default: 25, set to 0 to disable)
    warmup_epochs = int(config["training"].get("warmup_epochs", 25))
    if warmup_epochs <= 0:
        warmup_epochs = 0  # Disable warmup
    warmup_start_ratio = 0.01  # Start warmup at 1% of target LR
    
    # For "path" parameterization, we'll update LR dynamically (layer-wise)
    # Start with initial LR, but it will be updated after each epoch
    use_layerwise_lr = (parameterization == "path" and isinstance(model, LazarusMLP))
    optimal_lrs_dict = None  # Will be set if layer-wise LR is used
    target_lrs_dict = None  # Store target LRs (before warmup scaling)
    
    if parameterization == "path":
        # Compute optimal initial LR using path recipe
        print(f"  Computing optimal initial LR for path parameterization...")
        print(f"  Using initial LR fallback (η₀): {initial_lr:.6e}")
        
        if automatic_lr:
            # Automatic mode: compute LR based on target gate mobility M*
            print(f"  Automatic mode enabled: computing LR from target mobility M*={target_mobility}")
            try:
                if use_layerwise_lr:
                    # Compute M_g with LR=1, then calculate η* = η₀ · M* / M_g[layer]
                    optimal_lrs_dict = compute_initial_lr_from_target_mobility(
                        model, train_loader, device, n_classes, alpha,
                        eta_0=initial_lr, target_mobility=target_mobility, n_batches=1
                    )
                    # Validate and clamp LRs to reasonable range
                    for layer_idx in optimal_lrs_dict:
                        lr_val = optimal_lrs_dict[layer_idx]
                        if np.isnan(lr_val) or np.isinf(lr_val) or lr_val <= 0:
                            optimal_lrs_dict[layer_idx] = initial_lr
                            print(f"  Warning: Layer {layer_idx} LR was invalid, using fallback: {initial_lr:.6e}")
                        else:
                            optimal_lrs_dict[layer_idx] = lr_val
                    # Use median as initial LR for all layers (will be updated per-layer)
                    valid_lrs = [v for v in optimal_lrs_dict.values() if not (np.isnan(v) or np.isinf(v))]
                    if len(valid_lrs) > 0:
                        lr = float(np.median(valid_lrs))
                    else:
                        lr = initial_lr
                        print(f"  Warning: All computed LRs were invalid, using fallback: {lr:.6e}")
                    print(f"  Initial LR (median): {lr:.6e}")
                    print(f"  Per-layer LRs: {[f'L{i}:{lr_val:.6e}' for i, lr_val in optimal_lrs_dict.items()]}")
                    # Verify layer coverage for LazarusMLP
                    if isinstance(model, LazarusMLP):
                        expected_layers = set(range(model.depth))  # Blocks 0 to depth-1
                        computed_layers = set(optimal_lrs_dict.keys())
                        missing = expected_layers - computed_layers
                        if missing:
                            print(f"  Warning: Missing LRs for layers: {missing}")
                        else:
                            print(f"  ✓ Automatic LR computed for all {model.depth} blocks")
                    # Store target LRs for warmup
                    target_lrs_dict = optimal_lrs_dict.copy()
                else:
                    # For non-LazarusMLP, compute single LR
                    optimal_lrs_dict = compute_initial_lr_from_target_mobility(
                        model, train_loader, device, n_classes, alpha,
                        eta_0=initial_lr, target_mobility=target_mobility, n_batches=1
                    )
                    # Use median as single LR
                    lr = float(np.median(list(optimal_lrs_dict.values())))
                    print(f"  Initial LR: {lr:.6e}")
                    target_lrs_dict = optimal_lrs_dict.copy()
            except Exception as e:
                print(f"  Warning: Failed to compute automatic LR, falling back to original method: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to original method
                try:
                    if use_layerwise_lr:
                        optimal_lrs_dict = compute_optimal_lr_path(
                            model, train_loader, device, n_classes, alpha, n_batches=1, return_per_layer=True
                        )
                        lr = float(np.median(list(optimal_lrs_dict.values())))
                        print(f"  Fallback: Optimal initial LR (median): {lr:.6e}")
                    else:
                        lr = compute_optimal_lr_path(
                            model, train_loader, device, n_classes, alpha, n_batches=1, return_per_layer=False
                        )
                        print(f"  Fallback: Optimal initial LR: {lr:.6e}")
                except Exception as e2:
                    print(f"  Warning: Fallback method also failed, using default LR: {e2}")
                    lr = initial_lr
                    optimal_lrs_dict = None
        else:
            # Original method: η ≈ median(d_f) / E[||∇w||]
            try:
                if use_layerwise_lr:
                    optimal_lrs_dict = compute_optimal_lr_path(
                        model, train_loader, device, n_classes, alpha, n_batches=1, return_per_layer=True
                    )
                    # Use median as initial LR for all layers (will be updated per-layer)
                    lr = float(np.median(list(optimal_lrs_dict.values())))
                    print(f"  Optimal initial LR (median): {lr:.6e}")
                    print(f"  Per-layer LRs: {[f'L{i}:{lr_val:.6e}' for i, lr_val in optimal_lrs_dict.items()]}")
                    # Store target LRs for warmup
                    target_lrs_dict = optimal_lrs_dict.copy()
                else:
                    optimal_lr = compute_optimal_lr_path(
                        model, train_loader, device, n_classes, alpha, n_batches=1, return_per_layer=False
                    )
                    lr = optimal_lr
                    print(f"  Optimal initial LR: {lr:.6e}")
                    target_lrs_dict = {0: lr}  # Store as single-layer dict for consistency
            except Exception as e:
                print(f"  Warning: Failed to compute optimal LR, using fallback: {e}")
                lr = initial_lr
                optimal_lrs_dict = None
    
    # Create optimizer with layer-wise parameter groups for path parameterization
    if use_layerwise_lr:
        # Apply initial warmup factor for epoch 0 (before training starts)
        initial_warmup_factor = warmup_start_ratio if (parameterization == "path" and warmup_epochs > 0) else 1.0
        
        # Group parameters by layer for LazarusMLP
        param_groups = []
        
        # Input projection (layer -1, distinct from blocks)
        INPUT_PROJ_LAYER = -1
        base_lr_input = optimal_lrs_dict.get(INPUT_PROJ_LAYER, lr) if (parameterization == "path" and optimal_lrs_dict is not None) else lr
        param_groups.append({
            'params': list(model.input_proj.parameters()),
            'lr': base_lr_input * initial_warmup_factor,
            'layer': INPUT_PROJ_LAYER
        })
        
        # Residual blocks (layers 0 to depth-1)
        for l, block in enumerate(model.blocks):
            base_lr_l = optimal_lrs_dict.get(l, lr) if (parameterization == "path" and optimal_lrs_dict is not None) else lr
            param_groups.append({
                'params': list(block.parameters()),
                'lr': base_lr_l * initial_warmup_factor,
                'layer': l
            })
        
        # Output readout (use last block's LR or median)
        readout_base_lr = optimal_lrs_dict.get(model.depth - 1, lr) if (parameterization == "path" and optimal_lrs_dict is not None) else lr
        param_groups.append({
            'params': list(model.readout.parameters()),
            'lr': readout_base_lr * initial_warmup_factor,
            'layer': 'readout'
        })
        
        if parameterization == "path":
            if warmup_epochs > 0:
                print(f"  Applied initial warmup factor: {initial_warmup_factor:.4f} (will warmup over {warmup_epochs} epochs)")
            else:
                print(f"  Warmup disabled (warmup_epochs=0), using full LRs from start")
        
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(param_groups, lr=lr)  # lr is used as default/fallback
        elif optimizer_type == "adam":
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
        elif optimizer_type == "muon":
            # Muon optimizer: similar to Adam but with different hyperparameters
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}. Options: sgd, adam, muon")
    else:
        # Standard optimizer (single LR for all parameters)
        # Apply initial warmup factor for epoch 0 if path parameterization
        initial_warmup_factor = warmup_start_ratio if (parameterization == "path" and warmup_epochs > 0) else 1.0
        initial_lr_warmup = lr * initial_warmup_factor
        
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr_warmup)
        elif optimizer_type == "adam":
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr_warmup)
        elif optimizer_type == "muon":
            # Muon optimizer: similar to Adam but with different hyperparameters
            # Using AdamW with default parameters (can be customized if needed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr_warmup)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}. Options: sgd, adam, muon")
        
        if parameterization == "path":
            if warmup_epochs > 0:
                print(f"  Applied initial warmup factor: {initial_warmup_factor:.4f} (will warmup over {warmup_epochs} epochs)")
            else:
                print(f"  Warmup disabled (warmup_epochs=0), using full LRs from start")
    
    loss_fn = nn.MSELoss()
    
    # History tracking
    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "M_g_avg": [],
        "C_def": [],
        "H_Lambda": [],
        "lr_per_layer": {},  # Dict mapping layer_idx -> list of LRs over epochs
        "grad_norms_per_layer": {}  # Dict mapping layer_idx -> list of gradient norms over epochs
    }
    
    # Get device ID for logging (to distinguish parallel workers)
    device_id = device.index if hasattr(device, 'index') and device.index is not None else "?"
    device_tag = f"[GPU {device_id}]" if torch.cuda.is_available() else "[CPU]"
    
    print(f"\n{device_tag} {'='*60}")
    print(f"{device_tag} Training with {parameterization} parameterization")
    print(f"{device_tag} {'='*60}")
    print(f"{device_tag} Epochs: {epochs}, LR: {lr}, Optimizer: {optimizer_type}")
    print(f"{device_tag} Metrics tracked every {metrics_freq} epochs")
    
    # Initialize LR tracking at epoch 0
    if use_layerwise_lr:
        for param_group in optimizer.param_groups:
            layer_idx = param_group.get('layer')
            if layer_idx is not None:
                if layer_idx not in history["lr_per_layer"]:
                    history["lr_per_layer"][layer_idx] = []
                history["lr_per_layer"][layer_idx].append(param_group['lr'])
    else:
        if 'global' not in history["lr_per_layer"]:
            history["lr_per_layer"]['global'] = []
        history["lr_per_layer"]['global'].append(lr)
    
    # Training loop
    for epoch in range(epochs + 1):
        # Initialize gradient norms tracking for this epoch
        grad_norms_epoch = None
        
        # Training step
        if epoch > 0:
            model.train()
            # Track gradient norms per layer (before clipping) - compute on first batch
            first_batch = True
            
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                
                if n_classes == 1:
                    loss = loss_fn(yhat, yb)
                else:
                    if yb.dim() > 1:
                        yb = yb.view(-1)
                    yb_class = (yb / alpha).long()
                    yb_class = torch.clamp(yb_class, 0, n_classes - 1)
                    yb_onehot = torch.zeros_like(yhat)
                    src_values = torch.ones_like(yb.unsqueeze(1)) * alpha
                    yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values)
                    loss = loss_fn(yhat, yb_onehot)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norms per layer BEFORE clipping (on first batch only)
                if first_batch:
                    try:
                        grad_norms_epoch = compute_gradient_norms_per_layer(
                            model, xb, yb, loss_fn, device, n_classes, alpha
                        )
                    except Exception as e:
                        # If computation fails, skip tracking for this epoch
                        grad_norms_epoch = None
                    first_batch = False
                
                # Optional gradient clipping to prevent exploding updates
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            # For "path" parameterization: update LR for next epoch (layer-wise if LazarusMLP)
            if parameterization == "path" and epoch > 0:  # Update LR after each epoch (except epoch 0)
                # Check if model has NaN/Inf parameters before computing LR
                has_nan_params = False
                for param in model.parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        has_nan_params = True
                        break
                
                if has_nan_params:
                    # Model has NaN - skip LR update, keep current LRs, just print loss
                    train_loss_epoch = evaluate_loss(model, train_loader, device, n_classes, alpha)
                    test_loss_epoch = evaluate_loss(model, test_loader, device, n_classes, alpha)
                    # Get current LRs for display
                    current_lrs = {}
                    for param_group in optimizer.param_groups:
                        layer_idx = param_group.get('layer')
                        if layer_idx == 'readout':
                            current_lrs['readout'] = param_group['lr']
                        elif isinstance(layer_idx, int):
                            current_lrs[layer_idx] = param_group['lr']
                    
                    # Format current LRs
                    lr_items = []
                    for k, v in current_lrs.items():
                        if k == 'readout':
                            lr_items.append((999999, 'readout', v))
                        else:
                            lr_items.append((k, f'L{k}', v))
                    lr_items.sort(key=lambda x: x[0])
                    
                    if len(lr_items) <= 10:
                        lr_str = ", ".join([f"{name}:{v:.6e}" for _, name, v in lr_items])
                    else:
                        first = ", ".join([f"{name}:{v:.6e}" for _, name, v in lr_items[:3]])
                        last = ", ".join([f"{name}:{v:.6e}" for _, name, v in lr_items[-3:]])
                        lr_str = f"{first}, ..., {last}"
                    
                    print(f"{device_tag} [Epoch {epoch}] Train Loss: {train_loss_epoch:.6f}, Test Loss: {test_loss_epoch:.6f}, LRs: {lr_str} (NaN model - LR update skipped)")
                else:
                    try:
                        if use_layerwise_lr:
                            # Get current learning rates for each layer
                            # Note: input_proj uses layer -1, blocks use 0 to depth-1
                            current_lrs_dict = {}
                            for param_group in optimizer.param_groups:
                                layer_idx = param_group.get('layer')
                                if isinstance(layer_idx, int):
                                    current_lrs_dict[layer_idx] = param_group['lr']
                            
                            # Compute gate mobility M_g,l for each layer using current LRs
                            M_g_per_layer = compute_gate_mobility_per_layer_path(
                                model, train_loader, current_lrs_dict, device, n_classes, alpha, n_batches=1
                            )
                            
                            # Check if computation failed
                            if M_g_per_layer is None or all(v is None for v in M_g_per_layer.values()):
                                # M_g computation failed - reduce LRs conservatively (model may be diverging)
                                updated_lrs_dict = {}
                                for l in current_lrs_dict:
                                    # Reduce by min_scale and cap at lr_max
                                    updated_lrs_dict[l] = min(current_lrs_dict[l] * lr_update_min_scale, lr_max)
                                
                                # Update parameter groups
                                # Note: input_proj uses layer -1, blocks use 0 to depth-1, readout uses 'readout'
                                updated_lrs = {}
                                for param_group in optimizer.param_groups:
                                    layer_idx = param_group.get('layer')
                                    if layer_idx == 'readout':
                                        target_lr = updated_lrs_dict.get(model.depth - 1, param_group['lr'])
                                        param_group['lr'] = target_lr
                                        updated_lrs['readout'] = target_lr
                                    elif isinstance(layer_idx, int):
                                        # Handle both input_proj (layer -1) and blocks (layers 0 to depth-1)
                                        target_lr = updated_lrs_dict.get(layer_idx, param_group['lr'])
                                        param_group['lr'] = target_lr
                                        updated_lrs[layer_idx] = target_lr
                                
                                # Track LRs
                                for layer_idx, lr_val in updated_lrs.items():
                                    if layer_idx not in history["lr_per_layer"]:
                                        history["lr_per_layer"][layer_idx] = []
                                    while len(history["lr_per_layer"][layer_idx]) < epoch:
                                        history["lr_per_layer"][layer_idx].append(None)
                                    history["lr_per_layer"][layer_idx].append(lr_val)
                                
                                # M_g computation failed - reduce LRs conservatively
                                train_loss_epoch = evaluate_loss(model, train_loader, device, n_classes, alpha)
                                test_loss_epoch = evaluate_loss(model, test_loader, device, n_classes, alpha)
                                # Get current LRs for display
                                current_lrs = {}
                                for param_group in optimizer.param_groups:
                                    layer_idx = param_group.get('layer')
                                    if layer_idx == 'readout':
                                        current_lrs['readout'] = param_group['lr']
                                    elif isinstance(layer_idx, int):
                                        current_lrs[layer_idx] = param_group['lr']
                                
                                # Format updated LRs for display
                                lr_items = []
                                for k, v in updated_lrs.items():
                                    if k == 'readout':
                                        lr_items.append((999999, 'readout', v))
                                    else:
                                        lr_items.append((k, f'L{k}', v))
                                lr_items.sort(key=lambda x: x[0])
                                
                                if len(lr_items) <= 10:
                                    lr_str = ", ".join([f"{name}:{v:.6e}" for _, name, v in lr_items])
                                else:
                                    first = ", ".join([f"{name}:{v:.6e}" for _, name, v in lr_items[:3]])
                                    last = ", ".join([f"{name}:{v:.6e}" for _, name, v in lr_items[-3:]])
                                    lr_str = f"{first}, ..., {last}"
                                
                                print(f"{device_tag} [Epoch {epoch}] Train Loss: {train_loss_epoch:.6f}, Test Loss: {test_loss_epoch:.6f}, LRs: {lr_str} (M_g failed - reduced LRs)")
                            else:
                                # Update LRs using damped mobility-based rule
                                updated_lrs_dict = update_lr_damped_mobility(
                                    current_lrs_dict, M_g_per_layer, target_mobility,
                                    lr_update_alpha, lr_update_eps, lr_update_min_scale, lr_update_max_scale, lr_max
                                )
                                
                                # Apply warmup scaling if in warmup period
                                warmup_factor = 1.0
                                if epoch <= warmup_epochs and warmup_epochs > 0:
                                    # Linear warmup from warmup_start_ratio to 1.0
                                    warmup_factor = warmup_start_ratio + (1.0 - warmup_start_ratio) * (epoch / warmup_epochs)
                                    warmup_factor = max(warmup_start_ratio, min(1.0, warmup_factor))  # Clamp
                                
                                # Update each parameter group with its layer's updated LR (with warmup scaling)
                                # Note: input_proj uses layer -1, blocks use 0 to depth-1, readout uses 'readout'
                                updated_lrs = {}
                                for param_group in optimizer.param_groups:
                                    layer_idx = param_group.get('layer')
                                    if layer_idx == 'readout':
                                        # Use last block's LR for readout
                                        target_lr = updated_lrs_dict.get(model.depth - 1, param_group['lr'])
                                        new_lr = target_lr * warmup_factor
                                        param_group['lr'] = new_lr
                                        updated_lrs['readout'] = new_lr
                                    elif isinstance(layer_idx, int):
                                        # Handle both input_proj (layer -1) and blocks (layers 0 to depth-1)
                                        target_lr = updated_lrs_dict.get(layer_idx, param_group['lr'])
                                        new_lr = target_lr * warmup_factor
                                        param_group['lr'] = new_lr
                                        updated_lrs[layer_idx] = new_lr
                                
                                if epoch <= warmup_epochs and warmup_epochs > 0:
                                    print(f"{device_tag}   Warmup: epoch {epoch}/{warmup_epochs}, factor={warmup_factor:.4f}")
                                
                                # Track LRs per layer (every epoch for path parameterization)
                                for layer_idx, lr_val in updated_lrs.items():
                                    if layer_idx not in history["lr_per_layer"]:
                                        history["lr_per_layer"][layer_idx] = []
                                    # Pad with None for epochs before this one if needed
                                    while len(history["lr_per_layer"][layer_idx]) < epoch:
                                        history["lr_per_layer"][layer_idx].append(None)
                                    history["lr_per_layer"][layer_idx].append(lr_val)
                                
                                # Print LR and loss every epoch for path parameterization
                                train_loss_epoch = evaluate_loss(model, train_loader, device, n_classes, alpha)
                                test_loss_epoch = evaluate_loss(model, test_loader, device, n_classes, alpha)
                                
                                # Format LRs for display - sort by layer index (handle 'readout' separately)
                                lr_items = []
                                for k, v in updated_lrs.items():
                                    if k == 'readout':
                                        lr_items.append((999999, 'readout', v))  # Put readout at end
                                    else:
                                        lr_items.append((k, f'L{k}', v))
                                lr_items.sort(key=lambda x: x[0])  # Sort by numeric key
                                
                                if len(lr_items) <= 10:
                                    lr_str = ", ".join([f"{name}:{v:.6e}" for _, name, v in lr_items])
                                else:
                                    first = ", ".join([f"{name}:{v:.6e}" for _, name, v in lr_items[:3]])
                                    last = ", ".join([f"{name}:{v:.6e}" for _, name, v in lr_items[-3:]])
                                    lr_str = f"{first}, ..., {last}"
                                
                                warmup_str = f" (warmup: {warmup_factor:.4f})" if (epoch <= warmup_epochs and warmup_epochs > 0) else ""
                                print(f"{device_tag} [Epoch {epoch}] Train Loss: {train_loss_epoch:.6f}, Test Loss: {test_loss_epoch:.6f}, LRs: {lr_str}{warmup_str}")
                        else:
                            # Single global LR (for non-LazarusMLP models)
                            # Get current LR
                            current_lr_global = optimizer.param_groups[0]['lr']
                            current_lrs_dict = {0: current_lr_global}  # Use layer 0 as placeholder
                            
                            # Compute gate mobility M_g using current LR
                            M_g_per_layer = compute_gate_mobility_per_layer_path(
                                model, train_loader, current_lrs_dict, device, n_classes, alpha, n_batches=1
                            )
                            
                            # Check if computation failed
                            if M_g_per_layer is None or M_g_per_layer.get(0) is None:
                                # M_g computation failed - keep current LR
                                train_loss_epoch = evaluate_loss(model, train_loader, device, n_classes, alpha)
                                test_loss_epoch = evaluate_loss(model, test_loader, device, n_classes, alpha)
                                print(f"{device_tag} [Epoch {epoch}] Train Loss: {train_loss_epoch:.6f}, Test Loss: {test_loss_epoch:.6f}, LR: {current_lr_global:.6e} (M_g computation failed - keeping current)")
                            else:
                                # Update LR using damped mobility-based rule
                                updated_lrs_dict = update_lr_damped_mobility(
                                    current_lrs_dict, M_g_per_layer, target_mobility,
                                    lr_update_alpha, lr_update_eps, lr_update_min_scale, lr_update_max_scale, lr_max
                                )
                                updated_lr = updated_lrs_dict.get(0, current_lr_global)
                                
                                # Apply warmup scaling if in warmup period
                                warmup_factor = 1.0
                                if epoch <= warmup_epochs and warmup_epochs > 0:
                                    # Linear warmup from warmup_start_ratio to 1.0
                                    warmup_factor = warmup_start_ratio + (1.0 - warmup_start_ratio) * (epoch / warmup_epochs)
                                    warmup_factor = max(warmup_start_ratio, min(1.0, warmup_factor))  # Clamp
                                
                                # Update optimizer learning rate (with warmup scaling)
                                actual_lr = updated_lr * warmup_factor
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = actual_lr
                                
                                # Track LR
                                if 'global' not in history["lr_per_layer"]:
                                    history["lr_per_layer"]['global'] = []
                                while len(history["lr_per_layer"]['global']) < epoch:
                                    history["lr_per_layer"]['global'].append(None)
                                history["lr_per_layer"]['global'].append(actual_lr)
                                
                                # Print LR and loss every epoch
                                train_loss_epoch = evaluate_loss(model, train_loader, device, n_classes, alpha)
                                test_loss_epoch = evaluate_loss(model, test_loader, device, n_classes, alpha)
                                warmup_str = f" (warmup: {warmup_factor:.4f})" if (epoch <= warmup_epochs and warmup_epochs > 0) else ""
                                print(f"{device_tag} [Epoch {epoch}] Train Loss: {train_loss_epoch:.6f}, Test Loss: {test_loss_epoch:.6f}, LR: {actual_lr:.6e}{warmup_str}")
                    except Exception as e:
                        # Print error but continue training
                        train_loss_epoch = evaluate_loss(model, train_loader, device, n_classes, alpha)
                        test_loss_epoch = evaluate_loss(model, test_loader, device, n_classes, alpha)
                        print(f"{device_tag} [Epoch {epoch}] Train Loss: {train_loss_epoch:.6f}, Test Loss: {test_loss_epoch:.6f}, [Path LR] Error: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue training with current LRs - don't let LR update failure stop training
        
        # Compute train and test loss every epoch (for history tracking)
        train_loss = evaluate_loss(model, train_loader, device, n_classes, alpha)
        test_loss = evaluate_loss(model, test_loader, device, n_classes, alpha)
        
        # Log train/test error every 10 epochs (if not computing full metrics)
        if epoch % 10 == 0 and epoch % metrics_freq != 0:
            print(f"{device_tag} [Epoch {epoch}] Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        # Save train/test loss to history every epoch
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        
        # Compute metrics at specified frequency
        if epoch % metrics_freq == 0:
            print(f"\n{device_tag} [Epoch {epoch}] Computing metrics...")
            
            # Train and test loss are already computed above, just print them
            print(f"{device_tag}   Train loss: {train_loss:.6f}, Test loss: {test_loss:.6f}")
            
            # Compute M_g
            if compute_metrics:
                try:
                    print(f"{device_tag}   Computing M_g...")
                    # For path parameterization with layer-wise LRs, use actual per-layer LRs
                    if parameterization == "path" and use_layerwise_lr:
                        # Get current per-layer LRs from optimizer
                        current_lrs_dict = {}
                        for param_group in optimizer.param_groups:
                            layer_idx = param_group.get('layer')
                            if isinstance(layer_idx, int):
                                current_lrs_dict[layer_idx] = param_group['lr']
                        
                        # Compute M_g using actual per-layer LRs
                        M_g_per_layer = compute_gate_mobility_per_layer_path(
                            model, train_loader, current_lrs_dict, device, n_classes, alpha, n_batches=m_g_n_batches
                        )
                        
                        # Filter out None, inf and nan values
                        valid_m_g = [v for v in M_g_per_layer.values() if v is not None and not (np.isinf(v) or np.isnan(v))]
                        if len(valid_m_g) > 0:
                            M_g_avg = np.mean(valid_m_g)
                        else:
                            M_g_avg = float('nan')
                            print(f"{device_tag}   Warning: All M_g values are inf or nan")
                        print(f"{device_tag}   M_g (avg, using actual per-layer LRs): {M_g_avg:.6f}")
                    else:
                        # Use standard computation with scalar LR
                        if isinstance(model, LazarusMLP):
                            m_g_result = _compute_gate_mobility_lazarus(
                                model, train_loader, lr, device,
                                n_batches=m_g_n_batches, n_classes=n_classes, alpha=alpha,
                                return_distributions=False
                            )
                        else:
                            m_g_result = compute_gate_mobility(
                                model, train_loader, lr, device,
                                n_batches=m_g_n_batches, n_classes=n_classes, alpha=alpha,
                                return_distributions=False
                            )
                        M_g_dict = m_g_result["M_g"]
                        # Filter out inf and nan values, handle empty list
                        valid_m_g = [v for v in M_g_dict.values() if not (np.isinf(v) or np.isnan(v))]
                        if len(valid_m_g) > 0:
                            M_g_avg = np.mean(valid_m_g)
                        else:
                            M_g_avg = float('nan')
                            print(f"{device_tag}   Warning: All M_g values are inf or nan")
                        print(f"{device_tag}   M_g (avg): {M_g_avg:.6f}")
                except Exception as e:
                    print(f"{device_tag}   Warning: Failed to compute M_g: {e}")
                    import traceback
                    traceback.print_exc()
                    M_g_avg = float('nan')
            else:
                M_g_avg = float('nan')
            
            # Compute C_def and H_Lambda (path kernel metrics)
            C_def = float('nan')
            H_Lambda = float('nan')
            if compute_metrics:
                try:
                    print(f"{device_tag}   Computing path kernel metrics (C_def, H_Lambda)...")
                    # Collect path factors
                    pack = collect_path_factors(
                    model, train_loader, device,
                    mode="routing_gain",
                    include_input=True,
                    max_samples=kernel_max_samples
                    )
                    X = pack["X"]
                    E_list = pack["E_list"]
                    
                    if X is not None and len(E_list) > 0:
                        # Check for valid data
                        if X.shape[0] == 0 or any(E.shape[0] == 0 for E in E_list):
                            print(f"{device_tag}   Warning: Empty path factors (X.shape={X.shape if X is not None else None}, E_list shapes={[E.shape for E in E_list]})")
                        else:
                            # Check for NaN/Inf in inputs
                            if torch.isnan(X).any() or torch.isinf(X).any():
                                print(f"{device_tag}   Warning: X contains NaN/Inf values")
                            for i, E in enumerate(E_list):
                                if torch.isnan(E).any() or torch.isinf(E).any():
                                    print(f"{device_tag}   Warning: E_list[{i}] contains NaN/Inf values")
                            
                            # Compute Path Deformation Capacity with improved numerical stability
                            try:
                                # Normalize each factor by its Frobenius norm to prevent overflow
                                # 
                                # Mathematical justification:
                                # - Σ_norm = (X @ X.T) / trace(X @ X.T)
                                # - If X' = X/α, then Σ' = (1/α²) * (X @ X.T)
                                # - trace(Σ') = (1/α²) * trace(X @ X.T)
                                # - Σ'_norm = Σ' / trace(Σ') = Σ_norm (scaling cancels)
                                # - Same logic applies to H: H_norm is invariant to uniform scaling
                                # - Therefore C_def = ||H_norm - Σ_norm||_F is preserved
                                #
                                # This normalization prevents overflow in the Hadamard product
                                # while preserving the metric's value.
                                
                                # Normalize X by Frobenius norm
                                X_fro_norm = torch.norm(X, p='fro')
                                if X_fro_norm > 1e-8:
                                    X_normalized = X / X_fro_norm
                                else:
                                    X_normalized = X
                                
                                # Normalize each E by Frobenius norm
                                E_list_normalized = []
                                for E in E_list:
                                    E_fro_norm = torch.norm(E, p='fro')
                                    if E_fro_norm > 1e-8:
                                        E_list_normalized.append(E / E_fro_norm)
                                    else:
                                        E_list_normalized.append(E)
                                
                                # Now compute with normalized inputs (metric is preserved)
                                C_def = compute_path_deformation_capacity(
                                    X_normalized, E_list_normalized, device,
                                    block_size=2048, dtype=torch.float32, use_tf32=True
                                )
                                
                                # Check for NaN/Inf
                                if np.isnan(C_def) or np.isinf(C_def):
                                    print(f"{device_tag}   Warning: C_def is {C_def} even after Frobenius normalization")
                                    # Try to diagnose: check if traces are valid
                                    try:
                                        Sigma = X_normalized @ X_normalized.T
                                        trace_Sigma = torch.trace(Sigma).item()
                                        print(f"{device_tag}     Trace(Sigma_norm) = {trace_Sigma:.6e}")
                                        # Check H trace
                                        factors = [X_normalized] + E_list_normalized
                                        H = compute_path_kernel_matrix(factors, device, block_size=2048, dtype=torch.float32, use_tf32=True)
                                        trace_H = torch.trace(H).item()
                                        print(f"{device_tag}     Trace(H) = {trace_H:.6e}")
                                        if torch.isnan(H).any() or torch.isinf(H).any():
                                            nan_count = torch.isnan(H).sum().item()
                                            inf_count = torch.isinf(H).sum().item()
                                            print(f"{device_tag}     H contains {nan_count} NaN and {inf_count} Inf values")
                                    except Exception as diag_e:
                                        print(f"{device_tag}     Diagnostic failed: {diag_e}")
                                else:
                                    print(f"{device_tag}   C_def: {C_def:.6f}")
                            except Exception as e:
                                print(f"{device_tag}   Error computing C_def: {e}")
                                C_def = float('nan')
                            
                            # Compute Path Covariance Entropy
                            # Use ORIGINAL (unnormalized) E_list for H_Lambda
                            # H_Lambda measures entropy of path overlaps, which needs raw values
                            # But check for NaN/Inf first and skip if present
                            try:
                                # Check if E_list has NaN/Inf before computing
                                has_nan_inf = False
                                for E in E_list:
                                    if torch.isnan(E).any() or torch.isinf(E).any():
                                        has_nan_inf = True
                                        break
                                
                                # Debug: summarize E_list stats to diagnose entropy collapse
                                if not has_nan_inf:
                                    try:
                                        e_shapes = [tuple(e.shape) for e in E_list]
                                        e_min = [float(torch.min(e).item()) for e in E_list]
                                        e_max = [float(torch.max(e).item()) for e in E_list]
                                        e_finite = [int(torch.isfinite(e).sum().item()) for e in E_list]
                                        print(f"{device_tag}   E_list stats - shapes:{e_shapes}, finite counts:{e_finite}, min:{e_min}, max:{e_max}")
                                    except Exception as _e_stats:
                                        print(f"{device_tag}   E_list stats failed: {_e_stats}")
                                
                                if has_nan_inf:
                                    H_Lambda = float('nan')
                                    print(f"{device_tag}   H_Lambda: nan (E_list contains NaN/Inf)")
                                else:
                                    # Try with normalize_factors=False first to preserve distribution
                                    H_Lambda = compute_path_covariance_entropy(
                                        E_list, device,  # Use original E_list
                                        n_bins=100, block_size=2048, dtype=torch.float32, use_tf32=True,
                                        normalize_factors=False, epsilon=1e-12, use_float64=True
                                    )
                                    
                                    # If entropy is 0, it means all Lambda values are identical
                                    # This happens when routing is very uniform (all paths have similar gains)
                                    # This is a legitimate result, not a bug
                                    if H_Lambda == 0.0:
                                        # Try with normalization to see if it helps reveal structure
                                        H_Lambda_norm = compute_path_covariance_entropy(
                                            E_list, device,
                                            n_bins=100, block_size=2048, dtype=torch.float32, use_tf32=True,
                                            normalize_factors=True, epsilon=1e-12, use_float64=True
                                        )
                                        if H_Lambda_norm > 0:
                                            H_Lambda = H_Lambda_norm
                                            print(f"{device_tag}   H_Lambda: {H_Lambda:.6f} (using normalized factors)")
                                        else:
                                            # Both give 0 - all Lambda values are identical
                                            # This means routing is completely uniform (all paths have same gain)
                                            H_Lambda = 0.0
                                            print(f"{device_tag}   H_Lambda: {H_Lambda:.6f} (all Lambda values identical - uniform routing)")
                                    else:
                                        print(f"{device_tag}   H_Lambda: {H_Lambda:.6f}")
                                    
                                    if np.isnan(H_Lambda) or np.isinf(H_Lambda):
                                        print(f"{device_tag}   H_Lambda: {H_Lambda}")
                            except Exception as e:
                                print(f"{device_tag}   Error computing H_Lambda: {e}")
                                import traceback
                                traceback.print_exc()
                                H_Lambda = float('nan')
                    else:
                        print(f"{device_tag}   Warning: Could not collect path factors (X={X is not None}, E_list len={len(E_list) if E_list else 0})")
                except Exception as e:
                    print(f"{device_tag}   Warning: Failed to compute path kernel metrics: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Track LRs per layer (at metrics frequency) - only if not already tracked this epoch
            # (for path parameterization, LRs are tracked every epoch, so skip here)
            if parameterization != "path" or not use_layerwise_lr:
                if use_layerwise_lr:
                    # Get current LRs from optimizer
                    for param_group in optimizer.param_groups:
                        layer_idx = param_group.get('layer')
                        if layer_idx is not None:
                            if layer_idx not in history["lr_per_layer"]:
                                history["lr_per_layer"][layer_idx] = []
                            history["lr_per_layer"][layer_idx].append(param_group['lr'])
                else:
                    # Single LR for all layers
                    if 'global' not in history["lr_per_layer"]:
                        history["lr_per_layer"]['global'] = []
                    history["lr_per_layer"]['global'].append(lr)
            
            # Track gradient norms per layer (at metrics frequency)
            if grad_norms_epoch is not None:
                for layer_idx, grad_norm in grad_norms_epoch.items():
                    if layer_idx not in history["grad_norms_per_layer"]:
                        history["grad_norms_per_layer"][layer_idx] = []
                    history["grad_norms_per_layer"][layer_idx].append(grad_norm)
            
            # Store metrics in history (train_loss and test_loss already saved above)
            history["M_g_avg"].append(M_g_avg)
            history["C_def"].append(C_def)
            history["H_Lambda"].append(H_Lambda)
        else:
            # For epochs where metrics are not computed, append NaN
            history["M_g_avg"].append(float('nan'))
            history["C_def"].append(float('nan'))
            history["H_Lambda"].append(float('nan'))
        
        # Save intermediate results (only at metrics frequency to avoid too frequent I/O)
        if epoch % metrics_freq == 0:
            # Convert numpy/torch types to native Python types for JSON serialization
            def convert_to_native(obj):
                """Recursively convert numpy/torch types to native Python types."""
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                elif isinstance(obj, torch.Tensor):
                    return obj.item() if obj.numel() == 1 else obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                else:
                    return obj
            
            history_serializable = convert_to_native(history)
            results_file = os.path.join(out_dir, f"history_{architecture}_{parameterization}.json")
            with open(results_file, 'w') as f:
                json.dump(history_serializable, f, indent=2)
    
    return history


def plot_metrics(history: Dict, parameterization: str, out_dir: str):
    """Plot metrics vs epochs."""
    epochs = history["epochs"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Train/Test Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], marker='o', label='Train Loss', linewidth=2)
    ax.plot(epochs, history["test_loss"], marker='s', label='Test Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'Train/Test Loss ({parameterization})')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # M_g
    ax = axes[0, 1]
    m_g_vals = [v for v in history["M_g_avg"] if not np.isnan(v) and not np.isinf(v)]
    m_g_epochs = [e for e, v in zip(epochs, history["M_g_avg"]) if not np.isnan(v) and not np.isinf(v)]
    if len(m_g_vals) > 0:
        ax.plot(m_g_epochs, m_g_vals, marker='o', label='M_g (avg)', linewidth=2, color='C2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('M_g (Gate Mobility Number)')
    ax.set_title(f'M_g vs Epoch ({parameterization})')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # C_def
    ax = axes[1, 0]
    c_def_vals = [v for v in history["C_def"] if not np.isnan(v)]
    c_def_epochs = [e for e, v in zip(epochs, history["C_def"]) if not np.isnan(v)]
    if len(c_def_vals) > 0:
        ax.plot(c_def_epochs, c_def_vals, marker='o', label='C_def', linewidth=2, color='C3')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('C_def (Path Deformation Capacity)')
    ax.set_title(f'C_def vs Epoch ({parameterization})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # H_Lambda
    ax = axes[1, 1]
    h_lambda_vals = [v for v in history["H_Lambda"] if not np.isnan(v)]
    h_lambda_epochs = [e for e, v in zip(epochs, history["H_Lambda"]) if not np.isnan(v)]
    if len(h_lambda_vals) > 0:
        ax.plot(h_lambda_epochs, h_lambda_vals, marker='o', label='H_Λ', linewidth=2, color='C4')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('H_Λ (Path Covariance Entropy)')
    ax.set_title(f'H_Λ vs Epoch ({parameterization})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'metrics_{parameterization}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics plot: {filename}")


def plot_lr_per_layer(history: Dict, parameterization: str, out_dir: str):
    """Plot learning rate per layer vs epoch with color coding."""
    if "lr_per_layer" not in history or not history["lr_per_layer"]:
        print(f"No LR per layer data for {parameterization}, skipping plot")
        return
    
    epochs = history["epochs"]
    lr_data = history["lr_per_layer"]
    
    # Get all layer indices and sort them
    layer_indices = []
    for key in lr_data.keys():
        if key == 'readout':
            layer_indices.append((999999, 'readout'))
        elif isinstance(key, int):
            layer_indices.append((key, f'L{key}'))
        else:
            layer_indices.append((0, str(key)))
    layer_indices.sort(key=lambda x: x[0])
    
    if not layer_indices:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Use viridis colormap with continuous gradient
    # Normalize layer indices to [0, 1] for smooth color gradient
    if len(layer_indices) > 1:
        # Extract numeric layer keys for normalization (exclude 'readout' and other non-numeric)
        numeric_keys = [k[0] for k in layer_indices if isinstance(k[0], int) and k[0] != 999999]
        if numeric_keys:
            min_key = min(numeric_keys)
            max_key = max(numeric_keys)
            key_range = max_key - min_key if max_key > min_key else 1
        else:
            key_range = len(layer_indices) - 1 if len(layer_indices) > 1 else 1
            min_key = 0
    else:
        key_range = 1
        min_key = 0
    
    cmap = plt.cm.get_cmap('viridis')
    
    for idx, (layer_key, layer_name) in enumerate(layer_indices):
        # Map layer index to [0, 1] for color gradient
        if isinstance(layer_key, int) and layer_key != 999999:
            # Normalize based on actual layer number
            if key_range > 0:
                color_val = (layer_key - min_key) / key_range
            else:
                color_val = 0.5
        else:
            # For non-numeric layers (like 'readout'), use end of colormap
            color_val = 1.0 if layer_key == 999999 else idx / max(len(layer_indices) - 1, 1)
        
        color = cmap(color_val)
        
        # Get the data (handle both key types)
        lr_values = None
        if layer_key in lr_data:
            lr_values = lr_data[layer_key]
        elif layer_name in lr_data:
            lr_values = lr_data[layer_name]
        elif str(layer_key) in lr_data:
            lr_values = lr_data[str(layer_key)]
        
        if lr_values is None:
            continue
        
        # Filter out None values and get corresponding epochs
        valid_lrs = []
        valid_epochs = []
        for e, lr_val in enumerate(lr_values):
            if lr_val is not None and not np.isnan(lr_val) and not np.isinf(lr_val):
                valid_lrs.append(lr_val)
                # For path parameterization, LRs are tracked every epoch (lr_values length = total epochs)
                # For others, they're tracked at metrics frequency (lr_values length = len(epochs))
                if len(epochs) == len(lr_values):
                    # Same length - use metrics epochs
                    valid_epochs.append(epochs[e])
                else:
                    # Different length - path parameterization tracks every epoch
                    # Use epoch index directly
                    valid_epochs.append(e)
        
        if len(valid_lrs) > 0:
            ax.plot(valid_epochs, valid_lrs, marker='o', label=layer_name, 
                   linewidth=2, color=color, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(f'Learning Rate per Layer vs Epoch ({parameterization})', fontsize=14)
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'lr_per_layer_{parameterization}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved LR per layer plot: {filename}")


def plot_gradient_norms_per_layer(history: Dict, parameterization: str, out_dir: str):
    """Plot gradient norms per layer vs epoch with color coding."""
    if "grad_norms_per_layer" not in history or not history["grad_norms_per_layer"]:
        print(f"No gradient norms per layer data for {parameterization}, skipping plot")
        return
    
    epochs = history["epochs"]
    grad_data = history["grad_norms_per_layer"]
    
    # Get all layer indices and sort them
    layer_indices = []
    for key in grad_data.keys():
        if isinstance(key, int):
            layer_indices.append((key, f'L{key}'))
        else:
            layer_indices.append((0, str(key)))
    layer_indices.sort(key=lambda x: x[0])
    
    if not layer_indices:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Use viridis colormap with continuous gradient
    # Normalize layer indices to [0, 1] for smooth color gradient
    if len(layer_indices) > 1:
        numeric_keys = [k[0] for k in layer_indices if isinstance(k[0], int)]
        if numeric_keys:
            min_key = min(numeric_keys)
            max_key = max(numeric_keys)
            key_range = max_key - min_key if max_key > min_key else 1
        else:
            key_range = len(layer_indices) - 1 if len(layer_indices) > 1 else 1
            min_key = 0
    else:
        key_range = 1
        min_key = 0
    
    cmap = plt.cm.get_cmap('viridis')
    
    for idx, (layer_key, layer_name) in enumerate(layer_indices):
        if layer_key not in grad_data:
            continue
        
        # Map layer index to [0, 1] for color gradient
        if isinstance(layer_key, int):
            if key_range > 0:
                color_val = (layer_key - min_key) / key_range
            else:
                color_val = 0.5
        else:
            color_val = idx / max(len(layer_indices) - 1, 1)
        
        color = cmap(color_val)
        
        grad_norms = grad_data[layer_key]
        
        # Filter out None/NaN/Inf values and get corresponding epochs
        valid_norms = []
        valid_epochs = []
        for e, norm_val in zip(range(len(grad_norms)), grad_norms):
            if norm_val is not None and not np.isnan(norm_val) and not np.isinf(norm_val) and norm_val > 0:
                valid_norms.append(norm_val)
                if e < len(epochs):
                    valid_epochs.append(epochs[e])
        
        if len(valid_norms) > 0:
            ax.plot(valid_epochs, valid_norms, marker='o', label=layer_name, 
                   linewidth=2, color=color, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm (pre-clip)', fontsize=12)
    ax.set_title(f'Gradient Norms per Layer vs Epoch ({parameterization})', fontsize=14)
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'grad_norms_per_layer_{parameterization}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved gradient norms per layer plot: {filename}")


def _worker_thread_parameterization(
    job_queue, result_queue, gpu_id,
    input_dim, n_classes, cfg, architecture, depth, width, widths, out_dir
):
    """Worker thread that runs parameterization training jobs on a specific GPU."""
    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")
    
    print(f"[GPU {gpu_id}] Worker started on {device}")
    
    # Build data loaders for this worker
    cfg_fixed = copy.deepcopy(cfg)
    if isinstance(cfg_fixed["dataset"].get("alpha"), list):
        cfg_fixed["dataset"]["alpha"] = cfg_fixed["dataset"]["alpha"][0]
    if isinstance(cfg_fixed["dataset"].get("n_train"), list):
        cfg_fixed["dataset"]["n_train"] = cfg_fixed["dataset"]["n_train"][0]
    
    Xtr, ytr, Xva, yva, Xte, yte, meta = build_mnist_datasets(cfg_fixed)
    
    train_dataset = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), 
                                  torch.tensor(ytr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(Xte, dtype=torch.float32), 
                                 torch.tensor(yte, dtype=torch.float32))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(cfg["training"]["batch_size"]), 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )
    
    job_count = 0
    while True:
        try:
            job = job_queue.get(timeout=1)
            if job is None:  # sentinel
                break
            
            param = job
            job_count += 1
            
            start_time = time.time()
            print(f"[GPU {gpu_id}] Starting parameterization: {param}")
            
            try:
                # Parse parameterization name to extract base parameterization and optimizer
                base_param, optimizer_override = parse_parameterization_name(param)
                
                # Ensure we're on the correct device
                if torch.cuda.is_available() and gpu_id is not None:
                    torch.cuda.set_device(gpu_id)
                
                # For "path" parameterization, always use LazarusMLP architecture
                if base_param == "path":
                    actual_architecture = "lazarus"
                    model = LazarusMLP(
                        d_in=input_dim,
                        widths=widths,
                        bias=cfg["model"].get("bias", True),
                        activation=cfg["model"].get("activation", "relu"),
                        n_classes=n_classes
                    ).to(device)
                else:
                    actual_architecture = architecture
                    # Create model based on architecture
                    # Check if base_param contains "batchnorm" to use MLPBatchNorm instead of MLP
                    use_batchnorm = "batchnorm" in base_param.lower()
                    
                    if architecture == "standard":
                        if use_batchnorm:
                            model = MLPBatchNorm(
                                d_in=input_dim,
                                widths=widths,
                                bias=cfg["model"].get("bias", True),
                                activation=cfg["model"].get("activation", "relu"),
                                n_classes=n_classes
                            ).to(device)
                        else:
                            model = MLP(
                                d_in=input_dim,
                                widths=widths,
                                bias=cfg["model"].get("bias", True),
                                activation=cfg["model"].get("activation", "relu"),
                                n_classes=n_classes
                            ).to(device)
                    elif architecture == "resnet":
                        model = MLPResNet(
                            d_in=input_dim,
                            widths=widths,
                            bias=cfg["model"].get("bias", True),
                            activation=cfg["model"].get("activation", "relu"),
                            n_classes=n_classes
                        ).to(device)
                    elif architecture == "lazarus":
                        model = LazarusMLP(
                            d_in=input_dim,
                            widths=widths,
                            bias=cfg["model"].get("bias", True),
                            activation=cfg["model"].get("activation", "relu"),
                            n_classes=n_classes
                        ).to(device)
                    else:
                        raise ValueError(f"Unknown architecture: {architecture}")
                    
                    # Initialize with parameterization (use base parameterization for initialization)
                    initialize_parameterization(model, base_param, device)
                
                # Train (pass optimizer override if specified)
                history = train_with_parameterization(
                    model, train_loader, test_loader, cfg, device, base_param, actual_architecture, out_dir,
                    optimizer_override=optimizer_override
                )
                
                # Plot individual metrics (use original param name for file naming)
                plot_metrics(history, f"{actual_architecture}_{param}", out_dir)
                # Plot LR per layer and gradient norms per layer
                plot_lr_per_layer(history, f"{actual_architecture}_{param}", out_dir)
                plot_gradient_norms_per_layer(history, f"{actual_architecture}_{param}", out_dir)
                
                total_time = time.time() - start_time
                print(f"[GPU {gpu_id}] Completed {param} in {total_time:.1f}s")
                
                key = f"{actual_architecture}_{param}"
                result_queue.put(("success", key, history))
                
                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                print(f"[GPU {gpu_id}] ERROR in {param}: {str(e)}")
                result_queue.put(("error", param, error_msg))
                
        except Empty:
            continue
    
    print(f"[GPU {gpu_id}] Worker finished ({job_count} jobs completed)")


def _run_parallel_training(
    parameterizations, architecture, depth, width, widths,
    input_dim, n_classes, cfg, cfg_fixed,
    train_dataset, test_dataset, out_dir, num_gpus, gpu_ids
) -> Dict:
    """Run parameterizations in parallel across multiple GPUs."""
    job_queue = Queue()
    result_queue = Queue()
    
    # Add all parameterizations as jobs
    for param in parameterizations:
        job_queue.put(param)
    
    # Add sentinels to stop workers
    for _ in range(num_gpus):
        job_queue.put(None)
    
    # Start worker threads
    workers = []
    for gpu_idx in range(num_gpus):
        gpu_id = gpu_ids[gpu_idx] if torch.cuda.is_available() else None
        t = Thread(
            target=_worker_thread_parameterization,
            args=(
                job_queue, result_queue, gpu_id,
                input_dim, n_classes, cfg, architecture, depth, width, widths, out_dir
            )
        )
        t.daemon = False
        t.start()
        workers.append(t)
        time.sleep(0.2)  # Small delay to avoid race conditions
    
    print(f"Started {len(workers)} worker threads across {num_gpus} GPU(s)\n")
    
    # Collect results
    all_histories = {}
    completed = 0
    total_jobs = len(parameterizations)
    consecutive_timeouts = 0
    max_consecutive_timeouts = 10  # Allow up to 10 consecutive timeouts before giving up
    
    print(f"Waiting for {total_jobs} parameterization(s) to complete...")
    
    while completed < total_jobs:
        try:
            # Use a longer timeout (30 minutes) to accommodate long training runs
            status, key, result = result_queue.get(timeout=1800)  # 30 minute timeout
            consecutive_timeouts = 0  # Reset timeout counter on success
            if status == "success":
                all_histories[key] = result
                completed += 1
                print(f"Progress: {completed}/{total_jobs} parameterizations completed")
            elif status == "error":
                print(f"ERROR: Failed to train parameterization {key}")
                completed += 1  # Count errors too
        except Empty:
            consecutive_timeouts += 1
            if consecutive_timeouts >= max_consecutive_timeouts:
                print(f"Warning: {max_consecutive_timeouts} consecutive timeouts. Some jobs may still be running...")
                print(f"Completed: {completed}/{total_jobs}, continuing to wait for workers...")
                break
            # Don't print warning for first few timeouts - jobs might just be taking longer
            if consecutive_timeouts % 5 == 0:  # Print every 5th timeout
                print(f"Still waiting... ({completed}/{total_jobs} completed, {consecutive_timeouts} timeouts)")
    
    # Wait for all workers to finish (with longer timeout)
    print("Waiting for all worker threads to finish...")
    for t in workers:
        t.join(timeout=300)  # 5 minute timeout per worker
    
    return all_histories


def plot_comparison(all_histories: Dict[str, Dict], out_dir: str):
    """Plot comparison across parameterizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'standard': 'C0', 'mup': 'C1', 'ntk': 'C2'}
    
    # Train/Test Loss
    ax = axes[0, 0]
    for param, history in all_histories.items():
        epochs = history["epochs"]
        ax.plot(epochs, history["train_loss"], marker='o', label=f'Train ({param})', 
                linewidth=2, color=colors.get(param, 'gray'), linestyle='-')
        ax.plot(epochs, history["test_loss"], marker='s', label=f'Test ({param})', 
                linewidth=2, color=colors.get(param, 'gray'), linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Train/Test Loss Comparison')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # M_g
    ax = axes[0, 1]
    for param, history in all_histories.items():
        epochs = history["epochs"]
        m_g_vals = [v for v in history["M_g_avg"] if not np.isnan(v) and not np.isinf(v)]
        m_g_epochs = [e for e, v in zip(epochs, history["M_g_avg"]) if not np.isnan(v) and not np.isinf(v)]
        if len(m_g_vals) > 0:
            ax.plot(m_g_epochs, m_g_vals, marker='o', label=f'M_g ({param})', 
                    linewidth=2, color=colors.get(param, 'gray'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('M_g (Gate Mobility Number)')
    ax.set_title('M_g Comparison')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # C_def
    ax = axes[1, 0]
    for param, history in all_histories.items():
        epochs = history["epochs"]
        c_def_vals = [v for v in history["C_def"] if not np.isnan(v)]
        c_def_epochs = [e for e, v in zip(epochs, history["C_def"]) if not np.isnan(v)]
        if len(c_def_vals) > 0:
            ax.plot(c_def_epochs, c_def_vals, marker='o', label=f'C_def ({param})', 
                    linewidth=2, color=colors.get(param, 'gray'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('C_def (Path Deformation Capacity)')
    ax.set_title('C_def Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # H_Lambda
    ax = axes[1, 1]
    for param, history in all_histories.items():
        epochs = history["epochs"]
        h_lambda_vals = [v for v in history["H_Lambda"] if not np.isnan(v)]
        h_lambda_epochs = [e for e, v in zip(epochs, history["H_Lambda"]) if not np.isnan(v)]
        if len(h_lambda_vals) > 0:
            ax.plot(h_lambda_epochs, h_lambda_vals, marker='o', label=f'H_Λ ({param})', 
                    linewidth=2, color=colors.get(param, 'gray'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('H_Λ (Path Covariance Entropy)')
    ax.set_title('H_Λ Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'metrics_comparison.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {filename}")


def plot_gradient_norms_comparison(all_histories: Dict[str, Dict], out_dir: str):
    """Plot gradient norms comparison across parameterizations (meaned per layer)."""
    # First, collect all layer indices across all histories
    all_layer_indices = set()
    for history in all_histories.values():
        if "grad_norms_per_layer" in history:
            all_layer_indices.update(history["grad_norms_per_layer"].keys())
    
    if not all_layer_indices:
        print("No gradient norms data available for comparison")
        return
    
    # Sort layer indices
    sorted_layers = sorted([l for l in all_layer_indices if isinstance(l, int)])
    
    # Compute mean gradient norm per layer for each parameterization
    param_mean_grad_norms = {}
    for param, history in all_histories.items():
        if "grad_norms_per_layer" not in history:
            continue
        
        epochs = history["epochs"]
        grad_data = history["grad_norms_per_layer"]
        
        # Compute mean gradient norm per layer across all epochs
        layer_means = {}
        for layer_idx in sorted_layers:
            if layer_idx in grad_data:
                norms = [v for v in grad_data[layer_idx] if v is not None and not np.isnan(v) and not np.isinf(v) and v > 0]
                if len(norms) > 0:
                    layer_means[layer_idx] = np.mean(norms)
        
        if layer_means:
            param_mean_grad_norms[param] = layer_means
    
    if not param_mean_grad_norms:
        return
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors_map = {'standard': 'C0', 'mup': 'C1', 'ntk': 'C2', 'mup_L': 'C3', 'path': 'C4'}
    
    for param, layer_means in param_mean_grad_norms.items():
        layers = sorted(layer_means.keys())
        means = [layer_means[l] for l in layers]
        color = colors_map.get(param.split('_')[-1] if '_' in param else param, 'gray')
        ax.plot(layers, means, marker='o', label=param, linewidth=2, color=color, markersize=6)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean Gradient Norm (pre-clip)', fontsize=12)
    ax.set_title('Mean Gradient Norms per Layer Comparison', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'grad_norms_comparison.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved gradient norms comparison plot: {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="parameterizations/config_mnist_parameterizations.yaml")
    args = ap.parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    
    # Device
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Fix config to handle list values
    cfg_fixed = copy.deepcopy(cfg)
    if isinstance(cfg_fixed["dataset"].get("alpha"), list):
        cfg_fixed["dataset"]["alpha"] = cfg_fixed["dataset"]["alpha"][0]
    if isinstance(cfg_fixed["dataset"].get("n_train"), list):
        cfg_fixed["dataset"]["n_train"] = cfg_fixed["dataset"]["n_train"][0]
    
    # Build dataset
    Xtr, ytr, Xva, yva, Xte, yte, meta = build_mnist_datasets(cfg_fixed)
    input_dim = meta["d"]
    n_classes = meta["n_classes"]
    
    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), 
                                  torch.tensor(ytr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(Xte, dtype=torch.float32), 
                                 torch.tensor(yte, dtype=torch.float32))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(cfg["training"]["batch_size"]), 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )
    
    # Output directory
    out_dir = os.path.join(project_root, "outputs", f"{cfg['experiment_name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(out_dir)
    save_json(cfg, os.path.join(out_dir, "config.json"))
    print(f"Output directory: {out_dir}")
    
    # Get parameterizations to run
    param_cfg = cfg["training"].get("parameterization", "standard")
    if isinstance(param_cfg, list):
        parameterizations = param_cfg
    else:
        parameterizations = [param_cfg]
    
    # Get architecture and model config
    architecture = cfg["model"].get("architecture", "standard")
    depth = int(cfg["model"].get("depth", 4))
    width = int(cfg["model"].get("width", 1024))
    # Create fixed-width hidden layers list
    widths = [width] * depth
    
    # Detect available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        print(f"Found {num_gpus} GPU(s): {gpu_ids}")
        for gpu_id in gpu_ids:
            print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        num_gpus = 1
        gpu_ids = [None]
        print("No GPUs found, using CPU")
    
    # Use multi-GPU parallelization if we have multiple parameterizations and GPUs
    if len(parameterizations) > 1 and num_gpus > 1:
        print(f"\n{'='*60}")
        print(f"Running {len(parameterizations)} parameterizations in parallel across {num_gpus} GPU(s)")
        print(f"{'='*60}\n")
        all_histories = _run_parallel_training(
            parameterizations, architecture, depth, width, widths,
            input_dim, n_classes, cfg, cfg_fixed,
            train_dataset, test_dataset, out_dir, num_gpus, gpu_ids
        )
    else:
        # Sequential execution (single GPU or single parameterization)
        print(f"\n{'='*60}")
        print(f"Running {len(parameterizations)} parameterization(s) sequentially")
        print(f"{'='*60}\n")
        all_histories = {}
        
        for param in parameterizations:
            # Parse parameterization name to extract base parameterization and optimizer
            base_param, optimizer_override = parse_parameterization_name(param)
            
            # For "path" parameterization, always use LazarusMLP architecture
            # (Lazarus initialization is built into LazarusMLP.__init__)
            if base_param == "path":
                actual_architecture = "lazarus"
                print(f"\n{'='*60}")
                print(f"Running {param} parameterization (using LazarusMLP architecture)")
                if optimizer_override:
                    print(f"Optimizer override: {optimizer_override}")
                print(f"Depth: {depth}, Width: {width}")
                print(f"{'='*60}")
                
                # Create LazarusMLP (initialization is already done in __init__)
                model = LazarusMLP(
                    d_in=input_dim,
                    widths=widths,
                    bias=cfg["model"].get("bias", True),
                    activation=cfg["model"].get("activation", "relu"),
                    n_classes=n_classes
                )
                # No need to call initialize_parameterization - LazarusMLP initializes itself
            else:
                actual_architecture = architecture
                print(f"\n{'='*60}")
                print(f"Running {param} parameterization with {architecture} architecture")
                if optimizer_override:
                    print(f"Optimizer override: {optimizer_override}")
                print(f"Depth: {depth}, Width: {width}")
                print(f"{'='*60}")
                
                # Create model based on architecture
                # Check if base_param contains "batchnorm" to use MLPBatchNorm instead of MLP
                use_batchnorm = "batchnorm" in base_param.lower()
                
                if architecture == "standard":
                    if use_batchnorm:
                        model = MLPBatchNorm(
                            d_in=input_dim,
                            widths=widths,
                            bias=cfg["model"].get("bias", True),
                            activation=cfg["model"].get("activation", "relu"),
                            n_classes=n_classes
                        )
                    else:
                        model = MLP(
                            d_in=input_dim,
                            widths=widths,
                            bias=cfg["model"].get("bias", True),
                            activation=cfg["model"].get("activation", "relu"),
                            n_classes=n_classes
                        )
                elif architecture == "resnet":
                    model = MLPResNet(
                        d_in=input_dim,
                        widths=widths,
                        bias=cfg["model"].get("bias", True),
                        activation=cfg["model"].get("activation", "relu"),
                        n_classes=n_classes
                    )
                elif architecture == "lazarus":
                    model = LazarusMLP(
                        d_in=input_dim,
                        widths=widths,
                        bias=cfg["model"].get("bias", True),
                        activation=cfg["model"].get("activation", "relu"),
                        n_classes=n_classes
                    )
                else:
                    raise ValueError(f"Unknown architecture: {architecture}. Options: standard, resnet, lazarus")
                
                # Initialize with parameterization (use base parameterization for initialization)
                initialize_parameterization(model, base_param, device)
            
            # Train (pass optimizer override if specified)
            history = train_with_parameterization(
                model, train_loader, test_loader, cfg, device, base_param, actual_architecture, out_dir,
                optimizer_override=optimizer_override
            )
            
            # Store with architecture in key for comparison
            key = f"{actual_architecture}_{param}"
            all_histories[key] = history
            
            # Plot individual metrics
            plot_metrics(history, f"{actual_architecture}_{param}", out_dir)
            # Plot LR per layer and gradient norms per layer
            plot_lr_per_layer(history, f"{actual_architecture}_{param}", out_dir)
            plot_gradient_norms_per_layer(history, f"{actual_architecture}_{param}", out_dir)
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Plot comparison (works for both parallel and sequential execution)
    if len(all_histories) > 1:
        print(f"\n{'='*60}")
        print("Creating comparison plots...")
        print(f"{'='*60}")
        plot_comparison(all_histories, out_dir)
        plot_gradient_norms_comparison(all_histories, out_dir)
    
    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
