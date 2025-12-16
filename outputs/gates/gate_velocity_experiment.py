"""
Gate Velocity / Gate Mobility Number Experiment

This experiment computes the "Gate Mobility Number" (M_g) which measures
the energy cost required to flip a gate at different depths.

M_g = (η * E[||∇w||]) / E[d_f]

where:
- d_f(i) = |h_i(x)| / ||x|| is the distance to sign flip for neuron i
- η is the learning rate
- E[||∇w||] is the expected gradient norm
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
from typing import Dict, List, Tuple

import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.data.models.ffnn import MLP
from src.data.mnist import build_mnist_datasets
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.save_io import ensure_dir

# Import model variants from local files (same directory)
# Add script directory to path for local imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from mlp_resnet import MLPResNet
from mlp_batchnorm import MLPBatchNorm
from mlp_skip import MLPSkip


@torch.no_grad()
def compute_distance_to_flip(model, x_batch: torch.Tensor, device: torch.device) -> Dict[int, torch.Tensor]:
    """
    Compute distance to sign flip d_f(i) = |h_i(x)| / ||x|| for each neuron in each layer.
    GPU-optimized version.
    
    Args:
        model: The neural network model
        x_batch: Input batch (batch_size, d_in)
        device: Device to run on
    
    Returns:
        Dictionary mapping layer index to tensor of distances (batch_size, width_l)
    """
    model.eval()
    x_batch = x_batch.to(device, non_blocking=True)
    
    distances = {}
    h = x_batch  # Start with input
    
    for l in range(model.depth):
        # Get pre-activation: h_i(x) = w_i · x + b_i
        linear = model.linears[l]
        u = linear(h)  # (batch_size, width_l)
        
        # Compute ||x|| for each sample in batch
        # x is h (the input to this layer)
        x_norm = torch.norm(h, dim=1, keepdim=True)  # (batch_size, 1)
        
        # Avoid division by zero
        x_norm = torch.clamp(x_norm, min=1e-8)
        
        # Distance to flip: d_f(i) = |h_i(x)| / ||x||
        d_f = torch.abs(u) / x_norm  # (batch_size, width_l)
        
        distances[l] = d_f
        
        # Update h for next layer (after activation)
        h = model.activation(u)
    
    return distances


def compute_gradient_norms(model, x_batch: torch.Tensor, y_batch: torch.Tensor, 
                           loss_fn, device: torch.device, n_classes: int = 1, alpha: float = 1.0) -> Dict[int, float]:
    """
    Compute gradient norms for each layer. GPU-optimized.
    
    Args:
        model: The neural network model
        x_batch: Input batch
        y_batch: Target batch
        loss_fn: Loss function
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
        # Binary classification: MSE loss
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
        loss = loss_fn(yhat, yb_onehot)
    
    # Backward pass
    loss.backward()
    
    gradient_norms = {}
    for l in range(model.depth):
        weight = model.linears[l].weight
        if weight.grad is not None:
            grad_norm = torch.norm(weight.grad).item()
            gradient_norms[l] = grad_norm
        else:
            gradient_norms[l] = 0.0
    
    return gradient_norms


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
        Dictionary with M_g values and optionally d_f distributions
    """
    model = model.to(device)
    
    # Loss function (MSE)
    loss_fn = nn.MSELoss()
    
    # Use torch tensors for accumulation (GPU-friendly)
    grad_norm_sums = torch.zeros(model.depth, device=device)
    distance_sums = torch.zeros(model.depth, device=device)
    
    # Store distributions if requested
    d_f_distributions = {l: [] for l in range(model.depth)} if return_distributions else None
    
    batch_count = 0
    
    for x_batch, y_batch in train_loader:
        if batch_count >= n_batches:
            break
        
        # Compute distances to flip
        distances = compute_distance_to_flip(model, x_batch, device)
        
        # Compute gradient norms
        grad_norms = compute_gradient_norms(model, x_batch, y_batch, loss_fn, device, n_classes, alpha)
        
        # Accumulate on GPU
        for l in range(model.depth):
            # Average distance over batch and neurons (keep on GPU)
            mean_dist = distances[l].mean()
            distance_sums[l] += mean_dist
            
            # Store distribution if requested (flatten and move to CPU)
            if return_distributions:
                d_f_flat = distances[l].flatten().cpu().numpy()
                d_f_distributions[l].extend(d_f_flat)
            
            # Gradient norm (convert to tensor)
            grad_norm_sums[l] += grad_norms[l]
        
        batch_count += 1
    
    # Compute M_g for each layer
    M_g = {}
    for l in range(model.depth):
        E_grad_norm = (grad_norm_sums[l] / batch_count).item()
        E_d_f = (distance_sums[l] / batch_count).item()
        
        if E_d_f > 0:
            M_g[l] = (lr * E_grad_norm) / E_d_f
        else:
            M_g[l] = float('inf')
    
    result = {"M_g": M_g}
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
    architecture: str = "standard"  # "standard", "batchnorm", "skip", "resnet"
) -> Dict:
    """
    Run the gate velocity experiment for all depth/width combinations.
    
    Args:
        depths: List of depths (number of hidden layers)
        widths: List of widths (neurons per layer)
        device: Device to run on
        config_path: Path to config file
        n_batches: Number of batches to average over
        batch_size: Batch size for data loading
    
    Returns:
        Dictionary with results: {depth: {width: {layer: M_g}}}
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
            
            # Store results (average M_g across layers)
            avg_M_g = np.mean(list(M_g_dict.values()))
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
                "d_f_distributions": d_f_distributions_serializable,
                "depth": depth,
                "width": width
            }
            
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


def plot_heatmaps(results: Dict, out_dir: str, suffix: str = ""):
    """
    Create heatmaps of M_g vs depth and width.
    
    Args:
        results: Results dictionary from run_experiment
        out_dir: Output directory for plots
    """
    ensure_dir(out_dir)
    
    # Extract data for heatmap
    depths = sorted(results.keys())
    widths = sorted(set(w for d in results.values() for w in d.keys()))
    
    # Create matrix for average M_g
    M_g_matrix = np.zeros((len(depths), len(widths)))
    
    for i, depth in enumerate(depths):
        for j, width in enumerate(widths):
            if width in results[depth]:
                M_g_matrix[i, j] = results[depth][width]["M_g_avg"]
            else:
                M_g_matrix[i, j] = np.nan
    
    # Normal scale heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(M_g_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='M_g (Gate Mobility Number)')
    plt.xticks(range(len(widths)), widths)
    plt.yticks(range(len(depths)), depths)
    plt.xlabel('Width (neurons per layer)')
    plt.ylabel('Depth (number of layers)')
    # Determine title suffix based on architecture
    if "_resnet" in suffix:
        title_suffix = " (ResNet: Skip + BatchNorm)"
    elif "_batchnorm" in suffix:
        title_suffix = " (BatchNorm Only)"
    elif "_skip" in suffix:
        title_suffix = " (Skip Connections Only)"
    else:
        title_suffix = " (Standard)"
    plt.title(f'Gate Mobility Number M_g (Normal Scale){title_suffix}')
    
    # Add text annotations
    for i in range(len(depths)):
        for j in range(len(widths)):
            if not np.isnan(M_g_matrix[i, j]):
                text = plt.text(j, i, f'{M_g_matrix[i, j]:.3f}',
                              ha="center", va="center", color="white" if M_g_matrix[i, j] > M_g_matrix[~np.isnan(M_g_matrix)].mean() else "black")
    
    plt.tight_layout()
    filename = f'gate_mobility_heatmap_normal{suffix}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()
    
    # Log scale heatmap
    # Replace inf and very small values for log scale
    M_g_matrix_log = np.copy(M_g_matrix)
    M_g_matrix_log[M_g_matrix_log <= 0] = np.nan
    M_g_matrix_log = np.log10(M_g_matrix_log)
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(M_g_matrix_log, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='log10(M_g)')
    plt.xticks(range(len(widths)), widths)
    plt.yticks(range(len(depths)), depths)
    plt.xlabel('Width (neurons per layer)')
    plt.ylabel('Depth (number of layers)')
    # Determine title suffix based on architecture
    if "_resnet" in suffix:
        title_suffix = " (ResNet: Skip + BatchNorm)"
    elif "_batchnorm" in suffix:
        title_suffix = " (BatchNorm Only)"
    elif "_skip" in suffix:
        title_suffix = " (Skip Connections Only)"
    else:
        title_suffix = " (Standard)"
    plt.title(f'Gate Mobility Number M_g (Log Scale){title_suffix}')
    
    # Add text annotations
    for i in range(len(depths)):
        for j in range(len(widths)):
            if not np.isnan(M_g_matrix_log[i, j]):
                text = plt.text(j, i, f'{M_g_matrix_log[i, j]:.3f}',
                              ha="center", va="center", color="white" if M_g_matrix_log[i, j] > np.nanmean(M_g_matrix_log) else "black")
    
    plt.tight_layout()
    filename = f'gate_mobility_heatmap_log{suffix}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()
    
    print(f"Saved heatmaps{suffix} to {out_dir}")


def main():
    # Configuration
    depths =  [1,2,3, 4, 7,12,16]
    widths = [16,32, 64,128,256,512, 1024,2048,5124,10248]
    
    # Flag to control JSON saving (set to True to save results)
    save_json_results = False
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Output directory (relative to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir)  # outputs/gates
    ensure_dir(out_dir)
    
    # Config path (relative to project root)
    config_path = os.path.join(project_root, "configs", "config_mnist.yaml")
    
    # Run experiments for all architectures
    print("Starting Gate Velocity Experiment...")
    print(f"Depths: {depths}")
    print(f"Widths: {widths}")
    print(f"Output directory: {out_dir}")
    
    # Number of samples to use: n_batches * batch_size
    n_batches = 50  # Increase this to use more batches (e.g., 50 = 51,200 samples)
    batch_size = 1024  # Increase this for larger batches (e.g., 2048 = larger batches)
    
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
        architecture="standard"
    )
    
    # Save standard results (if flag is enabled)
    if save_json_results:
        results_file_standard = os.path.join(out_dir, "gate_mobility_results_standard.json")
        with open(results_file_standard, 'w') as f:
            json.dump(results_standard, f, indent=2)
        print(f"Saved standard results to {results_file_standard}")
    
    # Create heatmaps for standard
    plot_heatmaps(results_standard, out_dir, suffix="_standard")
    
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
        architecture="batchnorm"
    )
    
    # Save batch norm results (if flag is enabled)
    if save_json_results:
        results_file_batchnorm = os.path.join(out_dir, "gate_mobility_results_batchnorm.json")
        with open(results_file_batchnorm, 'w') as f:
            json.dump(results_batchnorm, f, indent=2)
        print(f"Saved batch norm results to {results_file_batchnorm}")
    
    # Create heatmaps for batch norm
    plot_heatmaps(results_batchnorm, out_dir, suffix="_batchnorm")
    
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
        architecture="skip"
    )
    
    # Save skip connections results (if flag is enabled)
    if save_json_results:
        results_file_skip = os.path.join(out_dir, "gate_mobility_results_skip.json")
        with open(results_file_skip, 'w') as f:
            json.dump(results_skip, f, indent=2)
        print(f"Saved skip connections results to {results_file_skip}")
    
    # Create heatmaps for skip connections
    plot_heatmaps(results_skip, out_dir, suffix="_skip")
    
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
        architecture="resnet"
    )
    
    # Save ResNet results (if flag is enabled)
    if save_json_results:
        results_file_resnet = os.path.join(out_dir, "gate_mobility_results_resnet.json")
        with open(results_file_resnet, 'w') as f:
            json.dump(results_resnet, f, indent=2)
        print(f"Saved ResNet results to {results_file_resnet}")
    
    # Create heatmaps for ResNet
    plot_heatmaps(results_resnet, out_dir, suffix="_resnet")
    
    # Create d_f distribution plots for ResNet
    plot_d_f_distributions(results_resnet, out_dir, suffix="_resnet")
    
    print("\n" + "="*60)
    print("All Experiments Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

