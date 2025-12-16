"""
Plotting Utilities for Parameterization Experiments

Contains functions for visualizing:
- Training metrics (loss, M_g, C_def, H_Lambda)
- Learning rate per layer
- Gradient norms per layer
- Kernel metrics (rank, CKA, Wasserstein, eigenvalue spectra)
- Gradient eigenvalue evolution
- Comparison plots across parameterizations
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_metrics(history: Dict, parameterization: str, out_dir: str):
    """
    Plot metrics vs epochs.
    
    Creates a 2x2 grid with:
    - Train/Test Loss
    - M_g (Gate Mobility Number)
    - C_def (Path Deformation Capacity)
    - H_Lambda (Path Covariance Entropy)
    
    Args:
        history: Training history dictionary
        parameterization: Parameterization name (for labels)
        out_dir: Output directory for saving the plot
    """
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
    """
    Plot learning rate per layer vs epoch with color coding.
    
    Args:
        history: Training history dictionary
        parameterization: Parameterization name (for labels)
        out_dir: Output directory for saving the plot
    """
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
    if len(layer_indices) > 1:
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
            if key_range > 0:
                color_val = (layer_key - min_key) / key_range
            else:
                color_val = 0.5
        else:
            color_val = 1.0 if layer_key == 999999 else idx / max(len(layer_indices) - 1, 1)
        
        color = cmap(color_val)
        
        # Get the data
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
                if len(epochs) == len(lr_values):
                    valid_epochs.append(epochs[e])
                else:
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
    """
    Plot gradient norms per layer vs epoch with color coding.
    
    Args:
        history: Training history dictionary
        parameterization: Parameterization name (for labels)
        out_dir: Output directory for saving the plot
    """
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
        
        # Filter out None/NaN/Inf values
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


def plot_comparison(all_histories: Dict[str, Dict], out_dir: str):
    """
    Plot comparison across parameterizations.
    
    Creates a 2x2 grid comparing:
    - Train/Test Loss
    - M_g
    - C_def
    - H_Lambda
    
    Args:
        all_histories: Dict mapping parameterization name -> history dict
        out_dir: Output directory for saving the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'standard': 'C0', 'mup': 'C1', 'ntk': 'C2', 'mup_L': 'C3', 'path': 'C4'}
    
    # Train/Test Loss
    ax = axes[0, 0]
    for param, history in all_histories.items():
        epochs = history["epochs"]
        color = colors.get(param.split('_')[-1] if '_' in param else param, 'gray')
        ax.plot(epochs, history["train_loss"], marker='o', label=f'Train ({param})', 
                linewidth=2, color=color, linestyle='-')
        ax.plot(epochs, history["test_loss"], marker='s', label=f'Test ({param})', 
                linewidth=2, color=color, linestyle='--')
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
            color = colors.get(param.split('_')[-1] if '_' in param else param, 'gray')
            ax.plot(m_g_epochs, m_g_vals, marker='o', label=f'M_g ({param})', 
                    linewidth=2, color=color)
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
            color = colors.get(param.split('_')[-1] if '_' in param else param, 'gray')
            ax.plot(c_def_epochs, c_def_vals, marker='o', label=f'C_def ({param})', 
                    linewidth=2, color=color)
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
            color = colors.get(param.split('_')[-1] if '_' in param else param, 'gray')
            ax.plot(h_lambda_epochs, h_lambda_vals, marker='o', label=f'H_Λ ({param})', 
                    linewidth=2, color=color)
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
    """
    Plot gradient norms comparison across parameterizations (meaned per layer).
    
    Args:
        all_histories: Dict mapping parameterization name -> history dict
        out_dir: Output directory for saving the plot
    """
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


def plot_kernel_metrics(history: Dict, parameterization: str, out_dir: str):
    """
    Plot kernel metrics vs epochs.
    
    Creates a 2x3 grid with:
    - Path kernel effective rank
    - Path kernel CKA (vs initial)
    - Path kernel Wasserstein distance
    - Hidden kernel effective rank
    - Hidden kernel CKA (vs initial)
    - Hidden kernel Wasserstein distance
    
    Args:
        history: Training history dictionary
        parameterization: Parameterization name (for labels)
        out_dir: Output directory for saving the plot
    """
    # Check if kernel metrics are available
    if "path_kernel_rank" not in history or len(history.get("path_kernel_rank", [])) == 0:
        print(f"No kernel metrics available for {parameterization}, skipping kernel plots")
        return
    
    epochs = history["epochs"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Path kernel rank
    ax = axes[0, 0]
    vals = history.get("path_kernel_rank", [])
    valid_vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
    valid_epochs = [e for e, v in zip(epochs, vals) if not np.isnan(v) and not np.isinf(v)]
    if len(valid_vals) > 0:
        ax.plot(valid_epochs, valid_vals, marker='o', linewidth=2, color='C0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Effective Rank')
    ax.set_title(f'Path Kernel Rank ({parameterization})')
    ax.grid(True, alpha=0.3)
    
    # Path kernel CKA
    ax = axes[0, 1]
    vals = history.get("path_kernel_cka", [])
    valid_vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
    valid_epochs = [e for e, v in zip(epochs, vals) if not np.isnan(v) and not np.isinf(v)]
    if len(valid_vals) > 0:
        ax.plot(valid_epochs, valid_vals, marker='o', linewidth=2, color='C1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CKA (vs Initial)')
    ax.set_title(f'Path Kernel CKA ({parameterization})')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Path kernel Wasserstein
    ax = axes[0, 2]
    vals = history.get("path_kernel_wasserstein", [])
    valid_vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
    valid_epochs = [e for e, v in zip(epochs, vals) if not np.isnan(v) and not np.isinf(v)]
    if len(valid_vals) > 0:
        ax.plot(valid_epochs, valid_vals, marker='o', linewidth=2, color='C2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title(f'Path Kernel Eigenvalue Drift ({parameterization})')
    ax.grid(True, alpha=0.3)
    
    # Hidden kernel rank
    ax = axes[1, 0]
    vals = history.get("hidden_kernel_rank", [])
    valid_vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
    valid_epochs = [e for e, v in zip(epochs, vals) if not np.isnan(v) and not np.isinf(v)]
    if len(valid_vals) > 0:
        ax.plot(valid_epochs, valid_vals, marker='o', linewidth=2, color='C3')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Effective Rank')
    ax.set_title(f'Hidden Kernel Rank ({parameterization})')
    ax.grid(True, alpha=0.3)
    
    # Hidden kernel CKA
    ax = axes[1, 1]
    vals = history.get("hidden_kernel_cka", [])
    valid_vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
    valid_epochs = [e for e, v in zip(epochs, vals) if not np.isnan(v) and not np.isinf(v)]
    if len(valid_vals) > 0:
        ax.plot(valid_epochs, valid_vals, marker='o', linewidth=2, color='C4')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CKA (vs Initial)')
    ax.set_title(f'Hidden Kernel CKA ({parameterization})')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Hidden kernel Wasserstein
    ax = axes[1, 2]
    vals = history.get("hidden_kernel_wasserstein", [])
    valid_vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
    valid_epochs = [e for e, v in zip(epochs, vals) if not np.isnan(v) and not np.isinf(v)]
    if len(valid_vals) > 0:
        ax.plot(valid_epochs, valid_vals, marker='o', linewidth=2, color='C5')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title(f'Hidden Kernel Eigenvalue Drift ({parameterization})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'kernel_metrics_{parameterization}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved kernel metrics plot: {filename}")


def plot_eigenvalue_spectra(history: Dict, parameterization: str, out_dir: str):
    """
    Plot initial vs final eigenvalue spectra for path and hidden kernels.
    
    Args:
        history: Training history dictionary
        parameterization: Parameterization name (for labels)
        out_dir: Output directory for saving the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Path kernel spectrum
    ax = axes[0]
    initial_eigs = history.get("initial_path_kernel_eigs")
    final_eigs = history.get("final_path_kernel_eigs")
    
    if initial_eigs is not None and len(initial_eigs) > 0:
        ax.semilogy(range(len(initial_eigs)), initial_eigs, 'o-', label='Initial', linewidth=2, markersize=4)
    if final_eigs is not None and len(final_eigs) > 0:
        ax.semilogy(range(len(final_eigs)), final_eigs, 's-', label='Final', linewidth=2, markersize=4)
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title(f'Path Kernel Spectrum ({parameterization})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hidden kernel spectrum
    ax = axes[1]
    initial_eigs = history.get("initial_hidden_kernel_eigs")
    final_eigs = history.get("final_hidden_kernel_eigs")
    
    if initial_eigs is not None and len(initial_eigs) > 0:
        ax.semilogy(range(len(initial_eigs)), initial_eigs, 'o-', label='Initial', linewidth=2, markersize=4)
    if final_eigs is not None and len(final_eigs) > 0:
        ax.semilogy(range(len(final_eigs)), final_eigs, 's-', label='Final', linewidth=2, markersize=4)
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title(f'Hidden Kernel Spectrum ({parameterization})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'eigenvalue_spectra_{parameterization}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved eigenvalue spectra plot: {filename}")


def plot_gradient_eigenvalues(history: Dict, parameterization: str, out_dir: str):
    """
    Plot gradient eigenvalue statistics over training.
    
    Shows:
    - Mean and max of concatenated gradient eigenvalues vs epoch
    - Distribution of gradient eigenvalues (final epoch)
    
    Args:
        history: Training history dictionary
        parameterization: Parameterization name (for labels)
        out_dir: Output directory for saving the plot
    """
    grad_eigs_list = history.get("gradient_eigenvalues_concat", [])
    
    if not grad_eigs_list or all(len(eigs) == 0 for eigs in grad_eigs_list):
        print(f"No gradient eigenvalue data for {parameterization}, skipping plot")
        return
    
    epochs = history["epochs"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Compute statistics per epoch
    mean_eigs = []
    max_eigs = []
    sum_eigs = []
    valid_epochs = []
    
    for i, eigs in enumerate(grad_eigs_list):
        if eigs is not None and len(eigs) > 0:
            eigs_arr = np.array(eigs)
            eigs_arr = eigs_arr[~np.isnan(eigs_arr)]
            if len(eigs_arr) > 0:
                mean_eigs.append(np.mean(eigs_arr))
                max_eigs.append(np.max(eigs_arr))
                sum_eigs.append(np.sum(eigs_arr))
                valid_epochs.append(epochs[i] if i < len(epochs) else i)
    
    # Mean gradient EV
    ax = axes[0]
    if len(mean_eigs) > 0:
        ax.semilogy(valid_epochs, mean_eigs, 'o-', linewidth=2, color='C0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Gradient Eigenvalue')
    ax.set_title(f'Mean Gradient EV ({parameterization})')
    ax.grid(True, alpha=0.3)
    
    # Max gradient EV
    ax = axes[1]
    if len(max_eigs) > 0:
        ax.semilogy(valid_epochs, max_eigs, 'o-', linewidth=2, color='C1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Max Gradient Eigenvalue')
    ax.set_title(f'Max Gradient EV ({parameterization})')
    ax.grid(True, alpha=0.3)
    
    # Distribution at final epoch
    ax = axes[2]
    # Find last non-empty entry
    final_eigs = None
    for eigs in reversed(grad_eigs_list):
        if eigs is not None and len(eigs) > 0:
            final_eigs = np.array(eigs)
            final_eigs = final_eigs[~np.isnan(final_eigs)]
            break
    
    if final_eigs is not None and len(final_eigs) > 0:
        # Plot log histogram
        log_eigs = np.log10(final_eigs + 1e-12)
        ax.hist(log_eigs, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('log10(Eigenvalue)')
        ax.set_ylabel('Count')
    ax.set_title(f'Gradient EV Distribution (Final, {parameterization})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'gradient_eigenvalues_{parameterization}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved gradient eigenvalues plot: {filename}")


def plot_kernel_metrics_comparison(all_histories: Dict[str, Dict], out_dir: str):
    """
    Plot kernel metrics comparison across parameterizations.
    
    Creates a 2x3 grid comparing all parameterizations:
    - Path kernel rank
    - Path kernel CKA
    - Path kernel Wasserstein
    - Hidden kernel rank
    - Hidden kernel CKA
    - Hidden kernel Wasserstein
    
    Args:
        all_histories: Dict mapping parameterization name -> history dict
        out_dir: Output directory for saving the plot
    """
    # Check if any history has kernel metrics
    has_kernel_metrics = any(
        "path_kernel_rank" in h and len(h.get("path_kernel_rank", [])) > 0
        for h in all_histories.values()
    )
    
    if not has_kernel_metrics:
        print("No kernel metrics available for comparison")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors_map = {'standard': 'C0', 'mup': 'C1', 'ntk': 'C2', 'mup_L': 'C3', 'path': 'C4', 
                  'batchnorm': 'C5', 'adam': 'C6', 'muon': 'C7'}
    
    metrics_config = [
        ("path_kernel_rank", "Path Kernel Rank", axes[0, 0], False),
        ("path_kernel_cka", "Path Kernel CKA", axes[0, 1], True),
        ("path_kernel_wasserstein", "Path Kernel Wasserstein", axes[0, 2], False),
        ("hidden_kernel_rank", "Hidden Kernel Rank", axes[1, 0], False),
        ("hidden_kernel_cka", "Hidden Kernel CKA", axes[1, 1], True),
        ("hidden_kernel_wasserstein", "Hidden Kernel Wasserstein", axes[1, 2], False),
    ]
    
    for metric_key, title, ax, is_cka in metrics_config:
        for param, history in all_histories.items():
            if metric_key not in history:
                continue
            
            epochs = history["epochs"]
            vals = history[metric_key]
            valid_vals = [v for v in vals if v is not None and not np.isnan(v) and not np.isinf(v)]
            valid_epochs = [e for e, v in zip(epochs, vals) if v is not None and not np.isnan(v) and not np.isinf(v)]
            
            if len(valid_vals) > 0:
                # Get color based on last part of param name
                param_key = param.split('_')[-1] if '_' in param else param
                color = colors_map.get(param_key, 'gray')
                ax.plot(valid_epochs, valid_vals, marker='o', label=param, linewidth=2, color=color, markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title.split()[-1])  # Last word of title
        ax.set_title(title)
        if is_cka:
            ax.set_ylim([0, 1.05])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'kernel_metrics_comparison.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved kernel metrics comparison plot: {filename}")


def plot_all_kernel_metrics(history: Dict, parameterization: str, out_dir: str):
    """
    Plot all kernel-related metrics for a single training run.
    
    Calls:
    - plot_kernel_metrics
    - plot_eigenvalue_spectra
    - plot_gradient_eigenvalues
    
    Args:
        history: Training history dictionary
        parameterization: Parameterization name (for labels)
        out_dir: Output directory for saving the plot
    """
    plot_kernel_metrics(history, parameterization, out_dir)
    plot_eigenvalue_spectra(history, parameterization, out_dir)
    plot_gradient_eigenvalues(history, parameterization, out_dir)

