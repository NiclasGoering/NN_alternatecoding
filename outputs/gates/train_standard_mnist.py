"""
Train standard MLP on MNIST with different depths and widths.
Optimized for H100 GPUs with multi-GPU support.
Computes train and test error and creates heatmaps.
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
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional
from queue import Queue, Empty
from threading import Thread

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.data.models.ffnn import MLP
from src.data.mnist import build_mnist_datasets
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.save_io import ensure_dir, save_json


@torch.no_grad()
def evaluate_error(model, loader, device, n_classes: int, alpha: float) -> Tuple[float, float]:
    """
    Evaluate classification error (0-1 loss) and MSE loss on a dataset.
    
    Returns:
        (error_rate, mse_loss)
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_mse = 0.0
    
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        yhat = model(xb)
        
        if n_classes == 1:
            # Binary classification
            predictions = torch.sign(yhat).squeeze()
            targets = yb.squeeze()
            correct = (predictions == targets).sum().item()
            mse = torch.mean((yhat - yb) ** 2).item()
        else:
            # Multi-class classification
            if yb.dim() > 1:
                yb = yb.view(-1)
            yb_class = (yb / alpha).long()
            yb_class = torch.clamp(yb_class, 0, n_classes - 1)
            
            # Predictions: argmax of output
            predictions = torch.argmax(yhat, dim=1)
            correct = (predictions == yb_class).sum().item()
            
            # MSE loss: convert to one-hot
            yb_onehot = torch.zeros_like(yhat)
            src_values = (torch.ones_like(yb.unsqueeze(1)) * alpha).to(dtype=yhat.dtype)
            yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values)
            mse = torch.mean((yhat - yb_onehot) ** 2).item()
        
        total_correct += correct
        total_samples += xb.size(0)
        total_mse += mse * xb.size(0)
    
    error_rate = 1.0 - (total_correct / total_samples) if total_samples > 0 else 1.0
    mse_loss = total_mse / total_samples if total_samples > 0 else 0.0
    
    return error_rate, mse_loss


@torch.no_grad()
def compute_gating_diversity(model, loader, device) -> Dict[str, float]:
    """
    Compute gating diversity metric D_l for each layer.
    
    For each layer l:
    - p_l = E_b,i[1[u_l,i(x_b) > 0]] (gate ON probability)
    - D_l = 4 * p_l * (1 - p_l) (diversity score, max=1 when p_l=0.5)
    
    Returns:
        Dictionary with:
        - D_l: List of diversity scores per layer
        - D_min: Minimum diversity across layers
        - D_mean: Mean diversity across layers
    """
    model.eval()
    all_gate_states = []  # List of lists: one per layer
    
    # Collect gate states from a batch
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        
        # Forward pass and collect pre-activations
        h = xb
        layer_gate_states = []
        
        for l in range(model.depth):
            u = model.linears[l](h)  # Pre-activation
            gate_states = (u > 0).float()  # 1 if ON, 0 if OFF
            layer_gate_states.append(gate_states)
            h = model.activation(u)
        
        all_gate_states.append(layer_gate_states)
        break  # Only need one batch
    
    if not all_gate_states:
        return {"D_l": [], "D_min": 0.0, "D_mean": 0.0}
    
    # Compute p_l and D_l for each layer
    D_l = []
    for l in range(model.depth):
        # Concatenate gate states across batch
        gate_states = torch.cat([batch[l] for batch in all_gate_states], dim=0)
        # p_l = mean gate ON probability
        p_l = gate_states.mean().item()
        # D_l = 4 * p_l * (1 - p_l)
        D_l_value = 4.0 * p_l * (1.0 - p_l)
        D_l.append(D_l_value)
    
    D_min = min(D_l) if D_l else 0.0
    D_mean = np.mean(D_l) if D_l else 0.0
    
    return {
        "D_l": D_l,
        "D_min": D_min,
        "D_mean": D_mean
    }


def compute_gate_flip_fraction(
    model,
    loader,
    device,
    config: Dict,
    n_classes: int,
    alpha: float
) -> Dict[str, float]:
    """
    Compute gate flip fraction p_flip after one SGD step.
    
    For each layer l:
    1. Forward pass on batch, store pre-activations u_l(x)
    2. Backward pass, compute gradients, do one SGD step (on a copy)
    3. Forward pass again, get new pre-activations u_l'(x)
    4. Compute fraction where sign changed
    
    Returns:
        Dictionary with:
        - p_flip_l: List of flip fractions per layer
        - p_flip_min: Minimum flip fraction across layers
        - p_flip_mean: Mean flip fraction across layers
    """
    # Get a batch
    xb, yb = next(iter(loader))
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    
    # Store original pre-activations (use eval mode)
    model.eval()
    h = xb
    u_original = []
    with torch.no_grad():
        for l in range(model.depth):
            u = model.linears[l](h)
            u_original.append(u.detach().clone())
            h = model.activation(u)
    
    # Create a copy of the model for the update
    # Get input dimension from first layer
    d_in = model.linears[0].weight.shape[1]
    widths_list = [model.linears[l].weight.shape[0] for l in range(model.depth)]
    has_bias = model.linears[0].bias is not None
    
    model_copy = type(model)(
        d_in=d_in,
        widths=widths_list,
        bias=has_bias,
        activation=model.activation_name,
        n_classes=model.n_classes
    ).to(device)
    
    # Copy weights
    for l in range(model.depth):
        model_copy.linears[l].weight.data = model.linears[l].weight.data.clone()
        if model_copy.linears[l].bias is not None:
            model_copy.linears[l].bias.data = model.linears[l].bias.data.clone()
    model_copy.readout.weight.data = model.readout.weight.data.clone()
    if model_copy.readout.bias is not None:
        model_copy.readout.bias.data = model.readout.bias.data.clone()
    
    # Setup optimizer for one step
    lr = float(config["training"]["lr_w"])
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr, momentum=0.0)  # No momentum for clean step
    loss_fn = nn.MSELoss()
    
    # Forward pass on copy (in train mode for gradients)
    model_copy.train()
    yhat = model_copy(xb)
    
    # Compute loss
    if n_classes == 1:
        loss = loss_fn(yhat, yb)
    else:
        if yb.dim() > 1:
            yb = yb.view(-1)
        yb_class = (yb / alpha).long()
        yb_class = torch.clamp(yb_class, 0, n_classes - 1)
        yb_onehot = torch.zeros_like(yhat)
        src_values = (torch.ones_like(yb.unsqueeze(1)) * alpha).to(dtype=yhat.dtype)
        yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values)
        loss = loss_fn(yhat, yb_onehot)
    
    # Backward and step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Forward pass again to get new pre-activations
    model_copy.eval()
    h_new = xb
    u_new = []
    with torch.no_grad():
        for l in range(model_copy.depth):
            u = model_copy.linears[l](h_new)
            u_new.append(u)
            h_new = model_copy.activation(u)
    
    # Compute flip fractions
    p_flip_l = []
    for l in range(model.depth):
        u_orig = u_original[l]
        u_n = u_new[l]
        
        # Compute sign changes
        sign_orig = (u_orig > 0).float()
        sign_new = (u_n > 0).float()
        flips = (sign_orig != sign_new).float()
        
        # Fraction of neuron-sample pairs that flipped
        p_flip = flips.mean().item()
        p_flip_l.append(p_flip)
    
    p_flip_min = min(p_flip_l) if p_flip_l else 0.0
    p_flip_mean = np.mean(p_flip_l) if p_flip_l else 0.0
    
    # Clean up
    del model_copy
    torch.cuda.empty_cache()
    
    return {
        "p_flip_l": p_flip_l,
        "p_flip_min": p_flip_min,
        "p_flip_mean": p_flip_mean
    }


def train_model(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict,
    device: torch.device,
    depth: int,
    width: int
) -> Dict:
    """
    Train model and return final train/test errors.
    
    Returns:
        Dictionary with:
        - train_error: Final training error rate
        - test_error: Final test error rate
        - train_loss: Final training MSE loss
        - test_loss: Final test MSE loss
        - depth: Model depth
        - width: Model width
    """
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["lr_w"])
    optimizer_type = config["training"].get("optimizer", "sgd").lower()
    if isinstance(optimizer_type, list):
        optimizer_type = optimizer_type[0]
    grad_clip = float(config["training"].get("grad_clip_max_norm", 0.0))
    
    # Get n_classes and alpha
    n_classes = model.n_classes
    alpha = float(config.get("dataset", {}).get("alpha", 1.0))
    if isinstance(alpha, list):
        alpha = alpha[0]
    
    # Setup optimizer
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    loss_fn = nn.MSELoss()
    
    # Get device ID for logging
    device_id = device.index if hasattr(device, 'index') and device.index is not None else "?"
    device_tag = f"[GPU {device_id}]" if torch.cuda.is_available() else "[CPU]"
    
    # Logging frequency (every 10% of epochs or at least every 100 epochs)
    log_every = max(100, epochs // 10)
    
    print(f"{device_tag} Starting training: Depth={depth}, Width={width}, Epochs={epochs}, LR={lr}")
    
    # Compute metrics at initialization
    print(f"{device_tag} Computing initialization metrics...")
    p_flip_init = compute_gate_flip_fraction(model, train_loader, device, config, n_classes, alpha)
    D_init = compute_gating_diversity(model, train_loader, device)
    print(f"{device_tag} Init - p_flip_min: {p_flip_init['p_flip_min']:.6f}, D_min: {D_init['D_min']:.4f}")
    
    # Training loop with mixed precision for H100 optimization
    model.train()
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            
            # Use mixed precision for forward pass
            with torch.cuda.amp.autocast():
                yhat = model(xb)
                
                if n_classes == 1:
                    loss = loss_fn(yhat, yb)
                else:
                    if yb.dim() > 1:
                        yb = yb.view(-1)
                    yb_class = (yb / alpha).long()
                    yb_class = torch.clamp(yb_class, 0, n_classes - 1)
                    yb_onehot = torch.zeros_like(yhat)
                    src_values = (torch.ones_like(yb.unsqueeze(1)) * alpha).to(dtype=yhat.dtype)
                    yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values)
                    loss = loss_fn(yhat, yb_onehot)
            
            epoch_loss += loss.item()
            n_batches += 1
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        
        # Periodic logging
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            train_error, train_loss_eval = evaluate_error(model, train_loader, device, n_classes, alpha)
            test_error, test_loss_eval = evaluate_error(model, test_loader, device, n_classes, alpha)
            
            print(f"{device_tag} Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_loss:.6f} | Train Error: {train_error:.4f} | "
                  f"Test Error: {test_error:.4f} | "
                  f"Depth={depth}, Width={width}")
    
    # Evaluate final errors
    train_error, train_loss = evaluate_error(model, train_loader, device, n_classes, alpha)
    test_error, test_loss = evaluate_error(model, test_loader, device, n_classes, alpha)
    
    # Compute metrics after training
    print(f"{device_tag} Computing final metrics...")
    p_flip_final = compute_gate_flip_fraction(model, train_loader, device, config, n_classes, alpha)
    D_final = compute_gating_diversity(model, train_loader, device)
    print(f"{device_tag} Final - p_flip_min: {p_flip_final['p_flip_min']:.6f}, D_min: {D_final['D_min']:.4f}")
    
    print(f"{device_tag} ✓ COMPLETED Depth={depth}, Width={width} | "
          f"Final Train Error: {train_error:.4f} | Final Test Error: {test_error:.4f}")
    
    return {
        "train_error": float(train_error),
        "test_error": float(test_error),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "depth": depth,
        "width": width,
        # Initialization metrics
        "p_flip_min_init": float(p_flip_init["p_flip_min"]),
        "p_flip_mean_init": float(p_flip_init["p_flip_mean"]),
        "D_min_init": float(D_init["D_min"]),
        "D_mean_init": float(D_init["D_mean"]),
        # Final metrics
        "p_flip_min_final": float(p_flip_final["p_flip_min"]),
        "p_flip_mean_final": float(p_flip_final["p_flip_mean"]),
        "D_min_final": float(D_final["D_min"]),
        "D_mean_final": float(D_final["D_mean"]),
    }


def _worker_thread(
    job_queue, result_queue, gpu_id,
    input_dim, n_classes, cfg, depths, widths, out_dir
):
    """Worker thread for parallel GPU training."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    # Build datasets once per worker
    Xtr, ytr, Xva, yva, Xte, yte, meta = build_mnist_datasets(cfg)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    test_dataset = TensorDataset(torch.tensor(Xte), torch.tensor(yte))
    
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = 8  # Optimize for H100 (more workers for data loading)
    pin_memory = True
    prefetch_factor = 2  # Prefetch batches for faster loading
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    
    while True:
        try:
            job = job_queue.get(timeout=1)
            if job is None:  # Poison pill
                break
            
            depth, width = job
            start_time = time.time()
            
            device_id = device.index if hasattr(device, 'index') and device.index is not None else "?"
            device_tag = f"[GPU {device_id}]"
            
            print(f"{device_tag} {'='*70}")
            print(f"{device_tag} Starting job: Depth={depth}, Width={width}")
            print(f"{device_tag} {'='*70}")
            
            # Create model
            widths_list = [width] * depth
            model = MLP(
                d_in=input_dim,
                widths=widths_list,
                bias=cfg["model"].get("bias", True),
                activation=cfg["model"].get("activation", "relu"),
                n_classes=n_classes
            ).to(device)
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            print(f"{device_tag} Model created: {n_params:,} parameters")
            
            # Standard initialization
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            # Train
            result = train_model(model, train_loader, test_loader, cfg, device, depth, width)
            result["training_time"] = time.time() - start_time
            result["n_params"] = n_params
            
            print(f"{device_tag} Training completed in {result['training_time']:.2f}s")
            print(f"{device_tag} {'='*70}\n")
            
            result_queue.put(("success", result, None))
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Empty:
            continue
        except Exception as e:
            result_queue.put(("error", None, str(e)))
            import traceback
            traceback.print_exc()


def create_heatmaps(results: List[Dict], out_dir: str):
    """Create normal and log-scale heatmaps for train and test errors."""
    # Extract unique depths and widths
    depths = sorted(set(r["depth"] for r in results))
    widths = sorted(set(r["width"] for r in results))
    
    # Create matrices
    train_error_matrix = np.full((len(depths), len(widths)), np.nan)
    test_error_matrix = np.full((len(depths), len(widths)), np.nan)
    
    for r in results:
        d_idx = depths.index(r["depth"])
        w_idx = widths.index(r["width"])
        train_error_matrix[d_idx, w_idx] = r["train_error"]
        test_error_matrix[d_idx, w_idx] = r["test_error"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Normal scale heatmaps
    im1 = axes[0, 0].imshow(train_error_matrix, aspect='auto', cmap='viridis_r', origin='lower')
    axes[0, 0].set_title('Train Error (Normal Scale)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Width', fontsize=12)
    axes[0, 0].set_ylabel('Depth', fontsize=12)
    axes[0, 0].set_xticks(range(len(widths)))
    axes[0, 0].set_xticklabels(widths)
    axes[0, 0].set_yticks(range(len(depths)))
    axes[0, 0].set_yticklabels(depths)
    plt.colorbar(im1, ax=axes[0, 0], label='Error Rate')
    
    im2 = axes[0, 1].imshow(test_error_matrix, aspect='auto', cmap='viridis_r', origin='lower')
    axes[0, 1].set_title('Test Error (Normal Scale)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Width', fontsize=12)
    axes[0, 1].set_ylabel('Depth', fontsize=12)
    axes[0, 1].set_xticks(range(len(widths)))
    axes[0, 1].set_xticklabels(widths)
    axes[0, 1].set_yticks(range(len(depths)))
    axes[0, 1].set_yticklabels(depths)
    plt.colorbar(im2, ax=axes[0, 1], label='Error Rate')
    
    # Log scale heatmaps
    train_error_log = np.log10(train_error_matrix + 1e-8)  # Add small epsilon to avoid log(0)
    test_error_log = np.log10(test_error_matrix + 1e-8)
    
    im3 = axes[1, 0].imshow(train_error_log, aspect='auto', cmap='viridis_r', origin='lower')
    axes[1, 0].set_title('Train Error (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Width', fontsize=12)
    axes[1, 0].set_ylabel('Depth', fontsize=12)
    axes[1, 0].set_xticks(range(len(widths)))
    axes[1, 0].set_xticklabels(widths)
    axes[1, 0].set_yticks(range(len(depths)))
    axes[1, 0].set_yticklabels(depths)
    plt.colorbar(im3, ax=axes[1, 0], label='log10(Error Rate)')
    
    im4 = axes[1, 1].imshow(test_error_log, aspect='auto', cmap='viridis_r', origin='lower')
    axes[1, 1].set_title('Test Error (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Width', fontsize=12)
    axes[1, 1].set_ylabel('Depth', fontsize=12)
    axes[1, 1].set_xticks(range(len(widths)))
    axes[1, 1].set_xticklabels(widths)
    axes[1, 1].set_yticks(range(len(depths)))
    axes[1, 1].set_yticklabels(depths)
    plt.colorbar(im4, ax=axes[1, 1], label='log10(Error Rate)')
    
    plt.tight_layout()
    
    # Save normal scale heatmap
    fig_normal, axes_normal = plt.subplots(1, 2, figsize=(16, 6))
    im1_n = axes_normal[0].imshow(train_error_matrix, aspect='auto', cmap='viridis_r', origin='lower')
    axes_normal[0].set_title('Train Error (Normal Scale)', fontsize=14, fontweight='bold')
    axes_normal[0].set_xlabel('Width', fontsize=12)
    axes_normal[0].set_ylabel('Depth', fontsize=12)
    axes_normal[0].set_xticks(range(len(widths)))
    axes_normal[0].set_xticklabels(widths)
    axes_normal[0].set_yticks(range(len(depths)))
    axes_normal[0].set_yticklabels(depths)
    plt.colorbar(im1_n, ax=axes_normal[0], label='Error Rate')
    
    im2_n = axes_normal[1].imshow(test_error_matrix, aspect='auto', cmap='viridis_r', origin='lower')
    axes_normal[1].set_title('Test Error (Normal Scale)', fontsize=14, fontweight='bold')
    axes_normal[1].set_xlabel('Width', fontsize=12)
    axes_normal[1].set_ylabel('Depth', fontsize=12)
    axes_normal[1].set_xticks(range(len(widths)))
    axes_normal[1].set_xticklabels(widths)
    axes_normal[1].set_yticks(range(len(depths)))
    axes_normal[1].set_yticklabels(depths)
    plt.colorbar(im2_n, ax=axes_normal[1], label='Error Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_heatmap_normal.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_normal)
    
    # Save log scale heatmap
    fig_log, axes_log = plt.subplots(1, 2, figsize=(16, 6))
    im3_l = axes_log[0].imshow(train_error_log, aspect='auto', cmap='viridis_r', origin='lower')
    axes_log[0].set_title('Train Error (Log Scale)', fontsize=14, fontweight='bold')
    axes_log[0].set_xlabel('Width', fontsize=12)
    axes_log[0].set_ylabel('Depth', fontsize=12)
    axes_log[0].set_xticks(range(len(widths)))
    axes_log[0].set_xticklabels(widths)
    axes_log[0].set_yticks(range(len(depths)))
    axes_log[0].set_yticklabels(depths)
    plt.colorbar(im3_l, ax=axes_log[0], label='log10(Error Rate)')
    
    im4_l = axes_log[1].imshow(test_error_log, aspect='auto', cmap='viridis_r', origin='lower')
    axes_log[1].set_title('Test Error (Log Scale)', fontsize=14, fontweight='bold')
    axes_log[1].set_xlabel('Width', fontsize=12)
    axes_log[1].set_ylabel('Depth', fontsize=12)
    axes_log[1].set_xticks(range(len(widths)))
    axes_log[1].set_xticklabels(widths)
    axes_log[1].set_yticks(range(len(depths)))
    axes_log[1].set_yticklabels(depths)
    plt.colorbar(im4_l, ax=axes_log[1], label='log10(Error Rate)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_heatmap_log.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_log)
    
    plt.close(fig)
    
    # Create heatmaps for p_flip metrics
    if any("p_flip_min_init" in r for r in results):
        # p_flip_init heatmaps
        p_flip_init_matrix = np.full((len(depths), len(widths)), np.nan)
        p_flip_final_matrix = np.full((len(depths), len(widths)), np.nan)
        
        for r in results:
            if "p_flip_min_init" in r:
                d_idx = depths.index(r["depth"])
                w_idx = widths.index(r["width"])
                p_flip_init_matrix[d_idx, w_idx] = r["p_flip_min_init"]
                p_flip_final_matrix[d_idx, w_idx] = r["p_flip_min_final"]
        
        # Normal scale
        fig_pflip, axes_pflip = plt.subplots(1, 2, figsize=(16, 6))
        im1_p = axes_pflip[0].imshow(p_flip_init_matrix, aspect='auto', cmap='viridis', origin='lower')
        axes_pflip[0].set_title('p_flip_min (Init)', fontsize=14, fontweight='bold')
        axes_pflip[0].set_xlabel('Width', fontsize=12)
        axes_pflip[0].set_ylabel('Depth', fontsize=12)
        axes_pflip[0].set_xticks(range(len(widths)))
        axes_pflip[0].set_xticklabels(widths)
        axes_pflip[0].set_yticks(range(len(depths)))
        axes_pflip[0].set_yticklabels(depths)
        plt.colorbar(im1_p, ax=axes_pflip[0], label='Gate Flip Fraction')
        
        im2_p = axes_pflip[1].imshow(p_flip_final_matrix, aspect='auto', cmap='viridis', origin='lower')
        axes_pflip[1].set_title('p_flip_min (Final)', fontsize=14, fontweight='bold')
        axes_pflip[1].set_xlabel('Width', fontsize=12)
        axes_pflip[1].set_ylabel('Depth', fontsize=12)
        axes_pflip[1].set_xticks(range(len(widths)))
        axes_pflip[1].set_xticklabels(widths)
        axes_pflip[1].set_yticks(range(len(depths)))
        axes_pflip[1].set_yticklabels(depths)
        plt.colorbar(im2_p, ax=axes_pflip[1], label='Gate Flip Fraction')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "p_flip_heatmap_normal.png"), dpi=150, bbox_inches='tight')
        plt.close(fig_pflip)
        
        # Log scale
        p_flip_init_log = np.log10(p_flip_init_matrix + 1e-8)
        p_flip_final_log = np.log10(p_flip_final_matrix + 1e-8)
        
        fig_pflip_log, axes_pflip_log = plt.subplots(1, 2, figsize=(16, 6))
        im1_pl = axes_pflip_log[0].imshow(p_flip_init_log, aspect='auto', cmap='viridis', origin='lower')
        axes_pflip_log[0].set_title('p_flip_min (Init, Log Scale)', fontsize=14, fontweight='bold')
        axes_pflip_log[0].set_xlabel('Width', fontsize=12)
        axes_pflip_log[0].set_ylabel('Depth', fontsize=12)
        axes_pflip_log[0].set_xticks(range(len(widths)))
        axes_pflip_log[0].set_xticklabels(widths)
        axes_pflip_log[0].set_yticks(range(len(depths)))
        axes_pflip_log[0].set_yticklabels(depths)
        plt.colorbar(im1_pl, ax=axes_pflip_log[0], label='log10(Gate Flip Fraction)')
        
        im2_pl = axes_pflip_log[1].imshow(p_flip_final_log, aspect='auto', cmap='viridis', origin='lower')
        axes_pflip_log[1].set_title('p_flip_min (Final, Log Scale)', fontsize=14, fontweight='bold')
        axes_pflip_log[1].set_xlabel('Width', fontsize=12)
        axes_pflip_log[1].set_ylabel('Depth', fontsize=12)
        axes_pflip_log[1].set_xticks(range(len(widths)))
        axes_pflip_log[1].set_xticklabels(widths)
        axes_pflip_log[1].set_yticks(range(len(depths)))
        axes_pflip_log[1].set_yticklabels(depths)
        plt.colorbar(im2_pl, ax=axes_pflip_log[1], label='log10(Gate Flip Fraction)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "p_flip_heatmap_log.png"), dpi=150, bbox_inches='tight')
        plt.close(fig_pflip_log)
    
    # Create heatmaps for D_min metrics
    if any("D_min_init" in r for r in results):
        D_init_matrix = np.full((len(depths), len(widths)), np.nan)
        D_final_matrix = np.full((len(depths), len(widths)), np.nan)
        
        for r in results:
            if "D_min_init" in r:
                d_idx = depths.index(r["depth"])
                w_idx = widths.index(r["width"])
                D_init_matrix[d_idx, w_idx] = r["D_min_init"]
                D_final_matrix[d_idx, w_idx] = r["D_min_final"]
        
        # Normal scale
        fig_D, axes_D = plt.subplots(1, 2, figsize=(16, 6))
        im1_D = axes_D[0].imshow(D_init_matrix, aspect='auto', cmap='viridis', origin='lower')
        axes_D[0].set_title('D_min (Init)', fontsize=14, fontweight='bold')
        axes_D[0].set_xlabel('Width', fontsize=12)
        axes_D[0].set_ylabel('Depth', fontsize=12)
        axes_D[0].set_xticks(range(len(widths)))
        axes_D[0].set_xticklabels(widths)
        axes_D[0].set_yticks(range(len(depths)))
        axes_D[0].set_yticklabels(depths)
        plt.colorbar(im1_D, ax=axes_D[0], label='Gating Diversity')
        
        im2_D = axes_D[1].imshow(D_final_matrix, aspect='auto', cmap='viridis', origin='lower')
        axes_D[1].set_title('D_min (Final)', fontsize=14, fontweight='bold')
        axes_D[1].set_xlabel('Width', fontsize=12)
        axes_D[1].set_ylabel('Depth', fontsize=12)
        axes_D[1].set_xticks(range(len(widths)))
        axes_D[1].set_xticklabels(widths)
        axes_D[1].set_yticks(range(len(depths)))
        axes_D[1].set_yticklabels(depths)
        plt.colorbar(im2_D, ax=axes_D[1], label='Gating Diversity')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "D_min_heatmap_normal.png"), dpi=150, bbox_inches='tight')
        plt.close(fig_D)
    
    print(f"Saved heatmaps to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_mnist_standard.yaml",
                    help="Path to config file")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory (default: based on experiment_name)")
    args = ap.parse_args()
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    cfg = load_config(config_path)
    
    # Set seed
    set_seed(cfg.get("seed", 123))
    
    # Setup output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        exp_name = cfg.get("experiment_name", "standard_mnist")
        out_dir = os.path.join(os.path.dirname(__file__), "results", exp_name)
    ensure_dir(out_dir)
    
    # Save config
    save_json(cfg, os.path.join(out_dir, "config.json"))
    
    # Get model config
    model_cfg = cfg["model"]
    depths = model_cfg.get("depths", [2, 4, 8, 16, 32, 64])
    widths = model_cfg.get("widths", [64, 128, 256, 512])
    
    # Build datasets to get input dimension and n_classes
    Xtr, ytr, Xva, yva, Xte, yte, meta = build_mnist_datasets(cfg)
    input_dim = Xtr.shape[1]
    n_classes = meta["n_classes"]
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {n_classes}")
    print(f"Depths to test: {depths}")
    print(f"Widths to test: {widths}")
    print(f"Total combinations: {len(depths) * len(widths)}")
    
    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    
    start_time_total = time.time()  # Track total experiment time
    
    if n_gpus == 0:
        print("WARNING: No GPUs available, using CPU (will be very slow)")
        device = torch.device("cpu")
        # Run sequentially on CPU
        results = []
        train_dataset = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
        test_dataset = TensorDataset(torch.tensor(Xte), torch.tensor(yte))
        batch_size = int(cfg["training"]["batch_size"])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        n_jobs = len(depths) * len(widths)
        print(f"\n{'='*70}")
        print(f"Starting training on CPU...")
        print(f"Total jobs: {n_jobs} (Depths: {len(depths)}, Widths: {len(widths)})")
        print(f"{'='*70}\n")
        
        completed = 0
        for depth in depths:
            for width in widths:
                completed += 1
                print(f"\n{'='*70}")
                print(f"Job {completed}/{n_jobs}: Depth={depth}, Width={width}")
                print(f"{'='*70}")
                
                widths_list = [width] * depth
                model = MLP(
                    d_in=input_dim,
                    widths=widths_list,
                    bias=model_cfg.get("bias", True),
                    activation=model_cfg.get("activation", "relu"),
                    n_classes=n_classes
                ).to(device)
                
                # Standard initialization
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                
                result = train_model(model, train_loader, test_loader, cfg, device, depth, width)
                results.append(result)
                
                # Save results incrementally
                results_file = os.path.join(out_dir, "results.json")
                save_json(results, results_file)
                
                elapsed = time.time() - start_time_total
                avg_time = elapsed / completed
                remaining = n_jobs - completed
                eta = (avg_time * remaining) / 3600
                print(f"Progress: {completed}/{n_jobs} ({100*completed/n_jobs:.1f}%) | "
                      f"ETA: {eta:.2f}h | ✓ Results saved\n")
                
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # Use multi-GPU parallel training
        n_workers = min(n_gpus, 4)  # Use up to 4 GPUs
        print(f"Using {n_workers} GPUs for parallel training")
        
        # Create job queue
        job_queue = Queue()
        for depth in depths:
            for width in widths:
                job_queue.put((depth, width))
        
        # Add poison pills
        for _ in range(n_workers):
            job_queue.put(None)
        
        # Result queue
        result_queue = Queue()
        
        # Start worker threads
        workers = []
        for gpu_id in range(n_workers):
            worker = Thread(
                target=_worker_thread,
                args=(job_queue, result_queue, gpu_id, input_dim, n_classes, cfg, depths, widths, out_dir)
            )
            worker.start()
            workers.append(worker)
        
        # Collect results
        results = []
        n_jobs = len(depths) * len(widths)
        completed = 0
        errors = []
        
        print(f"\n{'='*70}")
        print(f"Starting training on {n_workers} GPUs...")
        print(f"Total jobs: {n_jobs} (Depths: {len(depths)}, Widths: {len(widths)})")
        print(f"{'='*70}\n")
        
        results_file = os.path.join(out_dir, "results.json")
        
        while completed < n_jobs:
            try:
                status, result, error = result_queue.get(timeout=600)  # 10 min timeout per job
                if status == "success":
                    results.append(result)
                    completed += 1
                    elapsed = time.time() - start_time_total
                    avg_time_per_job = elapsed / completed if completed > 0 else 0
                    remaining_jobs = n_jobs - completed
                    eta_seconds = avg_time_per_job * remaining_jobs
                    eta_hours = eta_seconds / 3600
                    
                    # Save results incrementally after each completion
                    save_json(results, results_file)
                    
                    print(f"\n{'='*70}")
                    print(f"Progress: {completed}/{n_jobs} completed ({100*completed/n_jobs:.1f}%)")
                    print(f"Elapsed: {elapsed/3600:.2f}h | Avg time/job: {avg_time_per_job/60:.1f}min")
                    print(f"ETA: {eta_hours:.2f}h ({eta_seconds/60:.1f}min)")
                    print(f"Latest result: Depth={result['depth']}, Width={result['width']}")
                    print(f"  Train Error: {result['train_error']:.4f}, Test Error: {result['test_error']:.4f}")
                    print(f"✓ Results saved to {results_file}")
                    print(f"{'='*70}\n")
                elif status == "error":
                    errors.append(error)
                    completed += 1
                    print(f"\n❌ ERROR ({completed}/{n_jobs}): {error}\n")
            except Empty:
                print("⚠️  Warning: Timeout waiting for result (job may still be running)")
                # Don't break, continue waiting
                continue
        
        # Wait for workers to finish
        for worker in workers:
            worker.join()
        
        if errors:
            print(f"\n{len(errors)} errors occurred during training")
    
    # Save results to JSON (final save, in case any were missed)
    results_file = os.path.join(out_dir, "results.json")
    save_json(results, results_file)
    print(f"✓ Final results saved to {results_file}")
    
    total_time = time.time() - start_time_total
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Total jobs: {n_jobs}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    if results:
        avg_train_error = np.mean([r['train_error'] for r in results])
        avg_test_error = np.mean([r['test_error'] for r in results])
        min_test_error = min([r['test_error'] for r in results])
        max_test_error = max([r['test_error'] for r in results])
        print(f"\nError Statistics:")
        print(f"  Average Train Error: {avg_train_error:.4f}")
        print(f"  Average Test Error: {avg_test_error:.4f}")
        print(f"  Best Test Error: {min_test_error:.4f}")
        print(f"  Worst Test Error: {max_test_error:.4f}")
    if total_time > 0:
        print(f"\nTotal time: {total_time/3600:.2f}h ({total_time/60:.1f}min)")
    print(f"\nSaved results to {results_file}")
    print(f"{'='*70}\n")
    
    # Create heatmaps
    if results:
        print("Creating heatmaps...")
        create_heatmaps(results, out_dir)
        print("✓ Heatmaps created successfully")
    else:
        print("⚠️  No results to plot")


if __name__ == "__main__":
    main()

