"""
Training Loop for Different Parameterization Schemes

Contains the main training function that supports:
- Standard, mup, ntk, mup_L, and path parameterizations
- Layer-wise learning rate scheduling (for path parameterization)
- Gate mobility-based LR updates
- Multiple optimizer support (SGD, Adam, Muon)
"""
from __future__ import annotations
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from src.models.lazarus import LazarusMLP
from src.analysis.mobility import (
    compute_gate_mobility_lazarus,
    compute_initial_lr_from_target_mobility,
    compute_optimal_lr_path,
    compute_gate_mobility_per_layer_path,
    update_lr_damped_mobility,
    compute_gradient_norms_per_layer,
)


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
    
    Args:
        model: Neural network model (MLP, MLPResNet, MLPBatchNorm, or LazarusMLP)
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        config: Configuration dictionary
        device: Device to train on
        parameterization: Parameterization scheme name
        architecture: Architecture name (for logging)
        out_dir: Output directory for saving results
        optimizer_override: Optional optimizer to use instead of config default
    
    Returns:
        Dictionary with training history containing:
        - epochs: List of epoch numbers
        - train_loss: List of train losses
        - test_loss: List of test losses
        - M_g_avg: List of average M_g values
        - C_def: List of C_def values
        - H_Lambda: List of H_Lambda values
        - lr_per_layer: Dict mapping layer_idx -> list of LRs
        - grad_norms_per_layer: Dict mapping layer_idx -> list of gradient norms
    """
    # Import here to avoid circular imports
    from src.analysis.path_kernel import collect_path_factors
    from src.analysis.mobility import (
        compute_path_deformation_capacity,
        compute_path_covariance_entropy,
        compute_path_kernel_matrix,
        compute_gate_mobility,
    )
    
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
            optimizer = torch.optim.SGD(param_groups, lr=lr)
        elif optimizer_type == "adam":
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
        elif optimizer_type == "muon":
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}. Options: sgd, adam, muon")
    else:
        # Standard optimizer (single LR for all parameters)
        initial_warmup_factor = warmup_start_ratio if (parameterization == "path" and warmup_epochs > 0) else 1.0
        initial_lr_warmup = lr * initial_warmup_factor
        
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr_warmup)
        elif optimizer_type == "adam":
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr_warmup)
        elif optimizer_type == "muon":
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr_warmup)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}. Options: sgd, adam, muon")
        
        if parameterization == "path":
            if warmup_epochs > 0:
                print(f"  Applied initial warmup factor: {initial_warmup_factor:.4f} (will warmup over {warmup_epochs} epochs)")
            else:
                print(f"  Warmup disabled (warmup_epochs=0), using full LRs from start")
    
    loss_fn = nn.MSELoss()
    
    # Get kernel tracking settings
    track_kernel_metrics = config.get("logging", {}).get("track_kernel_metrics", True)
    topk_spectrum = int(config.get("logging", {}).get("topk_spectrum", 50))
    
    # History tracking
    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "M_g_avg": [],
        "C_def": [],
        "H_Lambda": [],
        "lr_per_layer": {},  # Dict mapping layer_idx -> list of LRs over epochs
        "grad_norms_per_layer": {},  # Dict mapping layer_idx -> list of gradient norms over epochs
        # Kernel tracking metrics
        "path_kernel_rank": [],
        "path_kernel_numerical_rank": [],
        "path_kernel_cka": [],  # CKA between initial and current
        "path_kernel_wasserstein": [],  # Wasserstein distance from initial eigenvalues
        "hidden_kernel_rank": [],
        "hidden_kernel_numerical_rank": [],
        "hidden_kernel_cka": [],
        "hidden_kernel_wasserstein": [],
        "gradient_eigenvalues_concat": [],  # Concatenated gradient EVs per epoch
        # Initial and final spectra (saved at end)
        "initial_path_kernel_eigs": None,
        "final_path_kernel_eigs": None,
        "initial_hidden_kernel_eigs": None,
        "final_hidden_kernel_eigs": None,
    }
    
    # Storage for initial eigenvalues (for CKA/Wasserstein computation)
    initial_path_eigs = None
    initial_hidden_eigs = None
    
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
    
    # Import kernel tracking module
    from src.analysis.kernel_tracking import (
        compute_all_kernel_metrics,
        compute_gradient_eigenvalues_per_layer,
        concatenate_gradient_eigenvalues,
    )
    
    # Training loop
    for epoch in range(epochs + 1):
        # Initialize gradient norms tracking for this epoch
        grad_norms_epoch = None
        grad_eigs_epoch = None
        
        # At epoch 0, compute initial kernel metrics
        if epoch == 0 and track_kernel_metrics:
            print(f"{device_tag} Computing initial kernel metrics...")
            try:
                initial_metrics = compute_all_kernel_metrics(
                    model, train_loader, device, config,
                    initial_path_eigs=None,
                    initial_hidden_eigs=None
                )
                # Store eigenvalues for later comparison (no full kernels needed)
                initial_path_eigs = initial_metrics.get("path_kernel_eigs")
                initial_hidden_eigs = initial_metrics.get("hidden_kernel_eigs")
                
                # Store initial eigenvalues
                if initial_path_eigs is not None:
                    history["initial_path_kernel_eigs"] = initial_path_eigs.tolist()
                if initial_hidden_eigs is not None:
                    history["initial_hidden_kernel_eigs"] = initial_hidden_eigs.tolist()
                
                # Format rank for printing (handle nan gracefully)
                path_rank = initial_metrics.get('path_kernel_rank', float('nan'))
                hidden_rank = initial_metrics.get('hidden_kernel_rank', float('nan'))
                path_rank_str = f"{path_rank:.2f}" if not np.isnan(path_rank) else "N/A"
                hidden_rank_str = f"{hidden_rank:.2f}" if not np.isnan(hidden_rank) else "N/A"
                print(f"{device_tag}   Initial path kernel rank: {path_rank_str}")
                print(f"{device_tag}   Initial hidden kernel rank: {hidden_rank_str}")
            except Exception as e:
                import traceback
                print(f"{device_tag}   Warning: Failed to compute initial kernel metrics: {e}")
                print(traceback.format_exc())
        
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
                        # Also compute gradient eigenvalues if tracking kernel metrics
                        if track_kernel_metrics:
                            grad_eigs_epoch = compute_gradient_eigenvalues_per_layer(
                                model, xb, yb, loss_fn, device, n_classes, alpha, topk=topk_spectrum
                            )
                    except Exception as e:
                        grad_norms_epoch = None
                        grad_eigs_epoch = None
                    first_batch = False
                
                # Optional gradient clipping
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            # For "path" parameterization: update LR for next epoch
            if parameterization == "path" and epoch > 0:
                _update_path_lr(
                    model, optimizer, train_loader, device, n_classes, alpha,
                    use_layerwise_lr, target_mobility, lr_update_alpha, lr_update_eps,
                    lr_update_min_scale, lr_update_max_scale, lr_max,
                    warmup_epochs, warmup_start_ratio, epoch, history, device_tag
                )
        
        # Compute train and test loss every epoch
        train_loss = evaluate_loss(model, train_loader, device, n_classes, alpha)
        test_loss = evaluate_loss(model, test_loader, device, n_classes, alpha)
        
        # Log train/test error every 10 epochs
        if epoch % 10 == 0 and epoch % metrics_freq != 0:
            print(f"{device_tag} [Epoch {epoch}] Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        # Save train/test loss to history every epoch
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        
        # Compute metrics at specified frequency
        if epoch % metrics_freq == 0:
            _compute_and_log_metrics(
                model, train_loader, optimizer, device, n_classes, alpha, lr,
                parameterization, use_layerwise_lr, compute_metrics,
                m_g_n_batches, kernel_max_samples, epoch, history, device_tag,
                grad_norms_epoch, collect_path_factors, compute_path_deformation_capacity,
                compute_path_covariance_entropy, compute_path_kernel_matrix, compute_gate_mobility,
                train_loss, test_loss, out_dir, architecture
            )
            
            # Compute kernel metrics if enabled
            if track_kernel_metrics:
                try:
                    kernel_metrics = compute_all_kernel_metrics(
                        model, train_loader, device, config,
                        initial_path_eigs=initial_path_eigs,
                        initial_hidden_eigs=initial_hidden_eigs
                    )
                    
                    history["path_kernel_rank"].append(kernel_metrics.get("path_kernel_rank", float('nan')))
                    history["path_kernel_numerical_rank"].append(kernel_metrics.get("path_kernel_numerical_rank", 0))
                    history["path_kernel_cka"].append(kernel_metrics.get("path_kernel_cka", float('nan')))
                    history["path_kernel_wasserstein"].append(kernel_metrics.get("path_kernel_wasserstein", float('nan')))
                    history["hidden_kernel_rank"].append(kernel_metrics.get("hidden_kernel_rank", float('nan')))
                    history["hidden_kernel_numerical_rank"].append(kernel_metrics.get("hidden_kernel_numerical_rank", 0))
                    history["hidden_kernel_cka"].append(kernel_metrics.get("hidden_kernel_cka", float('nan')))
                    history["hidden_kernel_wasserstein"].append(kernel_metrics.get("hidden_kernel_wasserstein", float('nan')))
                    
                    # Store final eigenvalues (will be overwritten each time, last one is final)
                    if kernel_metrics.get("path_kernel_eigs") is not None:
                        history["final_path_kernel_eigs"] = kernel_metrics["path_kernel_eigs"].tolist()
                    if kernel_metrics.get("hidden_kernel_eigs") is not None:
                        history["final_hidden_kernel_eigs"] = kernel_metrics["hidden_kernel_eigs"].tolist()
                    
                    # Format for printing (handle nan gracefully)
                    path_rank = kernel_metrics.get('path_kernel_rank', float('nan'))
                    hidden_rank = kernel_metrics.get('hidden_kernel_rank', float('nan'))
                    path_cka = kernel_metrics.get('path_kernel_cka', float('nan'))
                    hidden_cka = kernel_metrics.get('hidden_kernel_cka', float('nan'))
                    path_rank_str = f"{path_rank:.2f}" if not np.isnan(path_rank) else "N/A"
                    hidden_rank_str = f"{hidden_rank:.2f}" if not np.isnan(hidden_rank) else "N/A"
                    path_cka_str = f"{path_cka:.4f}" if not np.isnan(path_cka) else "N/A"
                    hidden_cka_str = f"{hidden_cka:.4f}" if not np.isnan(hidden_cka) else "N/A"
                    print(f"{device_tag}   Path kernel rank: {path_rank_str}, CKA: {path_cka_str}")
                    print(f"{device_tag}   Hidden kernel rank: {hidden_rank_str}, CKA: {hidden_cka_str}")
                except Exception as e:
                    import traceback
                    print(f"{device_tag}   Warning: Failed to compute kernel metrics: {e}")
                    print(traceback.format_exc())
                    history["path_kernel_rank"].append(float('nan'))
                    history["path_kernel_numerical_rank"].append(0)
                    history["path_kernel_cka"].append(float('nan'))
                    history["path_kernel_wasserstein"].append(float('nan'))
                    history["hidden_kernel_rank"].append(float('nan'))
                    history["hidden_kernel_numerical_rank"].append(0)
                    history["hidden_kernel_cka"].append(float('nan'))
                    history["hidden_kernel_wasserstein"].append(float('nan'))
            
            # Track gradient eigenvalues
            if grad_eigs_epoch is not None:
                concat_eigs = concatenate_gradient_eigenvalues(grad_eigs_epoch)
                history["gradient_eigenvalues_concat"].append(concat_eigs.tolist())
            else:
                history["gradient_eigenvalues_concat"].append([])
        else:
            # For epochs where metrics are not computed, append NaN
            history["M_g_avg"].append(float('nan'))
            history["C_def"].append(float('nan'))
            history["H_Lambda"].append(float('nan'))
            if track_kernel_metrics:
                history["path_kernel_rank"].append(float('nan'))
                history["path_kernel_numerical_rank"].append(0)
                history["path_kernel_cka"].append(float('nan'))
                history["path_kernel_wasserstein"].append(float('nan'))
                history["hidden_kernel_rank"].append(float('nan'))
                history["hidden_kernel_numerical_rank"].append(0)
                history["hidden_kernel_cka"].append(float('nan'))
                history["hidden_kernel_wasserstein"].append(float('nan'))
                history["gradient_eigenvalues_concat"].append([])
        
        # Save intermediate results (only at metrics frequency)
        if epoch % metrics_freq == 0:
            _save_history(history, out_dir, architecture, parameterization)
    
    return history


def _update_path_lr(
    model, optimizer, train_loader, device, n_classes, alpha,
    use_layerwise_lr, target_mobility, lr_update_alpha, lr_update_eps,
    lr_update_min_scale, lr_update_max_scale, lr_max,
    warmup_epochs, warmup_start_ratio, epoch, history, device_tag
):
    """Update learning rates for path parameterization using gate mobility."""
    # Check if model has NaN/Inf parameters
    has_nan_params = False
    for param in model.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            has_nan_params = True
            break
    
    if has_nan_params:
        # Model has NaN - skip LR update
        train_loss_epoch = evaluate_loss(model, train_loader, device, n_classes, alpha)
        test_loss_epoch = evaluate_loss(model, DataLoader([]), device, n_classes, alpha)  # Empty, will be 0
        print(f"{device_tag} [Epoch {epoch}] Train Loss: {train_loss_epoch:.6f} (NaN model - LR update skipped)")
        return
    
    try:
        if use_layerwise_lr:
            # Get current learning rates for each layer
            current_lrs_dict = {}
            for param_group in optimizer.param_groups:
                layer_idx = param_group.get('layer')
                if isinstance(layer_idx, int):
                    current_lrs_dict[layer_idx] = param_group['lr']
            
            # Compute gate mobility M_g,l for each layer
            M_g_per_layer = compute_gate_mobility_per_layer_path(
                model, train_loader, current_lrs_dict, device, n_classes, alpha, n_batches=1
            )
            
            if M_g_per_layer is None or all(v is None for v in M_g_per_layer.values()):
                # M_g computation failed - reduce LRs conservatively
                updated_lrs_dict = {}
                for l in current_lrs_dict:
                    updated_lrs_dict[l] = min(current_lrs_dict[l] * lr_update_min_scale, lr_max)
                
                _apply_updated_lrs(model, optimizer, updated_lrs_dict, history, epoch)
            else:
                # Update LRs using damped mobility-based rule
                updated_lrs_dict = update_lr_damped_mobility(
                    current_lrs_dict, M_g_per_layer, target_mobility,
                    lr_update_alpha, lr_update_eps, lr_update_min_scale, lr_update_max_scale, lr_max
                )
                
                # Apply warmup scaling if in warmup period
                warmup_factor = 1.0
                if epoch <= warmup_epochs and warmup_epochs > 0:
                    warmup_factor = warmup_start_ratio + (1.0 - warmup_start_ratio) * (epoch / warmup_epochs)
                    warmup_factor = max(warmup_start_ratio, min(1.0, warmup_factor))
                
                _apply_updated_lrs_with_warmup(model, optimizer, updated_lrs_dict, warmup_factor, history, epoch, device_tag, warmup_epochs)
        else:
            # Single global LR (for non-LazarusMLP models)
            current_lr_global = optimizer.param_groups[0]['lr']
            current_lrs_dict = {0: current_lr_global}
            
            M_g_per_layer = compute_gate_mobility_per_layer_path(
                model, train_loader, current_lrs_dict, device, n_classes, alpha, n_batches=1
            )
            
            if M_g_per_layer is None or M_g_per_layer.get(0) is None:
                pass  # Keep current LR
            else:
                updated_lrs_dict = update_lr_damped_mobility(
                    current_lrs_dict, M_g_per_layer, target_mobility,
                    lr_update_alpha, lr_update_eps, lr_update_min_scale, lr_update_max_scale, lr_max
                )
                updated_lr = updated_lrs_dict.get(0, current_lr_global)
                
                # Apply warmup
                warmup_factor = 1.0
                if epoch <= warmup_epochs and warmup_epochs > 0:
                    warmup_factor = warmup_start_ratio + (1.0 - warmup_start_ratio) * (epoch / warmup_epochs)
                    warmup_factor = max(warmup_start_ratio, min(1.0, warmup_factor))
                
                actual_lr = updated_lr * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = actual_lr
                
                # Track LR
                if 'global' not in history["lr_per_layer"]:
                    history["lr_per_layer"]['global'] = []
                while len(history["lr_per_layer"]['global']) < epoch:
                    history["lr_per_layer"]['global'].append(None)
                history["lr_per_layer"]['global'].append(actual_lr)
    except Exception as e:
        print(f"{device_tag} [Epoch {epoch}] [Path LR] Error: {e}")
        import traceback
        traceback.print_exc()


def _apply_updated_lrs(model, optimizer, updated_lrs_dict, history, epoch):
    """Apply updated learning rates to optimizer parameter groups."""
    updated_lrs = {}
    for param_group in optimizer.param_groups:
        layer_idx = param_group.get('layer')
        if layer_idx == 'readout':
            target_lr = updated_lrs_dict.get(model.depth - 1, param_group['lr'])
            param_group['lr'] = target_lr
            updated_lrs['readout'] = target_lr
        elif isinstance(layer_idx, int):
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


def _apply_updated_lrs_with_warmup(model, optimizer, updated_lrs_dict, warmup_factor, history, epoch, device_tag, warmup_epochs):
    """Apply updated learning rates with warmup scaling."""
    updated_lrs = {}
    for param_group in optimizer.param_groups:
        layer_idx = param_group.get('layer')
        if layer_idx == 'readout':
            target_lr = updated_lrs_dict.get(model.depth - 1, param_group['lr'])
            new_lr = target_lr * warmup_factor
            param_group['lr'] = new_lr
            updated_lrs['readout'] = new_lr
        elif isinstance(layer_idx, int):
            target_lr = updated_lrs_dict.get(layer_idx, param_group['lr'])
            new_lr = target_lr * warmup_factor
            param_group['lr'] = new_lr
            updated_lrs[layer_idx] = new_lr
    
    if epoch <= warmup_epochs and warmup_epochs > 0:
        print(f"{device_tag}   Warmup: epoch {epoch}/{warmup_epochs}, factor={warmup_factor:.4f}")
    
    # Track LRs per layer
    for layer_idx, lr_val in updated_lrs.items():
        if layer_idx not in history["lr_per_layer"]:
            history["lr_per_layer"][layer_idx] = []
        while len(history["lr_per_layer"][layer_idx]) < epoch:
            history["lr_per_layer"][layer_idx].append(None)
        history["lr_per_layer"][layer_idx].append(lr_val)


def _compute_and_log_metrics(
    model, train_loader, optimizer, device, n_classes, alpha, lr,
    parameterization, use_layerwise_lr, compute_metrics,
    m_g_n_batches, kernel_max_samples, epoch, history, device_tag,
    grad_norms_epoch, collect_path_factors, compute_path_deformation_capacity,
    compute_path_covariance_entropy, compute_path_kernel_matrix, compute_gate_mobility,
    train_loss, test_loss, out_dir, architecture
):
    """Compute and log metrics at specified frequency."""
    print(f"\n{device_tag} [Epoch {epoch}] Computing metrics...")
    print(f"{device_tag}   Train loss: {train_loss:.6f}, Test loss: {test_loss:.6f}")
    
    # Compute M_g
    M_g_avg = float('nan')
    if compute_metrics:
        try:
            print(f"{device_tag}   Computing M_g...")
            if parameterization == "path" and use_layerwise_lr:
                current_lrs_dict = {}
                for param_group in optimizer.param_groups:
                    layer_idx = param_group.get('layer')
                    if isinstance(layer_idx, int):
                        current_lrs_dict[layer_idx] = param_group['lr']
                
                M_g_per_layer = compute_gate_mobility_per_layer_path(
                    model, train_loader, current_lrs_dict, device, n_classes, alpha, n_batches=m_g_n_batches
                )
                
                valid_m_g = [v for v in M_g_per_layer.values() if v is not None and not (np.isinf(v) or np.isnan(v))]
                if len(valid_m_g) > 0:
                    M_g_avg = np.mean(valid_m_g)
                else:
                    M_g_avg = float('nan')
                    print(f"{device_tag}   Warning: All M_g values are inf or nan")
                print(f"{device_tag}   M_g (avg, using actual per-layer LRs): {M_g_avg:.6f}")
            else:
                if isinstance(model, LazarusMLP):
                    m_g_result = compute_gate_mobility_lazarus(
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
    
    # Compute C_def and H_Lambda (path kernel metrics)
    C_def = float('nan')
    H_Lambda = float('nan')
    if compute_metrics:
        try:
            print(f"{device_tag}   Computing path kernel metrics (C_def, H_Lambda)...")
            pack = collect_path_factors(
                model, train_loader, device,
                mode="routing_gain",
                include_input=True,
                max_samples=kernel_max_samples
            )
            X = pack["X"]
            E_list = pack["E_list"]
            
            if X is not None and len(E_list) > 0:
                if X.shape[0] == 0 or any(E.shape[0] == 0 for E in E_list):
                    print(f"{device_tag}   Warning: Empty path factors")
                else:
                    # Normalize for numerical stability
                    X_fro_norm = torch.norm(X, p='fro')
                    if X_fro_norm > 1e-8:
                        X_normalized = X / X_fro_norm
                    else:
                        X_normalized = X
                    
                    E_list_normalized = []
                    for E in E_list:
                        E_fro_norm = torch.norm(E, p='fro')
                        if E_fro_norm > 1e-8:
                            E_list_normalized.append(E / E_fro_norm)
                        else:
                            E_list_normalized.append(E)
                    
                    try:
                        C_def = compute_path_deformation_capacity(
                            X_normalized, E_list_normalized, device,
                            block_size=2048, dtype=torch.float32, use_tf32=True
                        )
                        if not np.isnan(C_def) and not np.isinf(C_def):
                            print(f"{device_tag}   C_def: {C_def:.6f}")
                    except Exception as e:
                        print(f"{device_tag}   Error computing C_def: {e}")
                        C_def = float('nan')
                    
                    try:
                        has_nan_inf = any(torch.isnan(E).any() or torch.isinf(E).any() for E in E_list)
                        if not has_nan_inf:
                            H_Lambda = compute_path_covariance_entropy(
                                E_list, device,
                                n_bins=100, block_size=2048, dtype=torch.float32, use_tf32=True,
                                normalize_factors=False, epsilon=1e-12, use_float64=True
                            )
                            print(f"{device_tag}   H_Lambda: {H_Lambda:.6f}")
                    except Exception as e:
                        print(f"{device_tag}   Error computing H_Lambda: {e}")
                        H_Lambda = float('nan')
        except Exception as e:
            print(f"{device_tag}   Warning: Failed to compute path kernel metrics: {e}")
    
    # Track LRs per layer (at metrics frequency) - only if not already tracked
    if parameterization != "path" or not use_layerwise_lr:
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
    
    # Track gradient norms per layer (at metrics frequency)
    if grad_norms_epoch is not None:
        for layer_idx, grad_norm in grad_norms_epoch.items():
            if layer_idx not in history["grad_norms_per_layer"]:
                history["grad_norms_per_layer"][layer_idx] = []
            history["grad_norms_per_layer"][layer_idx].append(grad_norm)
    
    # Store metrics in history
    history["M_g_avg"].append(M_g_avg)
    history["C_def"].append(C_def)
    history["H_Lambda"].append(H_Lambda)


def _save_history(history, out_dir, architecture, parameterization):
    """Save history to JSON file."""
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

