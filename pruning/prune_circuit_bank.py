#!/usr/bin/env python3
"""
Circuit Bank Distillation: Prune network to top-K eigenpaths with top-M paths per eigenpath.

Usage:
    python prune_circuit_bank.py --model_path <path_to_model_final.pt> --config_path <config.yaml>
    
    Optional:
    --k_values 5 10 15 20        # Number of eigenpaths to use
    --k_sub_values 1 3 5 10       # Number of top paths per eigenpath
    --epochs 100                   # Training epochs for readout
    --device cuda:0                # Device to use
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.path_kernel import compute_path_kernel_eigs, collect_path_factors
from src.analysis.path_analysis import _beam_top_paths, _mean_transmittance_per_layer
from src.data.models.ffnn import MLP
from src.utils.save_io import save_json, ensure_dir


def load_model_and_config(model_path: str, config_path: Optional[str] = None):
    """Load model from checkpoint and infer config if needed."""
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Infer model structure from state dict
    first_layer_key = None
    for key in state_dict.keys():
        if key.startswith('linears.0.weight'):
            first_layer_key = key
            break
    
    if first_layer_key is None:
        raise ValueError(f"Could not find first layer weight. Keys: {list(state_dict.keys())[:10]}")
    
    d_in = state_dict[first_layer_key].shape[1]
    
    # Infer widths from layer weights
    widths = []
    l = 0
    while f'linears.{l}.weight' in state_dict:
        w = state_dict[f'linears.{l}.weight']
        widths.append(w.shape[0])
        l += 1
    
    # Check for readout
    if 'readout.weight' in state_dict:
        n_classes = state_dict['readout.weight'].shape[0]
    else:
        n_classes = 1
    
    # Check for bias
    bias = f'linears.0.bias' in state_dict
    
    # Load config if provided
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Infer from model
        config = {
            "model": {
                "widths": widths,
                "bias": bias,
                "n_classes": n_classes
            },
            "dataset": {
                "d": d_in
            }
        }
    
    # Create model
    model = MLP(
        d_in=d_in,
        widths=widths,
        bias=bias,
        activation="relu",
        n_classes=n_classes
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, config


def extract_paths_from_eigenpath(
    model,
    eigenpath_idx: int,
    evecs: torch.Tensor,  # (P_train, K)
    Phi: torch.Tensor,   # (P_train, D_paths)
    k_sub: int,
    factors: Dict,  # Already collected path factors
    device: torch.device,
    mode: str = "routing_gain"
) -> List[List[int]]:
    """
    Extract top-k_sub individual paths from an eigenpath.
    
    Strategy:
    1. Project eigenpath to path space: eigenpath_path_features = evec.T @ Phi
    2. Decompose into per-layer importance scores
    3. Use beam search weighted by eigenpath contributions to find top paths
    """
    model.eval()
    
    # Project eigenpath to path space
    eigenpath = evecs[:, eigenpath_idx]  # (P_train,)
    
    # Use provided factors (already collected with correct max_samples)
    E_list = factors["E_list"]
    
    # Compute weighted mean transmittance per layer
    # Weight E by eigenpath: eigenpath[i] weights sample i's contribution
    mean_E_weighted = []
    
    for l, E in enumerate(E_list):
        # E is (P_train, d_l), eigenpath is (P_train,)
        # Ensure they have the same number of samples
        min_samples = min(E.shape[0], eigenpath.shape[0])
        E_subset = E[:min_samples]  # (min_samples, d_l)
        eigenpath_subset = eigenpath[:min_samples]  # (min_samples,)
        
        # Weight each sample's transmittance by its eigenpath coefficient
        weighted_E = (E_subset * eigenpath_subset.abs().unsqueeze(1)).mean(dim=0)  # (d_l,)
        # Move to CPU for _beam_top_paths (which expects CPU tensors)
        mean_E_weighted.append(weighted_E.cpu())
    
    # Use beam search with weighted transmittance
    paths = _beam_top_paths(model, mean_E_weighted, beam=min(50, k_sub * 10), top_k=k_sub)
    
    return paths


def create_pruned_model(
    model: MLP,
    important_paths: List[List[int]],
    device: torch.device
) -> MLP:
    """
    Create a pruned model that only includes units/connections in important_paths.
    
    Strategy: Keep all units that appear in any path, but only connections used by paths.
    """
    L = len(model.linears)
    
    # Collect unique units per layer
    # units_per_layer[0] = input layer, units_per_layer[1..L] = hidden layers
    units_per_layer = [set() for _ in range(L + 1)]
    
    # Also track which input features are used
    input_features_used = set()
    
    # First, collect which hidden neurons are used
    for path in important_paths:
        if len(path) == 0:
            continue
        
        # For first hidden layer, path[0] is the unit index
        if len(path) > 0:
            units_per_layer[1].add(path[0])  # First hidden layer
        
        for l in range(1, len(path)):
            if l <= L:
                units_per_layer[l].add(path[l-1])  # Source unit (from previous layer)
                if l < L:
                    units_per_layer[l+1].add(path[l])  # Target unit (in next layer)
    
    # Now determine which input features connect to the first hidden layer neurons
    # Look at the weight matrix: W[hidden_neuron_idx, input_feature_idx]
    first_layer_weights = model.linears[0].weight  # (d_out, d_in)
    
    for hidden_neuron_idx in units_per_layer[1]:
        # Get weights connecting to this hidden neuron
        weights_to_neuron = first_layer_weights[hidden_neuron_idx, :]  # (d_in,)
        # Keep input features with non-zero (or significant) weights
        # Use a threshold: keep features with |weight| > some small threshold
        # Or keep top-k features by weight magnitude
        threshold = 1e-6  # Small threshold to filter near-zero weights
        significant_inputs = torch.where(torch.abs(weights_to_neuron) > threshold)[0]
        input_features_used.update(significant_inputs.cpu().tolist())
    
    # If no input features found (shouldn't happen), keep all as fallback
    if len(input_features_used) == 0:
        print("    Warning: No input features found for paths, keeping all inputs")
        input_features_used = set(range(model.linears[0].in_features))
    
    units_per_layer[0] = input_features_used
    
    # Convert to sorted lists, with fallback to all units if empty
    widths = [model.linears[0].in_features] + [w.out_features for w in model.linears]
    units_per_layer = [sorted(list(units)) if len(units) > 0 else list(range(width)) 
                       for units, width in zip(units_per_layer, widths)]
    
    # Create mapping from original indices to pruned indices
    idx_maps = [{} for _ in range(L + 1)]
    for l, units in enumerate(units_per_layer):
        for new_idx, old_idx in enumerate(units):
            idx_maps[l][old_idx] = new_idx
    
    # Create pruned model with reduced widths
    pruned_widths = [len(units) for units in units_per_layer[1:]]
    # Ensure we preserve n_classes from original model
    n_classes = getattr(model, 'n_classes', 1)
    pruned_model = MLP(
        d_in=len(units_per_layer[0]),
        widths=pruned_widths,
        bias=model.linears[0].bias is not None,
        activation="relu",
        n_classes=n_classes
    ).to(device)
    
    # Copy weights for kept connections
    with torch.no_grad():
        # First layer: only copy weights for connections that are actually in paths
        old_W = model.linears[0].weight  # (d_out, d_in)
        new_W = pruned_model.linears[0].weight  # (pruned_d_out, pruned_d_in)
        
        # Initialize new weights to zero (so unused connections are zero)
        new_W.zero_()
        
        for old_out_idx in units_per_layer[1]:
            new_out_idx = idx_maps[1][old_out_idx]
            for old_in_idx in units_per_layer[0]:
                new_in_idx = idx_maps[0][old_in_idx]
                new_W[new_out_idx, new_in_idx] = old_W[old_out_idx, old_in_idx]
        
        if model.linears[0].bias is not None:
            for old_out_idx in units_per_layer[1]:
                new_out_idx = idx_maps[1][old_out_idx]
                pruned_model.linears[0].bias[new_out_idx] = model.linears[0].bias[old_out_idx]
        
        # Hidden layers: only copy weights for connections that are in paths
        for l in range(1, L):
            old_W = model.linears[l].weight
            new_W = pruned_model.linears[l].weight
            
            # Initialize to zero
            new_W.zero_()
            
            for old_out_idx in units_per_layer[l+1]:
                new_out_idx = idx_maps[l+1][old_out_idx]
                for old_in_idx in units_per_layer[l]:
                    new_in_idx = idx_maps[l][old_in_idx]
                    new_W[new_out_idx, new_in_idx] = old_W[old_out_idx, old_in_idx]
            
            if model.linears[l].bias is not None:
                for old_out_idx in units_per_layer[l+1]:
                    new_out_idx = idx_maps[l+1][old_out_idx]
                    pruned_model.linears[l].bias[new_out_idx] = model.linears[l].bias[old_out_idx]
        
        # Readout layer
        old_W = model.readout.weight  # (n_classes, d_last)
        new_W = pruned_model.readout.weight  # (n_classes, pruned_d_last)
        
        for old_in_idx in units_per_layer[L]:
            new_in_idx = idx_maps[L][old_in_idx]
            new_W[:, new_in_idx] = old_W[:, old_in_idx]
        
        if model.readout.bias is not None:
            pruned_model.readout.bias.data = model.readout.bias.data.clone()
    
    return pruned_model, idx_maps, units_per_layer


def freeze_all_except_readout(model: MLP):
    """Freeze all parameters except readout layer."""
    # Freeze all linear layers
    for linear in model.linears:
        for param in linear.parameters():
            param.requires_grad = False
    
    # Only train readout
    for param in model.readout.parameters():
        param.requires_grad = True


def train_readout_only(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    n_classes: int,
    alpha: float = 1.0,
    input_feature_indices: Optional[List[int]] = None
) -> Dict[str, List[float]]:
    """
    Train only the readout layer.
    
    Args:
        input_feature_indices: If provided, slice inputs to only these features (for pruned models)
    """
    model.train()
    optimizer = optim.Adam(model.readout.parameters(), lr=1e-3)
    
    history = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Slice input features if needed (for pruned models)
            if input_feature_indices is not None:
                xb = xb[:, input_feature_indices]
            optimizer.zero_grad()
            
            yhat = model(xb)
            
            if n_classes == 1:
                # Binary classification
                loss = nn.functional.mse_loss(yhat, yb)
                pred = torch.sign(yhat)
                train_correct += (pred == yb).float().sum().item()
            else:
                # Multi-class: MSE with one-hot
                if yb.dim() > 1:
                    yb = yb.view(-1)
                yb_class = (yb / alpha).long()
                # Clamp to valid range based on actual model output size
                actual_n_classes = yhat.shape[1]
                yb_class = torch.clamp(yb_class, 0, actual_n_classes - 1)
                # Ensure yb_class is on the same device as yhat
                yb_class = yb_class.to(yhat.device)
                yb_onehot = torch.zeros_like(yhat)
                # Use scatter with proper indexing - ensure indices are valid
                valid_mask = (yb_class >= 0) & (yb_class < actual_n_classes)
                if valid_mask.any():
                    src_values = torch.ones_like(yb.unsqueeze(1)) * alpha
                    yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values.to(yhat.device))
                loss = nn.functional.mse_loss(yhat, yb_onehot)
                pred = yhat.argmax(dim=1)
                train_correct += (pred == yb_class).float().sum().item()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_total += xb.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                # Slice input features if needed (for pruned models)
                if input_feature_indices is not None:
                    xb = xb[:, input_feature_indices]
                yhat = model(xb)
                
                if n_classes == 1:
                    loss = nn.functional.mse_loss(yhat, yb)
                    pred = torch.sign(yhat)
                    val_correct += (pred == yb).float().sum().item()
            else:
                if yb.dim() > 1:
                    yb = yb.view(-1)
                yb_class = (yb / alpha).long()
                # Clamp to valid range based on actual model output size
                actual_n_classes = yhat.shape[1]
                yb_class = torch.clamp(yb_class, 0, actual_n_classes - 1)
                # Ensure yb_class is on the same device as yhat
                yb_class = yb_class.to(yhat.device)
                yb_onehot = torch.zeros_like(yhat)
                # Use scatter with proper indexing - ensure indices are valid
                valid_mask = (yb_class >= 0) & (yb_class < actual_n_classes)
                if valid_mask.any():
                    src_values = torch.ones_like(yb.unsqueeze(1)) * alpha
                    yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values.to(yhat.device))
                loss = nn.functional.mse_loss(yhat, yb_onehot)
                pred = yhat.argmax(dim=1)
                val_correct += (pred == yb_class).float().sum().item()
                
                val_loss += loss.item() * xb.size(0)
                val_total += xb.size(0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss / train_total)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss / val_total)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    return history


def evaluate_model(
    model: MLP,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    alpha: float = 1.0,
    input_feature_indices: Optional[List[int]] = None
) -> Tuple[float, float]:
    """
    Evaluate model accuracy and loss.
    
    Args:
        model: The model to evaluate
        loader: DataLoader
        device: Device
        n_classes: Number of classes
        alpha: Alpha scaling factor
        input_feature_indices: If provided, slice inputs to only these features (for pruned models)
    """
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # Slice input features if needed (for pruned models)
            if input_feature_indices is not None:
                xb = xb[:, input_feature_indices]
            yhat = model(xb)
            
            if n_classes == 1:
                loss = nn.functional.mse_loss(yhat, yb)
                pred = torch.sign(yhat)
                correct += (pred == yb).float().sum().item()
            else:
                if yb.dim() > 1:
                    yb = yb.view(-1)
                yb_class = (yb / alpha).long()
                # Clamp to valid range based on actual model output size
                actual_n_classes = yhat.shape[1]
                yb_class = torch.clamp(yb_class, 0, actual_n_classes - 1)
                # Ensure yb_class is on the same device as yhat
                yb_class = yb_class.to(yhat.device)
                yb_onehot = torch.zeros_like(yhat)
                # Use scatter with proper indexing - ensure indices are valid
                valid_mask = (yb_class >= 0) & (yb_class < actual_n_classes)
                if valid_mask.any():
                    src_values = torch.ones_like(yb.unsqueeze(1)) * alpha
                    yb_onehot.scatter_(1, yb_class.unsqueeze(1), src_values.to(yhat.device))
                loss = nn.functional.mse_loss(yhat, yb_onehot)
                pred = yhat.argmax(dim=1)
                correct += (pred == yb_class).float().sum().item()
            
            loss_sum += loss.item() * xb.size(0)
            total += xb.size(0)
    
    acc = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / total if total > 0 else 0.0
    return acc, avg_loss


def build_dataloaders_from_model_path(model_path: str, config: Dict):
    """Build dataloaders from model directory structure."""
    dataset_dir = Path(model_path).parent
    
    # Try to load dataset meta
    meta_path = dataset_dir / "dataset_meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    else:
        meta = config.get("dataset", {})
    
    # Try to infer dataset type and load
    dataset_name = meta.get("name", "").lower()
    if not dataset_name:
        # Try to infer from config
        dataset_name = config.get("dataset", {}).get("name", "").lower()
    
    # Ensure config has dataset section
    if "dataset" not in config:
        config["dataset"] = {}
    config["dataset"].update(meta)
    
    if dataset_name == "mnist" or "mnist" in str(model_path).lower():
        from src.data.mnist import build_mnist_datasets, MNISTDataset
        
        # Build datasets
        Xtr, ytr, Xva, yva, Xte, yte, meta_loaded = build_mnist_datasets(config)
        
        # Create datasets
        train_dataset = MNISTDataset(Xtr, ytr)
        val_dataset = MNISTDataset(Xva, yva)
        test_dataset = MNISTDataset(Xte, yte)
        
        # Create loaders
        batch_size = config.get("training", {}).get("batch_size", 128)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, meta_loaded
    
    elif dataset_name == "hierarchical_xor" or "xor" in str(model_path).lower():
        from src.data.hierarchical_xor import build_hierarchical_xor_datasets, HierarchicalXORDataset
        
        Xtr, ytr, Xva, yva, Xte, yte, meta_loaded = build_hierarchical_xor_datasets(config)
        
        train_dataset = HierarchicalXORDataset(Xtr, ytr)
        val_dataset = HierarchicalXORDataset(Xva, yva)
        test_dataset = HierarchicalXORDataset(Xte, yte)
        
        batch_size = config.get("training", {}).get("batch_size", 128)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, meta_loaded
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please provide dataloaders manually or ensure dataset_meta.json exists.")


def main():
    # ============================================================================
    # CONFIGURATION - Edit these values directly
    # ============================================================================
    model_path = "/home/goring/NN_alternatecoding/outputs/04_12/mnist_run_1_20251204_004314/n50000_lam0.0/sgd/model_final.pt"
    config_path = None  # Optional: path to config YAML (will infer from model if None)
    
    k_values = [1,2,5, 10, ]  # Number of eigenpaths to use
    k_sub_values = [1,2, 5, 10]  # Number of top paths per eigenpath
    
    epochs = 150  # Training epochs for readout
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"  # Device to use
    mode = "routing"  # Path kernel mode: "routing", "routing_gain", "routing_posdev"
    max_samples = 2500  # Max samples for eigenpath computation
    # ============================================================================
    
    device = torch.device(device_str)
    results_dir = Path(__file__).parent / "results"
    ensure_dir(str(results_dir))
    
    print(f"Loading model from {model_path}")
    model, config = load_model_and_config(model_path, config_path)
    model = model.to(device)
    
    # Build dataloaders
    print(f"\nBuilding dataloaders...")
    try:
        train_loader, val_loader, test_loader, meta = build_dataloaders_from_model_path(
            model_path, config
        )
    except Exception as e:
        print(f"Error building dataloaders: {e}")
        print("Please ensure the model directory contains dataset_meta.json or provide config_path")
        return
    
    # Get dataset info
    n_classes = meta.get("n_classes", config.get("model", {}).get("n_classes", model.n_classes))
    alpha = meta.get("alpha", config.get("dataset", {}).get("alpha", 1.0))
    
    print(f"Model: {config['model']['widths']}, n_classes={n_classes}, alpha={alpha}")
    
    print(f"\nComputing top-K eigenpaths...")
    kern_results = compute_path_kernel_eigs(
        model, train_loader, device=device, mode=mode,
        include_input=True, k=max(k_values), n_iter=30,
        block_size=1024, max_samples=max_samples, verbose=True
    )
    
    evecs = kern_results["evecs"].to(device)  # (P_train, K_max)
    evals = kern_results["evals"].to(device)  # (K_max,)
    
    # Get Phi for path extraction
    factors = collect_path_factors(
        model, train_loader, device=device, mode=mode,
        include_input=True, max_samples=max_samples
    )
    
    # Build Phi matrix
    X = factors.get("X")
    E_list = factors["E_list"]
    Phi_parts = []
    if X is not None:
        Phi_parts.append(X)
    Phi_parts.extend(E_list)
    Phi = torch.cat(Phi_parts, dim=1).to(device)  # (P_train, D_paths)
    
    print(f"Phi shape: {Phi.shape}, evecs shape: {evecs.shape}")
    
    # Store results
    all_results = []
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Processing K={k} eigenpaths")
        print(f"{'='*60}")
        
        # Get top-k eigenpaths
        top_k_evecs = evecs[:, :k]  # (P_train, k)
        
        for k_sub in k_sub_values:
            print(f"\n  K={k}, k_sub={k_sub}: Extracting paths...")
            
            # Extract paths from each eigenpath
            all_paths = []
            for eig_idx in range(k):
                paths = extract_paths_from_eigenpath(
                    model, eig_idx, top_k_evecs, Phi, k_sub,
                    factors, device, mode
                )
                all_paths.extend(paths)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_paths = []
            for path in all_paths:
                path_tuple = tuple(path)
                if path_tuple not in seen:
                    seen.add(path_tuple)
                    unique_paths.append(path)
            
            print(f"    Extracted {len(unique_paths)} unique paths")
            
            # Create pruned model
            print(f"    Creating pruned model...")
            pruned_model, idx_maps, units_per_layer = create_pruned_model(
                model, unique_paths, device
            )
            
            # Count parameters
            original_params = sum(p.numel() for p in model.parameters())
            pruned_params = sum(p.numel() for p in pruned_model.parameters())
            compression = original_params / pruned_params if pruned_params > 0 else float('inf')
            
            print(f"    Compression: {compression:.2f}x ({original_params} -> {pruned_params} params)")
            
            # Plot pruned network heatmap
            heatmap_path = results_dir / f"network_heatmap_K{k}_k_sub{k_sub}.png"
            plot_pruned_network_heatmap(
                model, pruned_model, unique_paths, str(heatmap_path),
                title=f"Pruned Network: K={k}, k_sub={k_sub} ({len(unique_paths)} paths)"
            )
            
            # Freeze everything except readout
            freeze_all_except_readout(pruned_model)
            
            # Get input feature indices for this pruned model
            input_feature_indices = units_per_layer[0]
            pruned_hidden_widths = [len(units) for units in units_per_layer[1:]]
            
            # Train readout only
            print(f"    Training readout for {epochs} epochs...")
            print(f"    Pruned model: {len(input_feature_indices)} input features, {pruned_hidden_widths} hidden widths")
            history = train_readout_only(
                pruned_model, train_loader, val_loader, epochs,
                device, n_classes, alpha, input_feature_indices
            )
            
            # Final evaluation
            train_acc, train_loss = evaluate_model(pruned_model, train_loader, device, n_classes, alpha, input_feature_indices)
            val_acc, val_loss = evaluate_model(pruned_model, val_loader, device, n_classes, alpha, input_feature_indices)
            test_acc, test_loss = evaluate_model(pruned_model, test_loader, device, n_classes, alpha, input_feature_indices)
            
            result = {
                "k": k,
                "k_sub": k_sub,
                "num_paths": len(unique_paths),
                "compression": compression,
                "original_params": original_params,
                "pruned_params": pruned_params,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "history": history
            }
            
            all_results.append(result)
            print(f"    Final: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
    
    # Save results
    results_file = results_dir / "pruning_results.json"
    save_json(all_results, str(results_file))
    print(f"\nResults saved to {results_file}")
    
    # Create plots
    print(f"\nCreating plots...")
    plot_results(all_results, results_dir)
    print(f"Plots saved to {results_dir}")


def plot_pruned_network_heatmap(
    original_model: MLP,
    pruned_model: MLP,
    important_paths: List[List[int]],
    out_path: str,
    title: str = "Pruned Network Structure"
):
    """
    Plot pruned network weight matrices stacked vertically, showing only non-zero connections.
    
    Args:
        original_model: The original (unpruned) MLP model
        pruned_model: The pruned MLP model
        important_paths: List of paths (each path is [i0, i1, ..., i_{L-1}])
        out_path: Path to save the plot
        title: Plot title
    """
    L = len(pruned_model.linears)
    
    # Get weight matrices from pruned model
    weight_matrices = []
    layer_names = []
    
    # First layer: input to first hidden
    W0 = pruned_model.linears[0].weight.detach().cpu().numpy()  # (d_out, d_in)
    weight_matrices.append(W0)
    layer_names.append(f"Input → Hidden 1\n({W0.shape[1]} → {W0.shape[0]})")
    
    # Hidden layers
    for l in range(1, L):
        W = pruned_model.linears[l].weight.detach().cpu().numpy()  # (d_out, d_in)
        weight_matrices.append(W)
        layer_names.append(f"Hidden {l} → Hidden {l+1}\n({W.shape[1]} → {W.shape[0]})")
    
    # Readout layer
    W_readout = pruned_model.readout.weight.detach().cpu().numpy()  # (n_classes, d_last)
    weight_matrices.append(W_readout)
    layer_names.append(f"Hidden {L} → Output\n({W_readout.shape[1]} → {W_readout.shape[0]})")
    
    # Create figure with subplots stacked vertically
    num_layers = len(weight_matrices)
    fig, axes = plt.subplots(num_layers, 1, figsize=(14, 2.5 * num_layers))
    
    if num_layers == 1:
        axes = [axes]
    
    # Find global min/max for consistent colormap (excluding zeros)
    all_weights = np.concatenate([W.flatten() for W in weight_matrices])
    non_zero_weights = all_weights[all_weights != 0]
    if len(non_zero_weights) > 0:
        vmin = non_zero_weights.min()
        vmax = non_zero_weights.max()
        # Use symmetric colormap if weights are both positive and negative
        if vmin < 0 and vmax > 0:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
    else:
        vmin, vmax = -1, 1
    
    # Create colormap: white for zero, red/blue for non-zero
    colors = ['white', 'lightblue', 'blue', 'darkblue', 'darkred', 'red', 'lightcoral']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('weight_map', colors, N=n_bins)
    
    # Plot each weight matrix
    for idx, (W, layer_name, ax) in enumerate(zip(weight_matrices, layer_names, axes)):
        # Create mask for non-zero weights
        mask = (W != 0)
        
        # Plot: white for zero, colored for non-zero
        # We'll plot absolute values and use sign for color direction
        W_plot = np.abs(W)
        W_plot[~mask] = 0  # Ensure zeros stay zero
        
        # Use different colormap for positive vs negative (if needed)
        # For simplicity, just show magnitude
        im = ax.imshow(W_plot, cmap='Reds', aspect='auto', 
                      vmin=0, vmax=vmax if vmax > 0 else 1,
                      interpolation='nearest')
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, W.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, W.shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
        
        # Set labels
        ax.set_ylabel(layer_name, fontsize=10, fontweight='bold', rotation=0, ha='right', va='center')
        
        # Only show x-axis labels on last subplot
        if idx == len(weight_matrices) - 1:
            ax.set_xlabel('Input Neuron Index', fontsize=10)
        else:
            ax.set_xticklabels([])
        
        # Show y-axis labels (output neuron indices)
        if W.shape[0] <= 20:
            ax.set_yticks(range(W.shape[0]))
            ax.set_yticklabels(range(W.shape[0]))
        else:
            # Show every Nth tick
            step = max(1, W.shape[0] // 10)
            ax.set_yticks(range(0, W.shape[0], step))
            ax.set_yticklabels(range(0, W.shape[0], step))
        
        # Count non-zero connections
        num_nonzero = np.sum(mask)
        total = W.size
        sparsity = 1.0 - (num_nonzero / total) if total > 0 else 1.0
        
        # Add text annotation with stats
        stats_text = f"Non-zero: {num_nonzero}/{total} ({sparsity*100:.1f}% sparse)"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Add colorbar for the last subplot
    cbar = fig.colorbar(im, ax=axes[-1], orientation='horizontal', pad=0.1)
    cbar.set_label('Weight Magnitude (|weight|)', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved network weight matrices: {out_path}")


def plot_results(results: List[Dict], results_dir: Path):
    """Create plots of accuracy vs K and k_sub."""
    # Extract data
    k_values = sorted(set(r["k"] for r in results))
    k_sub_values = sorted(set(r["k_sub"] for r in results))
    
    # Plot 1: Test accuracy vs K (for different k_sub)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Test accuracy vs K
    ax = axes[0]
    for k_sub in k_sub_values:
        data = [(r["k"], r["test_acc"]) for r in results if r["k_sub"] == k_sub]
        if data:
            k_vals, accs = zip(*sorted(data))
            ax.plot(k_vals, accs, marker='o', label=f'k_sub={k_sub}')
    ax.set_xlabel('K (number of eigenpaths)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Test accuracy vs k_sub (for different K)
    ax = axes[1]
    for k in k_values:
        data = [(r["k_sub"], r["test_acc"]) for r in results if r["k"] == k]
        if data:
            k_sub_vals, accs = zip(*sorted(data))
            ax.plot(k_sub_vals, accs, marker='s', label=f'K={k}')
    ax.set_xlabel('k_sub (paths per eigenpath)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy vs k_sub')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_vs_k.png", dpi=150)
    plt.close()
    
    # Plot 2: Compression vs Accuracy
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for r in results:
        ax.scatter(r["compression"], r["test_acc"], 
                  s=100, alpha=0.6, 
                  label=f'K={r["k"]}, k_sub={r["k_sub"]}')
    
    ax.set_xlabel('Compression Factor')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Compression vs Test Accuracy')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "compression_vs_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Heatmap of test accuracy
    if len(k_values) > 1 and len(k_sub_values) > 1:
        acc_matrix = np.zeros((len(k_sub_values), len(k_values)))
        for r in results:
            i = k_sub_values.index(r["k_sub"])
            j = k_values.index(r["k"])
            acc_matrix[i, j] = r["test_acc"]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        im = ax.imshow(acc_matrix, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticks(range(len(k_sub_values)))
        ax.set_yticklabels(k_sub_values)
        ax.set_xlabel('K (number of eigenpaths)')
        ax.set_ylabel('k_sub (paths per eigenpath)')
        ax.set_title('Test Accuracy Heatmap')
        plt.colorbar(im, ax=ax, label='Test Accuracy')
        plt.tight_layout()
        plt.savefig(results_dir / "accuracy_heatmap.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()

