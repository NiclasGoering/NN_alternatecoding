#!/usr/bin/env python3
"""
Generate all path analysis plots from saved experiment outputs.

Usage:
    # Edit configuration in main() function, then run:
    python plots/generate_path_analysis_plots.py

The script will automatically find all models in subdirectories matching:
    output_dir/n*_lam*/algorithm_name/model_final.pt
"""

from __future__ import annotations

import os
import sys
import json
import csv
import ast
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Optional dependency for clustering
try:
    from sklearn.cluster import KMeans
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.models.ffnn import GatedMLP
from src.data.hierarchical_xor import build_hierarchical_xor_datasets, HierarchicalXORDataset
from src.analysis.path_analysis import (
    run_full_analysis_at_checkpoint,
    plot_eig_spectrum,
    plot_nn_graph_with_paths,
    plot_path_cleanliness,
    plot_embedding_map,
    flow_centrality_heatmap,
    path_embedding,
    ablation_waterfall,
    plot_path_shapley_bars,
    circuit_overlap_matrix,
)
from src.analysis.path_kernel import compute_path_kernel_eigs


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def build_dataloaders_from_config(cfg: dict, n_train: int, lambda_val: float) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build dataloaders for a specific n_train and lambda value.
    Returns: (train_loader, val_loader, test_loader)
    """
    ds_cfg = cfg["dataset"]
    name = ds_cfg.get("name", "ksparse_parity").lower()
    
    if name == "hierarchical_xor":
        # Create a temporary config with specific n_train
        temp_cfg = cfg.copy()
        temp_cfg["dataset"] = ds_cfg.copy()
        temp_cfg["dataset"]["n_train"] = n_train
        
        Xtr, ytr, Xva, yva, Xte, yte, meta, groups_tr, groups_va, groups_te = build_hierarchical_xor_datasets(temp_cfg)
        n_groups = meta.get("n_groups")
        ds_tr = HierarchicalXORDataset(Xtr, ytr, groups=groups_tr, n_groups=n_groups)
        ds_va = HierarchicalXORDataset(Xva, yva, groups=groups_va, n_groups=n_groups)
        ds_te = HierarchicalXORDataset(Xte, yte, groups=groups_te, n_groups=n_groups)
        
        bs = cfg["training"]["batch_size"]
        train_loader = DataLoader(ds_tr, batch_size=min(bs, len(ds_tr)), shuffle=False, drop_last=False)
        val_loader = DataLoader(ds_va, batch_size=len(ds_va), shuffle=False)
        test_loader = DataLoader(ds_te, batch_size=len(ds_te), shuffle=False)
        
        return train_loader, val_loader, test_loader
    else:
        raise ValueError(f"Dataset {name} not yet supported in plotting script")


def load_model(model_path: str, cfg: dict) -> GatedMLP:
    """Load a saved model from state_dict."""
    # Infer d_in from the saved model's first layer weight shape
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Find the first linear layer weight (linears.0.weight)
    first_layer_key = None
    for key in state_dict.keys():
        if key.startswith('linears.0.weight'):
            first_layer_key = key
            break
    
    if first_layer_key is None:
        raise ValueError(f"Could not find first layer weight in model state_dict. Keys: {list(state_dict.keys())[:10]}")
    
    # d_in is the second dimension of the first layer weight
    d_in = state_dict[first_layer_key].shape[1]
    
    # Check if gates exist in state_dict (sgd_relu models don't have gates)
    has_gates = any(key.startswith('gates.') for key in state_dict.keys())
    use_gates = has_gates and cfg["model"]["use_gates"]
    
    model = GatedMLP(
        d_in=d_in,
        widths=cfg["model"]["widths"],
        bias=cfg["model"]["bias"],
        use_gates=use_gates
    )
    
    # Load state dict with strict=False to handle missing gates gracefully
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def parse_lambda_string(lam_str: str) -> float:
    """Parse lambda string like '1e-3', '0.001', '1e-4', '0' to float."""
    # Handle special cases
    if lam_str == '0' or lam_str == '0.0':
        return 0.0
    
    # Try direct float conversion first
    try:
        return float(lam_str)
    except ValueError:
        pass
    
    # Try replacing 'e' with 'E' for scientific notation
    try:
        return float(lam_str.replace('e', 'E'))
    except ValueError:
        pass
    
    # If all else fails, try to parse common patterns
    if lam_str.startswith('1e-') or lam_str.startswith('1E-'):
        try:
            exp = int(lam_str.split('-')[1])
            return 10.0 ** (-exp)
        except:
            pass
    
    raise ValueError(f"Could not parse lambda string: {lam_str}")


def find_experiment_combinations(output_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find all (n_train_dir, lambda_dir, algo) combinations in output directory.
    Returns list of (n_train_dir, lambda_dir, algo_name) tuples.
    """
    combinations = []
    output_path = Path(output_dir)
    
    # Find all subdirectories matching pattern n*_lam*
    for item in output_path.iterdir():
        if not item.is_dir():
            continue
        
        dir_name = item.name
        if not (dir_name.startswith('n') and '_lam' in dir_name):
            continue
        
        # Extract n_train and lambda from directory name
        # Format: n500_lam0.001 or n1000_lam1e-4
        parts = dir_name.split('_lam')
        if len(parts) != 2:
            continue
        
        n_train_dir = dir_name
        lambda_dir = dir_name
        
        # Look for algorithm subdirectories
        for algo_dir in item.iterdir():
            if not algo_dir.is_dir():
                continue
            
            algo_name = algo_dir.name
            # Check if model_final.pt exists
            model_path = algo_dir / "model_final.pt"
            if model_path.exists():
                combinations.append((n_train_dir, lambda_dir, algo_name))
    
    return combinations


def format_tag(n_train: int, lambda_val: float, algo: str) -> str:
    """Create a tag string for filenames."""
    # Format lambda nicely
    if lambda_val == 0.0:
        lam_str = "0.0"
    elif lambda_val >= 1.0:
        lam_str = f"{lambda_val:.1f}"
    elif lambda_val >= 0.1:
        lam_str = f"{lambda_val:.1f}"
    elif lambda_val >= 0.01:
        lam_str = f"{lambda_val:.2f}"
    elif lambda_val >= 0.001:
        lam_str = f"{lambda_val:.3f}"
    else:
        lam_str = f"{lambda_val:.4f}"
    
    return f"{algo}_n{n_train}_lam{lam_str}_final"


def generate_all_plots(
    output_dir: str,
    plots_dir: Optional[str] = None,
    max_samples_kernel: Optional[int] = 2000,
    max_samples_embed: Optional[int] = 2000,
    kernel_k: int = 48,
    mode: str = "routing_gain",
    device: Optional[str] = None,
):
    """
    Generate all path analysis plots for all models in the output directory.
    
    Args:
        output_dir: Path to experiment output directory (contains config.json)
        plots_dir: Directory to save plots (default: output_dir/plots/path_analysis_all)
        max_samples_kernel: Max samples for kernel computation
        max_samples_embed: Max samples for embedding computation
        kernel_k: Number of top eigenvalues to compute
        mode: Transmittance mode ("routing", "routing_gain", "routing_posdev")
        device: Device to use ("cuda", "cpu", or None for auto-detect)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    config_path = output_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found in {output_dir}")
    
    cfg = load_config(str(config_path))
    
    # Set up plots directory
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots in: {plots_dir}")
    print(f"Mode: {mode}")
    print(f"Max samples (kernel): {max_samples_kernel}")
    print(f"Max samples (embedding): {max_samples_embed}")
    print()
    
    # Find all experiment combinations
    combinations = find_experiment_combinations(str(output_path))
    
    if len(combinations) == 0:
        print(f"Warning: No models found in {output_dir}")
        print("Looking for directories matching pattern: n*_lam*/algorithm_name/model_final.pt")
        print("\nDirectory structure should be:")
        print("  output_dir/")
        print("    ├── config.json")
        print("    ├── n1000_lam1e-2/")
        print("    │   ├── alt_em_sgd/model_final.pt")
        print("    │   ├── sgd_relu/model_final.pt")
        print("    │   └── ...")
        print("    └── ...")
        return
    
    print(f"Found {len(combinations)} model combinations to process")
    print()
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    print(f"Using device: {device}")
    print()
    
    for idx, (n_train_dir, lambda_dir, algo_name) in enumerate(combinations, 1):
        # Skip sgd_relu models - they don't have gates and path analysis doesn't apply well
        if algo_name == "sgd_relu":
            print(f"  Skipping {algo_name} (no gates)")
            continue
        print(f"[{idx}/{len(combinations)}] Processing: {n_train_dir}/{algo_name}")
        
        # Parse n_train and lambda from directory name
        n_train_str = n_train_dir.split('_lam')[0][1:]  # Remove 'n' prefix
        lambda_str = n_train_dir.split('_lam')[1]
        
        try:
            n_train = int(n_train_str)
            lambda_val = parse_lambda_string(lambda_str)
        except ValueError as e:
            print(f"  Warning: Could not parse n_train or lambda from {n_train_dir}: {e}")
            continue
        
        # Path to model
        model_path = output_path / n_train_dir / algo_name / "model_final.pt"
        if not model_path.exists():
            print(f"  Warning: Model not found: {model_path}")
            continue
        
        # Load model
        try:
            model = load_model(str(model_path), cfg)
            model.to(device_obj)
            # Ensure all parameters are on the specified device
            for param in model.parameters():
                param.data = param.data.to(device_obj)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue
        
        # Build dataloaders
        try:
            train_loader, val_loader, test_loader = build_dataloaders_from_config(cfg, n_train, lambda_val)
        except Exception as e:
            print(f"  Error building dataloaders: {e}")
            continue
        
        # Create tag for filenames
        tag = format_tag(n_train, lambda_val, algo_name)
        algo_plots_dir = plots_dir / algo_name
        algo_plots_dir.mkdir(exist_ok=True)
        
        print(f"  Generating plots with tag: {tag}")
        
        # Initialize shared variables
        kern = None
        Epack = None
        
        # 1. Eigenvalue spectrum (path kernel)
        try:
            print("    - Computing path kernel eigenvalues...")
            # Ensure model is on the specified device
            model.to(device_obj)
            kern = compute_path_kernel_eigs(
                model,
                val_loader,
                device=device,  # Use specified device
                mode=mode,
                include_input=True,
                k=kernel_k,
                n_iter=30,
                block_size=1024,
                max_samples=max_samples_kernel,
                verbose=False,
            )
            spec_png = algo_plots_dir / f"eig_spectrum_{tag}.png"
            plot_eig_spectrum(kern["evals"], str(spec_png), title=f"Path-kernel spectrum [{tag}]")
        except Exception as e:
            print(f"    Warning: Eigenvalue spectrum failed: {e}")
        
        # 2. NN graph with top paths
        try:
            print("    - Generating NN graph with paths...")
            graph_png = algo_plots_dir / f"nn_graph_paths_{tag}.png"
            plot_nn_graph_with_paths(model, val_loader, str(graph_png), mode=mode, beam=24, top_k=3)
        except Exception as e:
            print(f"    Warning: NN graph failed: {e}")
        
        # 3. Path cleanliness
        try:
            print("    - Computing path cleanliness...")
            clean_png = algo_plots_dir / f"path_cleanliness_{tag}.png"
            plot_path_cleanliness(model, val_loader, str(clean_png), mode=mode, top_k=5)
        except Exception as e:
            print(f"    Warning: Path cleanliness failed: {e}")
        
        # 4. Embedding map
        try:
            print("    - Computing path embeddings...")
            if Epack is None:
                Epack = path_embedding(model, val_loader, device=device_obj, mode=mode, normalize=True, max_samples=max_samples_embed)
            emb_png = algo_plots_dir / f"path_embedding_{tag}.png"
            plot_embedding_map(Epack["E"], Epack["labels"], str(emb_png), title=f"Path-embedding [{tag}]")
        except Exception as e:
            print(f"    Warning: Embedding map failed: {e}")
        
        # 5. Flow centrality heatmap
        try:
            print("    - Computing flow centrality...")
            central_png = algo_plots_dir / f"flow_centrality_{tag}.png"
            flow_centrality_heatmap(model, val_loader, str(central_png), mode=mode)
        except Exception as e:
            print(f"    Warning: Flow centrality failed: {e}")
        
        # 6. Ablation waterfall (optional, can be slow)
        try:
            print("    - Computing ablation waterfall...")
            ablation_png = algo_plots_dir / f"ablation_waterfall_{tag}.png"
            ablation_waterfall(
                model,
                test_loader,
                str(ablation_png),
                top_units_per_layer=8,
                mode=mode,
            )
        except Exception as e:
            print(f"    Warning: Ablation waterfall failed: {e}")
        
        # 7. Path-Shapley bars (mutual information analysis)
        try:
            if not HAVE_SKLEARN:
                print("    - Skipping path-Shapley (scikit-learn not available)...")
            else:
                print("    - Computing path-Shapley (MI analysis)...")
                # Use eigenvectors from path kernel as circuit scores
                if kern is not None and "evecs" in kern:
                    if Epack is None:
                        Epack = path_embedding(model, val_loader, device=device_obj, mode=mode, normalize=True, max_samples=max_samples_embed)
                    evecs = kern["evecs"].detach().cpu().numpy()  # (P, k)
                    if Epack["labels"] is not None or Epack["y"] is not None:
                        y_data = Epack["labels"].numpy() if Epack["labels"] is not None else Epack["y"].numpy()
                        # Use top eigenvectors as circuit scores (limit to reasonable number for MI)
                        top_m = min(24, evecs.shape[1])  # Limit to 24 for reasonable computation
                        scores = evecs[:, :top_m]
                        shapley_png = algo_plots_dir / f"path_shapley_{tag}.png"
                        plot_path_shapley_bars(
                            scores,
                            y_data,
                            str(shapley_png),
                            title=f"Path-Shapley (MI proxy) [{tag}]"
                        )
        except Exception as e:
            print(f"    Warning: Path-Shapley failed: {e}")
        
        # 8. Circuit overlap matrix
        try:
            print("    - Computing circuit overlap matrix...")
            # Cluster embeddings and use centroids as prototypes
            if Epack is None:
                Epack = path_embedding(model, val_loader, device=device_obj, mode=mode, normalize=True, max_samples=max_samples_embed)
            E = Epack["E"].numpy()
            if E.shape[0] > 8:  # Need enough samples to cluster
                if not HAVE_SKLEARN:
                    print("    Warning: sklearn not available; skipping circuit overlap.")
                else:
                    k_clusters = min(8, E.shape[0] // 10, E.shape[0] - 1)  # Adaptive number of clusters
                    if k_clusters >= 2:
                        km = KMeans(n_clusters=k_clusters, random_state=1, n_init="auto")
                        km.fit(E)
                        prototypes = km.cluster_centers_  # (k, D)
                        overlap_png = algo_plots_dir / f"circuit_overlap_{tag}.png"
                        circuit_overlap_matrix(
                            prototypes,
                            str(overlap_png),
                            title=f"Circuit Overlap (cosine) [{tag}]"
                        )
        except Exception as e:
            print(f"    Warning: Circuit overlap failed: {e}")
        
        print(f"  ✓ Completed {tag}")
        print()
    
    print(f"All plots generated in: {plots_dir}")


def load_training_history(csv_path: str) -> Dict[str, List]:
    """Load training history CSV and return as dictionary of lists."""
    data = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value == '' or value is None:
                    data[key].append('')  # Keep empty values to maintain alignment
                    continue
                try:
                    # First try parsing as list (for top10_eigenvalues, etc.)
                    if value.strip().startswith('[') and value.strip().endswith(']'):
                        try:
                            parsed = ast.literal_eval(value)
                            data[key].append(parsed)
                            continue
                        except:
                            pass
                    
                    # Try to parse as float
                    if '.' in value or 'e' in value.lower() or 'E' in value:
                        try:
                            data[key].append(float(value))
                        except ValueError:
                            data[key].append(value)
                    else:
                        try:
                            data[key].append(int(value))
                        except ValueError:
                            data[key].append(value)
                except Exception:
                    data[key].append(value)
    return dict(data)


def plot_metric_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    metric_name: str = "path_kernel_effective_rank",
    algo_name: str = "alt_em_sgd",
):
    """
    Create a grid plot for a specific metric across all n_train and lambda combinations.
    
    Grid layout: n_train (rows) vs lambda (columns)
    Each subplot shows: epoch/cycle vs metric value
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all combinations
    combinations = find_experiment_combinations(str(output_path))
    
    # Filter by algorithm
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    # Parse n_train and lambda values
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    # Determine if cycle-based or epoch-based
    sample_path = output_path / algo_combinations[0][0] / algo_name / "training_history.csv"
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            header = f.readline().strip()
            is_cycle_based = 'cycle' in header
            time_col = 'cycle' if is_cycle_based else 'epoch'
    else:
        is_cycle_based = 'alt_em' in algo_name
        time_col = 'cycle' if is_cycle_based else 'epoch'
    
    # Collect data for each combination
    grid_data = {}
    for n_train_dir, lambda_dir, algo in algo_combinations:
        n_train = int(n_train_dir.split('_lam')[0][1:])
        lambda_val = parse_lambda_string(n_train_dir.split('_lam')[1])
        
        csv_path = output_path / n_train_dir / algo / "training_history.csv"
        if not csv_path.exists():
            continue
        
        try:
            history = load_training_history(str(csv_path))
            if metric_name not in history:
                print(f"  Debug: Metric '{metric_name}' not found in {csv_path.name}")
                continue
            if time_col not in history:
                print(f"  Debug: Time column '{time_col}' not found in {csv_path.name}")
                continue
            
            # Filter out empty values
            times = []
            values = []
            time_list = history[time_col]
            metric_list = history[metric_name]
            
            if len(time_list) != len(metric_list):
                print(f"  Warning: Mismatched lengths in {csv_path.name}: {len(time_list)} cycles vs {len(metric_list)} metric values")
            
            for t, v in zip(time_list, metric_list):
                # Skip empty values (empty strings, None, or empty lists)
                if v == '' or v is None:
                    continue
                if isinstance(v, list) and len(v) == 0:
                    continue
                try:
                    # Skip list values for scalar metrics (unless it's top10_eigenvalues which is handled separately)
                    if isinstance(v, list) and metric_name != "path_kernel_top10_eigenvalues":
                        continue
                    if isinstance(v, str) and v.strip().startswith('['):
                        # This is a string representation of a list, skip for scalar metrics
                        if metric_name != "path_kernel_top10_eigenvalues":
                            continue
                    times.append(float(t))
                    values.append(float(v))
                except (ValueError, TypeError) as e:
                    # Skip values that can't be converted to float
                    continue
            
            if len(times) > 0:
                grid_data[(n_train, lambda_val)] = (times, values)
                print(f"  Loaded {len(times)} data points for {metric_name} from {csv_path.name}")
            else:
                print(f"  Warning: No valid data points found for {metric_name} in {csv_path.name}")
        except Exception as e:
            print(f"Warning: Failed to load {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(grid_data) == 0:
        print(f"No data found for metric {metric_name} and algorithm {algo_name}")
        return
    
    # Create grid plot
    n_rows = len(n_train_values)
    n_cols = len(lambda_values)
    
    if n_rows == 0 or n_cols == 0:
        print(f"Warning: No data to plot for {metric_name} and {algo_name}")
        return
    
    # Ensure we have valid dimensions
    if n_rows == 0 or n_cols == 0:
        print(f"Warning: Invalid grid dimensions: {n_rows} rows x {n_cols} cols")
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=False)
    
    # Handle different subplot configurations - ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes_2d = [[axes]]
    elif n_rows == 1:
        # axes is a 1D array of length n_cols
        if hasattr(axes, '__len__') and len(axes) == n_cols:
            axes_2d = [list(axes) if isinstance(axes, np.ndarray) else axes]
        else:
            axes_2d = [[axes]]
    elif n_cols == 1:
        # axes is a 1D array of length n_rows
        if hasattr(axes, '__len__') and len(axes) == n_rows:
            axes_2d = [[ax] for ax in axes]
        else:
            axes_2d = [[axes]]
    else:
        # Already 2D array from matplotlib
        if isinstance(axes, np.ndarray):
            axes_2d = axes.tolist()
        else:
            axes_2d = axes
    
    # Format metric name for title
    metric_display = metric_name.replace('path_kernel_', '').replace('_', ' ').title()
    
    for i, n_train in enumerate(n_train_values):
        for j, lambda_val in enumerate(lambda_values):
            # Safety check for axes bounds
            if i >= len(axes_2d) or j >= len(axes_2d[i]):
                print(f"Warning: Index out of range for axes_2d[{i}][{j}], shape is {len(axes_2d)}x{len(axes_2d[0]) if axes_2d else 0}")
                continue
            ax = axes_2d[i][j]
            
            # Format lambda for title
            if lambda_val == 0.0:
                lam_str = "0"
            elif lambda_val >= 1.0:
                lam_str = f"{lambda_val:.1f}"
            elif lambda_val >= 0.1:
                lam_str = f"{lambda_val:.1f}"
            elif lambda_val >= 0.01:
                lam_str = f"{lambda_val:.2f}"
            elif lambda_val >= 0.001:
                lam_str = f"{lambda_val:.3f}"
            else:
                lam_str = f"{lambda_val:.0e}"
            
            if (n_train, lambda_val) in grid_data:
                times, values = grid_data[(n_train, lambda_val)]
                ax.plot(times, values, marker='o', markersize=3, linewidth=1.5)
                ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
            
            if i == n_rows - 1:
                ax.set_xlabel('Cycle' if is_cycle_based else 'Epoch', fontsize=9)
            if j == 0:
                ax.set_ylabel(metric_display, fontsize=9)
    
    plt.suptitle(f"{metric_display} - {algo_name}", fontsize=12, y=0.995)
    plt.tight_layout()
    
    # Save plot
    metric_safe = metric_name.replace('path_kernel_', '').replace('_', '-')
    out_path = plots_dir / f"metric_grid_{metric_safe}_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metric grid: {out_path}")


def plot_top10_eigenvalues_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
):
    """
    Create a grid plot for top 10 eigenvalues across all n_train and lambda combinations.
    Each eigenvalue is plotted with a different color.
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all combinations
    combinations = find_experiment_combinations(str(output_path))
    
    # Filter by algorithm
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    # Parse n_train and lambda values
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    # Determine if cycle-based or epoch-based
    sample_path = output_path / algo_combinations[0][0] / algo_name / "training_history.csv"
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            header = f.readline().strip()
            is_cycle_based = 'cycle' in header
            time_col = 'cycle' if is_cycle_based else 'epoch'
    else:
        is_cycle_based = 'alt_em' in algo_name
        time_col = 'cycle' if is_cycle_based else 'epoch'
    
    # Collect data for each combination
    grid_data = {}
    for n_train_dir, lambda_dir, algo in algo_combinations:
        n_train = int(n_train_dir.split('_lam')[0][1:])
        lambda_val = parse_lambda_string(n_train_dir.split('_lam')[1])
        
        csv_path = output_path / n_train_dir / algo / "training_history.csv"
        if not csv_path.exists():
            continue
        
        try:
            history = load_training_history(str(csv_path))
            if "path_kernel_top10_eigenvalues" not in history or time_col not in history:
                continue
            
            # Collect times and eigenvalue lists
            times = []
            evals_list = []
            for t, evals in zip(history[time_col], history["path_kernel_top10_eigenvalues"]):
                if evals != '' and evals is not None:
                    try:
                        if isinstance(evals, str):
                            evals = ast.literal_eval(evals)
                        if isinstance(evals, list) and len(evals) > 0:
                            times.append(float(t))
                            evals_list.append([float(e) for e in evals[:10]])  # Ensure max 10
                    except (ValueError, TypeError) as e:
                        continue
            
            if len(times) > 0:
                grid_data[(n_train, lambda_val)] = (times, evals_list)
        except Exception as e:
            print(f"Warning: Failed to load {csv_path}: {e}")
            continue
    
    if len(grid_data) == 0:
        print(f"No data found for top10 eigenvalues and algorithm {algo_name}")
        return
    
    # Create grid plot
    n_rows = len(n_train_values)
    n_cols = len(lambda_values)
    
    if n_rows == 0 or n_cols == 0:
        print(f"Warning: No data to plot for top10 eigenvalues and {algo_name}")
        return
    
    # Ensure we have valid dimensions
    if n_rows == 0 or n_cols == 0:
        print(f"Warning: Invalid grid dimensions: {n_rows} rows x {n_cols} cols")
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=False)
    
    # Handle different subplot configurations - ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes_2d = [[axes]]
    elif n_rows == 1:
        # axes is a 1D array of length n_cols
        if hasattr(axes, '__len__') and len(axes) == n_cols:
            axes_2d = [list(axes) if isinstance(axes, np.ndarray) else axes]
        else:
            axes_2d = [[axes]]
    elif n_cols == 1:
        # axes is a 1D array of length n_rows
        if hasattr(axes, '__len__') and len(axes) == n_rows:
            axes_2d = [[ax] for ax in axes]
        else:
            axes_2d = [[axes]]
    else:
        # Already 2D array from matplotlib
        if isinstance(axes, np.ndarray):
            axes_2d = axes.tolist()
        else:
            axes_2d = axes
    
    # Color map for 10 eigenvalues
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, n_train in enumerate(n_train_values):
        for j, lambda_val in enumerate(lambda_values):
            # Safety check for axes bounds
            if i >= len(axes_2d) or j >= len(axes_2d[i]):
                print(f"Warning: Index out of range for axes_2d[{i}][{j}], shape is {len(axes_2d)}x{len(axes_2d[0]) if axes_2d else 0}")
                continue
            ax = axes_2d[i][j]
            
            # Format lambda for title
            if lambda_val == 0.0:
                lam_str = "0"
            elif lambda_val >= 1.0:
                lam_str = f"{lambda_val:.1f}"
            elif lambda_val >= 0.1:
                lam_str = f"{lambda_val:.1f}"
            elif lambda_val >= 0.01:
                lam_str = f"{lambda_val:.2f}"
            elif lambda_val >= 0.001:
                lam_str = f"{lambda_val:.3f}"
            else:
                lam_str = f"{lambda_val:.0e}"
            
            if (n_train, lambda_val) in grid_data:
                times, evals_list = grid_data[(n_train, lambda_val)]
                # Determine how many eigenvalues we actually have
                max_eig_count = max(len(ev) for ev in evals_list) if evals_list else 0
                num_to_plot = min(10, max_eig_count)
                
                # Plot each eigenvalue with different color
                for eig_idx in range(num_to_plot):
                    evals_for_eig = [ev[eig_idx] if len(ev) > eig_idx else 0.0 for ev in evals_list]
                    # Filter out zero values that might be padding
                    if any(v > 0 for v in evals_for_eig):
                        ax.plot(times, evals_for_eig, marker='o', markersize=2, linewidth=1.0,
                               color=colors[eig_idx], label=f'λ{eig_idx+1}' if i == 0 and j == 0 else '',
                               alpha=0.8)
                
                ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                ax.grid(True, alpha=0.3, which='both')
                ax.set_yscale('log')  # Log scale for eigenvalues
            else:
                ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
            
            if i == n_rows - 1:
                ax.set_xlabel('Cycle' if is_cycle_based else 'Epoch', fontsize=9)
            if j == 0:
                ax.set_ylabel('Eigenvalue', fontsize=9)
    
    # Add legend to first subplot
    if len(grid_data) > 0:
        axes_2d[0][0].legend(loc='upper right', fontsize=7, ncol=2)
    
    plt.suptitle(f"Top 10 Eigenvalues - {algo_name}", fontsize=12, y=0.995)
    plt.tight_layout()
    
    # Save plot
    out_path = plots_dir / f"metric_grid_top10-eigenvalues_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved top10 eigenvalues grid: {out_path}")


def generate_metric_grids(
    output_dir: str,
    plots_dir: Optional[str] = None,
):
    """
    Generate grid plots for all path kernel metrics across all algorithms.
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all algorithms
    combinations = find_experiment_combinations(str(output_path))
    algorithms = sorted(set(algo for _, _, algo in combinations))
    
    # Define all path kernel metrics to plot
    metrics = [
        "path_kernel_effective_rank",
        "path_kernel_top_eigenvalue",
        "path_kernel_eigenvalue_sum",
        "path_kernel_variance_explained_train",
        "path_kernel_variance_explained_train_top10",
        "path_kernel_variance_explained_test",
        "path_kernel_variance_explained_test_top10",
    ]
    
    print(f"Generating metric grid plots for {len(metrics)} metrics across {len(algorithms)} algorithms...")
    print()
    
    for algo in algorithms:
        print(f"Processing algorithm: {algo}")
        for metric in metrics:
            try:
                plot_metric_grid(output_dir, plots_dir, metric, algo)
            except Exception as e:
                print(f"  Warning: Failed to plot {metric} for {algo}: {e}")
        
        # Also plot top 10 eigenvalues
        try:
            plot_top10_eigenvalues_grid(output_dir, plots_dir, algo)
        except Exception as e:
            print(f"  Warning: Failed to plot top10 eigenvalues for {algo}: {e}")
        print()
    
    print(f"All metric grid plots generated in: {plots_dir}")


def plot_eigenvalue_spectrum_initial_vs_final(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
    mode: str = "routing_gain",
    device: Optional[str] = None,
    max_samples: int = 5000,
    kernel_k: int = 48,
):
    """
    Create a grid plot showing eigenvalue spectrum (eigenvalue vs rank) for initial and final models.
    Each subplot shows both initial and final on the same plot, color-coded.
    Grid layout: n_train (rows) vs lambda (columns)
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all combinations
    combinations = find_experiment_combinations(str(output_path))
    
    # Filter by algorithm
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    # Parse n_train and lambda values
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    # Load config
    config_path = output_path / "config.json"
    if not config_path.exists():
        print(f"Warning: config.json not found in {output_dir}")
        return
    cfg = load_config(str(config_path))
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    # Create grid plot
    n_rows = len(n_train_values)
    n_cols = len(lambda_values)
    
    if n_rows == 0 or n_cols == 0:
        print(f"Warning: No data to plot for {algo_name}")
        return
    
    # Ensure we have valid dimensions
    if n_rows == 0 or n_cols == 0:
        print(f"Warning: Invalid grid dimensions: {n_rows} rows x {n_cols} cols")
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=True)
    
    # Handle different subplot configurations - ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes_2d = [[axes]]
    elif n_rows == 1:
        # axes is a 1D array of length n_cols
        if hasattr(axes, '__len__') and len(axes) == n_cols:
            axes_2d = [list(axes) if isinstance(axes, np.ndarray) else axes]
        else:
            axes_2d = [[axes]]
    elif n_cols == 1:
        # axes is a 1D array of length n_rows
        if hasattr(axes, '__len__') and len(axes) == n_rows:
            axes_2d = [[ax] for ax in axes]
        else:
            axes_2d = [[axes]]
    else:
        # Already 2D array from matplotlib
        if isinstance(axes, np.ndarray):
            axes_2d = axes.tolist()
        else:
            axes_2d = axes
    
    for i, n_train in enumerate(n_train_values):
        for j, lambda_val in enumerate(lambda_values):
            # Safety check for axes bounds
            if i >= len(axes_2d) or j >= len(axes_2d[i]):
                print(f"Warning: Index out of range for axes_2d[{i}][{j}], shape is {len(axes_2d)}x{len(axes_2d[0]) if axes_2d else 0}")
                continue
            ax = axes_2d[i][j]
            
            # Format lambda for title
            if lambda_val == 0.0:
                lam_str = "0"
            elif lambda_val >= 1.0:
                lam_str = f"{lambda_val:.1f}"
            elif lambda_val >= 0.1:
                lam_str = f"{lambda_val:.1f}"
            elif lambda_val >= 0.01:
                lam_str = f"{lambda_val:.2f}"
            elif lambda_val >= 0.001:
                lam_str = f"{lambda_val:.3f}"
            else:
                lam_str = f"{lambda_val:.0e}"
            
            # Find the directory - need to match the actual directory name format
            # Search for matching directory
            n_train_dir = None
            for n_dir, _, _ in algo_combinations:
                n_parsed = int(n_dir.split('_lam')[0][1:])
                lam_parsed = parse_lambda_string(n_dir.split('_lam')[1])
                if n_parsed == n_train and abs(lam_parsed - lambda_val) < 1e-10:
                    n_train_dir = n_dir
                    break
            
            if n_train_dir is None:
                ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                ax.text(0.5, 0.5, 'No models', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            model_init_path = output_path / n_train_dir / algo_name / "model_init.pt"
            model_final_path = output_path / n_train_dir / algo_name / "model_final.pt"
            
            if not model_init_path.exists() or not model_final_path.exists():
                ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                ax.text(0.5, 0.5, 'No models', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            try:
                # Build dataloaders
                train_loader, val_loader, test_loader = build_dataloaders_from_config(cfg, n_train, lambda_val)
                
                # Load initial model
                model_init = load_model(str(model_init_path), cfg)
                model_init.to(device_obj)
                for param in model_init.parameters():
                    param.data = param.data.to(device_obj)
                
                # Load final model
                model_final = load_model(str(model_final_path), cfg)
                model_final.to(device_obj)
                for param in model_final.parameters():
                    param.data = param.data.to(device_obj)
                
                # Compute eigenvalues for initial model
                try:
                    kern_init = compute_path_kernel_eigs(
                        model_init, val_loader, device=device, mode=mode,
                        include_input=True, k=kernel_k, n_iter=30, block_size=1024,
                        max_samples=max_samples, verbose=False
                    )
                    evals_init = kern_init["evals"].detach().cpu().numpy()
                    # Normalize by first eigenvalue
                    if len(evals_init) > 0 and evals_init[0] != 0:
                        evals_init_norm = evals_init / evals_init[0]
                    else:
                        evals_init_norm = evals_init
                except Exception as e:
                    print(f"  Warning: Failed to compute initial eigenvalues for n={n_train}, λ={lam_str}: {e}")
                    evals_init_norm = np.array([])
                
                # Compute eigenvalues for final model
                try:
                    kern_final = compute_path_kernel_eigs(
                        model_final, val_loader, device=device, mode=mode,
                        include_input=True, k=kernel_k, n_iter=30, block_size=1024,
                        max_samples=max_samples, verbose=False
                    )
                    evals_final = kern_final["evals"].detach().cpu().numpy()
                    # Normalize by first eigenvalue
                    if len(evals_final) > 0 and evals_final[0] != 0:
                        evals_final_norm = evals_final / evals_final[0]
                    else:
                        evals_final_norm = evals_final
                except Exception as e:
                    print(f"  Warning: Failed to compute final eigenvalues for n={n_train}, λ={lam_str}: {e}")
                    evals_final_norm = np.array([])
                
                # Plot both spectra
                if len(evals_init_norm) > 0:
                    ranks_init = np.arange(1, len(evals_init_norm) + 1)
                    ax.plot(ranks_init, evals_init_norm, marker='o', markersize=3, linewidth=1.5,
                           color='blue', label='Initial', alpha=0.7)
                
                if len(evals_final_norm) > 0:
                    ranks_final = np.arange(1, len(evals_final_norm) + 1)
                    ax.plot(ranks_final, evals_final_norm, marker='s', markersize=3, linewidth=1.5,
                           color='red', label='Final', alpha=0.7)
                
                ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, which='both')
                
                # Add legend to first subplot
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize=7)
                
            except Exception as e:
                print(f"  Warning: Failed to plot for n={n_train}, λ={lam_str}: {e}")
                ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
            
            if i == n_rows - 1:
                ax.set_xlabel('Rank', fontsize=9)
            if j == 0:
                ax.set_ylabel('Eigenvalue / $\\lambda_1$', fontsize=9)
    
    plt.suptitle(f"Eigenvalue Spectrum: Initial vs Final - {algo_name}", fontsize=12, y=0.995)
    plt.tight_layout()
    
    # Save plot
    out_path = plots_dir / f"eigenvalue_spectrum_init_vs_final_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved eigenvalue spectrum (init vs final): {out_path}")


def generate_eigenvalue_spectrum_plots(
    output_dir: str,
    plots_dir: Optional[str] = None,
    mode: str = "routing_gain",
    device: Optional[str] = None,
):
    """
    Generate eigenvalue spectrum plots (initial vs final) for all algorithms.
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all algorithms
    combinations = find_experiment_combinations(str(output_path))
    algorithms = sorted(set(algo for _, _, algo in combinations))
    
    print(f"Generating eigenvalue spectrum plots (initial vs final) for {len(algorithms)} algorithms...")
    print()
    
    for algo in algorithms:
        print(f"Processing algorithm: {algo}")
        try:
            plot_eigenvalue_spectrum_initial_vs_final(
                output_dir, plots_dir, algo, mode=mode, device=device
            )
        except Exception as e:
            print(f"  Warning: Failed to plot eigenvalue spectrum for {algo}: {e}")
        print()
    
    print(f"All eigenvalue spectrum plots generated in: {plots_dir}")


def load_checkpoint_metrics(json_path: str) -> List[Dict]:
    """Load checkpoint metrics JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint metrics from {json_path}: {e}")
        return []


def plot_path_shapley_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
):
    """
    Create grid plots for Path-Shapley metrics (MI main effects and synergy).
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all combinations
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    # Parse n_train and lambda values
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    # Determine if cycle-based or epoch-based
    sample_path = output_path / algo_combinations[0][0] / algo_name / "training_history.csv"
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            header = f.readline().strip()
            is_cycle_based = 'cycle' in header
            time_col = 'cycle' if is_cycle_based else 'epoch'
    else:
        is_cycle_based = 'alt_em' in algo_name
        time_col = 'cycle' if is_cycle_based else 'epoch'
    
    # Metrics to plot
    metrics = [
        ("path_shapley_mi_main_mean", "Path-Shapley MI Main (Mean)"),
        ("path_shapley_synergy_mean", "Path-Shapley Synergy (Mean)"),
    ]
    
    for metric_key, metric_display in metrics:
        grid_data = {}
        for n_train_dir, lambda_dir, algo in algo_combinations:
            n_train = int(n_train_dir.split('_lam')[0][1:])
            lambda_val = parse_lambda_string(n_train_dir.split('_lam')[1])
            
            json_path = output_path / n_train_dir / algo / "checkpoint_metrics.json"
            if not json_path.exists():
                continue
            
            try:
                checkpoint_data = load_checkpoint_metrics(str(json_path))
                if not checkpoint_data:
                    continue
                
                times = []
                values = []
                for entry in checkpoint_data:
                    if metric_key in entry and entry[metric_key] is not None:
                        t = entry.get(time_col)
                        if t is not None:
                            times.append(float(t))
                            values.append(float(entry[metric_key]))
                
                if len(times) > 0:
                    grid_data[(n_train, lambda_val)] = (times, values)
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")
                continue
        
        if len(grid_data) == 0:
            print(f"No data found for {metric_key} and algorithm {algo_name}")
            continue
        
        # Create grid plot
        n_rows = len(n_train_values)
        n_cols = len(lambda_values)
        
        if n_rows == 0 or n_cols == 0:
            continue
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=False)
        
        # Handle axes
        if n_rows == 1 and n_cols == 1:
            axes_2d = [[axes]]
        elif n_rows == 1:
            axes_2d = [list(axes) if isinstance(axes, np.ndarray) else axes] if hasattr(axes, '__len__') and len(axes) == n_cols else [[axes]]
        elif n_cols == 1:
            axes_2d = [[ax] for ax in axes] if hasattr(axes, '__len__') and len(axes) == n_rows else [[axes]]
        else:
            axes_2d = axes.tolist() if isinstance(axes, np.ndarray) else axes
        
        for i, n_train in enumerate(n_train_values):
            for j, lambda_val in enumerate(lambda_values):
                ax = axes_2d[i][j]
                
                # Format lambda
                if lambda_val == 0.0:
                    lam_str = "0"
                elif lambda_val >= 1.0:
                    lam_str = f"{lambda_val:.1f}"
                elif lambda_val >= 0.1:
                    lam_str = f"{lambda_val:.1f}"
                elif lambda_val >= 0.01:
                    lam_str = f"{lambda_val:.2f}"
                elif lambda_val >= 0.001:
                    lam_str = f"{lambda_val:.3f}"
                else:
                    lam_str = f"{lambda_val:.0e}"
                
                if (n_train, lambda_val) in grid_data:
                    times, values = grid_data[(n_train, lambda_val)]
                    ax.plot(times, values, marker='o', markersize=3, linewidth=1.5)
                    ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                
                if i == n_rows - 1:
                    ax.set_xlabel('Cycle' if is_cycle_based else 'Epoch', fontsize=9)
                if j == 0:
                    ax.set_ylabel(metric_display, fontsize=9)
        
        plt.suptitle(f"{metric_display} - {algo_name}", fontsize=12, y=0.995)
        plt.tight_layout()
        
        metric_safe = metric_key.replace('path_shapley_', '').replace('_', '-')
        out_path = plots_dir / f"metric_grid_{metric_safe}_{algo_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved Path-Shapley grid: {out_path}")


def plot_centroid_drift_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
):
    """
    Create grid plots for Centroid Drift metrics.
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all combinations
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    # Parse n_train and lambda values
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    # Determine if cycle-based or epoch-based
    sample_path = output_path / algo_combinations[0][0] / algo_name / "training_history.csv"
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            header = f.readline().strip()
            is_cycle_based = 'cycle' in header
            time_col = 'cycle' if is_cycle_based else 'epoch'
    else:
        is_cycle_based = 'alt_em' in algo_name
        time_col = 'cycle' if is_cycle_based else 'epoch'
    
    # Metrics to plot
    metrics = [
        ("centroid_drift", "Centroid Drift"),
        ("cluster_radius", "Cluster Radius"),
    ]
    
    for metric_key, metric_display in metrics:
        grid_data = {}
        for n_train_dir, lambda_dir, algo in algo_combinations:
            n_train = int(n_train_dir.split('_lam')[0][1:])
            lambda_val = parse_lambda_string(n_train_dir.split('_lam')[1])
            
            json_path = output_path / n_train_dir / algo / "checkpoint_metrics.json"
            if not json_path.exists():
                continue
            
            try:
                checkpoint_data = load_checkpoint_metrics(str(json_path))
                if not checkpoint_data:
                    continue
                
                # Collect all timepoints and values
                all_times = []
                all_values = []
                for entry in checkpoint_data:
                    if metric_key in entry and entry[metric_key] is not None:
                        t = entry.get(time_col)
                        if t is not None:
                            # metric_key might be a list (centroid_drift, cluster_radius)
                            vals = entry[metric_key]
                            if isinstance(vals, list):
                                # Plot each value in the list
                                for idx, v in enumerate(vals):
                                    if v is not None:
                                        all_times.append(float(t) + idx * 0.1)  # Slight offset for visualization
                                        all_values.append(float(v))
                            else:
                                all_times.append(float(t))
                                all_values.append(float(vals))
                
                if len(all_times) > 0:
                    grid_data[(n_train, lambda_val)] = (all_times, all_values)
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")
                continue
        
        if len(grid_data) == 0:
            print(f"No data found for {metric_key} and algorithm {algo_name}")
            continue
        
        # Create grid plot
        n_rows = len(n_train_values)
        n_cols = len(lambda_values)
        
        if n_rows == 0 or n_cols == 0:
            continue
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=False)
        
        # Handle axes
        if n_rows == 1 and n_cols == 1:
            axes_2d = [[axes]]
        elif n_rows == 1:
            axes_2d = [list(axes) if isinstance(axes, np.ndarray) else axes] if hasattr(axes, '__len__') and len(axes) == n_cols else [[axes]]
        elif n_cols == 1:
            axes_2d = [[ax] for ax in axes] if hasattr(axes, '__len__') and len(axes) == n_rows else [[axes]]
        else:
            axes_2d = axes.tolist() if isinstance(axes, np.ndarray) else axes
        
        for i, n_train in enumerate(n_train_values):
            for j, lambda_val in enumerate(lambda_values):
                ax = axes_2d[i][j]
                
                # Format lambda
                if lambda_val == 0.0:
                    lam_str = "0"
                elif lambda_val >= 1.0:
                    lam_str = f"{lambda_val:.1f}"
                elif lambda_val >= 0.1:
                    lam_str = f"{lambda_val:.1f}"
                elif lambda_val >= 0.01:
                    lam_str = f"{lambda_val:.2f}"
                elif lambda_val >= 0.001:
                    lam_str = f"{lambda_val:.3f}"
                else:
                    lam_str = f"{lambda_val:.0e}"
                
                if (n_train, lambda_val) in grid_data:
                    times, values = grid_data[(n_train, lambda_val)]
                    ax.plot(times, values, marker='o', markersize=2, linewidth=1.0, alpha=0.7)
                    ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_title(f"n={n_train}, λ={lam_str}", fontsize=9)
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                
                if i == n_rows - 1:
                    ax.set_xlabel('Cycle' if is_cycle_based else 'Epoch', fontsize=9)
                if j == 0:
                    ax.set_ylabel(metric_display, fontsize=9)
        
        plt.suptitle(f"{metric_display} - {algo_name}", fontsize=12, y=0.995)
        plt.tight_layout()
        
        metric_safe = metric_key.replace('_', '-')
        out_path = plots_dir / f"metric_grid_{metric_safe}_{algo_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved Centroid Drift grid: {out_path}")


def plot_lineage_sankey_from_checkpoints(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
):
    """
    Plot Lineage Sankey diagrams from checkpoint metrics.
    This loads the flow data and cluster sizes from checkpoint_metrics.json.
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all combinations
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    # For each combination, load checkpoint metrics and plot Sankey
    for n_train_dir, lambda_dir, algo in algo_combinations:
        n_train = int(n_train_dir.split('_lam')[0][1:])
        lambda_val = parse_lambda_string(n_train_dir.split('_lam')[1])
        
        json_path = output_path / n_train_dir / algo / "checkpoint_metrics.json"
        if not json_path.exists():
            continue
        
        try:
            checkpoint_data = load_checkpoint_metrics(str(json_path))
            if not checkpoint_data:
                continue
            
            # Extract lineage data
            flows_list = []
            cluster_sizes_list = []
            for entry in checkpoint_data:
                if "lineage_flows" in entry and entry["lineage_flows"]:
                    flows_list.append(entry["lineage_flows"])
                if "lineage_cluster_sizes" in entry and entry["lineage_cluster_sizes"]:
                    cluster_sizes_list.append(entry["lineage_cluster_sizes"])
            
            if len(flows_list) == 0 or len(cluster_sizes_list) == 0:
                continue
            
            # Use the plotting function from path_analysis
            try:
                from src.analysis.path_analysis import plot_lineage_sankey
                
                # Convert cluster sizes to embeddings format (dummy embeddings based on sizes)
                # This is a workaround - ideally we'd have the actual embeddings
                # For now, we'll skip this plot and note that it needs actual embeddings
                print(f"  Note: Lineage Sankey requires actual embeddings, skipping for {n_train_dir}/{algo}")
                continue
            except Exception as e:
                print(f"Warning: Failed to plot Lineage Sankey for {n_train_dir}/{algo}: {e}")
                continue
        except Exception as e:
            print(f"Warning: Failed to process {json_path}: {e}")
            continue


def generate_checkpoint_metric_plots(
    output_dir: str,
    plots_dir: Optional[str] = None,
):
    """
    Generate plots for checkpoint-based metrics (Path-Shapley, Centroid Drift, Lineage Sankey).
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all algorithms
    combinations = find_experiment_combinations(str(output_path))
    algorithms = sorted(set(algo for _, _, algo in combinations))
    
    print(f"Generating checkpoint metric plots for {len(algorithms)} algorithms...")
    print()
    
    for algo in algorithms:
        print(f"Processing algorithm: {algo}")
        try:
            plot_path_shapley_grid(output_dir, plots_dir, algo)
        except Exception as e:
            print(f"  Warning: Failed to plot Path-Shapley for {algo}: {e}")
        
        try:
            plot_centroid_drift_grid(output_dir, plots_dir, algo)
        except Exception as e:
            print(f"  Warning: Failed to plot Centroid Drift for {algo}: {e}")
        
        try:
            plot_lineage_sankey_from_checkpoints(output_dir, plots_dir, algo)
        except Exception as e:
            print(f"  Warning: Failed to plot Lineage Sankey for {algo}: {e}")
        print()
    
    print(f"All checkpoint metric plots generated in: {plots_dir}")


def format_lambda_str(lambda_val: float) -> str:
    """Format lambda value as string for titles."""
    if lambda_val == 0.0:
        return "0"
    elif lambda_val >= 1.0:
        return f"{lambda_val:.1f}"
    elif lambda_val >= 0.1:
        return f"{lambda_val:.1f}"
    elif lambda_val >= 0.01:
        return f"{lambda_val:.2f}"
    elif lambda_val >= 0.001:
        return f"{lambda_val:.3f}"
    else:
        return f"{lambda_val:.0e}"


def model_has_gates(model) -> bool:
    """Check if model has gates (sgd_relu models don't have gates)."""
    return hasattr(model, 'use_gates') and model.use_gates and model.gates is not None


def add_grid_row_column_labels(fig, n_rows: int, n_cols: int, lambda_values: List[float], n_train_values: List[int]):
    """Add row (n_train) and column (lambda) labels to a grid plot figure.
    
    Note: x-axis (columns) = lambda, y-axis (rows) = n_train
    """
    # Add row labels (n_train) on the left
    for i, n_train in enumerate(n_train_values):
        fig.text(0.02, 0.5 + (n_rows - 1 - i) * (1.0 / n_rows) - 0.5/n_rows, f"n={n_train}", 
                 rotation=90, va='center', ha='center', fontsize=9)
    # Add column labels (lambda) on the top
    for j, lambda_val in enumerate(lambda_values):
        fig.text(0.5 + j * (1.0 / n_cols) - 0.5/n_cols, 0.98, f"λ={format_lambda_str(lambda_val)}", 
                 ha='center', va='top', fontsize=9)


def ensure_axes_2d(axes, n_rows: int, n_cols: int):
    """Ensure axes is always 2D list for consistent indexing."""
    if n_rows == 1 and n_cols == 1:
        return [[axes]]
    elif n_rows == 1:
        if hasattr(axes, '__len__') and len(axes) == n_cols:
            return [list(axes) if isinstance(axes, np.ndarray) else axes]
        else:
            return [[axes]]
    elif n_cols == 1:
        if hasattr(axes, '__len__') and len(axes) == n_rows:
            return [[ax] for ax in axes]
        else:
            return [[axes]]
    else:
        if isinstance(axes, np.ndarray):
            return axes.tolist()
        else:
            return axes


def plot_ablation_waterfall_to_ax(
    model,
    loader_test,
    ax,
    *,
    top_units_per_layer: int = 8,
    mode: str = "routing_gain",
):
    """Plot ablation waterfall to an axes object."""
    from copy import deepcopy
    from src.analysis.path_analysis import _mean_transmittance_per_layer
    
    base = deepcopy(model)
    dev = next(iter(model.parameters())).device
    
    # centrality per unit
    mean_E = _mean_transmittance_per_layer(base, loader_test, device=dev, mode=mode)
    L = len(base.linears)
    flows = []
    for l in range(L):
        if l < L - 1:
            W_next = base.linears[l+1].weight.detach().abs().to("cpu")
            out_sum = W_next.sum(dim=0)
        else:
            out_sum = base.readout.weight.detach().abs().to("cpu").squeeze(0)
        c = (out_sum * mean_E[l].to("cpu"))
        assert out_sum.numel() == mean_E[l].numel(), f"ablation centrality mismatch: out_sum={out_sum.shape}, mean_E[{l}]={mean_E[l].shape}"
        flows.append(c)
    
    # build ablation order
    order = []
    for l, c in enumerate(flows):
        topk = torch.topk(c, k=min(top_units_per_layer, c.numel())).indices.tolist()
        order.extend([(l, u) for u in topk])
    
    # measure baseline error
    def _mse(model, loader):
        L = n = 0.0
        for xb, yb in loader:
            xb = xb.to(dev); yb = yb.to(dev)
            yhat = model(xb)
            L += torch.mean((yhat - yb)**2).item() * xb.size(0)
            n += xb.size(0)
        return L / max(1.0, n)
    
    errs = [_mse(base, loader_test)]
    cur = base
    for (l, u) in order:
        if hasattr(cur, "gates") and cur.gates is not None:
            cur.gates[l].a_plus.data[u] = 0.0
            cur.gates[l].a_minus.data[u] = 0.0
        errs.append(_mse(cur, loader_test))
    
    # plot to axes
    xs = np.arange(len(errs))
    ax.plot(xs, errs, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("ablation step", fontsize=8)
    ax.set_ylabel("test MSE", fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)


def plot_eig_spectrum_to_ax(evals: torch.Tensor, ax, *, title: str = ""):
    """Plot eigenvalue spectrum to an axes object."""
    lam = evals.detach().cpu().numpy().astype(np.float64)
    
    if len(lam) > 0 and lam[0] > 0:
        lam_normalized = lam / lam[0]
    else:
        lam_normalized = lam
    
    xs = np.arange(1, 1 + lam_normalized.shape[0])
    ax.plot(xs, lam_normalized, marker="o", markersize=3, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("rank", fontsize=8)
    ax.set_ylabel("eigenvalue / λ₁", fontsize=8)
    if title:
        ax.set_title(title, fontsize=8)
    ax.grid(True, ls="--", alpha=0.3, which="both")


def plot_circuit_overlap_to_ax(prototypes: np.ndarray, ax, *, title: str = ""):
    """Plot circuit overlap matrix to an axes object."""
    P = prototypes
    num = P @ P.T
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    S = num / (norms * norms.T)
    im = ax.imshow(S, cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xlabel("circuit", fontsize=8)
    ax.set_ylabel("circuit", fontsize=8)


def plot_flow_centrality_to_ax(
    model,
    loader,
    ax,
    *,
    mode: str = "routing_gain",
):
    """Plot flow centrality heatmap to an axes object."""
    from src.analysis.path_analysis import _mean_transmittance_per_layer
    
    dev = next(iter(model.parameters())).device
    mean_E = _mean_transmittance_per_layer(model, loader, device=dev, mode=mode)
    
    C_rows = []
    L = len(model.linears)
    for l in range(L):
        if l < L - 1:
            W_next = model.linears[l+1].weight.detach().abs().to("cpu")
            out_sum = W_next.sum(dim=0)
        else:
            W_ro = model.readout.weight.detach().abs().to("cpu").squeeze(0)
            out_sum = W_ro
        c = (out_sum * mean_E[l].to("cpu"))
        C_rows.append(c.numpy())
    
    # Concatenate and plot
    all_C = np.concatenate(C_rows)
    max_width = max(len(c) for c in C_rows)
    # Create padded matrix
    C_matrix = np.zeros((L, max_width))
    for l, c in enumerate(C_rows):
        C_matrix[l, :len(c)] = c
    
    im = ax.imshow(C_matrix, cmap="viridis", aspect='auto', interpolation='nearest')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("unit", fontsize=8)
    ax.set_ylabel("layer", fontsize=8)
    ax.set_title("Flow Centrality", fontsize=8)


def plot_path_cleanliness_to_ax(
    model,
    loader,
    ax,
    *,
    mode: str = "routing_gain",
    top_k: int = 5,
):
    """Plot path cleanliness to an axes object."""
    from src.analysis.path_analysis import _mean_transmittance_per_layer, _beam_top_paths
    
    mean_E = _mean_transmittance_per_layer(model, loader, device=None, mode=mode)
    paths = _beam_top_paths(model, mean_E, beam=24, top_k=top_k)
    
    if len(paths) == 0:
        ax.text(0.5, 0.5, 'No paths', ha='center', va='center', transform=ax.transAxes)
        return
    
    dev = next(iter(model.parameters())).device
    xb0, yb0 = next(iter(loader))
    xb0 = xb0.to(dev)
    _, cache0 = model(xb0, return_cache=True)
    L = len(cache0["z"])
    
    argmax_list: List[List[int]] = [[] for _ in range(L)]
    for xb, _ in loader:
        xb = xb.to(dev)
        _, cache = model(xb, return_cache=True)
        zs = cache["z"]
        if hasattr(model, "gates") and model.gates is not None:
            ap_am = [(g.a_plus.detach().to(dev), g.a_minus.detach().to(dev)) for g in model.gates]
        else:
            ap_am = [(None, None) for _ in range(L)]
        
        for l in range(L):
            z = zs[l].float()
            a_plus, a_minus = ap_am[l]
            if mode == "routing":
                E = z
            elif mode == "routing_posdev":
                if a_plus is None:
                    E = z
                else:
                    ap = (a_plus - 1.0).clamp_min(0.0)
                    E = z * ap.unsqueeze(0)
            else:
                if a_plus is None:
                    E = z
                else:
                    E = z * a_plus.unsqueeze(0) + (1.0 - z) * a_minus.unsqueeze(0)
            idx = torch.argmax(E, dim=1)
            argmax_list[l].extend(idx.detach().to("cpu").tolist())
    
    fracs = []
    for p in paths:
        if len(p) == 0:
            continue
        layer_fracs = []
        for l, j in enumerate(p):
            if l >= len(argmax_list):
                break
            hits = sum(1 for idx in argmax_list[l] if idx == j)
            layer_fracs.append(hits / max(1, len(argmax_list[l])))
        if len(layer_fracs) > 0:
            fracs.append(layer_fracs)
    
    if not fracs:
        ax.text(0.5, 0.5, 'No valid paths', ha='center', va='center', transform=ax.transAxes)
        return
    
    Lp = len(fracs[0]) if fracs else 0
    xs = np.arange(Lp)
    for k, f in enumerate(fracs):
        ax.plot(xs, f, marker="o", markersize=2, linewidth=1.0, label=f"path {k}", alpha=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("layer", fontsize=8)
    ax.set_ylabel("cleanliness", fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)
    if len(fracs) <= 5:
        ax.legend(fontsize=6)


def plot_path_embedding_to_ax(E: torch.Tensor, labels: Optional[torch.Tensor], ax, *, title: str = ""):
    """Plot path embedding map to an axes object."""
    X = E.detach().cpu().numpy()
    y = labels.detach().cpu().numpy() if labels is not None else None
    
    try:
        from sklearn.manifold import TSNE
        HAVE_SK = True
    except:
        HAVE_SK = False
    
    try:
        import umap
        HAVE_UMAP = True
    except:
        HAVE_UMAP = False
    
    if HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=1)
        Z = reducer.fit_transform(X)
    elif HAVE_SK:
        Z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=1).fit_transform(X)
    else:
        Xc = X - X.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2][None, :]
    
    if y is None:
        ax.scatter(Z[:,0], Z[:,1], s=4, alpha=0.7)
    else:
        classes = np.unique(y)
        for c in classes:
            m = (y == c)
            ax.scatter(Z[m,0], Z[m,1], s=6, alpha=0.8, label=str(c))
        if len(classes) <= 10:
            ax.legend(fontsize=6, markerscale=1.5)
    if title:
        ax.set_title(title, fontsize=8)


def generate_ablation_waterfall_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
    mode: str = "routing_gain",
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Generate ablation waterfall grid plot: x-axis=n_train, y-axis=lambda."""
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    # Parse n_train and lambda values - note: x-axis is lambda, y-axis is n_train
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    config_path = output_path / "config.json"
    if not config_path.exists():
        print(f"Warning: config.json not found in {output_dir}")
        return
    cfg = load_config(str(config_path))
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    n_cols = len(lambda_values)   # x-axis: lambda
    n_rows = len(n_train_values)  # y-axis: dataset size
    
    if n_rows == 0 or n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows), sharex=False, sharey=False)
    axes_2d = ensure_axes_2d(axes, n_rows, n_cols)
    
    for i, n_train in enumerate(n_train_values):
        for j, lambda_val in enumerate(lambda_values):
            ax = axes_2d[i][j]
            
            # Find matching directory
            n_train_dir = None
            for n_dir, _, _ in algo_combinations:
                n_parsed = int(n_dir.split('_lam')[0][1:])
                lam_parsed = parse_lambda_string(n_dir.split('_lam')[1])
                if n_parsed == n_train and abs(lam_parsed - lambda_val) < 1e-10:
                    n_train_dir = n_dir
                    break
            
            if n_train_dir is None:
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            model_path = output_path / n_train_dir / algo_name / "model_final.pt"
            if not model_path.exists():
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No model', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            try:
                model = load_model(str(model_path), cfg)
                model.to(device_obj)
                for param in model.parameters():
                    param.data = param.data.to(device_obj)
                
                if not model_has_gates(model):
                    ax.set_title("", fontsize=7)
                    ax.text(0.5, 0.5, 'No gates\n(RELU only)', ha='center', va='center', transform=ax.transAxes, alpha=0.5, fontsize=6)
                else:
                    train_loader, val_loader, test_loader = build_dataloaders_from_config(cfg, n_train, lambda_val)
                    plot_ablation_waterfall_to_ax(model, test_loader, ax, top_units_per_layer=8, mode=mode)
                    ax.set_title("", fontsize=7)
            except Exception as e:
                print(f"  Warning: Failed ablation waterfall for n={n_train}, λ={format_lambda_str(lambda_val)}: {e}")
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
    
    # Add row/column labels
    add_grid_row_column_labels(fig, n_rows, n_cols, lambda_values, n_train_values)
    
    plt.suptitle(f"Ablation Waterfall - {algo_name}", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    out_path = plots_dir / f"grid_ablation_waterfall_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation waterfall grid: {out_path}")


def generate_circuit_overlap_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
    mode: str = "routing_gain",
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Generate circuit overlap grid plot: x-axis=n_train, y-axis=lambda."""
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    config_path = output_path / "config.json"
    if not config_path.exists():
        return
    cfg = load_config(str(config_path))
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    n_cols = len(lambda_values)   # x-axis: lambda
    n_rows = len(n_train_values)  # y-axis: dataset size
    
    if n_rows == 0 or n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), sharex=False, sharey=False)
    axes_2d = ensure_axes_2d(axes, n_rows, n_cols)
    
    for i, n_train in enumerate(n_train_values):
        for j, lambda_val in enumerate(lambda_values):
            ax = axes_2d[i][j]
            
            n_train_dir = None
            for n_dir, _, _ in algo_combinations:
                n_parsed = int(n_dir.split('_lam')[0][1:])
                lam_parsed = parse_lambda_string(n_dir.split('_lam')[1])
                if n_parsed == n_train and abs(lam_parsed - lambda_val) < 1e-10:
                    n_train_dir = n_dir
                    break
            
            if n_train_dir is None:
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            model_path = output_path / n_train_dir / algo_name / "model_final.pt"
            if not model_path.exists():
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No model', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            try:
                model = load_model(str(model_path), cfg)
                model.to(device_obj)
                for param in model.parameters():
                    param.data = param.data.to(device_obj)
                
                train_loader, val_loader, test_loader = build_dataloaders_from_config(cfg, n_train, lambda_val)
                
                # Compute embeddings and cluster
                Epack = path_embedding(model, val_loader, device=device_obj, mode=mode, normalize=True, max_samples=max_samples)
                E = Epack["E"].numpy()
                
                if E.shape[0] > 8:
                    if HAVE_SKLEARN:
                        k_clusters = min(8, E.shape[0] // 10, E.shape[0] - 1)
                        if k_clusters >= 2:
                            km = KMeans(n_clusters=k_clusters, random_state=1, n_init="auto")
                            km.fit(E)
                            prototypes = km.cluster_centers_
                            plot_circuit_overlap_to_ax(prototypes, ax)
                            ax.set_title("", fontsize=7)
                        else:
                            ax.set_title("", fontsize=7)
                            ax.text(0.5, 0.5, 'Not enough\nclusters', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                    else:
                        ax.set_title("", fontsize=7)
                        ax.text(0.5, 0.5, 'scikit-learn\nrequired', ha='center', va='center', transform=ax.transAxes, alpha=0.5, fontsize=6)
                else:
                    ax.set_title("", fontsize=7)
                    ax.text(0.5, 0.5, 'Not enough\nsamples', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
            except Exception as e:
                print(f"  Warning: Failed circuit overlap for n={n_train}, λ={format_lambda_str(lambda_val)}: {e}")
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
    
    # Add row/column labels
    add_grid_row_column_labels(fig, n_rows, n_cols, lambda_values, n_train_values)
    
    plt.suptitle(f"Circuit Overlap - {algo_name}", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    out_path = plots_dir / f"grid_circuit_overlap_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved circuit overlap grid: {out_path}")


def generate_eigen_spectrum_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
    mode: str = "routing_gain",
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    kernel_k: int = 48,
):
    """Generate eigenvalue spectrum grid plot: x-axis=n_train, y-axis=lambda."""
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    config_path = output_path / "config.json"
    if not config_path.exists():
        return
    cfg = load_config(str(config_path))
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    n_cols = len(n_train_values)
    n_rows = len(lambda_values)
    
    if n_rows == 0 or n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), sharex=False, sharey=True)
    axes_2d = ensure_axes_2d(axes, n_rows, n_cols)
    
    for i, lambda_val in enumerate(lambda_values):
        for j, n_train in enumerate(n_train_values):
            ax = axes_2d[i][j]
            
            n_train_dir = None
            for n_dir, _, _ in algo_combinations:
                n_parsed = int(n_dir.split('_lam')[0][1:])
                lam_parsed = parse_lambda_string(n_dir.split('_lam')[1])
                if n_parsed == n_train and abs(lam_parsed - lambda_val) < 1e-10:
                    n_train_dir = n_dir
                    break
            
            if n_train_dir is None:
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            model_path = output_path / n_train_dir / algo_name / "model_final.pt"
            if not model_path.exists():
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No model', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            try:
                model = load_model(str(model_path), cfg)
                model.to(device_obj)
                for param in model.parameters():
                    param.data = param.data.to(device_obj)
                
                train_loader, val_loader, test_loader = build_dataloaders_from_config(cfg, n_train, lambda_val)
                
                kern = compute_path_kernel_eigs(
                    model, val_loader, device=device, mode=mode,
                    include_input=True, k=kernel_k, n_iter=30, block_size=1024,
                    max_samples=max_samples, verbose=False
                )
                plot_eig_spectrum_to_ax(kern["evals"], ax)
                ax.set_title("", fontsize=7)
            except Exception as e:
                print(f"  Warning: Failed eigen spectrum for n={n_train}, λ={format_lambda_str(lambda_val)}: {e}")
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
    
    # Add row/column labels
    add_grid_row_column_labels(fig, n_rows, n_cols, lambda_values, n_train_values)
    
    plt.suptitle(f"Eigenvalue Spectrum - {algo_name}", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    out_path = plots_dir / f"grid_eigen_spectrum_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved eigen spectrum grid: {out_path}")


def generate_flow_centrality_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
    mode: str = "routing_gain",
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Generate flow centrality grid plot: x-axis=n_train, y-axis=lambda."""
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    config_path = output_path / "config.json"
    if not config_path.exists():
        return
    cfg = load_config(str(config_path))
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    n_cols = len(lambda_values)   # x-axis: lambda
    n_rows = len(n_train_values)  # y-axis: dataset size
    
    if n_rows == 0 or n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), sharex=False, sharey=False)
    axes_2d = ensure_axes_2d(axes, n_rows, n_cols)
    
    for i, n_train in enumerate(n_train_values):
        for j, lambda_val in enumerate(lambda_values):
            ax = axes_2d[i][j]
            
            n_train_dir = None
            for n_dir, _, _ in algo_combinations:
                n_parsed = int(n_dir.split('_lam')[0][1:])
                lam_parsed = parse_lambda_string(n_dir.split('_lam')[1])
                if n_parsed == n_train and abs(lam_parsed - lambda_val) < 1e-10:
                    n_train_dir = n_dir
                    break
            
            if n_train_dir is None:
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            model_path = output_path / n_train_dir / algo_name / "model_final.pt"
            if not model_path.exists():
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No model', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            try:
                model = load_model(str(model_path), cfg)
                model.to(device_obj)
                for param in model.parameters():
                    param.data = param.data.to(device_obj)
                
                if not model_has_gates(model):
                    ax.set_title("", fontsize=7)
                    ax.text(0.5, 0.5, 'No gates\n(RELU only)', ha='center', va='center', transform=ax.transAxes, alpha=0.5, fontsize=6)
                else:
                    train_loader, val_loader, test_loader = build_dataloaders_from_config(cfg, n_train, lambda_val)
                    plot_flow_centrality_to_ax(model, val_loader, ax, mode=mode)
                    ax.set_title("", fontsize=7)
            except Exception as e:
                print(f"  Warning: Failed flow centrality for n={n_train}, λ={format_lambda_str(lambda_val)}: {e}")
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
    
    # Add row/column labels
    add_grid_row_column_labels(fig, n_rows, n_cols, lambda_values, n_train_values)
    
    plt.suptitle(f"Flow Centrality - {algo_name}", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    out_path = plots_dir / f"grid_flow_centrality_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved flow centrality grid: {out_path}")


def plot_nn_graph_to_ax(
    model,
    loader,
    ax,
    *,
    mode: str = "routing_gain",
    beam: int = 24,
    top_k: int = 3,
):
    """Plot simplified NN graph with paths to an axes object."""
    from src.analysis.path_analysis import _mean_transmittance_per_layer, _beam_top_paths
    try:
        import networkx as nx
        HAVE_NX = True
    except ImportError:
        HAVE_NX = False
    
    if not HAVE_NX:
        ax.text(0.5, 0.5, 'networkx\nrequired', ha='center', va='center', transform=ax.transAxes, alpha=0.5, fontsize=6)
        return
    
    mean_E = _mean_transmittance_per_layer(model, loader, device=None, mode=mode)
    paths = _beam_top_paths(model, mean_E, beam=beam, top_k=top_k)
    
    if len(paths) == 0:
        ax.text(0.5, 0.5, 'No paths', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
        ax.axis('off')
        return
    
    # Create simplified graph visualization
    G = nx.DiGraph()
    widths = [model.linears[l].out_features for l in range(len(model.linears))]
    d_in = model.linears[0].in_features
    
    # Add layer nodes (simplified - just mark layers, not all units)
    layer_x = {}
    x = 0
    G.add_node(("L0", -1), layer=0)
    layer_x[0] = [("L0", -1)]
    for l, w in enumerate(widths, start=1):
        # Only add a few representative nodes per layer for visualization
        num_nodes_to_show = min(5, w)
        nodes = [(f"L{l}", i * (w // max(1, num_nodes_to_show - 1)) if num_nodes_to_show > 1 else 0) for i in range(num_nodes_to_show)]
        for n in nodes:
            G.add_node(n, layer=l)
        layer_x[l] = nodes
    Lmax = len(widths) + 1
    G.add_node(("OUT", -1), layer=Lmax)
    layer_x[Lmax] = [("OUT", -1)]
    
    # Add edges from top paths
    used_edges = set()
    for p in paths:
        if len(p) == 0:
            continue
        # Map path units to displayed nodes (simplified mapping)
        displayed_path = []
        for l, unit_idx in enumerate(p):
            if l < len(layer_x) - 1:
                # Find closest displayed node
                closest = min(layer_x[l+1], key=lambda n: abs(n[1] - (unit_idx % layer_x[l+1][-1][1] if layer_x[l+1][-1][1] > 0 else 0)))
                displayed_path.append(closest)
        
        if len(displayed_path) > 0:
            used_edges.add((("L0", -1), displayed_path[0]))
            for i in range(len(displayed_path) - 1):
                used_edges.add((displayed_path[i], displayed_path[i+1]))
            used_edges.add((displayed_path[-1], ("OUT", -1)))
    
    G.add_edges_from(list(used_edges))
    
    # Positions
    pos = {}
    for layer, nodes in layer_x.items():
        n = len(nodes)
        for i, nkey in enumerate(nodes):
            pos[nkey] = (layer, (i - n/2) / max(n, 1) * 2.0)
    
    # Draw simplified graph
    ax.axis('off')
    nx.draw(G, pos, ax=ax, with_labels=False, node_size=20, arrows=False, width=1, alpha=0.6)
    
    # Highlight paths
    for k, p in enumerate(paths[:top_k]):
        if len(p) == 0:
            continue
        # Simplified path visualization
        displayed_path = []
        for l, unit_idx in enumerate(p):
            if l < len(layer_x) - 1:
                closest = min(layer_x[l+1], key=lambda n: abs(n[1] - (unit_idx % layer_x[l+1][-1][1] if layer_x[l+1][-1][1] > 0 else 0)))
                displayed_path.append(closest)
        
        if len(displayed_path) > 0:
            edges = [(("L0",-1), displayed_path[0])]
            for i in range(len(displayed_path) - 1):
                edges.append((displayed_path[i], displayed_path[i+1]))
            edges.append((displayed_path[-1], ("OUT",-1)))
            nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, width=2, 
                                  edge_color=f"C{k}", alpha=0.8, arrows=False)


def generate_nn_graphs_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
    mode: str = "routing_gain",
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Generate NN graphs grid plot: x-axis=n_train, y-axis=lambda."""
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    config_path = output_path / "config.json"
    if not config_path.exists():
        return
    cfg = load_config(str(config_path))
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    n_cols = len(n_train_values)
    n_rows = len(lambda_values)
    
    if n_rows == 0 or n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows), sharex=False, sharey=False)
    axes_2d = ensure_axes_2d(axes, n_rows, n_cols)
    
    for i, lambda_val in enumerate(lambda_values):
        for j, n_train in enumerate(n_train_values):
            ax = axes_2d[i][j]
            
            n_train_dir = None
            for n_dir, _, _ in algo_combinations:
                n_parsed = int(n_dir.split('_lam')[0][1:])
                lam_parsed = parse_lambda_string(n_dir.split('_lam')[1])
                if n_parsed == n_train and abs(lam_parsed - lambda_val) < 1e-10:
                    n_train_dir = n_dir
                    break
            
            if n_train_dir is None:
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                continue
            
            model_path = output_path / n_train_dir / algo_name / "model_final.pt"
            if not model_path.exists():
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No model', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                continue
            
            try:
                model = load_model(str(model_path), cfg)
                model.to(device_obj)
                for param in model.parameters():
                    param.data = param.data.to(device_obj)
                
                if not model_has_gates(model):
                    ax.set_title("", fontsize=7)
                    ax.text(0.5, 0.5, 'No gates\n(RELU only)', ha='center', va='center', transform=ax.transAxes, alpha=0.5, fontsize=6)
                    ax.axis('off')
                else:
                    train_loader, val_loader, test_loader = build_dataloaders_from_config(cfg, n_train, lambda_val)
                    plot_nn_graph_to_ax(model, val_loader, ax, mode=mode, beam=24, top_k=3)
                    ax.set_title("", fontsize=7)
            except Exception as e:
                print(f"  Warning: Failed NN graph for n={n_train}, λ={format_lambda_str(lambda_val)}: {e}")
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.axis('off')
    
    # Add row/column labels
    add_grid_row_column_labels(fig, n_rows, n_cols, lambda_values, n_train_values)
    
    plt.suptitle(f"NN Graphs with Paths - {algo_name}", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    out_path = plots_dir / f"grid_nn_graphs_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved NN graphs grid: {out_path}")


def generate_path_cleanliness_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
    mode: str = "routing_gain",
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Generate path cleanliness grid plot: x-axis=n_train, y-axis=lambda."""
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    config_path = output_path / "config.json"
    if not config_path.exists():
        return
    cfg = load_config(str(config_path))
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    n_cols = len(n_train_values)
    n_rows = len(lambda_values)
    
    if n_rows == 0 or n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), sharex=False, sharey=True)
    axes_2d = ensure_axes_2d(axes, n_rows, n_cols)
    
    for i, lambda_val in enumerate(lambda_values):
        for j, n_train in enumerate(n_train_values):
            ax = axes_2d[i][j]
            
            n_train_dir = None
            for n_dir, _, _ in algo_combinations:
                n_parsed = int(n_dir.split('_lam')[0][1:])
                lam_parsed = parse_lambda_string(n_dir.split('_lam')[1])
                if n_parsed == n_train and abs(lam_parsed - lambda_val) < 1e-10:
                    n_train_dir = n_dir
                    break
            
            if n_train_dir is None:
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            model_path = output_path / n_train_dir / algo_name / "model_final.pt"
            if not model_path.exists():
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No model', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            try:
                model = load_model(str(model_path), cfg)
                model.to(device_obj)
                for param in model.parameters():
                    param.data = param.data.to(device_obj)
                
                if not model_has_gates(model):
                    ax.set_title("", fontsize=7)
                    ax.text(0.5, 0.5, 'No gates\n(RELU only)', ha='center', va='center', transform=ax.transAxes, alpha=0.5, fontsize=6)
                else:
                    train_loader, val_loader, test_loader = build_dataloaders_from_config(cfg, n_train, lambda_val)
                    plot_path_cleanliness_to_ax(model, val_loader, ax, mode=mode, top_k=5)
                    ax.set_title("", fontsize=7)
            except Exception as e:
                print(f"  Warning: Failed path cleanliness for n={n_train}, λ={format_lambda_str(lambda_val)}: {e}")
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
    
    # Add row/column labels
    add_grid_row_column_labels(fig, n_rows, n_cols, lambda_values, n_train_values)
    
    plt.suptitle(f"Path Cleanliness - {algo_name}", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    out_path = plots_dir / f"grid_path_cleanliness_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved path cleanliness grid: {out_path}")


def generate_path_embedding_grid(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
    mode: str = "routing_gain",
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Generate path embedding grid plot: x-axis=n_train, y-axis=lambda."""
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    config_path = output_path / "config.json"
    if not config_path.exists():
        return
    cfg = load_config(str(config_path))
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    n_cols = len(lambda_values)   # x-axis: lambda
    n_rows = len(n_train_values)  # y-axis: dataset size
    
    if n_rows == 0 or n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), sharex=False, sharey=False)
    axes_2d = ensure_axes_2d(axes, n_rows, n_cols)
    
    for i, n_train in enumerate(n_train_values):
        for j, lambda_val in enumerate(lambda_values):
            ax = axes_2d[i][j]
            
            n_train_dir = None
            for n_dir, _, _ in algo_combinations:
                n_parsed = int(n_dir.split('_lam')[0][1:])
                lam_parsed = parse_lambda_string(n_dir.split('_lam')[1])
                if n_parsed == n_train and abs(lam_parsed - lambda_val) < 1e-10:
                    n_train_dir = n_dir
                    break
            
            if n_train_dir is None:
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            model_path = output_path / n_train_dir / algo_name / "model_final.pt"
            if not model_path.exists():
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'No model', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            try:
                model = load_model(str(model_path), cfg)
                model.to(device_obj)
                for param in model.parameters():
                    param.data = param.data.to(device_obj)
                
                train_loader, val_loader, test_loader = build_dataloaders_from_config(cfg, n_train, lambda_val)
                Epack = path_embedding(model, val_loader, device=device_obj, mode=mode, normalize=True, max_samples=max_samples)
                plot_path_embedding_to_ax(Epack["E"], Epack["labels"], ax)
                ax.set_title("", fontsize=7)
            except Exception as e:
                print(f"  Warning: Failed path embedding for n={n_train}, λ={format_lambda_str(lambda_val)}: {e}")
                ax.set_title("", fontsize=7)
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
    
    # Add row/column labels
    add_grid_row_column_labels(fig, n_rows, n_cols, lambda_values, n_train_values)
    
    plt.suptitle(f"Path Embedding - {algo_name}", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    out_path = plots_dir / f"grid_path_embedding_{algo_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved path embedding grid: {out_path}")


def generate_train_test_error_grids(
    output_dir: str,
    plots_dir: Optional[str] = None,
    algo_name: str = "alt_em_sgd",
):
    """Generate train and test error grid plots: x-axis=lambda, y-axis=n_train.
    
    Each subplot shows error over time (epoch/cycle).
    """
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = find_experiment_combinations(str(output_path))
    algo_combinations = [(n, lam, algo) for n, lam, algo in combinations if algo == algo_name]
    
    if len(algo_combinations) == 0:
        print(f"No combinations found for algorithm: {algo_name}")
        return
    
    n_train_values = sorted(set(int(n.split('_lam')[0][1:]) for n, _, _ in algo_combinations))
    lambda_values = sorted(set(parse_lambda_string(n.split('_lam')[1]) for n, _, _ in algo_combinations))
    
    # Determine if cycle-based or epoch-based
    sample_path = output_path / algo_combinations[0][0] / algo_name / "training_history.csv"
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            header = f.readline().strip()
            is_cycle_based = 'cycle' in header
            time_col = 'cycle' if is_cycle_based else 'epoch'
    else:
        is_cycle_based = 'alt_em' in algo_name
        time_col = 'cycle' if is_cycle_based else 'epoch'
    
    # Generate plots for both train_loss and test_loss
    for metric_name, metric_display in [("train_loss", "Train Loss"), ("test_loss", "Test Loss")]:
        grid_data = {}
        for n_train_dir, lambda_dir, algo in algo_combinations:
            n_train = int(n_train_dir.split('_lam')[0][1:])
            lambda_val = parse_lambda_string(n_train_dir.split('_lam')[1])
            
            csv_path = output_path / n_train_dir / algo / "training_history.csv"
            if not csv_path.exists():
                continue
            
            try:
                history = load_training_history(str(csv_path))
                if metric_name not in history or time_col not in history:
                    continue
                
                times = []
                values = []
                time_list = history[time_col]
                metric_list = history[metric_name]
                
                for t, v in zip(time_list, metric_list):
                    if v == '' or v is None:
                        continue
                    try:
                        times.append(float(t))
                        values.append(float(v))
                    except (ValueError, TypeError):
                        continue
                
                if len(times) > 0:
                    grid_data[(n_train, lambda_val)] = (times, values)
            except Exception as e:
                continue
        
        if len(grid_data) == 0:
            print(f"  No data found for {metric_name}")
            continue
        
        n_cols = len(lambda_values)   # x-axis: lambda
        n_rows = len(n_train_values)  # y-axis: dataset size
        
        if n_rows == 0 or n_cols == 0:
            continue
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows), sharex=False, sharey=False)
        axes_2d = ensure_axes_2d(axes, n_rows, n_cols)
        
        for i, n_train in enumerate(n_train_values):
            for j, lambda_val in enumerate(lambda_values):
                ax = axes_2d[i][j]
                
                if (n_train, lambda_val) in grid_data:
                    times, values = grid_data[(n_train, lambda_val)]
                    ax.plot(times, values, marker='o', markersize=2, linewidth=1.5, alpha=0.8)
                    ax.set_title("", fontsize=7)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_title("", fontsize=7)
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        # Add row/column labels
        add_grid_row_column_labels(fig, n_rows, n_cols, lambda_values, n_train_values)
        
        plt.suptitle(f"{metric_display} - {algo_name}", fontsize=11, y=0.995)
        plt.tight_layout(rect=[0.03, 0, 1, 0.97])
        
        metric_safe = metric_name.replace('_', '-')
        out_path = plots_dir / f"grid_{metric_safe}_{algo_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric_display} grid: {out_path}")


def generate_all_grid_plots(
    output_dir: str,
    plots_dir: Optional[str] = None,
    max_samples_kernel: Optional[int] = 2000,
    max_samples_embed: Optional[int] = 2000,
    kernel_k: int = 48,
    mode: str = "routing_gain",
    device: Optional[str] = None,
):
    """Generate all grid plots for all algorithms."""
    output_path = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_path / "plots" / "path_analysis_all"
    else:
        plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = find_experiment_combinations(str(output_path))
    algorithms = sorted(set(algo for _, _, algo in combinations))
    
    print(f"Generating grid plots for {len(algorithms)} algorithms (including sgd_relu)...")
    print()
    
    for algo in algorithms:
        print(f"Processing algorithm: {algo}")
        try:
            generate_ablation_waterfall_grid(output_dir, plots_dir, algo, mode, device, max_samples_embed)
        except Exception as e:
            print(f"  Warning: Failed ablation waterfall grid for {algo}: {e}")
        
        try:
            generate_circuit_overlap_grid(output_dir, plots_dir, algo, mode, device, max_samples_embed)
        except Exception as e:
            print(f"  Warning: Failed circuit overlap grid for {algo}: {e}")
        
        try:
            generate_eigen_spectrum_grid(output_dir, plots_dir, algo, mode, device, max_samples_kernel, kernel_k)
        except Exception as e:
            print(f"  Warning: Failed eigen spectrum grid for {algo}: {e}")
        
        try:
            generate_flow_centrality_grid(output_dir, plots_dir, algo, mode, device, max_samples_embed)
        except Exception as e:
            print(f"  Warning: Failed flow centrality grid for {algo}: {e}")
        
        try:
            generate_path_cleanliness_grid(output_dir, plots_dir, algo, mode, device, max_samples_embed)
        except Exception as e:
            print(f"  Warning: Failed path cleanliness grid for {algo}: {e}")
        
        try:
            generate_path_embedding_grid(output_dir, plots_dir, algo, mode, device, max_samples_embed)
        except Exception as e:
            print(f"  Warning: Failed path embedding grid for {algo}: {e}")
        
        try:
            generate_nn_graphs_grid(output_dir, plots_dir, algo, mode, device, max_samples_embed)
        except Exception as e:
            print(f"  Warning: Failed NN graphs grid for {algo}: {e}")
        
        try:
            generate_train_test_error_grids(output_dir, plots_dir, algo)
        except Exception as e:
            print(f"  Warning: Failed train/test error grids for {algo}: {e}")
        
        print()
    
    print(f"All grid plots generated in: {plots_dir}")


def main():
    # ============================================================
    # Configuration - set values here
    # ============================================================
    # Path to experiment output directory (must contain config.json)
    output_dir = "/home/goring/NN_alternatecoding/outputs/27_11/hierarchical_xor_run_2_20251127_035024"
    
    # Directory to save plots (None = use default: output_dir/plots/path_analysis_all)
    plots_dir = None
    
    # Maximum samples for kernel computation
    max_samples_kernel = 5000
    
    # Maximum samples for embedding computation
    max_samples_embed = 5000
    
    # Number of top eigenvalues to compute for path kernel
    kernel_k = 48
    
    # Transmittance mode: "routing", "routing_gain", or "routing_posdev"
    mode = "routing_gain"
    
    # Device to use: "cuda", "cpu", or None for auto-detect
    device = None  # Will auto-detect (uses CUDA if available)
    
    # ============================================================
    # Generate plots
    # ============================================================
    
    # Expand user path (handles ~) and resolve to absolute path
    output_dir = os.path.expanduser(output_dir)
    output_dir = os.path.abspath(output_dir)
    
    print(f"Output directory: {output_dir}")
    print(f"Mode: {mode}")
    print(f"Max samples (kernel): {max_samples_kernel}")
    print(f"Max samples (embedding): {max_samples_embed}")
    print(f"Kernel k: {kernel_k}")
    print()
    
    # NOTE: Individual plots are now replaced by grid plots
    # Uncomment the following if you want individual plots too:
    # # Generate individual plots
    # generate_all_plots(
    #     output_dir=output_dir,
    #     plots_dir=plots_dir,
    #     max_samples_kernel=max_samples_kernel,
    #     max_samples_embed=max_samples_embed,
    #     kernel_k=kernel_k,
    #     mode=mode,
    #     device=device,
    # )
    
    # Generate metric grid plots
    print("\n" + "="*60)
    print("Generating metric grid plots...")
    print("="*60 + "\n")
    generate_metric_grids(
        output_dir=output_dir,
        plots_dir=plots_dir,
    )
    
    # Generate eigenvalue spectrum plots (initial vs final)
    print("\n" + "="*60)
    print("Generating eigenvalue spectrum plots (initial vs final)...")
    print("="*60 + "\n")
    generate_eigenvalue_spectrum_plots(
        output_dir=output_dir,
        plots_dir=plots_dir,
        mode=mode,
        device=device,
    )
    
    # Generate checkpoint metric plots (Path-Shapley, Centroid Drift, Lineage Sankey)
    print("\n" + "="*60)
    print("Generating checkpoint metric plots...")
    print("="*60 + "\n")
    generate_checkpoint_metric_plots(
        output_dir=output_dir,
        plots_dir=plots_dir,
    )
    
    # Generate grid plots for all visualization types (x-axis=n_train, y-axis=lambda)
    print("\n" + "="*60)
    print("Generating grid plots for all visualization types...")
    print("="*60 + "\n")
    generate_all_grid_plots(
        output_dir=output_dir,
        plots_dir=plots_dir,
        max_samples_kernel=max_samples_kernel,
        max_samples_embed=max_samples_embed,
        kernel_k=kernel_k,
        mode=mode,
        device=device,
    )


if __name__ == "__main__":
    main()

