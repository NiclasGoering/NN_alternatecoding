#!/usr/bin/env python3
"""
Generate all path analysis plots from saved experiment outputs.

Usage:
    # Pass output directory as argument:
    python plots/generate_path_analysis_plots.py /path/to/output/dir
    
    # Or use defaults (edit default path in main() function):
    python plots/generate_path_analysis_plots.py
    
    # With custom options:
    python plots/generate_path_analysis_plots.py /path/to/output/dir --max-samples-kernel 3000 --mode routing_gain

The script will automatically find all models in subdirectories matching:
    output_dir/n*_lam*/algorithm_name/model_final.pt
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple

import torch
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


def main():
    import argparse
    
    # ============================================================
    # Command-line arguments
    # ============================================================
    parser = argparse.ArgumentParser(
        description="Generate path analysis plots for all models in an experiment output directory."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to experiment output directory (must contain config.json). Example: /path/to/outputs/experiment_name"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: output_dir/plots/path_analysis_all)"
    )
    parser.add_argument(
        "--max-samples-kernel",
        type=int,
        default=5000,
        help="Maximum samples for kernel computation (default: 5000)"
    )
    parser.add_argument(
        "--max-samples-embed",
        type=int,
        default=5000,
        help="Maximum samples for embedding computation (default: 5000)"
    )
    parser.add_argument(
        "--kernel-k",
        type=int,
        default=48,
        help="Number of top eigenvalues to compute for path kernel (default: 48)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="routing_gain",
        choices=["routing", "routing_gain", "routing_posdev"],
        help="Transmittance mode (default: routing_gain)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Device to use for computation: 'cuda', 'cpu', or 'auto' (default: auto-detect, uses CUDA if available)"
    )
    
    args = parser.parse_args()
    
    # ============================================================
    # Get output directory
    # ============================================================
    if args.output_dir is None:
        # Default path - edit this if you want a different default
        # Or just pass the path as an argument: python generate_path_analysis_plots.py /path/to/output/dir
        output_dir = "/home/goring/NN_alternatecoding/outputs/24_11/hierarchical_xor_run_3_20251124_191204"
        print("No output directory provided. Using default:")
    else:
        output_dir = args.output_dir
    
    # Expand user path (handles ~) and resolve to absolute path
    output_dir = os.path.expanduser(output_dir)
    output_dir = os.path.abspath(output_dir)
    
    # ============================================================
    # Generate plots
    # ============================================================
    
    # Handle device argument
    device_arg = args.device
    if device_arg == "auto" or device_arg is None:
        device_arg = None  # Will auto-detect in generate_all_plots
    
    print(f"Output directory: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Max samples (kernel): {args.max_samples_kernel}")
    print(f"Max samples (embedding): {args.max_samples_embed}")
    print(f"Kernel k: {args.kernel_k}")
    print()
    
    generate_all_plots(
        output_dir=output_dir,
        plots_dir=args.plots_dir,
        max_samples_kernel=args.max_samples_kernel,
        max_samples_embed=args.max_samples_embed,
        kernel_k=args.kernel_k,
        mode=args.mode,
        device=device_arg,
    )


if __name__ == "__main__":
    main()

