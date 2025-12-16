#!/usr/bin/env python3
"""
Clean Entry Point for Parameterization Experiments

This script provides a modular, clean interface to run neural network training
experiments with different parameterization schemes:
- standard: Xavier/Kaiming initialization
- mup: Maximal Update Parametrization
- ntk: Neural Tangent Kernel parametrization
- mup_L: mup with Lazarus depth scaling
- path: Path parameterization with LazarusMLP

Usage:
    python run_experiment_parameterization.py --config config_mnist_parameterizations.yaml
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from queue import Queue, Empty
from threading import Thread

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- CLEAN MODULAR IMPORTS ---
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.save_io import ensure_dir, save_json
from src.data.mnist import build_mnist_datasets

# Models
from src.models import MLP, MLPResNet, MLPBatchNorm, LazarusMLP

# Training
from src.training import (
    train_with_parameterization,
    initialize_parameterization,
    parse_parameterization_name,
)

# Plotting
from src.utils.plotting import (
    plot_metrics,
    plot_lr_per_layer,
    plot_gradient_norms_per_layer,
    plot_comparison,
    plot_gradient_norms_comparison,
    plot_kernel_metrics,
    plot_eigenvalue_spectra,
    plot_gradient_eigenvalues,
    plot_kernel_metrics_comparison,
    plot_all_kernel_metrics,
)


def create_model(
    architecture: str,
    parameterization: str,
    input_dim: int,
    widths: list[int],
    n_classes: int,
    config: dict
) -> tuple:
    """
    Create a model based on architecture and parameterization.
    
    Args:
        architecture: Model architecture name ("standard", "resnet", "lazarus")
        parameterization: Parameterization scheme
        input_dim: Input dimension
        widths: List of hidden layer widths
        n_classes: Number of output classes
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, actual_architecture_name)
    """
    bias = config["model"].get("bias", True)
    activation = config["model"].get("activation", "relu")
    
    # Parse parameterization to get base name
    base_param, _ = parse_parameterization_name(parameterization)
    
    # Path parameterization always uses LazarusMLP
    if base_param == "path":
        return LazarusMLP(
            d_in=input_dim,
            widths=widths,
            bias=bias,
            activation=activation,
            n_classes=n_classes
        ), "lazarus"
    
    # Check if base_param contains "batchnorm"
    use_batchnorm = "batchnorm" in base_param.lower()
    
    if architecture == "standard":
        if use_batchnorm:
            return MLPBatchNorm(
                d_in=input_dim,
                widths=widths,
                bias=bias,
                activation=activation,
                n_classes=n_classes
            ), "standard"
        else:
            return MLP(
                d_in=input_dim,
                widths=widths,
                bias=bias,
                activation=activation,
                n_classes=n_classes
            ), "standard"
    elif architecture == "resnet":
        return MLPResNet(
            d_in=input_dim,
            widths=widths,
            bias=bias,
            activation=activation,
            n_classes=n_classes
        ), "resnet"
    elif architecture == "lazarus":
        return LazarusMLP(
            d_in=input_dim,
            widths=widths,
            bias=bias,
            activation=activation,
            n_classes=n_classes
        ), "lazarus"
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Options: standard, resnet, lazarus")


def run_single_parameterization(
    param: str,
    architecture: str,
    input_dim: int,
    widths: list[int],
    n_classes: int,
    config: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: str
) -> tuple[str, dict]:
    """
    Run a single parameterization experiment.
    
    Args:
        param: Parameterization name
        architecture: Base architecture name
        input_dim: Input dimension
        widths: List of hidden layer widths
        n_classes: Number of output classes
        config: Configuration dictionary
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to run on
        out_dir: Output directory
        
    Returns:
        Tuple of (key, history) where key is "{architecture}_{param}"
    """
    # Parse parameterization name
    base_param, optimizer_override = parse_parameterization_name(param)
    
    # Create model
    model, actual_architecture = create_model(
        architecture, param, input_dim, widths, n_classes, config
    )
    model = model.to(device)
    
    print(f"\n{'='*60}")
    print(f"Running {param} parameterization with {actual_architecture} architecture")
    if optimizer_override:
        print(f"Optimizer override: {optimizer_override}")
    print(f"Depth: {len(widths)}, Width: {widths[0]}")
    print(f"{'='*60}")
    
    # Initialize (only for non-path parameterizations, since LazarusMLP initializes itself)
    if base_param != "path":
        initialize_parameterization(model, base_param, device)
    
    # Train
    history = train_with_parameterization(
        model, train_loader, test_loader, config, device,
        base_param, actual_architecture, out_dir,
        optimizer_override=optimizer_override
    )
    
    # Plot individual metrics
    key = f"{actual_architecture}_{param}"
    plot_metrics(history, key, out_dir)
    plot_lr_per_layer(history, key, out_dir)
    plot_gradient_norms_per_layer(history, key, out_dir)
    
    # Plot kernel metrics if available
    track_kernel_metrics = config.get("logging", {}).get("track_kernel_metrics", True)
    if track_kernel_metrics:
        plot_all_kernel_metrics(history, key, out_dir)
    
    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return key, history


def worker_thread_parameterization(
    job_queue: Queue,
    result_queue: Queue,
    gpu_id: int | None,
    input_dim: int,
    n_classes: int,
    config: dict,
    architecture: str,
    widths: list[int],
    out_dir: str
):
    """Worker thread that runs parameterization training jobs on a specific GPU."""
    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")
    
    print(f"[GPU {gpu_id}] Worker started on {device}")
    
    # Build data loaders for this worker
    cfg_fixed = copy.deepcopy(config)
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
        batch_size=int(config["training"]["batch_size"]), 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["training"]["batch_size"]),
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
                key, history = run_single_parameterization(
                    param, architecture, input_dim, widths, n_classes,
                    config, train_loader, test_loader, device, out_dir
                )
                
                total_time = time.time() - start_time
                print(f"[GPU {gpu_id}] Completed {param} in {total_time:.1f}s")
                
                result_queue.put(("success", key, history))
                
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                print(f"[GPU {gpu_id}] ERROR in {param}: {str(e)}")
                result_queue.put(("error", param, error_msg))
                
        except Empty:
            continue
    
    print(f"[GPU {gpu_id}] Worker finished ({job_count} jobs completed)")


def run_parallel_training(
    parameterizations: list[str],
    architecture: str,
    widths: list[int],
    input_dim: int,
    n_classes: int,
    config: dict,
    out_dir: str,
    num_gpus: int,
    gpu_ids: list[int]
) -> dict:
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
            target=worker_thread_parameterization,
            args=(
                job_queue, result_queue, gpu_id,
                input_dim, n_classes, config, architecture, widths, out_dir
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
    
    print(f"Waiting for {total_jobs} parameterization(s) to complete...")
    
    while completed < total_jobs:
        try:
            status, key, result = result_queue.get(timeout=1800)  # 30 minute timeout
            if status == "success":
                all_histories[key] = result
                completed += 1
                print(f"Progress: {completed}/{total_jobs} parameterizations completed")
            elif status == "error":
                print(f"ERROR: Failed to train parameterization {key}")
                completed += 1
        except Empty:
            print(f"Warning: Timeout waiting for results ({completed}/{total_jobs} completed)")
            break
    
    # Wait for all workers to finish
    print("Waiting for all worker threads to finish...")
    for t in workers:
        t.join(timeout=300)
    
    return all_histories


def main():
    ap = argparse.ArgumentParser(description="Run parameterization experiments")
    ap.add_argument("--config", type=str, default="parameterizations/config_mnist_parameterizations.yaml",
                    help="Path to config YAML file")
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
    
    # Run experiments
    if len(parameterizations) > 1 and num_gpus > 1:
        # Multi-GPU parallel execution
        print(f"\n{'='*60}")
        print(f"Running {len(parameterizations)} parameterizations in parallel across {num_gpus} GPU(s)")
        print(f"{'='*60}\n")
        all_histories = run_parallel_training(
            parameterizations, architecture, widths,
            input_dim, n_classes, cfg, out_dir, num_gpus, gpu_ids
        )
    else:
        # Sequential execution
        print(f"\n{'='*60}")
        print(f"Running {len(parameterizations)} parameterization(s) sequentially")
        print(f"{'='*60}\n")
        all_histories = {}
        
        for param in parameterizations:
            key, history = run_single_parameterization(
                param, architecture, input_dim, widths, n_classes,
                cfg, train_loader, test_loader, device, out_dir
            )
            all_histories[key] = history
    
    # Plot comparison (if multiple parameterizations)
    if len(all_histories) > 1:
        print(f"\n{'='*60}")
        print("Creating comparison plots...")
        print(f"{'='*60}")
        plot_comparison(all_histories, out_dir)
        plot_gradient_norms_comparison(all_histories, out_dir)
        
        # Plot kernel metrics comparison if enabled
        track_kernel_metrics = cfg.get("logging", {}).get("track_kernel_metrics", True)
        if track_kernel_metrics:
            plot_kernel_metrics_comparison(all_histories, out_dir)
    
    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

