# src/run_experiment_mnist.py
from __future__ import annotations
import argparse, time, os, json, copy
import torch
from torch.utils.data import DataLoader
from threading import Thread
from queue import Queue, Empty

from .utils.config   import load_config
from .utils.seed     import set_seed
from .utils.save_io  import ensure_dir, save_json, save_csv, save_model
from .data.models.ffnn import MLP

# MNIST dataset
from .data.mnist import build_mnist_datasets, MNISTDataset

# CIFAR-10 dataset
from .data.cifar10 import build_cifar10_datasets, CIFAR10Dataset


def build_dataloaders(cfg, n_train_override=None, alpha_override=None):
    """
    Build dataloaders for MNIST or CIFAR-10. 
    If n_train_override is provided, use that instead of cfg["dataset"]["n_train"].
    If alpha_override is provided, use that instead of cfg["dataset"]["alpha"].
    """
    ds_cfg = cfg["dataset"]
    name = ds_cfg.get("name", "mnist").lower()

    if name == "mnist":
        temp_cfg = copy.deepcopy(cfg)
        if n_train_override is not None:
            temp_cfg["dataset"]["n_train"] = n_train_override
        if alpha_override is not None:
            temp_cfg["dataset"]["alpha"] = alpha_override

        Xtr, ytr, Xva, yva, Xte, yte, meta = build_mnist_datasets(temp_cfg)
        ds_tr = MNISTDataset(Xtr, ytr)
        ds_va = MNISTDataset(Xva, yva)
        ds_te = MNISTDataset(Xte, yte)
        bs    = cfg["training"]["batch_size"]
        train_loader      = DataLoader(ds_tr, batch_size=min(bs, len(ds_tr)), shuffle=True,  drop_last=False)
        full_train_loader = DataLoader(ds_tr, batch_size=len(ds_tr),           shuffle=False)
        val_loader        = DataLoader(ds_va, batch_size=len(ds_va),           shuffle=False)
        test_loader       = DataLoader(ds_te, batch_size=len(ds_te),           shuffle=False)
        if "d" not in meta:
            meta["d"] = int(Xtr.shape[1])
        return train_loader, full_train_loader, val_loader, test_loader, meta
    elif name == "cifar10":
        temp_cfg = copy.deepcopy(cfg)
        if n_train_override is not None:
            temp_cfg["dataset"]["n_train"] = n_train_override
        if alpha_override is not None:
            temp_cfg["dataset"]["alpha"] = alpha_override

        Xtr, ytr, Xva, yva, Xte, yte, meta = build_cifar10_datasets(temp_cfg)
        ds_tr = CIFAR10Dataset(Xtr, ytr)
        ds_va = CIFAR10Dataset(Xva, yva)
        ds_te = CIFAR10Dataset(Xte, yte)
        bs    = cfg["training"]["batch_size"]
        train_loader      = DataLoader(ds_tr, batch_size=min(bs, len(ds_tr)), shuffle=True,  drop_last=False)
        full_train_loader = DataLoader(ds_tr, batch_size=len(ds_tr),           shuffle=False)
        val_loader        = DataLoader(ds_va, batch_size=len(ds_va),           shuffle=False)
        test_loader       = DataLoader(ds_te, batch_size=len(ds_te),           shuffle=False)
        if "d" not in meta:
            meta["d"] = int(Xtr.shape[1])
        return train_loader, full_train_loader, val_loader, test_loader, meta
    else:
        raise ValueError(f"Unknown dataset name: {name}. Only 'mnist' and 'cifar10' are supported in this script.")


def run_algorithm(
    algo_name,
    input_dim,
    train_loader,
    full_train_loader,
    val_loader,
    test_loader,
    cfg,
    out_dir,
    meta,
    device,
    lambda_identity_override=None
):
    """
    Run one algorithm, save model, history, and final metrics.
    Reuses the same sgd.py training function.
    """
    from .algos.sgd import train_sgd

    algo_dir = os.path.join(out_dir, algo_name)
    ensure_dir(algo_dir)

    # Track checkpoint metrics if the trainer populates them (currently unused)
    checkpoint_metrics_history = []

    # Create path_analysis subdirectory
    path_analysis_dir = os.path.join(algo_dir, "path_analysis")
    ensure_dir(path_analysis_dir)

    # Override λ_id if requested (sweeps) - optional, only if regularization section exists
    train_cfg = copy.deepcopy(cfg)
    if lambda_identity_override is not None:
        train_cfg.setdefault("regularization", {})
        train_cfg["regularization"]["lambda_identity"] = lambda_identity_override
    
    # Add path_analysis output directory to config only if enabled in logging
    enable_path_analysis = cfg.get("logging", {}).get("enable_path_analysis", False)
    if enable_path_analysis:
        train_cfg["path_analysis_out_dir"] = path_analysis_dir
    # Always set algo_dir in config so plotting code can use it
    train_cfg["algo_dir"] = algo_dir

    # Fresh model - move to device immediately to ensure correct GPU assignment
    activation = train_cfg["model"].get("activation", "relu")
    n_classes = meta.get("n_classes", 1)  # Get n_classes from meta (10 for multiclass, 1 for binary)
    model = MLP(
        input_dim,
        train_cfg["model"]["widths"],
        bias=train_cfg["model"]["bias"],
        activation=activation,
        n_classes=n_classes
    )
    # Move model to device immediately to ensure it's on the correct GPU
    model = model.to(device)
    
    # Save a copy of the initial model for comparison
    model_initial = None
    if enable_path_analysis:
        # Create a deep copy of the initial model
        model_initial = copy.deepcopy(model)
        model_initial.eval()
    
    # Save initial model only if save_model is enabled
    if train_cfg.get("logging", {}).get("save_model", True):
        save_model(model, os.path.join(algo_dir, "model_init.pt"))

    print(f"\n{'='*60}\nRunning {algo_name}\n{'='*60}")

    # Train (pass test_loader for potential early stopping)
    if algo_name == "sgd":
        hist = train_sgd(model, train_loader, val_loader, train_cfg, test_loader=test_loader)
    else:
        raise ValueError(f"Unknown algo={algo_name}. Only 'sgd' is supported.")

    # Final evaluation
    n_classes = meta.get("n_classes", 1)
    trA, trL = _eval(model, full_train_loader, device, n_classes)
    vaA, vaL = _eval(model, val_loader, device, n_classes)
    teA, teL = _eval(model, test_loader, device, n_classes)

    # Save artifacts
    # Save final model only if save_model is enabled
    if cfg.get("logging", {}).get("save_model", True):
        save_model(model, os.path.join(algo_dir, "model_final.pt"))
    save_json(meta,  os.path.join(algo_dir, "dataset_meta.json"))
    
    # Compare initial vs final networks (circuit overlap and visualizations)
    if enable_path_analysis and model_initial is not None:
        try:
            from .analysis.circuit_comparison import compare_initial_vs_final_networks
            print(f"\n[run_algorithm] Comparing initial vs final networks...")
            compare_initial_vs_final_networks(
                model_initial=model_initial,
                model_final=model,
                loader=val_loader,
                out_dir=path_analysis_dir,
                mode="routing",
                max_samples=1000,
                device=device,
            )
        except Exception as e:
            import traceback
            print(f"[run_algorithm] Warning: Failed to compare initial vs final networks: {e}")
            print(f"[run_algorithm] Traceback: {traceback.format_exc()}")

    final_metrics = {
        "train_acc": trA, "train_loss": trL,
        "val_acc": vaA,   "val_loss": vaL,
        "test_acc": teA,  "test_loss": teL,
    }

    # Pull selected statistics from the final history entry (if provided by the trainer)
    # Only keep basic metrics (no path metrics, no slope metrics)
    if hist and len(hist) > 0:
        last = hist[-1]
        # Only keep effective rank if present (basic activation metric)
        if last.get("effective_rank_layers") is not None:
            final_metrics["final_effective_rank_layers"] = last["effective_rank_layers"]

    # Persist
    save_json(final_metrics, os.path.join(algo_dir, "final_metrics.json"))
    save_csv(hist,            os.path.join(algo_dir, "training_history.csv"))
    
    # Save checkpoint metrics history if available
    if checkpoint_metrics_history:
        save_json(checkpoint_metrics_history, os.path.join(algo_dir, "checkpoint_metrics.json"))

    # Also save losses.json for fast plotting
    train_losses = [h.get("train_loss", None) for h in hist]
    val_losses   = [h.get("val_loss",   None) for h in hist]
    losses_data  = {
        "train_losses": train_losses,
        "test_losses":  val_losses,
        "final_train_loss": trL,
        "final_test_loss":  teL
    }
    save_json(losses_data, os.path.join(algo_dir, "losses.json"))

    print(f"Completed {algo_name}. Results saved to {algo_dir}")
    return hist, final_metrics


def _worker_thread(job_queue, result_queue, gpu_id):
    """Worker thread that runs training jobs on a specific device."""
    # Set device for this thread - must be done before any CUDA operations
    if torch.cuda.is_available() and gpu_id is not None:
        device = f"cuda:{gpu_id}"
        # Set the default device for this thread's CUDA operations
        # Note: This is process-wide in PyTorch, but we ensure it's set before each operation
    else:
        device = "cpu"
    print(f"Worker started on {device} (GPU {gpu_id})")

    job_count = 0
    while True:
        try:
            job = job_queue.get(timeout=1)
            if job is None:  # sentinel
                break

            (algo_name, n_train, lambda_identity, alpha, optimizer, cfg, combo_dir) = job
            job_count += 1

            start_time = time.time()
            print(f"[{device}] Starting job {job_count}: {algo_name}, n_train={n_train}, lambda={lambda_identity}, alpha={alpha}, optimizer={optimizer}")
            try:
                # CRITICAL: Set device before any CUDA operations in this job
                # This ensures all tensors/models are created on the correct GPU
                if torch.cuda.is_available() and gpu_id is not None:
                    torch.cuda.set_device(gpu_id)
                    # Verify we're on the correct device
                    current_device = torch.cuda.current_device()
                    if current_device != gpu_id:
                        print(f"[{device}] WARNING: Device mismatch! Expected {gpu_id}, got {current_device}. Fixing...")
                        torch.cuda.set_device(gpu_id)
                
                # Build data loaders in this thread
                data_start = time.time()
                train_loader, full_train_loader, val_loader, test_loader, meta = build_dataloaders(
                    cfg, n_train_override=n_train, alpha_override=alpha
                )
                print(f"[{device}] Data loading took {time.time() - data_start:.2f}s")

                input_dim = int(meta.get("d", cfg["dataset"].get("d", 0)))

                cfg_copy = copy.deepcopy(cfg)
                cfg_copy["device"] = device
                # Override optimizer in config
                cfg_copy["training"]["optimizer"] = optimizer
                # Override alpha in config (use the single value, not the list)
                cfg_copy["dataset"]["alpha"] = alpha

                # Double-check device before training
                if torch.cuda.is_available() and gpu_id is not None:
                    torch.cuda.set_device(gpu_id)
                    actual_device = torch.cuda.current_device()
                    if actual_device != gpu_id:
                        print(f"[{device}] ERROR: Device mismatch before training! Expected {gpu_id}, got {actual_device}")
                
                train_start = time.time()
                hist, metrics = run_algorithm(
                    algo_name, input_dim,
                    train_loader, full_train_loader, val_loader, test_loader,
                    cfg_copy, combo_dir, meta, device,
                    lambda_identity_override=lambda_identity
                )
                train_time = time.time() - train_start
                total_time = time.time() - start_time
                print(f"[{device}] Completed {algo_name} (n={n_train}, λ={lambda_identity}, α={alpha}, opt={optimizer}) "
                      f"in {train_time:.1f}s (total: {total_time:.1f}s)")
                
                # Verify device after training (for debugging)
                if torch.cuda.is_available() and gpu_id is not None:
                    actual_device = torch.cuda.current_device()
                    if actual_device != gpu_id:
                        print(f"[{device}] WARNING: Device changed during training! Expected {gpu_id}, got {actual_device}")

                result_queue.put(("success", (algo_name, combo_dir), {"history": hist, "metrics": metrics}))
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                print(f"[{device}] ERROR in {algo_name} (n={n_train}, λ={lambda_identity}, α={alpha}, opt={optimizer}): {str(e)}")
                
                # Try to save error information even if training failed
                try:
                    algo_dir = os.path.join(combo_dir, algo_name)
                    ensure_dir(algo_dir)
                    error_info = {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "algo_name": algo_name,
                        "n_train": n_train,
                        "lambda_identity": lambda_identity,
                        "alpha": alpha,
                        "optimizer": optimizer
                    }
                    save_json(error_info, os.path.join(algo_dir, "error_info.json"))
                except:
                    pass  # If we can't save error info, at least continue
                
                result_queue.put(("error", (algo_name, combo_dir), error_msg))
        except Empty:
            continue
        except Exception as e:
            result_queue.put(("error", None, str(e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_mnist.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    # Devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        print(f"Found {num_gpus} GPU(s): {gpu_ids}")
    else:
        num_gpus = 1
        gpu_ids = [0]
        print("No GPUs found, using CPU")

    out_dir = os.path.join("outputs", f"{cfg['experiment_name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(out_dir)
    save_json(cfg, os.path.join(out_dir, "config.json"))

    # Algorithms to run (list or single string)
    algo_cfg = cfg["training"].get("algo", "sgd")
    if isinstance(algo_cfg, list):
        algorithms = algo_cfg
    else:
        algorithms = [algo_cfg]

    # n_train sweep (list or scalar)
    n_train_cfg = cfg["dataset"]["n_train"]
    n_train_values = n_train_cfg if isinstance(n_train_cfg, list) else [n_train_cfg]

    # lambda_identity sweep (list or scalar) - optional, defaults to [0.0]
    lam_cfg = cfg.get("regularization", {}).get("lambda_identity", 0.0)
    lambda_values = lam_cfg if isinstance(lam_cfg, list) else [lam_cfg]

    # alpha sweep (list or scalar) - optional, defaults to [1.0]
    alpha_cfg = cfg.get("dataset", {}).get("alpha", 1.0)
    alpha_values = alpha_cfg if isinstance(alpha_cfg, list) else [alpha_cfg]

    # optimizer sweep (list or scalar) - optional, defaults to ["adam"]
    optimizer_cfg = cfg.get("training", {}).get("optimizer", "adam")
    optimizer_values = optimizer_cfg if isinstance(optimizer_cfg, list) else [optimizer_cfg]
    # Normalize optimizer names to lowercase strings
    optimizer_values = [str(opt).lower() for opt in optimizer_values]

    # Build jobs
    job_queue = Queue()
    result_queue = Queue()
    all_jobs = []
    for n_train in n_train_values:
        for lam in lambda_values:
            for alpha in alpha_values:
                for optimizer in optimizer_values:
                    combo_suffix = f"n{n_train}_lam{lam}_alpha{alpha}_opt{optimizer}"
                    combo_dir = os.path.join(out_dir, combo_suffix)
                    ensure_dir(combo_dir)
                    for algo_name in algorithms:
                        job = (algo_name, n_train, lam, alpha, optimizer, cfg, combo_dir)
                        all_jobs.append(job)
                        job_queue.put(job)

    # Add sentinels
    for _ in range(num_gpus):
        job_queue.put(None)

    # Start workers - create one worker per GPU to ensure proper distribution
    workers = []
    for gpu_idx in range(num_gpus):
        gpu_id = gpu_ids[gpu_idx] if torch.cuda.is_available() else None
        device_str = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
        print(f"Starting worker thread {gpu_idx} on {device_str} (GPU ID: {gpu_id})")
        t = Thread(target=_worker_thread, args=(job_queue, result_queue, gpu_id))
        t.daemon = False  # Ensure threads don't die when main exits
        t.start()
        workers.append(t)
        # Small delay to ensure proper GPU initialization and avoid race conditions
        time.sleep(0.2)
    
    print(f"\nStarted {len(workers)} worker threads. Jobs will be distributed across {num_gpus} GPU(s).")

    # Collect results
    all_results = {}
    completed = 0
    total_jobs = len(all_jobs)

    main_start_time = time.time()
    print(f"\n{'='*60}\nStarting {total_jobs} training jobs across {num_gpus} device(s)\n{'='*60}\n")

    while completed < total_jobs:
        try:
            # Increased timeout: path analysis can take a long time, especially with large models
            # 3600 seconds = 1 hour per job (should be enough even for slow analysis)
            status, key, data = result_queue.get(timeout=3600)  # 1 hour timeout per job slot
            if status == "success":
                all_results[key] = data
                completed += 1
                elapsed = time.time() - main_start_time
                remaining = total_jobs - completed
                avg_time = elapsed / completed if completed > 0 else 0
                eta = avg_time * remaining if remaining > 0 else 0
                print(f"[MAIN] Progress: {completed}/{total_jobs} "
                      f"({100*completed/total_jobs:.1f}%) | Elapsed: {elapsed/60:.1f}m | "
                      f"ETA: {eta/60:.1f}m | {key[0]} in {key[1]}")
            elif status == "error":
                print(f"[MAIN] ERROR in job {key}: {data[:200]}")  # truncate long tracebacks in stdout
                completed += 1
        except Empty:
            print("[MAIN] Warning: Timeout waiting for results")
            break

    # Join workers
    for t in workers:
        t.join()

    # Save a compact summary (note: keys are (algo_name, dir); keep last one per algo)
    summary = {}
    for (algo_name, _dir), res in all_results.items():
        summary[algo_name] = {
            "final_train_loss": res["metrics"]["train_loss"],
            "final_test_loss":  res["metrics"]["test_loss"],
            "final_train_acc":  res["metrics"]["train_acc"],
            "final_test_acc":   res["metrics"]["test_acc"],
        }
    save_json(summary, os.path.join(out_dir, "summary.json"))

    print(f"\n{'='*60}\nAll algorithms completed. Results saved to {out_dir}\n{'='*60}")


@torch.no_grad()
def _eval(model, loader, device, n_classes=None):
    """Evaluate model for binary or multi-class classification."""
    model.eval()
    L = A = n = 0.0
    
    # Detect n_classes from model if not provided
    if n_classes is None:
        if hasattr(model, 'n_classes'):
            n_classes = model.n_classes
        elif hasattr(model, 'readout'):
            out_features = model.readout.out_features
            n_classes = out_features if out_features > 1 else None
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        
        if n_classes is None or n_classes == 1:
            # Binary classification: MSE loss and sign-based accuracy
            L += torch.mean((yhat - yb) ** 2).item() * xb.size(0)
            A += torch.sign(yhat).eq(yb).float().mean().item() * xb.size(0)
        else:
            # Multi-class classification: Cross-entropy loss and argmax accuracy
            import torch.nn.functional as F
            # Ensure yb is long for cross-entropy
            if yb.dim() > 1:
                yb = yb.view(-1)
            yb_long = yb.long()
            L += F.cross_entropy(yhat, yb_long).item() * xb.size(0)
            pred = yhat.argmax(dim=1)
            A += (pred == yb_long).float().mean().item() * xb.size(0)
        
        n += xb.size(0)
    
    model.train(False)
    return A / n, L / n


if __name__ == "__main__":
    main()

