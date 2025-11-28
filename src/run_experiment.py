# src/run_experiment.py
from __future__ import annotations
import argparse, time, os, json, copy
import torch
from torch.utils.data import DataLoader
from threading import Thread
from queue import Queue, Empty

from .utils.config   import load_config
from .utils.seed     import set_seed
from .utils.save_io  import ensure_dir, save_json, save_csv, save_model
from .data.models.ffnn import GatedMLP

# Parity dataset (default)
from .data.ksparse_parity import gen_ksparse_parity, ParityDataset

# Optional: synonym-tree dataset
_HAS_SYN_TREE = True
try:
    from .data.hierarchical_synonyms import build_synonym_tree_datasets, SynonymTreeDataset
except Exception:
    _HAS_SYN_TREE = False

# Hierarchical XOR dataset
from .data.hierarchical_xor import build_hierarchical_xor_datasets, HierarchicalXORDataset


def build_dataloaders(cfg, n_train_override=None):
    """
    Build dataloaders. If n_train_override is provided, use that instead of cfg["dataset"]["n_train"].
    """
    ds_cfg = cfg["dataset"]
    name = ds_cfg.get("name", "ksparse_parity").lower()

    if name == "synonym_tree":
        if not _HAS_SYN_TREE:
            raise ValueError("Requested dataset=name 'synonym_tree' but 'data/hierarchical_synonyms.py' is not available.")
        temp_cfg = copy.deepcopy(cfg)
        if n_train_override is not None:
            temp_cfg["dataset"]["n_train"] = n_train_override

        Xtr, ytr, Xva, yva, Xte, yte, meta = build_synonym_tree_datasets(temp_cfg)
        ds_tr = SynonymTreeDataset(Xtr, ytr)
        ds_va = SynonymTreeDataset(Xva, yva)
        ds_te = SynonymTreeDataset(Xte, yte)
        bs    = cfg["training"]["batch_size"]
        train_loader      = DataLoader(ds_tr, batch_size=min(bs, len(ds_tr)), shuffle=True,  drop_last=False)
        full_train_loader = DataLoader(ds_tr, batch_size=len(ds_tr),           shuffle=False)
        val_loader        = DataLoader(ds_va, batch_size=len(ds_va),           shuffle=False)
        test_loader       = DataLoader(ds_te, batch_size=len(ds_te),           shuffle=False)
        # 'meta' should include 'd' for input dim; if not, infer:
        if "d" not in meta:
            meta["d"] = int(Xtr.shape[1])
        return train_loader, full_train_loader, val_loader, test_loader, meta

    if name == "hierarchical_xor":
        temp_cfg = copy.deepcopy(cfg)
        if n_train_override is not None:
            temp_cfg["dataset"]["n_train"] = n_train_override

        Xtr, ytr, Xva, yva, Xte, yte, meta, groups_tr, groups_va, groups_te = build_hierarchical_xor_datasets(temp_cfg)
        n_groups = meta.get("n_groups")
        ds_tr = HierarchicalXORDataset(Xtr, ytr, groups=groups_tr, n_groups=n_groups)
        ds_va = HierarchicalXORDataset(Xva, yva, groups=groups_va, n_groups=n_groups)
        ds_te = HierarchicalXORDataset(Xte, yte, groups=groups_te, n_groups=n_groups)
        bs    = cfg["training"]["batch_size"]
        train_loader      = DataLoader(ds_tr, batch_size=min(bs, len(ds_tr)), shuffle=True,  drop_last=False)
        full_train_loader = DataLoader(ds_tr, batch_size=len(ds_tr),           shuffle=False)
        val_loader        = DataLoader(ds_va, batch_size=len(ds_va),           shuffle=False)
        test_loader       = DataLoader(ds_te, batch_size=len(ds_te),           shuffle=False)
        if "d" not in meta:
            meta["d"] = int(Xtr.shape[1])
        return train_loader, full_train_loader, val_loader, test_loader, meta

    # Default: ksparse_parity (backward compatible)
    d, k = ds_cfg["d"], ds_cfg["k"]
    ntr  = n_train_override if n_train_override is not None else ds_cfg["n_train"]
    nva, nte = ds_cfg["n_val"], ds_cfg["n_test"]
    x_dist   = ds_cfg.get("x_dist", "pm1")
    noise    = ds_cfg.get("label_noise", 0.0)
    seed     = cfg["seed"]

    Xtr, ytr, S = gen_ksparse_parity(d, k, ntr, x_dist=x_dist, label_noise=noise, seed=seed)
    Xva, yva, _ = gen_ksparse_parity(d, k, nva, x_dist=x_dist, label_noise=noise, seed=seed+1)
    Xte, yte, _ = gen_ksparse_parity(d, k, nte, x_dist=x_dist, label_noise=noise, seed=seed+2)

    ds_tr, ds_va, ds_te = ParityDataset(Xtr, ytr), ParityDataset(Xva, yva), ParityDataset(Xte, yte)
    bs = cfg["training"]["batch_size"]
    train_loader      = DataLoader(ds_tr, batch_size=min(bs, len(ds_tr)), shuffle=True,  drop_last=False)
    full_train_loader = DataLoader(ds_tr, batch_size=len(ds_tr),           shuffle=False)
    val_loader        = DataLoader(ds_va, batch_size=len(ds_va),           shuffle=False)
    test_loader       = DataLoader(ds_te, batch_size=len(ds_te),           shuffle=False)
    meta = {"name": "ksparse_parity", "d": int(d), "k": int(k), "S": list(map(int, S))}
    return train_loader, full_train_loader, val_loader, test_loader, meta


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
    """
    from .algos.sgd                import train_sgd_relu
    from .algos.alt_em_sgd         import train_alt_em_sgd
    from .algos.alt_em_closed_form import train_alt_em_closed_form
    from .algos.sgd_joint          import train_sgd_joint

    algo_dir = os.path.join(out_dir, algo_name)
    ensure_dir(algo_dir)

    # Create path_analysis subdirectory
    path_analysis_dir = os.path.join(algo_dir, "path_analysis")
    ensure_dir(path_analysis_dir)

    # Override λ_id if requested (sweeps)
    train_cfg = copy.deepcopy(cfg)
    if lambda_identity_override is not None:
        train_cfg.setdefault("regularization", {})
        train_cfg["regularization"]["lambda_identity"] = lambda_identity_override
    
    # Add path_analysis output directory to config
    train_cfg["path_analysis_out_dir"] = path_analysis_dir

    # Fresh model
    model = GatedMLP(
        input_dim,
        train_cfg["model"]["widths"],
        bias=train_cfg["model"]["bias"],
        use_gates=train_cfg["model"]["use_gates"]
    )
    # Save initial model only if save_model is enabled
    if train_cfg.get("logging", {}).get("save_model", True):
        save_model(model, os.path.join(algo_dir, "model_init.pt"))

    print(f"\n{'='*60}\nRunning {algo_name}\n{'='*60}")

    # Train (pass test_loader for potential early stopping)
    checkpoint_metrics_history = []
    if algo_name == "sgd_relu":
        hist = train_sgd_relu(model, train_loader, val_loader, train_cfg, test_loader=test_loader)
    elif algo_name == "alt_em_sgd":
        result = train_alt_em_sgd(model, train_loader, val_loader, train_cfg, test_loader=test_loader)
        if isinstance(result, tuple) and len(result) == 2:
            hist, checkpoint_metrics_history = result
        else:
            hist = result
    elif algo_name == "alt_em_closed_form":
        result = train_alt_em_closed_form(model, full_train_loader, val_loader, train_cfg, test_loader=test_loader)
        if isinstance(result, tuple) and len(result) == 2:
            hist, checkpoint_metrics_history = result
        else:
            hist = result
    elif algo_name == "sgd_joint":
        result = train_sgd_joint(model, train_loader, val_loader, train_cfg, test_loader=test_loader)
        if isinstance(result, tuple) and len(result) == 2:
            hist, checkpoint_metrics_history = result
        else:
            hist = result
    else:
        raise ValueError(f"Unknown algo={algo_name}")

    # Final evaluation
    trA, trL = _eval(model, full_train_loader, device)
    vaA, vaL = _eval(model, val_loader, device)
    teA, teL = _eval(model, test_loader, device)

    # Save artifacts
    # Save final model only if save_model is enabled
    if cfg.get("logging", {}).get("save_model", True):
        save_model(model, os.path.join(algo_dir, "model_final.pt"))
    save_json(meta,  os.path.join(algo_dir, "dataset_meta.json"))

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
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    print(f"Worker started on {device}")

    job_count = 0
    while True:
        try:
            job = job_queue.get(timeout=1)
            if job is None:  # sentinel
                break

            (algo_name, n_train, lambda_identity, cfg, combo_dir) = job
            job_count += 1

            start_time = time.time()
            print(f"[{device}] Starting job {job_count}: {algo_name}, n_train={n_train}, lambda={lambda_identity}")
            try:
                # Build data loaders in this thread
                data_start = time.time()
                train_loader, full_train_loader, val_loader, test_loader, meta = build_dataloaders(
                    cfg, n_train_override=n_train
                )
                print(f"[{device}] Data loading took {time.time() - data_start:.2f}s")

                input_dim = int(meta.get("d", cfg["dataset"].get("d", 0)))

                cfg_copy = copy.deepcopy(cfg)
                cfg_copy["device"] = device

                train_start = time.time()
                hist, metrics = run_algorithm(
                    algo_name, input_dim,
                    train_loader, full_train_loader, val_loader, test_loader,
                    cfg_copy, combo_dir, meta, device,
                    lambda_identity_override=lambda_identity
                )
                train_time = time.time() - train_start
                total_time = time.time() - start_time
                print(f"[{device}] Completed {algo_name} (n={n_train}, λ={lambda_identity}) "
                      f"in {train_time:.1f}s (total: {total_time:.1f}s)")

                result_queue.put(("success", (algo_name, combo_dir), {"history": hist, "metrics": metrics}))
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                print(f"[{device}] ERROR in {algo_name} (n={n_train}, λ={lambda_identity}): {str(e)}")
                
                # Try to save error information even if training failed
                try:
                    algo_dir = os.path.join(combo_dir, algo_name)
                    ensure_dir(algo_dir)
                    error_info = {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "algo_name": algo_name,
                        "n_train": n_train,
                        "lambda_identity": lambda_identity
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
    ap.add_argument("--config", type=str, default="config.yaml")
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
    algo_cfg = cfg["training"].get("algo", "alt_em_closed_form")
    if isinstance(algo_cfg, list):
        algorithms = algo_cfg
    else:
        algorithms = [algo_cfg]

    # n_train sweep (list or scalar)
    n_train_cfg = cfg["dataset"]["n_train"]
    n_train_values = n_train_cfg if isinstance(n_train_cfg, list) else [n_train_cfg]

    # lambda_identity sweep (list or scalar)
    lam_cfg = cfg["regularization"].get("lambda_identity", 0.0)
    lambda_values = lam_cfg if isinstance(lam_cfg, list) else [lam_cfg]

    # Build jobs
    job_queue = Queue()
    result_queue = Queue()
    all_jobs = []
    for n_train in n_train_values:
        for lam in lambda_values:
            combo_suffix = f"n{n_train}_lam{lam}"
            combo_dir = os.path.join(out_dir, combo_suffix)
            ensure_dir(combo_dir)
            for algo_name in algorithms:
                job = (algo_name, n_train, lam, cfg, combo_dir)
                all_jobs.append(job)
                job_queue.put(job)

    # Add sentinels
    for _ in range(num_gpus):
        job_queue.put(None)

    # Start workers
    workers = []
    for gpu_idx in range(num_gpus):
        gpu_id = gpu_ids[gpu_idx % len(gpu_ids)] if torch.cuda.is_available() else None
        t = Thread(target=_worker_thread, args=(job_queue, result_queue, gpu_id))
        t.start()
        workers.append(t)

    # Collect results
    all_results = {}
    completed = 0
    total_jobs = len(all_jobs)

    main_start_time = time.time()
    print(f"\n{'='*60}\nStarting {total_jobs} training jobs across {num_gpus} device(s)\n{'='*60}\n")

    while completed < total_jobs:
        try:
            status, key, data = result_queue.get(timeout=600)  # 10min timeout per job slot
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
def _eval(model, loader, device):
    model.eval()
    L = A = n = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        L += torch.mean((yhat - yb) ** 2).item() * xb.size(0)
        A += torch.sign(yhat).eq(yb).float().mean().item() * xb.size(0)
        n += xb.size(0)
    model.train(False)
    return A / n, L / n


if __name__ == "__main__":
    main()
