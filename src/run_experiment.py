from __future__ import annotations
import argparse, time, os, torch
from torch.utils.data import DataLoader
from .utils.config import load_config
from .utils.seed import set_seed
from .utils.save_io import ensure_dir, save_json, save_csv, save_model
from .data.ksparse_parity import gen_ksparse_parity, ParityDataset
from .data.models.ffnn import GatedMLP
from .algos.sgd import train_sgd_relu
from .algos.alt_em_sgd import train_alt_em_sgd
from .algos.alt_em_closed_form import train_alt_em_closed_form

def build_dataloaders(cfg):
    d,k = cfg["dataset"]["d"], cfg["dataset"]["k"]
    ntr, nva, nte = cfg["dataset"]["n_train"], cfg["dataset"]["n_val"], cfg["dataset"]["n_test"]
    x_dist, noise, seed = cfg["dataset"]["x_dist"], cfg["dataset"]["label_noise"], cfg["seed"]

    Xtr,ytr,S = gen_ksparse_parity(d,k,ntr,x_dist=x_dist,label_noise=noise,seed=seed)
    Xva,yva,_ = gen_ksparse_parity(d,k,nva,x_dist=x_dist,label_noise=noise,seed=seed+1)
    Xte,yte,_ = gen_ksparse_parity(d,k,nte,x_dist=x_dist,label_noise=noise,seed=seed+2)

    ds_tr, ds_va, ds_te = ParityDataset(Xtr,ytr), ParityDataset(Xva,yva), ParityDataset(Xte,yte)
    bs = cfg["training"]["batch_size"]
    train_loader      = DataLoader(ds_tr, batch_size=min(bs,len(ds_tr)), shuffle=True,  drop_last=False)
    full_train_loader = DataLoader(ds_tr, batch_size=len(ds_tr),           shuffle=False)
    val_loader        = DataLoader(ds_va, batch_size=len(ds_va),           shuffle=False)
    test_loader       = DataLoader(ds_te, batch_size=len(ds_te),           shuffle=False)
    return train_loader, full_train_loader, val_loader, test_loader, S

def run_algorithm(algo_name, train_loader, full_train_loader, val_loader, test_loader, cfg, out_dir, S, device):
    """Run a single algorithm and save results in its own directory."""
    algo_dir = os.path.join(out_dir, algo_name)
    ensure_dir(algo_dir)
    
    # Create a fresh model for each algorithm
    model = GatedMLP(cfg["dataset"]["d"], cfg["model"]["widths"],
                     bias=cfg["model"]["bias"], use_gates=cfg["model"]["use_gates"])
    save_model(model, os.path.join(algo_dir, "model_init.pt"))
    
    print(f"\n{'='*60}")
    print(f"Running {algo_name}")
    print(f"{'='*60}")
    
    # Train the model
    if algo_name == "sgd_relu":
        hist = train_sgd_relu(model, train_loader, val_loader, cfg)
    elif algo_name == "alt_em_sgd":
        hist = train_alt_em_sgd(model, train_loader, val_loader, cfg)
    elif algo_name == "alt_em_closed_form":
        hist = train_alt_em_closed_form(model, full_train_loader, val_loader, cfg)
    else:
        raise ValueError(f"Unknown algo={algo_name}")
    
    # Evaluate final metrics
    trA, trL = _eval(model, full_train_loader, device)
    vaA, vaL = _eval(model, val_loader, device)
    teA, teL = _eval(model, test_loader, device)
    
    # Save model
    save_model(model, os.path.join(algo_dir, "model_final.pt"))
    
    # Save final metrics
    final_metrics = {
        "train_acc": trA,
        "train_loss": trL,
        "val_acc": vaA,
        "val_loss": vaL,
        "test_acc": teA,
        "test_loss": teL,
        "subset_S": list(map(int, S))
    }
    save_json(final_metrics, os.path.join(algo_dir, "final_metrics.json"))
    
    # Save training history
    save_csv(hist, os.path.join(algo_dir, "training_history.csv"))
    
    # Extract and save train/test losses separately
    train_losses = [h.get("train_loss", None) for h in hist]
    test_losses = [h.get("val_loss", None) for h in hist]  # Using val_loss as test proxy during training
    
    # Save losses separately
    losses_data = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_train_loss": trL,
        "final_test_loss": teL
    }
    save_json(losses_data, os.path.join(algo_dir, "losses.json"))
    
    print(f"Completed {algo_name}. Results saved to {algo_dir}")
    return hist, final_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() or cfg["device"]=="cpu" else "cpu"

    # Create main output directory
    out_dir = os.path.join("outputs", f"{cfg['experiment_name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(out_dir)
    save_json(cfg, os.path.join(out_dir, "config.json"))

    # Build dataloaders once (same data for all algorithms)
    train_loader, full_train_loader, val_loader, test_loader, S = build_dataloaders(cfg)

    # Run all three algorithms sequentially
    algorithms = ["sgd_relu", "alt_em_sgd", "alt_em_closed_form"]
    all_results = {}
    
    for algo_name in algorithms:
        hist, metrics = run_algorithm(algo_name, train_loader, full_train_loader, 
                                     val_loader, test_loader, cfg, out_dir, S, device)
        all_results[algo_name] = {"history": hist, "metrics": metrics}
    
    # Save summary of all runs
    summary = {
        algo: {
            "final_train_loss": res["metrics"]["train_loss"],
            "final_test_loss": res["metrics"]["test_loss"],
            "final_train_acc": res["metrics"]["train_acc"],
            "final_test_acc": res["metrics"]["test_acc"]
        }
        for algo, res in all_results.items()
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))
    
    print(f"\n{'='*60}")
    print(f"All algorithms completed. Results saved to {out_dir}")
    print(f"{'='*60}")

@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    L=A=n=0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        L += torch.mean((yhat - yb)**2).item() * xb.size(0)
        A += torch.sign(yhat).eq(yb).float().mean().item() * xb.size(0)
        n += xb.size(0)
    model.train(False)
    return A/n, L/n

if __name__ == "__main__":
    main()
