from __future__ import annotations

import json
import os
import argparse
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Absolute imports so the module works with `python -m`
from src.analysis.path_kernel import collect_path_factors
from src.run_experiment_mnist import build_dataloaders
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.models.ffnn import MLP


@torch.enable_grad()
def _fgsm_flip_sign(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 0.15,
    clamp_min: float = -3.0,
    clamp_max: float = 3.0,
) -> torch.Tensor:
    """
    Single-step FGSM attack that pushes the prediction toward -y.
    Works for the binary {-1, +1} MNIST task used in this repo.
    """
    x_adv = x.detach().clone().requires_grad_(True)
    target = -y
    pred = model(x_adv)
    loss = nn.functional.mse_loss(pred, target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x_adv.grad.detach().sign()
    x_adv = x_adv + eps * grad_sign
    return x_adv.clamp_(clamp_min, clamp_max).detach()


@torch.no_grad()
def _gate_pattern(model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """Return per-layer binary gate activations for a single sample."""
    _, cache = model(x, return_cache=True)
    return [z.detach().bool() for z in cache["z"]]


@torch.no_grad()
def _gate_differences(
    gates_a: List[torch.Tensor], gates_b: List[torch.Tensor]
) -> List[Dict[str, float]]:
    """Per-layer Hamming stats between two gate patterns."""
    out: List[Dict[str, float]] = []
    for za, zb in zip(gates_a, gates_b):
        flips = (za ^ zb).float().sum().item()
        width = za.numel()
        out.append({"flips": flips, "fraction": flips / max(1, width), "width": width})
    return out


@torch.no_grad()
def _small_path_kernel(
    model: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    mode: str = "routing",
    include_input: bool = True,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Build the path kernel for a handful of samples (P x P).
    Uses the existing collect_path_factors helper.
    """
    device = device or next(model.parameters()).device
    ds = TensorDataset(xs, ys)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    pack = collect_path_factors(
        model,
        loader,
        device=device,
        mode=mode,
        include_input=include_input,
        max_samples=len(ds),
    )
    factors = []
    if pack["X"] is not None:
        factors.append(pack["X"])
    factors.extend(pack["E_list"])

    # K = (X X^T) ∘ ∏_ℓ (E_ℓ E_ℓ^T)
    K = torch.ones((len(ds), len(ds)), device=device, dtype=torch.float32)
    for F in factors:
        K *= F @ F.t()
    return K


@torch.no_grad()
def run_mnist_adversarial_path_check(
    model: nn.Module,
    test_loader: DataLoader,
    *,
    eps: float = 0.15,
    mode: str = "routing",
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Generate a single FGSM adversarial example and compare its path
    activation pattern against (a) the clean source and (b) a target
    sample from the opposite class.

    Returns a dict with predictions, gate flip stats, and small path
    kernel similarities that can be logged or printed.
    """
    device = device or next(model.parameters()).device
    model.eval()

    xb_full, yb_full = next(iter(test_loader))  # test_loader is full-batch in this repo
    xb_full, yb_full = xb_full.to(device), yb_full.to(device)

    # Pick a correctly classified source sample
    with torch.no_grad():
        preds = torch.sign(model(xb_full))
    correct_mask = preds.view(-1) == yb_full.view(-1)
    if not correct_mask.any():
        raise RuntimeError("No correctly classified samples found to attack.")
    idx_src = correct_mask.nonzero(as_tuple=False)[0].item()
    x_src = xb_full[idx_src : idx_src + 1]
    y_src = yb_full[idx_src : idx_src + 1]

    # Pick a target sample from the opposite class (closest in index)
    target_mask = (yb_full.view(-1) == -y_src.item())
    if not target_mask.any():
        raise RuntimeError("Could not find an opposite-class target sample.")
    idx_tgt = target_mask.nonzero(as_tuple=False)[0].item()
    x_tgt = xb_full[idx_tgt : idx_tgt + 1]
    y_tgt = yb_full[idx_tgt : idx_tgt + 1]

    # Craft adversarial example to flip the sign
    x_adv = _fgsm_flip_sign(model, x_src, y_src, eps=eps)

    # Predictions
    y_clean_pred = torch.sign(model(x_src)).item()
    y_adv_pred = torch.sign(model(x_adv)).item()
    y_tgt_pred = torch.sign(model(x_tgt)).item()

    # Gate patterns + differences
    g_clean = _gate_pattern(model, x_src)
    g_adv = _gate_pattern(model, x_adv)
    g_tgt = _gate_pattern(model, x_tgt)
    flips_clean_adv = _gate_differences(g_clean, g_adv)
    flips_clean_tgt = _gate_differences(g_clean, g_tgt)

    # Small path kernel on {clean, adv, target}
    xs = torch.cat([x_src, x_adv, x_tgt], dim=0)
    ys = torch.cat([y_src, y_src, y_tgt], dim=0)  # keep labels for completeness
    K = _small_path_kernel(
        model,
        xs,
        ys,
        mode=mode,
        include_input=True,
        device=device,
    ).detach().cpu()

    # Linf magnitude for reference
    linf = (x_adv - x_src).abs().max().item()

    return {
        "indices": {"src": idx_src, "tgt": idx_tgt},
        "labels": {"src": float(y_src.item()), "tgt": float(y_tgt.item())},
        "preds": {"clean": y_clean_pred, "adv": y_adv_pred, "target": y_tgt_pred},
        "linf_eps_used": eps,
        "linf_delta": linf,
        "gate_flips_clean_vs_adv": flips_clean_adv,
        "gate_flips_clean_vs_target": flips_clean_tgt,
        "path_kernel": {
            "K": K,
            "entries": {
                "clean_clean": float(K[0, 0]),
                "adv_adv": float(K[1, 1]),
                "target_target": float(K[2, 2]),
                "clean_adv": float(K[0, 1]),
                "clean_target": float(K[0, 2]),
                "adv_target": float(K[1, 2]),
            },
        },
    }


def pretty_print_result(res: Dict[str, object]) -> None:
    """Human-friendly print helper."""
    print(f"Source idx {res['indices']['src']} label={res['labels']['src']:+.0f}")
    print(f"Target idx {res['indices']['tgt']} label={res['labels']['tgt']:+.0f}")
    print(f"Preds -> clean: {res['preds']['clean']:+.0f} | adv: {res['preds']['adv']:+.0f} | target: {res['preds']['target']:+.0f}")
    print(f"Linf used ε={res['linf_eps_used']:.3f}, actual ||δ||_∞={res['linf_delta']:.3f}")
    print("\nGate flips clean→adv (count / frac):")
    for l, stats in enumerate(res["gate_flips_clean_vs_adv"]):
        print(f"  L{l}: {stats['flips']:.0f}/{stats['width']} ({stats['fraction']*100:.1f}%)")
    print("\nGate flips clean→target (count / frac):")
    for l, stats in enumerate(res["gate_flips_clean_vs_target"]):
        print(f"  L{l}: {stats['flips']:.0f}/{stats['width']} ({stats['fraction']*100:.1f}%)")
    print("\nPath-kernel entries on [clean, adv, target]:")
    for k, v in res["path_kernel"]["entries"].items():
        print(f"  {k}: {v:.4e}")


# ---------------------------
# Saving helpers (plots/files)
# ---------------------------


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _plot_gate_flips(res: Dict[str, object], out_path: str) -> None:
    flips_adv = res["gate_flips_clean_vs_adv"]
    flips_tgt = res["gate_flips_clean_vs_target"]
    layers = list(range(len(flips_adv)))
    fa = [f["fraction"] for f in flips_adv]
    ft = [f["fraction"] for f in flips_tgt]
    plt.figure(figsize=(6, 3))
    plt.plot(layers, fa, marker="o", label="clean→adv")
    plt.plot(layers, ft, marker="s", label="clean→target")
    plt.xlabel("layer")
    plt.ylabel("flip fraction")
    plt.ylim(0, 1)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_kernel(res: Dict[str, object], out_path: str) -> None:
    K = res["path_kernel"]["K"]
    labels = ["clean", "adv", "target"]
    plt.figure(figsize=(4, 3.6))
    im = plt.imshow(K, cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    plt.title("Path kernel on {clean, adv, target}")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_results(res: Dict[str, object], out_dir: str) -> None:
    _ensure_dir(out_dir)
    # JSON dump
    with open(os.path.join(out_dir, "adversarial_path_check.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda x: x.tolist() if torch.is_tensor(x) else x)
    # Text summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        def _write(s: str): f.write(s + "\n")
        _write(f"Source idx {res['indices']['src']} label={res['labels']['src']:+.0f}")
        _write(f"Target idx {res['indices']['tgt']} label={res['labels']['tgt']:+.0f}")
        _write(f"Preds -> clean: {res['preds']['clean']:+.0f} | adv: {res['preds']['adv']:+.0f} | target: {res['preds']['target']:+.0f}")
        _write(f"Linf used ε={res['linf_eps_used']:.3f}, actual ||δ||_∞={res['linf_delta']:.3f}")
        _write("\nGate flips clean→adv (count / frac):")
        for l, stats in enumerate(res["gate_flips_clean_vs_adv"]):
            _write(f"  L{l}: {stats['flips']:.0f}/{stats['width']} ({stats['fraction']*100:.1f}%)")
        _write("\nGate flips clean→target (count / frac):")
        for l, stats in enumerate(res["gate_flips_clean_vs_target"]):
            _write(f"  L{l}: {stats['flips']:.0f}/{stats['width']} ({stats['fraction']*100:.1f}%)")
        _write("\nPath-kernel entries on [clean, adv, target]:")
        for k, v in res["path_kernel"]["entries"].items():
            _write(f"  {k}: {v:.4e}")
    # Plots
    _plot_gate_flips(res, os.path.join(out_dir, "gate_flips.png"))
    _plot_kernel(res, os.path.join(out_dir, "path_kernel_heatmap.png"))
    # Save raw kernel as .npy
    import numpy as np
    np.save(os.path.join(out_dir, "path_kernel.npy"), res["path_kernel"]["K"].numpy())


# -------------
# CLI entrypoint
# -------------


def _load_model_from_config(cfg, device: str, checkpoint: str) -> nn.Module:
    input_dim = int(cfg["dataset"].get("d", 784))
    widths = cfg["model"]["widths"]
    bias = cfg["model"].get("bias", False)
    activation = cfg["model"].get("activation", "relu")
    model = MLP(input_dim, widths, bias=bias, activation=activation).to(device)
    state = torch.load(checkpoint, map_location=device)
    # Handle either full state dict or wrapped
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="Adversarial path-kernel check on MNIST (binary).")
    ap.add_argument("--config", type=str, default="configs/config_mnist.yaml", help="Config file used for the trained model.")
    ap.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint (.pt).")
    ap.add_argument("--eps", type=float, default=0.15, help="FGSM Linf epsilon.")
    ap.add_argument("--mode", type=str, default="routing", help="Path-kernel mode (routing or routing_gain).")
    ap.add_argument("--out-dir", type=str, default="outputs/adversarial", help="Where to save results.")
    ap.add_argument("--device", type=str, default=None, help="Device string, e.g., cuda:0 or cpu.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_config(cfg, device, args.model_path)

    # Build loaders (test_loader is full-batch)
    _, _, _, test_loader, _ = build_dataloaders(cfg)

    res = run_mnist_adversarial_path_check(
        model,
        test_loader,
        eps=args.eps,
        mode=args.mode,
        device=device,
    )
    save_results(res, args.out_dir)
    print(f"[adversarial] results saved to {args.out_dir}")


if __name__ == "__main__":
    main()

