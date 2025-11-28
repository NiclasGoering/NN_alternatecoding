"""
Standalone toy superposition experiment placed in a fresh folder so it
does not touch any existing code. Inspired by Elhage et al. (2022).

Features:
- Generates sparse synthetic features with decaying importance.
- Trains two shared-weight models:
    * linear:      x̂ = Wᵀ W x + b
    * relu_output: x̂ = ReLU(Wᵀ W x + b)
- Logs feature norms, interference, per-feature dimensionality.
- Plots WᵀW heatmaps and a simple path-kernel spectrum:
      K = (X Xᵀ) ∘ (G Gᵀ) where G is the ReLU routing mask.
- Summarizes orthogonality of learned columns of W.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover - dependency guard
    raise ImportError(
        "matplotlib is required for plotting. Install with `pip install -r requirements.txt`."
    ) from e

torch.set_default_dtype(torch.float32)


# -----------------------------
# Config and utilities
# -----------------------------

@dataclass
class ExperimentConfig:
    n_features: int = 80
    hidden_dim: int = 20
    n_samples: int = 5000
    sparsity_levels: Tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.1)
    importance_decay: float = 0.9
    bias: bool = True
    lr: float = 5e-3
    epochs: int = 2000
    seed: int = 123
    out_dir: str = "experiments/superposition_alt/runs"


def load_config(path: str | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(**raw)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# -----------------------------
# Data generation
# -----------------------------

def generate_sparse_features(
    n_samples: int,
    n_features: int,
    sparsity: float,
    importance_decay: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = torch.bernoulli(
        torch.full((n_samples, n_features), 1.0 - sparsity, device=device)
    )
    values = torch.rand((n_samples, n_features), device=device)
    X = active * values
    idx = torch.arange(n_features, device=device, dtype=torch.float32)
    importance = torch.pow(torch.full_like(idx, importance_decay), idx)
    return X, importance


# -----------------------------
# Models
# -----------------------------

class SuperpositionModel(torch.nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, *, use_relu_output: bool, bias: bool):
        super().__init__()
        self.use_relu_output = use_relu_output
        self.W = torch.nn.Parameter(torch.randn(hidden_dim, n_features) * 0.05)
        self.bias = torch.nn.Parameter(torch.zeros(n_features)) if bias else None

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        h = torch.matmul(x, self.W.t())
        pre = torch.matmul(h, self.W)
        if self.bias is not None:
            pre = pre + self.bias
        out = torch.relu(pre) if self.use_relu_output else pre
        cache = {"h": h, "pre_out": pre, "gates": (pre > 0).float()}
        return out, cache


# -----------------------------
# Training and metrics
# -----------------------------

def weighted_mse(pred: torch.Tensor, target: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * importance.unsqueeze(0)).mean()


def train_model(
    model: SuperpositionModel,
    X: torch.Tensor,
    importance: torch.Tensor,
    epochs: int,
    lr: float,
) -> List[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    for step in range(epochs):
        opt.zero_grad()
        pred, _ = model(X)
        loss = weighted_mse(pred, X, importance)
        loss.backward()
        opt.step()
        if (step + 1) % max(epochs // 10, 1) == 0:
            losses.append(loss.item())
    return losses


def compute_feature_metrics(W: torch.Tensor) -> Dict[str, torch.Tensor]:
    norms = torch.norm(W, dim=0)
    normalized = W / norms.clamp_min(1e-8)
    cos_sq = torch.matmul(normalized.t(), normalized) ** 2
    interference = cos_sq.sum(dim=1) - 1.0
    dimensionality = norms.pow(2) / cos_sq.sum(dim=1).clamp_min(1e-8)
    return {"norms": norms, "interference": interference, "dimensionality": dimensionality}


def compute_path_kernel(X: torch.Tensor, gates: torch.Tensor | None) -> torch.Tensor:
    Kx = torch.matmul(X, X.t())
    if gates is None:
        return Kx
    Kg = torch.matmul(gates, gates.t())
    return Kx * Kg


def orthogonality_summary(M: torch.Tensor) -> Dict[str, float]:
    """
    Computes cosine-similarity stats across columns of M.
    Used for both W (feature directions) and gates (routing patterns).
    """
    if M.ndim != 2:
        raise ValueError("orthogonality_summary expects a 2D matrix.")
    normalized = M / torch.norm(M, dim=0, keepdim=True).clamp_min(1e-8)
    cos = torch.matmul(normalized.t(), normalized)
    off_diag = cos - torch.eye(cos.shape[0], device=M.device)
    abs_off = off_diag.abs()
    return {
        "mean_abs_cos": abs_off.mean().item(),
        "max_abs_cos": abs_off.max().item(),
        "pct_below_0.1": (abs_off < 0.1).float().mean().item(),
        "pct_below_0.2": (abs_off < 0.2).float().mean().item(),
    }


# -----------------------------
# Plot helpers
# -----------------------------

def plot_wtw_heatmap(W: torch.Tensor, out_path: str, title: str) -> None:
    WtW = torch.matmul(W.t(), W).detach().cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.imshow(WtW, cmap="magma")
    plt.colorbar(label="W^T W")
    plt.title(title)
    plt.xlabel("feature j")
    plt.ylabel("feature i")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_feature_bars(values: torch.Tensor, out_path: str, title: str, ylabel: str) -> None:
    xs = np.arange(len(values))
    plt.figure(figsize=(6, 3))
    plt.bar(xs, values.detach().cpu().numpy(), width=0.9)
    plt.title(title)
    plt.xlabel("feature index")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_kernel_spectrum(K: torch.Tensor, out_path: str, title: str) -> None:
    evals = torch.linalg.eigvalsh(K.cpu())
    evals = torch.sort(torch.real(evals), descending=True).values
    xs = np.arange(1, 1 + len(evals))
    plt.figure(figsize=(5, 3))
    plt.plot(xs, evals.numpy(), marker="o", markersize=3)
    plt.yscale("log")
    plt.xlabel("rank")
    plt.ylabel("eigenvalue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# -----------------------------
# Main loop
# -----------------------------

def run_one_setting(
    cfg: ExperimentConfig,
    sparsity: float,
    model_kind: str,
    device: torch.device,
    out_root: str,
) -> Dict[str, float]:
    X, importance = generate_sparse_features(
        cfg.n_samples, cfg.n_features, sparsity, cfg.importance_decay, device
    )
    use_relu = model_kind == "relu_output"
    model = SuperpositionModel(
        n_features=cfg.n_features,
        hidden_dim=cfg.hidden_dim,
        use_relu_output=use_relu,
        bias=cfg.bias,
    ).to(device)

    losses = train_model(model, X, importance, cfg.epochs, cfg.lr)
    with torch.no_grad():
        pred, cache = model(X, return_cache=True)
        final_loss = weighted_mse(pred, X, importance).item()
        metrics = compute_feature_metrics(model.W.detach())
        ortho_w = orthogonality_summary(model.W.detach())
        gates = cache["gates"]
        ortho_gates = orthogonality_summary(gates) if gates is not None else None
        path_kernel = compute_path_kernel(X, cache["gates"] if use_relu else None)
        mean_pair_cos = torch.mean(
            torch.nn.functional.cosine_similarity(pred.flatten(), X.flatten(), dim=0)
        ).item()

    tag = f"sparsity_{sparsity:.2f}_{model_kind}"
    run_dir = os.path.join(out_root, tag)
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "final_loss": final_loss,
                "loss_trace": losses,
                "orthogonality_W": ortho_w,
                "orthogonality_gates": ortho_gates,
                "sparsity": sparsity,
                "model": model_kind,
                "mean_pred_target_cosine": mean_pair_cos,
            },
            f,
            indent=2,
        )
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    plot_wtw_heatmap(model.W.detach(), os.path.join(run_dir, "wtw_heatmap.png"), title=f"W^T W ({tag})")
    plot_feature_bars(metrics["norms"], os.path.join(run_dir, "feature_norms.png"), title=f"Feature norms ({tag})", ylabel="||W_i||")
    plot_feature_bars(metrics["interference"], os.path.join(run_dir, "interference.png"), title=f"Interference ({tag})", ylabel="Σ_j≠i (Wi·Wj)^2")
    plot_feature_bars(metrics["dimensionality"], os.path.join(run_dir, "dimensionality.png"), title=f"Dimensionality ({tag})", ylabel="D_i")
    plot_kernel_spectrum(path_kernel, os.path.join(run_dir, "path_kernel_spectrum.png"), title=f"Path kernel spectrum ({tag})")

    torch.save(
        {
            "W": model.W.detach().cpu(),
            "bias": model.bias.detach().cpu() if model.bias is not None else None,
            "path_kernel": path_kernel.cpu(),
            "feature_metrics": {k: v.cpu() for k, v in metrics.items()},
            "gates": gates.cpu(),
        },
        os.path.join(run_dir, "tensors.pt"),
    )
    return {
        "final_loss": final_loss,
        "sparsity": sparsity,
        "model": model_kind,
        "orthogonality_W": ortho_w,
        "orthogonality_gates": ortho_gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alternative toy superposition experiment (standalone).")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (optional).")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir

    set_seed(cfg.seed)
    device = torch.device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(cfg.out_dir, f"run_{timestamp}")
    ensure_dir(out_root)

    summary: List[Dict[str, float]] = []
    for sparsity in cfg.sparsity_levels:
        for model_kind in ("linear", "relu_output"):
            summary.append(run_one_setting(cfg, sparsity, model_kind, device, out_root))

    # Summary plot: orthogonality of W vs gates across sparsity and model
    def _plot_ortho(field: str, title: str, fname: str):
        plt.figure(figsize=(6, 4))
        for model_kind in ("linear", "relu_output"):
            xs = [s["sparsity"] for s in summary if s["model"] == model_kind]
            ys = [s[field]["mean_abs_cos"] if s[field] is not None else None for s in summary if s["model"] == model_kind]
            plt.plot(xs, ys, marker="o", label=model_kind)
        plt.xlabel("sparsity (P(feature=0))")
        plt.ylabel("mean |cos| (columns)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, fname), dpi=180)
        plt.close()

    _plot_ortho("orthogonality_W", "Orthogonality of W (feature directions)", "orthogonality_W_summary.png")
    _plot_ortho("orthogonality_gates", "Orthogonality of paths (routing/gates)", "orthogonality_paths_summary.png")

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Results saved to {out_root}")


if __name__ == "__main__":
    main()
