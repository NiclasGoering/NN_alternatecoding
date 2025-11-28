# Toy Superposition Experiment

This folder reproduces the classic "Toy Models of Superposition" setup with a tiny, self-contained script.

## What it does
- Generates sparse synthetic features with decaying importance.
- Trains two shallow models that share a single weight matrix:
  - `linear`: `x̂ = WᵀW x + b`
  - `relu_output`: `x̂ = ReLU(WᵀW x + b)` (captures the superposition regime).
- Sweeps several sparsity levels and logs:
  - Feature norms, interference, and per-feature dimensionality.
  - Heatmaps of `WᵀW`.
  - Simple path kernel spectra: `(X Xᵀ) ∘ (G Gᵀ)` where `G` is the ReLU routing mask.
  - Orthogonality stats of the learned paths (cosine similarity between columns of `W`).

## Quick start
```bash
pip install -r requirements.txt
python experiments/superposition_toy/run_experiment.py \
  --config experiments/superposition_toy/config.yaml
```

Outputs land in `experiments/superposition_toy/runs/run_<timestamp>/sparsity_*_{linear|relu_output}/`:
- `metrics.json`: losses, cosine stats, and sparsity/model tags.
- `wtw_heatmap.png`, `feature_norms.png`, `interference.png`, `dimensionality.png`
- `path_kernel_spectrum.png`
- `tensors.pt`: raw `W`, bias, path kernel, and per-feature metrics.

Adjust hyperparameters via the config file or CLI `--out-dir` / `--device`.
