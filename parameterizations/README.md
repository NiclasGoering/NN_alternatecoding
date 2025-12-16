# Parameterization Experiments

This directory contains the configuration and entry point for running neural network training experiments with different parameterization schemes.

## Directory Structure

After restructuring, the code is organized modularly:

```
NN_alternatecoding/
├── parameterizations/
│   ├── run_experiment_parameterization.py  # Clean entry point
│   ├── config_mnist_parameterizations.yaml # Configuration file
│   └── README.md                           # This file
│
└── src/
    ├── models/                  # Model architectures
    │   ├── __init__.py          # Exports all models
    │   ├── lazarus.py           # LazarusMLP (depth-aware residual MLP)
    │   ├── resnet.py            # MLPResNet (skip connections + batchnorm)
    │   └── batchnorm.py         # MLPBatchNorm (batchnorm only)
    │
    ├── training/                # Training loop and initialization
    │   ├── __init__.py          # Exports training functions
    │   ├── trainer.py           # train_with_parameterization()
    │   └── initialization.py    # initialize_parameterization()
    │
    ├── analysis/
    │   ├── mobility.py          # Gate mobility computations (M_g, d_f, LR)
    │   ├── path_kernel.py       # Path kernel analysis
    │   └── ...
    │
    └── utils/
        ├── plotting.py          # All plotting functions
        └── ...
```

## Supported Parameterizations

- **standard**: Xavier/Kaiming initialization
- **mup**: Maximal Update Parametrization (1/sqrt(width) scaling)
- **ntk**: Neural Tangent Kernel parametrization
- **mup_L**: mup with Lazarus depth scaling
- **path**: Path parameterization with LazarusMLP and adaptive layer-wise LR

### Optimizer Suffixes

You can combine parameterizations with different optimizers:
- `standard_adam`, `standard_sgd`, `standard_muon`
- `mup_adam`, `mup_sgd`
- etc.

### BatchNorm Modifier

Add `_batchnorm` to use BatchNorm:
- `standard_batchnorm`, `standard_batchnorm_adam`

## Usage

### Basic Usage

```bash
# From project root
python parameterizations/run_experiment_parameterization.py --config parameterizations/config_mnist_parameterizations.yaml
```

### Custom Config

```bash
python parameterizations/run_experiment_parameterization.py --config my_config.yaml
```

## Configuration File

Edit `config_mnist_parameterizations.yaml` to customize:

```yaml
experiment_name: mnist_parameterizations
seed: 123
device: cuda

dataset:
  name: mnist
  task_type: multiclass
  n_train: 50000
  n_test: 10000
  alpha: 1.0

model:
  architecture: resnet  # Options: standard, resnet, lazarus
  depth: 45
  width: 350
  activation: relu
  bias: true

training:
  epochs: 1000
  batch_size: 1024
  lr_w: 1e-4
  lr_w_path: 1e-4  # Base LR for path parameterization
  automatic: false  # Enable automatic LR from target mobility
  target_mobility: 0.3
  warmup_epochs: 0
  grad_clip_max_norm: 10.0
  optimizer: sgd
  # Can be single value or list to sweep
  parameterization: [standard, mup, path]

logging:
  metrics_every_n_epochs: 250
  compute_metrics: false
  m_g_n_batches: 8
```

## Importing Modules

The modular structure allows clean imports:

```python
# Models
from src.models import MLP, MLPResNet, MLPBatchNorm, LazarusMLP

# Training
from src.training import train_with_parameterization, initialize_parameterization

# Analysis
from src.analysis.mobility import compute_gate_mobility_lazarus

# Plotting
from src.utils.plotting import plot_metrics, plot_comparison
```

## Metrics Tracked

- **Train/Test Loss**: MSE loss per epoch
- **M_g**: Gate Mobility Number (per layer)
- **C_def**: Path Deformation Capacity
- **H_Lambda**: Path Covariance Entropy
- **LR per layer**: Learning rate evolution (for path parameterization)
- **Gradient norms**: Per-layer gradient norms (pre-clipping)

## Multi-GPU Support

When multiple GPUs are available and multiple parameterizations are requested, training runs in parallel across GPUs automatically.

## Legacy Files

The original `train_parameterizations.py` and `mlp_lazarus.py` are retained for reference but can be removed once the new structure is verified.
