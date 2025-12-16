# Parameterizations Experiment

This folder contains code for training FFNNs on MNIST with different parameterizations (standard, mup, ntk) and tracking metrics over training.

## Files

- `config_mnist_parameterizations.yaml`: Configuration file for the experiment
- `train_parameterizations.py`: Main training script

## Metrics Tracked

Every `metrics_every_n_epochs` epochs (default: 100), the following metrics are computed:

1. **Train Loss**: MSE loss on training set
2. **Test Loss**: MSE loss on test set
3. **M_g (Gate Mobility Number)**: Measures the energy cost required to flip a gate
   - Formula: `M_g = (η * E[||∇w||]) / E[d_f]`
4. **C_def (Path Deformation Capacity)**: Measures how much the path kernel H differs from input kernel Σ
   - Formula: `C_def = ||H_norm - Σ_norm||_F`
5. **H_Λ (Path Covariance Entropy)**: Entropy of the path overlap matrix Λ
   - Formula: `H_Λ = -∑_{i,j} P(Λ_ij) log P(Λ_ij)`

## Parameterizations

1. **standard**: Standard Xavier/Kaiming initialization
2. **mup (Maximal Update Parametrization)**: 
   - Hidden layers: scale by 1/sqrt(width)
   - Output layer: scale by 1/sqrt(width)
3. **ntk (Neural Tangent Kernel)**: 
   - All layers: scale by 1/sqrt(width)
4. **mup_L (Maximal Update Parametrization + Lazarus depth scaling)**: 
   - Combines mup initialization with Lazarus depth-aware scaling
   - For standard architectures: same as mup
   - For LazarusMLP: mup init + depth scaling (α = 1/sqrt(2*depth)) on branch outputs
5. **path**: Path-based adaptive learning rate
   - Uses Lazarus initialization (depth-aware scaling)
   - After each epoch, computes optimal LR: η ≈ median(d_f) / E[||∇w||]
   - Automatically updates learning rate for next epoch

## Usage

```bash
# Run with default config
python parameterizations/train_parameterizations.py

# Run with custom config
python parameterizations/train_parameterizations.py --config parameterizations/config_mnist_parameterizations.yaml
```

## Configuration

Edit `config_mnist_parameterizations.yaml` to customize:

- `model.architecture`: Architecture type ("standard", "resnet", or "lazarus")
- `model.depth`: Number of hidden layers
- `model.width`: Fixed width for all hidden layers
- `training.parameterization`: List of parameterizations to run (e.g., `[standard, mup, ntk, path]`)
- `training.epochs`: Number of training epochs
- `training.lr_w`: Learning rate
- `training.optimizer`: Optimizer type ("sgd" or "adam")
- `logging.metrics_every_n_epochs`: Frequency of metric computation (default: 100)
- `logging.m_g_n_batches`: Number of batches for M_g computation (default: 16)
- `logging.kernel_max_samples`: Max samples for kernel computation (default: 8192)

## Architectures

1. **standard**: Standard MLP with no skip connections or batch normalization
2. **resnet**: MLP with skip connections and batch normalization (ResNet-style)
3. **lazarus**: Deep residual MLP with depth-aware scaling initialization
   - Structure: Stack of L residual blocks
   - Block: x_{l+1} = x_l + Branch(x_l)
   - Branch: Linear(width, width) → ReLU → Linear(width, width)
   - Initialization: Kaiming Normal with α = 1/sqrt(2*depth) scaling on branch output
   - No BatchNorm, LayerNorm, or Dropout

## Output

Results are saved to `outputs/{experiment_name}_{timestamp}/`:

- `history_{architecture}_{parameterization}.json`: Training history for each architecture/parameterization combination
- `metrics_{architecture}_{parameterization}.png`: Individual plots for each combination
- `metrics_comparison.png`: Comparison plot across all combinations
- `config.json`: Saved configuration

## Dependencies

The script uses functions from `outputs/gates/gate_velocity_with_capacity.py` for computing M_g, C_def, and H_Λ metrics.

