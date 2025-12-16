# Circuit Bank Distillation

This module implements Circuit Bank Distillation: compressing a trained network into a small bank of K circuits by extracting top-K eigenpaths and top-M individual paths per eigenpath.

## Overview

The pruning process:
1. **Compute top-K eigenpaths** from the path kernel eigendecomposition
2. **Extract top-M individual paths** from each eigenpath using beam search
3. **Create pruned network** that only includes units/connections in the extracted paths
4. **Freeze everything except readout** to prevent rediscovery
5. **Train readout only** to maintain accuracy

## Usage

```bash
python prune_circuit_bank.py \
    --model_path <path_to_model_final.pt> \
    --config_path <path_to_config.yaml> \
    --k_values 5 10 15 20 \
    --k_sub_values 1 3 5 10 \
    --epochs 100 \
    --device cuda:0
```

### Arguments

- `--model_path` (required): Path to `model_final.pt` checkpoint
- `--config_path` (optional): Path to config YAML. If not provided, will infer from model structure
- `--k_values`: List of K values (number of eigenpaths). Default: `[5, 10, 15, 20]`
- `--k_sub_values`: List of k_sub values (paths per eigenpath). Default: `[1, 3, 5, 10]`
- `--epochs`: Training epochs for readout. Default: `100`
- `--device`: Device to use. Default: `cuda:0` if available, else `cpu`
- `--mode`: Path kernel mode. Default: `routing_gain`
- `--max_samples`: Max samples for eigenpath computation. Default: `1000`

### Example

```bash
# Prune MNIST model with different K and k_sub values
python prune_circuit_bank.py \
    --model_path outputs/10_12/mnist_run_1_20251210_005021/n50000_lam0.0_alpha1.0_optadam/sgd/model_final.pt \
    --k_values 5 10 15 20 \
    --k_sub_values 1 3 5 10 \
    --epochs 100
```

## Output

Results are saved to `pruning/results/`:

1. **`pruning_results.json`**: Detailed results for each (K, k_sub) combination:
   - Compression factor
   - Train/val/test accuracy and loss
   - Training history
   - Number of paths extracted

2. **`accuracy_vs_k.png`**: Plots showing:
   - Test accuracy vs K (for different k_sub)
   - Test accuracy vs k_sub (for different K)

3. **`compression_vs_accuracy.png`**: Scatter plot of compression factor vs test accuracy

4. **`accuracy_heatmap.png`**: Heatmap of test accuracy across (K, k_sub) combinations

## How It Works

### 1. Eigenpath Extraction
- Computes path kernel eigendecomposition to get top-K eigenpaths
- Each eigenpath is a linear combination of all paths that captures important patterns

### 2. Individual Path Extraction
For each eigenpath:
- Projects eigenpath to path space to get per-layer importance scores
- Uses beam search weighted by eigenpath contributions to find top-M individual paths
- Each path is a sequence `[i0, i1, ..., i_{L-1}]` of unit indices per layer

### 3. Network Pruning
- Collects all unique units that appear in any extracted path
- Creates a new smaller network with only those units
- Copies weights for kept connections from original model

### 4. Readout Training
- Freezes all linear layers and gates
- Trains only the readout layer
- Prevents rediscovery of new circuits

## Expected Results

- **Compression**: 10-20× parameter reduction
- **Accuracy**: ≥98-100% of original test accuracy
- **Interpretability**: Each path is a specific route through the network

## Notes

- The script automatically detects dataset type from `dataset_meta.json` in the model directory
- Currently supports MNIST and Hierarchical XOR datasets
- For other datasets, modify `build_dataloaders_from_model_path()` function

