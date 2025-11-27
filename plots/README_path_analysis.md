# Path Analysis Plotting Script

## Quick Start

Simply pass your output directory path:

```bash
python plots/generate_path_analysis_plots.py /home/goring/NN_alternatecoding/outputs/24_11/hierarchical_xor_run_3_20251124_191204
```

## Usage

### Basic Usage (pass path as argument)

```bash
python plots/generate_path_analysis_plots.py /path/to/output/directory
```

### With Custom Options

```bash
python plots/generate_path_analysis_plots.py /path/to/output/directory \
    --max-samples-kernel 3000 \
    --max-samples-embed 3000 \
    --kernel-k 32 \
    --mode routing_gain \
    --device cuda
```

### Without Arguments (uses default path)

Edit the default path in the `main()` function, then run:

```bash
python plots/generate_path_analysis_plots.py
```

## Command-Line Options

- `output_dir` (positional): Path to experiment output directory (must contain `config.json`)
- `--plots-dir`: Custom directory to save plots (default: `output_dir/plots/path_analysis_all`)
- `--max-samples-kernel`: Max samples for kernel computation (default: 5000)
- `--max-samples-embed`: Max samples for embedding computation (default: 5000)
- `--kernel-k`: Number of top eigenvalues to compute (default: 48)
- `--mode`: Transmittance mode - `routing`, `routing_gain`, or `routing_posdev` (default: `routing_gain`)
- `--device`: Device to use - `cuda`, `cpu`, or `auto` (default: `auto` - uses CUDA if available, otherwise CPU)

## Directory Structure

The script automatically finds all models matching this pattern:

```
output_dir/
├── config.json
├── n1000_lam1e-2/
│   ├── alt_em_sgd/
│   │   └── model_final.pt
│   ├── sgd_joint/
│   │   └── model_final.pt
│   └── ...
├── n2500_lam1.0/
│   └── ...
└── ...
```

## Output

Plots are saved to: `output_dir/plots/path_analysis_all/{algorithm_name}/`

Each model gets 8 plots:
1. `eig_spectrum_{tag}.png` - Eigenvalue spectrum
2. `nn_graph_paths_{tag}.png` - Network graph with top paths
3. `path_cleanliness_{tag}.png` - Path cleanliness scores
4. `path_embedding_{tag}.png` - 2D embedding visualization
5. `flow_centrality_{tag}.png` - Unit importance heatmap
6. `ablation_waterfall_{tag}.png` - Ablation analysis
7. `path_shapley_{tag}.png` - Mutual information analysis (and synergy heatmap)
8. `circuit_overlap_{tag}.png` - Circuit similarity matrix

## Notes

- The script automatically skips `sgd_relu` models (they don't have gates)
- Models without gates may have limited path analysis
- All analyses use the `routing_gain` mode by default
- Large datasets are automatically subsampled for computational efficiency
- **GPU Support**: By default, the script automatically uses CUDA if available for faster computation. Use `--device cpu` to force CPU usage.

