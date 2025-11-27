# Path Analysis Module - Complete Rundown

## Overview

The path analysis module (`src/analysis/path_analysis.py`) provides comprehensive tools for analyzing neural network paths through gated MLP architectures. It focuses on understanding how information flows through the network by analyzing routing decisions, gate activations, and path transmittances.

---

## Core Concepts

### Path Embedding e(x)
The **path embedding** is an invariant representation that concatenates per-layer "transmittance" vectors E_ℓ(p,:) across all layers. This creates a signature that captures how information is routed through the network without the Kronecker explosion that would come from enumerating all possible paths.

**Shape**: `[P, sum_l d_l]` where P is the number of samples and d_l are layer widths.

### Transmittance Modes
The module supports three modes for computing transmittance E_ℓ:

1. **`"routing"`**: Binary routing decisions
   - `E = z` (0/1 based on activation)

2. **`"routing_gain"`**: Routing weighted by gate gains
   - `E = z * a_plus + (1-z) * a_minus`
   - Combines routing decision with gate slope parameters

3. **`"routing_posdev"`**: Positive deviation from identity
   - `E = z * max(a_plus - 1, 0)`
   - Captures only the amplification beyond identity

---

## Main Analysis Functions

### 1. `path_embedding()` (lines 46-134)
**Purpose**: Compute invariant path embeddings for samples.

**Inputs**:
- `model`: The neural network model
- `loader`: DataLoader with samples
- `mode`: Transmittance mode ("routing"|"routing_gain"|"routing_posdev")
- `normalize`: Whether to standardize features (default: True)
- `max_samples`: Limit number of samples processed

**Outputs**:
- `E`: Path embedding tensor [P, sum_l d_l]
- `y`: Target values
- `labels`: Integer labels (if available)
- `widths`: Layer widths

**Process**:
1. Forward pass through model with cache enabled
2. Extract routing decisions (z) and gate parameters (a_plus, a_minus)
3. Compute transmittance E per layer based on mode
4. Concatenate all layer transmittances
5. Optionally normalize per-feature

---

### 2. `plot_eig_spectrum()` (lines 145-166)
**Purpose**: Plot eigenvalue spectrum of path kernel matrix.

**Visualization**:
- X-axis: Rank (eigenvalue index)
- Y-axis: Eigenvalue (log scale)
- Shows decay rate of path kernel eigenvalues

**Interpretation**: Faster decay = lower effective dimensionality of path space.

---

### 3. `plot_nn_graph_with_paths()` (lines 264-356)
**Purpose**: Visualize neural network as layered DAG with highlighted top-flow paths.

**Process**:
1. Compute mean transmittance per layer across dataset
2. Use beam search to find top-k paths maximizing:
   - Product of mean transmittance × |weight| across layers
3. Create NetworkX directed graph
4. Highlight top paths with different colors/linewidths

**Visualization**:
- Nodes: Input layer, hidden layer units, output node
- Edges: Only edges used in top-k paths
- Layout: Layered (left-to-right through network)
- Highlights: Top 3 paths shown with increasing line widths

---

### 4. `plot_path_cleanliness()` (lines 364-454)
**Purpose**: Measure how "clean" top paths are - i.e., how often chosen units rank #1.

**Process**:
1. Find top-k beam-searched paths
2. For each sample, compute argmax transmittance per layer
3. For each path, compute fraction of samples where path units match argmax

**Visualization**:
- X-axis: Layer index
- Y-axis: Cleanliness (fraction, 0-1)
- Multiple lines: One per top-k path
- Higher = path units are consistently the most active

**Interpretation**: Cleanliness near 1.0 means the path represents a dominant routing pattern.

---

### 5. `plot_embedding_map()` (lines 461-490)
**Purpose**: Create 2D visualization of path embeddings using dimensionality reduction.

**Methods** (fallback order):
1. **UMAP** (if available): Cosine metric, 30 neighbors
2. **t-SNE** (if sklearn available): Perplexity 30
3. **PCA** (fallback): First 2 principal components

**Visualization**:
- 2D scatter plot of samples in embedding space
- Color-coded by class labels (if available)
- Shows clustering/separation of samples by routing patterns

---

### 6. `plot_lineage_sankey()` (lines 505-587)
**Purpose**: Track cluster evolution over time (multiple checkpoints).

**Process**:
1. Cluster embeddings at each timepoint using KMeans
2. Align clusters across time using Hungarian algorithm (cosine distance)
3. Compute flow counts between aligned clusters
4. Draw Sankey-like diagram

**Visualization**:
- Timepoints as vertical bars (left to right)
- Cluster sizes shown as bar heights
- Flows between clusters as connecting lines
- Line thickness = flow magnitude

**Interpretation**: Shows how path patterns evolve during training.

---

### 7. `plot_centroid_drift_and_tightening()` (lines 594-631)
**Purpose**: Measure cluster dynamics over time.

**Metrics**:
- **Centroid drift**: Mean distance cluster centers move per step
- **Cluster radius**: Average within-cluster variance

**Visualization**:
- X-axis: Epoch/checkpoint index
- Y-axis: Metric value
- Two lines: drift and variance

**Interpretation**:
- Decreasing radius = clusters becoming more coherent
- Decreasing drift = convergence of path patterns

---

### 8. `plot_path_shapley_bars()` (lines 638-690)
**Purpose**: Mutual information analysis of path contributions.

**Process**:
1. Compute mutual information (MI) between each circuit component and labels
2. Compute pairwise synergy: MI(i,j) - MI(i) - MI(j)
3. Plot main effects and synergy heatmap

**Visualization**:
- Bar chart: Main effect MI per circuit
- Heatmap (if ≤24 circuits): Pairwise synergy matrix

**Interpretation**: Higher MI = stronger relationship to target; positive synergy = interaction effects.

---

### 9. `ablation_waterfall()` (lines 698-767)
**Purpose**: Measure impact of ablating top-centrality units.

**Process**:
1. Rank units by flow centrality: (mean transmittance × outgoing |W|)
2. Cumulatively set gate parameters to zero: `a_plus = a_minus = 0`
3. Measure test MSE after each ablation step

**Visualization**:
- X-axis: Ablation step (cumulative)
- Y-axis: Test MSE
- Line shows degradation curve

**Interpretation**: Steeper increase = more critical units being removed.

---

### 10. `flow_centrality_heatmap()` (lines 801-863)
**Purpose**: Visualize unit importance across layers.

**Centrality**: `mean_transmittance × sum(outgoing |weights|)`

**Visualization**:
- Rows: Layers
- Columns: Units within layer
- Color intensity: Flow centrality value
- Heatmap with magma colormap

**Interpretation**: Brighter = more central/important units for information flow.

---

### 11. `minimal_subgraph_per_class()` (lines 871-992)
**Purpose**: Extract class-specific routing subgraphs.

**Process**:
1. Compute class-conditional mean transmittance
2. Score edges: `|W| × mean_transmittance(source)`
3. Keep top fraction (default 15%) of edges per class
4. Visualize as minimal subgraph

**Visualization**:
- NetworkX graph per class
- Nodes: Units at each layer
- Edges: Top-flow edges for that class
- Edge width proportional to score

**Interpretation**: Shows class-specific routing patterns.

**Note**: This analysis is NOT included in the batch plotting script (`generate_path_analysis_plots.py`).

---

### 12. `circuit_overlap_matrix()` (lines 774-793)
**Purpose**: Measure similarity between circuit prototypes (cluster centroids).

**Metric**: Cosine similarity between prototypes

**Visualization**:
- Heatmap: Cosine similarity matrix
- Color scale: -1 to +1 (blue to red)

**Interpretation**: High similarity = circuits share similar routing patterns.

---

### 13. `run_full_analysis_at_checkpoint()` (lines 1000-1072)
**Purpose**: Orchestrator function that runs all major analyses at once.

**Generates**:
1. Eigenvalue spectrum
2. NN graph with top paths
3. Path cleanliness
4. Embedding map
5. Flow centrality heatmap

**Use case**: Called during training at checkpoints to track evolution.

---

## Plotting Script: `generate_path_analysis_plots.py`

### Overview
This script generates all path analysis plots for saved models in batch. It:

1. **Finds all models**: Scans output directory for `n*_lam*/algo_name/model_final.pt`
2. **Loads models**: Reconstructs models from saved state_dicts
3. **Builds dataloaders**: Creates validation/test loaders matching experiment config
4. **Generates plots**: Calls analysis functions and saves all visualizations

### Main Function: `generate_all_plots()`

**Configuration** (lines 393-414):
- `output_dir`: Experiment output directory
- `max_samples_kernel`: Limit for kernel computation (default: 5000)
- `max_samples_embed`: Limit for embedding computation (default: 5000)
- `kernel_k`: Number of eigenvalues to compute (default: 48)
- `mode`: Transmittance mode (default: "routing_gain")

**Plots Generated** (8 total analyses):

1. **Eigenvalue Spectrum** 
   - File: `eig_spectrum_{tag}.png`
   - Shows path kernel eigenvalue decay

2. **NN Graph with Paths**
   - File: `nn_graph_paths_{tag}.png`
   - Top 3 beam-searched paths highlighted

3. **Path Cleanliness**
   - File: `path_cleanliness_{tag}.png`
   - Cleanliness scores for top-5 paths

4. **Embedding Map**
   - File: `path_embedding_{tag}.png`
   - 2D UMAP/t-SNE/PCA visualization

5. **Flow Centrality Heatmap**
   - File: `flow_centrality_{tag}.png`
   - Unit importance across layers

6. **Ablation Waterfall**
   - File: `ablation_waterfall_{tag}.png`
   - Test error vs. cumulative ablation

7. **Path-Shapley Bars**
   - File: `path_shapley_{tag}.png` (and `path_shapley_{tag}_synergy.png`)
   - Mutual information analysis of path kernel eigenvectors
   - Shows main effects and pairwise synergy

8. **Circuit Overlap Matrix**
   - File: `circuit_overlap_{tag}.png`
   - Cosine similarity between cluster prototypes in embedding space

### Output Structure

```
plots/
└── path_analysis_all/
    └── {algo_name}/
        ├── eig_spectrum_{tag}.png
        ├── nn_graph_paths_{tag}.png
        ├── path_cleanliness_{tag}.png
        ├── path_embedding_{tag}.png
        ├── flow_centrality_{tag}.png
        ├── ablation_waterfall_{tag}.png
        └── minimal_subgraphs_{tag}/
            ├── class_0_subgraph.png
            ├── class_1_subgraph.png
            └── ...
```

### Tag Format
`{algo}_n{n_train}_lam{lambda}_final`

Example: `alt_em_closed_form_n500_lam0.001_final`

---

## Key Helper Functions

### `_mean_transmittance_per_layer()` (lines 174-211)
Computes average transmittance per layer across all samples. Used for path finding and centrality.

### `_beam_top_paths()` (lines 214-261)
Beam search algorithm to find top-k paths through network:
- Score = product of (mean transmittance × |weight|) across layers
- Maintains top `beam` candidates at each layer
- Returns top `top_k` final paths

### `_cluster_E()` (lines 497-502)
KMeans clustering of embeddings for lineage/sankey analysis.

---

## Dependencies

**Required**:
- `torch`, `numpy`, `matplotlib`

**Optional** (graceful fallback):
- `scikit-learn`: KMeans, t-SNE, mutual information, Hungarian algorithm
- `umap`: UMAP dimensionality reduction
- `networkx`: Graph visualization
- `scipy`: Linear assignment for cluster alignment

---

## Usage Example

### In training loop:
```python
from src.analysis.path_analysis import run_full_analysis_at_checkpoint

run_full_analysis_at_checkpoint(
    model=model,
    val_loader=val_loader,
    out_dir="outputs/analysis/",
    step_tag=f"epoch_{epoch}",
    kernel_k=48,
    kernel_mode="routing_gain",
)
```

### Batch plotting:
```python
python plots/generate_path_analysis_plots.py
```

Edit hyperparameters in `main()` function, then run.

---

## Summary of Visualizations

| Plot | What it Shows | Key Insight |
|------|---------------|-------------|
| Eigenvalue Spectrum | Decay of path kernel eigenvalues | Effective dimensionality |
| NN Graph + Paths | Network structure with highlighted top paths | Dominant routing patterns |
| Path Cleanliness | How consistently paths rank #1 | Path stability/dominance |
| Embedding Map | 2D projection of path embeddings | Sample clustering by routing |
| Flow Centrality | Unit importance heatmap | Critical nodes for information flow |
| Ablation Waterfall | Error increase with unit removal | Unit criticality |
| Minimal Subgraphs | Class-specific routing patterns | Class-dependent paths |
| Lineage Sankey | Cluster evolution over time | Path pattern dynamics (requires multiple checkpoints) |
| Centroid Drift | Cluster stability over training | Convergence of routing (requires multiple checkpoints) |
| Path-Shapley | Mutual information of circuits | Circuit relevance to target |
| Circuit Overlap | Similarity between prototypes | Shared routing patterns |

**Note**: Lineage Sankey and Centroid Drift require multiple checkpoints over training time, so they are not included in the single-model batch plotting script.

---

This analysis framework provides a comprehensive view of how information flows through gated neural networks, enabling deep insights into learned routing strategies and circuit structure.

