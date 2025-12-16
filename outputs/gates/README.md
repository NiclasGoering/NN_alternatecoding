# Gate Velocity / Gate Mobility Number Experiment

This experiment computes the "Gate Mobility Number" (M_g) which measures the energy cost required to flip a gate at different network depths and widths.

## Theory

For a neuron $i$ with pre-activation $h_i(x) = w \cdot x + b$, the distance in weight space to flip its sign is:

$$d_f(i) = \frac{|h_i(x)|}{||x||}$$

The Gate Mobility Number is defined as:

$$M_g = \frac{\eta \mathbb{E}[||\nabla w||]}{\mathbb{E}[d_f]}$$

where:
- $\eta$ is the learning rate
- $\mathbb{E}[||\nabla w||]$ is the expected gradient norm
- $\mathbb{E}[d_f]$ is the expected distance to flip

### Interpretation

- If $M_g \ll 1$: Lazy Regime (Gates are too "expensive" to flip)
- If $M_g \approx 1$: Rich Regime (Learning happens)

## Running the Experiment

From the project root:

```bash
python outputs/gates/gate_velocity_experiment.py
```

## Configuration

The experiment tests:
- **Depths**: [2, 4, 8, 16] layers
- **Widths**: [64, 256, 1024] neurons per layer
- Uses MNIST dataset (from `configs/config_mnist.yaml`)
- Averages over 10 batches
- Batch size: 1024

## Output

The script generates:
1. `gate_mobility_results.json` - Detailed results with M_g values per layer for each architecture
2. `gate_mobility_heatmap_normal.png` - Heatmap of M_g (normal scale) vs Depth and Width
3. `gate_mobility_heatmap_log.png` - Heatmap of M_g (log scale) vs Depth and Width

## GPU Optimization

The script is optimized for GPU execution:
- Uses `non_blocking` transfers
- Accumulates statistics on GPU
- Efficient batched operations
- Automatic GPU memory cleanup

