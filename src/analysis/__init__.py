"""Analysis tools for path kernels and neural network interpretability."""

from .path_kernel import (
    collect_path_factors,
    compute_path_kernel_eigs,
    compute_classwise_path_kernel_eigs,
    save_spectrum,
)

from .mobility import (
    compute_gate_mobility_lazarus,
    compute_initial_lr_from_target_mobility,
    compute_optimal_lr_path,
    compute_gate_mobility_per_layer_path,
    update_lr_damped_mobility,
    compute_gradient_norms_per_layer,
)

from .kernel_tracking import (
    compute_kernel_matrix,
    compute_path_kernel_matrix,
    compute_top_eigenvalues,
    compute_effective_rank,
    compute_numerical_rank,
    compute_cka,
    compute_wasserstein_distance,
    collect_hidden_layer_features,
    compute_gradient_eigenvalues_per_layer,
    compute_all_kernel_metrics,
    concatenate_gradient_eigenvalues,
)

from .path_analysis import (
    path_embedding,
    run_full_analysis_at_checkpoint,
    plot_eig_spectrum,
    plot_nn_graph_with_paths,
    plot_path_cleanliness,
    plot_embedding_map,
    plot_lineage_sankey,
    plot_centroid_drift_and_tightening,
    plot_path_shapley_bars,
    compute_interchange_intervention_accuracy,
    plot_iia_vs_epoch,
    ablation_waterfall,
    circuit_overlap_matrix,
    flow_centrality_heatmap,
    minimal_subgraph_per_class,
)

__all__ = [
    # Path kernel
    "collect_path_factors",
    "compute_path_kernel_eigs",
    "compute_classwise_path_kernel_eigs",
    "save_spectrum",
    # Mobility
    "compute_gate_mobility_lazarus",
    "compute_initial_lr_from_target_mobility",
    "compute_optimal_lr_path",
    "compute_gate_mobility_per_layer_path",
    "update_lr_damped_mobility",
    "compute_gradient_norms_per_layer",
    # Kernel tracking
    "compute_kernel_matrix",
    "compute_path_kernel_matrix",
    "compute_top_eigenvalues",
    "compute_effective_rank",
    "compute_numerical_rank",
    "compute_cka",
    "compute_wasserstein_distance",
    "collect_hidden_layer_features",
    "compute_gradient_eigenvalues_per_layer",
    "compute_all_kernel_metrics",
    "concatenate_gradient_eigenvalues",
    # Path analysis
    "path_embedding",
    "run_full_analysis_at_checkpoint",
    "plot_eig_spectrum",
    "plot_nn_graph_with_paths",
    "plot_path_cleanliness",
    "plot_embedding_map",
    "plot_lineage_sankey",
    "plot_centroid_drift_and_tightening",
    "plot_path_shapley_bars",
    "compute_interchange_intervention_accuracy",
    "plot_iia_vs_epoch",
    "ablation_waterfall",
    "circuit_overlap_matrix",
    "flow_centrality_heatmap",
    "minimal_subgraph_per_class",
]

