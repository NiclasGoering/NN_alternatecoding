"""Analysis tools for path kernels and neural network interpretability."""

from .path_kernel import (
    collect_path_factors,
    compute_path_kernel_eigs,
    compute_classwise_path_kernel_eigs,
    save_spectrum,
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
    "collect_path_factors",
    "compute_path_kernel_eigs",
    "compute_classwise_path_kernel_eigs",
    "save_spectrum",
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

