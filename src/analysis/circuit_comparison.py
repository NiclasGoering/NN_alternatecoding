# src/analysis/circuit_comparison.py
"""
Compare initial and final networks: circuit overlap and visualizations.
"""

from __future__ import annotations
import os
from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt

from .path_analysis import (
    path_embedding, circuit_overlap_matrix, plot_nn_graph_with_paths
)


@torch.no_grad()
def compare_initial_vs_final_networks(
    model_initial,
    model_final,
    loader,
    out_dir: str,
    *,
    mode: str = "routing",
    max_samples: int = 1000,
    device: Optional[str] = None,
):
    """
    Compare initial and final networks:
    1. Compute circuit overlap matrix
    2. Plot initial and final network visualizations side by side
    
    Args:
        model_initial: Initial (untrained) model
        model_final: Final (trained) model
        loader: DataLoader for computing path embeddings
        out_dir: Directory to save plots
        mode: Path kernel mode (default: "routing")
        max_samples: Maximum samples to use
        device: Device to use (default: from model)
    """
    _ensure_dir(out_dir)
    
    dev = device or next(iter(model_final.parameters())).device
    
    print(f"[circuit_comparison] Computing path embeddings for initial and final models...")
    
    # Step 1: Compute path embeddings for both models
    embed_initial = path_embedding(
        model_initial, loader, device=dev, mode=mode,
        normalize=True, max_samples=max_samples
    )
    embed_final = path_embedding(
        model_final, loader, device=dev, mode=mode,
        normalize=True, max_samples=max_samples
    )
    
    E_initial = embed_initial["E"].numpy()  # (P, D)
    E_final = embed_final["E"].numpy()  # (P, D)
    
    print(f"[circuit_comparison] Initial embedding shape: {E_initial.shape}")
    print(f"[circuit_comparison] Final embedding shape: {E_final.shape}")
    
    # Step 2: Compute circuit overlap
    # Use cluster centroids or representative samples as prototypes
    # For simplicity, we'll use the mean embedding per class if labels are available,
    # or use a subset of samples as prototypes
    
    labels = embed_final.get("labels")
    if labels is not None:
        # Compute centroids per class
        unique_labels = np.unique(labels.numpy())
        prototypes_initial = []
        prototypes_final = []
        
        for label in unique_labels:
            mask = labels.numpy() == label
            if mask.sum() > 0:
                # Use mean embedding per class as prototype
                proto_init = E_initial[mask].mean(axis=0)
                proto_final = E_final[mask].mean(axis=0)
                prototypes_initial.append(proto_init)
                prototypes_final.append(proto_final)
        
        prototypes_initial = np.array(prototypes_initial)  # (n_classes, D)
        prototypes_final = np.array(prototypes_final)  # (n_classes, D)
        
        print(f"[circuit_comparison] Computed {len(prototypes_initial)} class prototypes")
    else:
        # No labels available - use a subset of samples as prototypes
        n_prototypes = min(20, len(E_initial))
        indices = np.linspace(0, len(E_initial) - 1, n_prototypes, dtype=int)
        prototypes_initial = E_initial[indices]
        prototypes_final = E_final[indices]
        
        print(f"[circuit_comparison] Using {n_prototypes} sample prototypes")
    
    # Step 3: Compute overlap between initial and final prototypes
    # Concatenate prototypes to create a combined set
    prototypes_combined = np.vstack([prototypes_initial, prototypes_final])  # (2*n_prototypes, D)
    
    # Plot circuit overlap matrix
    overlap_path = os.path.join(out_dir, "circuit_overlap_initial_vs_final.png")
    circuit_overlap_matrix(
        prototypes_combined,
        overlap_path,
        title="Circuit Overlap: Initial (top) vs Final (bottom)"
    )
    
    # Step 4: Plot network visualizations side by side
    print(f"[circuit_comparison] Generating network visualizations...")
    
    graph_initial_path = os.path.join(out_dir, "network_graph_initial.png")
    graph_final_path = os.path.join(out_dir, "network_graph_final.png")
    graph_combined_path = os.path.join(out_dir, "network_graph_initial_vs_final.png")
    
    # Plot individual graphs
    plot_nn_graph_with_paths(
        model_initial, loader, graph_initial_path,
        mode=mode, beam=24, top_k=5
    )
    plot_nn_graph_with_paths(
        model_final, loader, graph_final_path,
        mode=mode, beam=24, top_k=5
    )
    
    # Create side-by-side comparison plot
    try:
        from PIL import Image
        
        img_initial = Image.open(graph_initial_path)
        img_final = Image.open(graph_final_path)
        
        # Create side-by-side image
        width = img_initial.width + img_final.width + 40  # Add padding
        height = max(img_initial.height, img_final.height) + 60  # Add padding for titles
        
        combined = Image.new('RGB', (width, height), 'white')
        combined.paste(img_initial, (10, 50))
        combined.paste(img_final, (img_initial.width + 30, 50))
        
        # Add titles using PIL (simple text)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((img_initial.width // 2, 10), "Initial Network", fill='black', font=font, anchor='mt')
        draw.text((img_initial.width + 30 + img_final.width // 2, 10), "Final Network", fill='black', font=font, anchor='mt')
        
        combined.save(graph_combined_path, dpi=180)
        print(f"[circuit_comparison] Saved combined network visualization -> {graph_combined_path}")
    except ImportError:
        print(f"[circuit_comparison] PIL not available, skipping combined visualization")
        print(f"[circuit_comparison] Individual visualizations saved: {graph_initial_path}, {graph_final_path}")
    except Exception as e:
        print(f"[circuit_comparison] Warning: Failed to create combined visualization: {e}")
        print(f"[circuit_comparison] Individual visualizations saved: {graph_initial_path}, {graph_final_path}")
    
    print(f"[circuit_comparison] Circuit comparison completed")


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

