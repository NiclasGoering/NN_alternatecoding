# src/analysis/path_analysis.py

from __future__ import annotations

import os
import math
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np

# Set non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Disable LaTeX rendering and use plain text formatting
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

# Optional dependencies (graceful fallback)
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_cosine_similarity
    from scipy.optimize import linear_sum_assignment
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    import umap  # type: ignore
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False


# -----------------------------
# Invariant path embedding e(x)
# -----------------------------

@torch.no_grad()
def path_embedding(
    model,
    loader,
    device: Optional[str] = None,
    *,
    mode: str = "routing",   # Only "routing" mode supported (binary indicator)
    normalize: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """
    e(x): concatenate per-layer 'transmittance' vectors E_ℓ(p,:) across layers.
    Shape: [P, sum_l d_l]. This is an invariant 'routing' signature
    (no Kronecker explosion) suitable for clustering/UMAP/t-SNE.
    """
    model.eval()
    dev = device or next(iter(model.parameters())).device

    # Peek to get shapes
    xb0, yb0 = next(iter(loader))
    xb0 = xb0.to(dev)
    _, cache0 = model(xb0, return_cache=True)
    L = len(cache0["z"])
    widths = [z.shape[1] for z in cache0["z"]]

    rows: List[torch.Tensor] = []
    y_rows: List[torch.Tensor] = []
    label_rows: List[torch.Tensor] = []

    seen = 0
    for xb, yb in loader:
        if max_samples is not None and seen >= max_samples:
            break
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        bsz = xb.shape[0]
        if max_samples is not None and (seen + bsz > max_samples):
            xb = xb[: (max_samples - seen)]
            yb = yb[: (max_samples - seen)]
            bsz = xb.shape[0]

        _, cache = model(xb, return_cache=True)
        zs = cache["z"]

        batch_parts = []
        for l in range(L):
            z = zs[l].float()
            # Only routing mode: E = z (binary indicator)
            E = z
            batch_parts.append(E)
        eb = torch.cat(batch_parts, dim=1)  # (B, sum d_l)

        rows.append(eb.detach())
        y_rows.append(yb.detach().view(-1))
        try:
            y_int = yb.detach().view(-1).to("cpu")
            if torch.allclose(y_int.round(), y_int):
                label_rows.append(y_int.long())
        except Exception:
            pass

        seen += bsz

    E = torch.cat(rows, dim=0).to("cpu")
    y = torch.cat(y_rows, dim=0).to("cpu")
    labels = torch.cat(label_rows, dim=0) if len(label_rows) == len(y_rows) else None

    if normalize:
        # per-feature standardization
        mu = E.mean(dim=0, keepdim=True)
        sig = E.std(dim=0, keepdim=True).clamp_min(1e-6)
        E = (E - mu) / sig

    return {"E": E, "y": y, "labels": labels, "widths": widths}


# ------------------------
# Core plotting primitives
# ------------------------

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def plot_eig_spectrum(evals: torch.Tensor, out_path: str, *, title: str = "Eigenvalue Spectrum"):
    _ensure_dir(os.path.dirname(out_path))
    lam = evals.detach().cpu().numpy().astype(np.float64)
    
    # Normalize by first (largest) eigenvalue
    if len(lam) > 0 and lam[0] > 0:
        lam_normalized = lam / lam[0]
    else:
        lam_normalized = lam
    
    xs = np.arange(1, 1 + lam_normalized.shape[0])
    plt.figure(figsize=(6,4))
    plt.plot(xs, lam_normalized, marker="o", markersize=4)
    plt.yscale("log")
    
    # Fix log scale formatting - use proper log formatter
    ax = plt.gca()
    from matplotlib.ticker import LogFormatterSciNotation
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
    ax.yaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 0.4)))
    
    plt.xlabel("rank")
    plt.ylabel("eigenvalue / λ₁")  # Indicate normalization
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.3, which="both")  # Show both major and minor grid
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved spectrum -> {out_path}")


# ---------------------------------
# Graph view + Top-3 flow paths
# ---------------------------------

@torch.no_grad()
def _mean_transmittance_per_layer(model, loader, device, mode="routing") -> List[torch.Tensor]:
    model.eval()
    dev = device or next(iter(model.parameters())).device
    xb0, yb0 = next(iter(loader))
    xb0 = xb0.to(dev)
    _, cache0 = model(xb0, return_cache=True)
    L = len(cache0["z"])

    sums = [0.0 for _ in range(L)]
    counts = 0
    for xb, _ in loader:
        xb = xb.to(dev)
        _, cache = model(xb, return_cache=True)
        zs = cache["z"]
        for l in range(L):
            z = zs[l].float()
            # Only routing mode: E = z (binary indicator)
            E = z
            sums[l] = (sums[l] + E.sum(dim=0)) if isinstance(sums[l], torch.Tensor) else E.sum(dim=0)
        counts += xb.shape[0]
    means = [ (s / counts).to("cpu") for s in sums ]
    return means


def _beam_top_paths(model, mean_E: List[torch.Tensor], beam: int = 24, top_k: int = 3) -> List[List[int]]:
    """
    Approximate top paths by beam search maximizing product of
    mean transmittance × |weight| across layers.
    """
    L = len(mean_E)
    if L == 0:
        return []

    # Pre-pull absolute weights on CPU
    W = [torch.as_tensor(model.linears[l].weight.detach().abs().to("cpu")) for l in range(len(model.linears))]

    # Start at gate layer 0
    scores = mean_E[0].clamp_min(1e-12).log()  # (w0,)
    paths = [[i] for i in range(scores.numel())]

    # Expand from gate layer l -> gate layer l+1 using W_{l+1}
    for l in range(L - 1):
        W_curr = W[l + 1]                   # (w_{l+1} x w_l)
        next_scores, next_paths = [], []

        for s, p in zip(scores.tolist(), paths):
            i = p[-1]                       # index in gate layer l  (0..w_l-1)
            # Sanity checks for dimensions
            assert W_curr.shape[1] == mean_E[l].numel(), f"W_{l+1} in-features != width(l): {W_curr.shape} vs {mean_E[l].shape}"
            assert W_curr.shape[0] == mean_E[l+1].numel(), f"W_{l+1} out-features != width(l+1): {W_curr.shape} vs {mean_E[l+1].shape}"
            trans = W_curr[:, i].clamp_min(1e-12).log() + mean_E[l+1].clamp_min(1e-12).log()  # (w_{l+1},)
            vals, idxs = torch.topk(trans, k=min(beam, trans.numel()))
            for v, j in zip(vals.tolist(), idxs.tolist()):
                next_scores.append(s + v)
                next_paths.append(p + [j])

        if not next_scores:
            break

        if len(next_scores) > beam:
            vals, idxs = torch.topk(torch.tensor(next_scores), k=beam)
            scores = vals
            paths = [next_paths[i] for i in idxs.tolist()]
        else:
            scores = torch.tensor(next_scores)
            paths = next_paths

    # keep top_k
    if scores.numel() > top_k:
        vals, idxs = torch.topk(scores, k=top_k)
        return [paths[i] for i in idxs.tolist()]
    return paths


def plot_nn_graph_with_paths(
    model,
    loader,
    out_path: str,
    *,
    mode: str = "routing",
    beam: int = 24,
    top_k: int = 3,
):
    """
    Draw MLP as layered DAG; highlight top_k beam-searched paths.
    """
    if not HAVE_NX:
        print("[analysis] networkx not installed; skipping NN graph plot.")
        return
    _ensure_dir(os.path.dirname(out_path))

    mean_E = _mean_transmittance_per_layer(model, loader, device=None, mode=mode)
    paths = _beam_top_paths(model, mean_E, beam=beam, top_k=top_k)
    
    # Handle empty paths case
    if len(paths) == 0:
        print(f"[analysis] Warning: No valid paths found for NN graph plot. Skipping.")
        return

    G = nx.DiGraph()
    widths = [model.linears[l].out_features for l in range(len(model.linears))]
    d_in = model.linears[0].in_features

    # Add layer nodes
    layer_x = {}
    x = 0
    # input layer as one "super-node"
    G.add_node(("L0", -1), layer=0)
    layer_x[0] = [("L0", -1)]
    # hidden layers
    for l, w in enumerate(widths, start=1):
        nodes = [(f"L{l}", i) for i in range(w)]
        for n in nodes: G.add_node(n, layer=l)
        layer_x[l] = nodes
    # output node
    Lmax = len(widths) + 1
    G.add_node(("OUT", -1), layer=Lmax)
    layer_x[Lmax] = [("OUT", -1)]

    # Add a sparse set of edges: only those used in top paths
    used_edges = set()
    for p in paths:
        if len(p) == 0:
            continue  # Skip empty paths
        # input -> first hidden
        used_edges.add((("L0", -1), ("L1", p[0])))
        # through hidden layers
        for l in range(1, len(p)):
            used_edges.add(((f"L{l}", p[l-1]), (f"L{l+1}", p[l])))
        # last hidden -> out
        used_edges.add(((f"L{len(p)}", p[-1]), ("OUT", -1)))

    G.add_edges_from(list(used_edges))

    # Positions for layered plot
    pos = {}
    # manual y spacing
    for layer, nodes in layer_x.items():
        n = len(nodes)
        for i, nkey in enumerate(nodes):
            pos[nkey] = (layer, (i - n/2) / max(n,1) * 2.0)

    plt.figure(figsize=(10, 4))
    nx.draw(G, pos, with_labels=False, node_size=60, arrows=False, width=2)
    # highlight each path in a different linewidth
    if len(paths) == 0:
        plt.title("No valid paths found")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"[analysis] saved NN graph (no paths) -> {out_path}")
        return
    for k, p in enumerate(paths):
        if len(p) == 0:
            continue  # Skip empty paths
        edges = [ (("L0",-1), ("L1",p[0])) ]
        for l in range(1, len(p)):
            edges.append(((f"L{l}", p[l-1]), (f"L{l+1}", p[l])))
        edges.append(((f"L{len(p)}", p[-1]), ("OUT",-1)))
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=3+1.5*(top_k-k), edge_color="C{}".format(k%10))
    plt.title("Top-Flow Paths (beam search)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved NN graph with paths -> {out_path}")


# ---------------------------
# Path cleanliness (top-k)
# ---------------------------

@torch.no_grad()
def plot_path_cleanliness(
    model,
    loader,
    out_path: str,
    *,
    mode: str = "routing",
    top_k: int = 5,
):
    """
    For each of the top_k beam paths, report per-layer 'cleanliness':
    fraction of samples where the chosen unit ranks top-1 by transmittance in its layer.
    """
    _ensure_dir(os.path.dirname(out_path))
    mean_E = _mean_transmittance_per_layer(model, loader, device=None, mode=mode)
    paths = _beam_top_paths(model, mean_E, beam=24, top_k=top_k)
    
    # Handle empty paths case
    if len(paths) == 0:
        print(f"[analysis] Warning: No valid paths found for cleanliness plot. Skipping.")
        return

    # collect per-sample E for ranking
    dev = next(iter(model.parameters())).device
    xb0, yb0 = next(iter(loader))
    xb0 = xb0.to(dev)
    _, cache0 = model(xb0, return_cache=True)
    L = len(cache0["z"])

    # We'll stream once to compute argmax per layer per sample
    argmax_counts = [torch.zeros(w, dtype=torch.long) for w in [e.numel() for e in mean_E]]
    total = 0
    # Instead of full histogram, compute per-layer argmax indices list
    argmax_list: List[List[int]] = [[] for _ in range(L)]
    for xb, _ in loader:
        xb = xb.to(dev)
        _, cache = model(xb, return_cache=True)
        zs = cache["z"]

        for l in range(L):
            z = zs[l].float()
            # Only routing mode: E = z (binary indicator)
            E = z
            idx = torch.argmax(E, dim=1)  # (B,)
            argmax_list[l].extend(idx.detach().to("cpu").tolist())
        total += xb.shape[0]

    # cleanliness: for path p, layer l, fraction argmax==p[l]
    fracs = []
    for p in paths:
        if len(p) == 0:
            continue  # Skip empty paths
        layer_fracs = []
        for l, j in enumerate(p):
            if l >= len(argmax_list):
                break  # Path has more layers than we have data for
            hits = sum(1 for idx in argmax_list[l] if idx == j)
            layer_fracs.append(hits / max(1, len(argmax_list[l])))
        if len(layer_fracs) > 0:
            fracs.append(layer_fracs)

    # plot
    plt.figure(figsize=(8, 4 + 0.2*top_k))
    Lp = len(fracs[0]) if fracs else 0
    xs = np.arange(Lp)
    for k, f in enumerate(fracs):
        plt.plot(xs, f, marker="o", label=f"path {k}")
    plt.ylim(1e-3, 1.0)
    plt.yscale("log")
    plt.xlabel("layer")
    plt.ylabel("cleanliness (top-1 share)")
    plt.grid(True, ls="--", alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved cleanliness plot -> {out_path}")


# -----------------------
# Embedding maps (UMAP/tSNE)
# -----------------------

def plot_embedding_map(E: torch.Tensor, labels: Optional[torch.Tensor], out_path: str, *, title: str):
    _ensure_dir(os.path.dirname(out_path))
    X = E.detach().cpu().numpy()
    y = labels.detach().cpu().numpy() if labels is not None else None

    if HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=1)
        Z = reducer.fit_transform(X)
    elif HAVE_SK:
        Z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=1).fit_transform(X)
    else:
        # PCA fallback
        Xc = X - X.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2][None, :]

    plt.figure(figsize=(6,5))
    if y is None:
        plt.scatter(Z[:,0], Z[:,1], s=6, alpha=0.7)
    else:
        classes = np.unique(y)
        for c in classes:
            m = (y == c)
            plt.scatter(Z[m,0], Z[m,1], s=8, alpha=0.8, label=str(c))
        plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved embedding map -> {out_path}")


# ----------------------------
# Lineage Sankey + clustering
# ----------------------------

def _cluster_E(E: torch.Tensor, k: int = 8, seed: int = 1):
    if not HAVE_SK:
        return None, None
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labs = km.fit_predict(E.numpy())
    return labs, km.cluster_centers_


def plot_lineage_sankey(
    E_time: List[torch.Tensor],
    out_path: str,
    *,
    k: int = 8,
    title: str = "Lineage Sankey",
):
    if not HAVE_SK:
        print("[analysis] scikit-learn/scipy missing; skipping Sankey.")
        return
    _ensure_dir(os.path.dirname(out_path))

    # Cluster each time slice
    labels_t = []
    cents_t = []
    for E in E_time:
        labs, cents = _cluster_E(E, k=k)
        labels_t.append(labs)
        cents_t.append(cents)

    # Align clusters across time via Hungarian on cosine distance of centroids
    aligned = [labels_t[0]]
    for t in range(1, len(labels_t)):
        C_prev = cents_t[t-1]
        C_cur  = cents_t[t]
        sim = (C_prev @ C_cur.T)
        # cosine normalize
        sim /= np.linalg.norm(C_prev, axis=1, keepdims=True) + 1e-8
        sim /= np.linalg.norm(C_cur,  axis=1, keepdims=True).T + 1e-8
        cost = 1.0 - sim
        r, c = linear_sum_assignment(cost)
        perm = np.argsort(c)  # map current -> aligned order
        aligned.append(perm[labels_t[t]])

    # Build simple layerwise flow counts
    flows = []
    for t in range(len(aligned) - 1):
        a = aligned[t]
        b = aligned[t+1]
        flow = np.zeros((k, k), dtype=int)
        for i in range(len(a)):
            flow[a[i], b[i]] += 1
        flows.append(flow)

    # Render a simple Sankey using rectangles and lines
    plt.figure(figsize=(10, 5))
    y_offsets = []
    for t in range(len(aligned)):
        counts = np.bincount(aligned[t], minlength=k)
        total = counts.sum()
        ys = np.cumsum(counts) / max(total, 1)
        y0 = np.concatenate([[0.0], ys])
        y_offsets.append(y0)
        # draw bars
        for j in range(k):
            plt.fill_between([t, t+0.2], [y0[j], y0[j]], [y0[j+1], y0[j+1]], alpha=0.4)

    # draw flows
    for t, flow in enumerate(flows):
        total_left = np.bincount(aligned[t], minlength=k)
        total_right = np.bincount(aligned[t+1], minlength=k)
        left_edges = y_offsets[t].copy()
        right_edges = y_offsets[t+1].copy()
        for i in range(k):
            for j in range(k):
                if flow[i,j] == 0: continue
                h = flow[i,j] / max(total_left[i], 1)
                h2 = flow[i,j] / max(total_right[j], 1)
                y0 = left_edges[i]
                y1 = right_edges[j]
                # draw connection
                plt.plot([t+0.2, t+0.8], [y0, y1], lw=2 + 6* (flow[i,j]/flow.max()), alpha=0.5)
                left_edges[i] += h
                right_edges[j] += h2

    plt.title(title)
    plt.xlim(-0.1, len(aligned)-0.1)
    plt.ylim(0, 1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved Sankey -> {out_path}")


# -----------------------------------
# Centroid drift and tightening
# -----------------------------------

def plot_centroid_drift_and_tightening(
    E_time: List[torch.Tensor],
    out_path: str,
    *,
    k: int = 8,
    title: str = "Centroid Drift & Tightening"
):
    if not HAVE_SK:
        print("[analysis] scikit-learn missing; skipping centroid drift plot.")
        return
    _ensure_dir(os.path.dirname(out_path))
    cents = []
    radii = []
    for E in E_time:
        labs, C = _cluster_E(E, k=k)
        cents.append(C)
        # within-cluster variance
        var = 0.0
        for j in range(k):
            Xj = E.numpy()[labs == j]
            if Xj.size == 0: continue
            var += ((Xj - C[j][None,:])**2).mean()
        radii.append(var)

    cents = np.stack(cents, axis=0)   # T x k x D
    drift = np.linalg.norm(cents[1:] - cents[:-1], axis=2).mean(axis=1)  # per step

    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(radii)), radii, label="cluster radius (var)")
    plt.plot(np.arange(1, len(drift)+1), drift, label="centroid drift")
    plt.yscale("log")
    plt.xlabel("epoch index")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.3, which="both")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved centroid drift -> {out_path}")


# -------------------------------------
# Path-Shapley (MI-based interactions)
# -------------------------------------

def plot_path_shapley_bars(
    scores: np.ndarray,  # shape [P, m] circuit scores (e.g., top-m eigenvector projections)
    y: np.ndarray,       # labels (binary or real)
    out_path: str,
    *,
    title: str = "Path-Shapley (MI proxy)",
):
    # Re-check for scikit-learn at runtime (in case it was installed after module import)
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        print("[analysis] scikit-learn missing; skipping path-shapley.")
        raise RuntimeError("scikit-learn is required for Path-Shapley analysis. Please install it: pip install scikit-learn")

    _ensure_dir(os.path.dirname(out_path))

    # main effects
    m = scores.shape[1]
    mi_main = np.zeros(m)
    for j in range(m):
        mi_main[j] = mutual_info_regression(scores[:, [j]], y, random_state=1)[0]

    # pairwise "synergy": MI([i,j]) - MI(i) - MI(j)
    mi_pair = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            s = np.stack([scores[:,i], scores[:,j]], axis=1)
            mi_ij = mutual_info_regression(s, y, random_state=1).sum()
            mi_pair[i,j] = mi_ij - mi_main[i] - mi_main[j]

    # bar plot for main, and show average positive synergy
    plt.figure(figsize=(8,4))
    xs = np.arange(m)
    plt.bar(xs, mi_main, label="main effect (MI)")
    plt.yscale("log")
    plt.xlabel("circuit index")
    plt.ylabel("mutual information")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved path-shapley bars -> {out_path}")

    # heatmap for synergy (optional)
    if m <= 24:
        plt.figure(figsize=(5,4))
        H = mi_pair + mi_pair.T
        plt.imshow(H, cmap="viridis")
        plt.colorbar(label="synergy (MI excess)")
        plt.title("Pairwise synergy")
        plt.tight_layout()
        p2 = out_path.replace(".png", "_synergy.png")
        plt.savefig(p2, dpi=180)
        plt.close()
        print(f"[analysis] saved synergy heatmap -> {p2}")


def compute_path_shapley_metrics(
    scores: np.ndarray,  # shape [P, m] circuit scores
    y: np.ndarray,       # labels
) -> Dict[str, object]:
    """
    Compute Path-Shapley metrics (MI main effects and synergy).
    Returns JSON-serializable dictionary.
    """
    if not HAVE_SK:
        return {
            "path_shapley_mi_main": [],
            "path_shapley_mi_main_mean": 0.0,
            "path_shapley_synergy_mean": 0.0,
        }
    
    try:
        from sklearn.feature_selection import mutual_info_regression
        
        m = scores.shape[1]
        mi_main = np.zeros(m)
        for j in range(m):
            mi_main[j] = mutual_info_regression(scores[:, [j]], y, random_state=1)[0]
        
        # Pairwise synergy: MI([i,j]) - MI(i) - MI(j)
        mi_pair = np.zeros((m, m))
        for i in range(m):
            for j in range(i+1, m):
                s = np.stack([scores[:,i], scores[:,j]], axis=1)
                mi_ij = mutual_info_regression(s, y, random_state=1).sum()
                mi_pair[i,j] = mi_ij - mi_main[i] - mi_main[j]
        
        # Symmetrize
        mi_pair = mi_pair + mi_pair.T
        
        # Compute mean synergy (only upper triangle, excluding diagonal)
        synergy_values = []
        for i in range(m):
            for j in range(i+1, m):
                if mi_pair[i,j] > 0:  # Only positive synergy
                    synergy_values.append(mi_pair[i,j])
        
        return {
            "path_shapley_mi_main": mi_main.tolist(),
            "path_shapley_mi_main_mean": float(mi_main.mean()),
            "path_shapley_synergy_mean": float(np.mean(synergy_values)) if len(synergy_values) > 0 else 0.0,
        }
    except Exception as e:
        print(f"[path_shapley_metrics] Warning: Computation failed: {e}")
        return {
            "path_shapley_mi_main": [],
            "path_shapley_mi_main_mean": 0.0,
            "path_shapley_synergy_mean": 0.0,
        }


def compute_centroid_drift_metrics(
    E_time: List[torch.Tensor],
    k: int = 8,
) -> Dict[str, object]:
    """
    Compute centroid drift and cluster radius metrics.
    Returns JSON-serializable dictionary.
    """
    if not HAVE_SK or len(E_time) < 2:
        return {
            "centroid_drift": [],
            "cluster_radius": [],
        }
    
    try:
        cents = []
        radii = []
        for E in E_time:
            labs, C = _cluster_E(E, k=k)
            if C is None:
                continue
            cents.append(C)
            # Within-cluster variance
            var = 0.0
            E_np = E.numpy() if isinstance(E, torch.Tensor) else E
            for j in range(k):
                Xj = E_np[labs == j]
                if Xj.size == 0:
                    continue
                var += ((Xj - C[j][None,:])**2).mean()
            radii.append(var)
        
        if len(cents) < 2:
            return {
                "centroid_drift": [],
                "cluster_radius": radii,
            }
        
        cents = np.stack(cents, axis=0)   # T x k x D
        drift = np.linalg.norm(cents[1:] - cents[:-1], axis=2).mean(axis=1)  # per step
        
        return {
            "centroid_drift": drift.tolist(),
            "cluster_radius": radii,
        }
    except Exception as e:
        print(f"[centroid_drift_metrics] Warning: Computation failed: {e}")
        return {
            "centroid_drift": [],
            "cluster_radius": [],
        }


def compute_lineage_sankey_metrics(
    E_time: List[torch.Tensor],
    k: int = 8,
) -> Dict[str, object]:
    """
    Compute lineage Sankey flow metrics.
    Returns JSON-serializable dictionary with flow data.
    """
    if not HAVE_SK or len(E_time) < 2:
        return {
            "lineage_flows": [],
            "lineage_cluster_sizes": [],
        }
    
    try:
        # Cluster each time slice
        labels_t = []
        cents_t = []
        for E in E_time:
            labs, cents = _cluster_E(E, k=k)
            if labs is None or cents is None:
                continue
            labels_t.append(labs)
            cents_t.append(cents)
        
        if len(labels_t) < 2:
            return {
                "lineage_flows": [],
                "lineage_cluster_sizes": [],
            }
        
        # Align clusters across time via Hungarian on cosine distance of centroids
        aligned = [labels_t[0]]
        for t in range(1, len(labels_t)):
            C_prev = cents_t[t-1]
            C_cur = cents_t[t]
            sim = (C_prev @ C_cur.T)
            # Cosine normalize
            sim /= np.linalg.norm(C_prev, axis=1, keepdims=True) + 1e-8
            sim /= np.linalg.norm(C_cur, axis=1, keepdims=True).T + 1e-8
            cost = 1.0 - sim
            r, c = linear_sum_assignment(cost)
            perm = np.argsort(c)  # map current -> aligned order
            aligned.append(perm[labels_t[t]])
        
        # Build flow counts
        flows = []
        cluster_sizes = []
        for t in range(len(aligned)):
            counts = np.bincount(aligned[t], minlength=k)
            cluster_sizes.append(counts.tolist())
        
        for t in range(len(aligned) - 1):
            a = aligned[t]
            b = aligned[t+1]
            flow = np.zeros((k, k), dtype=int)
            for i in range(len(a)):
                flow[a[i], b[i]] += 1
            flows.append(flow.tolist())
        
        return {
            "lineage_flows": flows,
            "lineage_cluster_sizes": cluster_sizes,
        }
    except Exception as e:
        print(f"[lineage_sankey_metrics] Warning: Computation failed: {e}")
        return {
            "lineage_flows": [],
            "lineage_cluster_sizes": [],
        }


# --------------------
# Ablation waterfall
# --------------------

@torch.no_grad()
def ablation_waterfall(
    model,
    loader_test,
    out_path: str,
    *,
    top_units_per_layer: int = 8,
    mode: str = "routing",
):
    """
    Rank units by flow centrality (expected transmittance × outgoing |W|),
    ablate cumulatively by zeroing out weights, and measure test error.
    """
    from copy import deepcopy

    _ensure_dir(os.path.dirname(out_path))
    base = deepcopy(model)
    dev = next(iter(model.parameters())).device

    # centrality per unit
    mean_E = _mean_transmittance_per_layer(base, loader_test, device=dev, mode=mode)
    L = len(base.linears)
    flows = []
    for l in range(L):
        if l < L - 1:
            W_next = base.linears[l+1].weight.detach().abs().to("cpu")  # (w_{l+1} x w_l)
            out_sum = W_next.sum(dim=0)                                 # (w_l,)
        else:
            out_sum = base.readout.weight.detach().abs().to("cpu").squeeze(0)  # (w_{L-1},)
        c = (out_sum * mean_E[l].to("cpu"))  # (w_l,)
        # Sanity check: ensure dimensions match
        assert out_sum.numel() == mean_E[l].numel(), f"ablation centrality mismatch: out_sum={out_sum.shape}, mean_E[{l}]={mean_E[l].shape}"
        flows.append(c)

    # build ablation order as list of (layer, unit)
    order = []
    for l, c in enumerate(flows):
        topk = torch.topk(c, k=min(top_units_per_layer, c.numel())).indices.tolist()
        order.extend([(l, u) for u in topk])

    # measure baseline error
    def _mse(model, loader):
        L = n = 0.0
        for xb, yb in loader:
            xb = xb.to(dev); yb = yb.to(dev)
            yhat = model(xb)
            L += torch.mean((yhat - yb)**2).item() * xb.size(0)
            n += xb.size(0)
        return L / max(1.0, n)

    errs = [ _mse(base, loader_test) ]
    # cumulative ablation - zero out weights instead of gates
    cur = base
    for (l, u) in order:
        # Zero out incoming weights to unit u in layer l
        if l > 0:
            cur.linears[l].weight.data[:, u] = 0.0
        # Zero out outgoing weights from unit u in layer l
        if l < L - 1:
            cur.linears[l+1].weight.data[u, :] = 0.0
        else:
            cur.readout.weight.data[0, u] = 0.0
        errs.append(_mse(cur, loader_test))

    # plot
    plt.figure(figsize=(7,4))
    xs = np.arange(len(errs))
    plt.plot(xs, errs, marker="o")
    plt.yscale("log")
    plt.xlabel("ablation step")
    plt.ylabel("test MSE")
    plt.title("Ablation waterfall (cumulative)")
    plt.grid(True, ls="--", alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved ablation waterfall -> {out_path}")


# ---------------------------
# Circuit overlap & clustering
# ---------------------------

def circuit_overlap_matrix(
    prototypes: np.ndarray,  # k x D (e.g., cluster centroids in E-space)
    out_path: str,
    *,
    title: str = "Circuit Overlap (cosine)"
):
    _ensure_dir(os.path.dirname(out_path))
    P = prototypes
    # cosine sim
    num = P @ P.T
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    S = num / (norms * norms.T)
    plt.figure(figsize=(5,4))
    plt.imshow(S, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="cosine")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved circuit overlap -> {out_path}")


# ---------------------------
# Flow centrality heatmap
# ---------------------------

@torch.no_grad()
def flow_centrality_heatmap(
    model,
    loader,
    out_path: str,
    *,
    mode: str = "routing",
):
    _ensure_dir(os.path.dirname(out_path))
    dev = next(iter(model.parameters())).device
    mean_E = _mean_transmittance_per_layer(model, loader, device=dev, mode=mode)

    # node centrality = expected transmittance × sum outgoing |W|
    C_rows = []  # one row per layer, variable width
    L = len(model.linears)
    for l in range(L):
        # outgoing |W| from gate layer l:
        if l < L - 1:
            # use next linear: W_{l+1} : (w_{l+1} x w_l)  -> sum over rows -> (w_l,)
            W_next = model.linears[l+1].weight.detach().abs().to("cpu")
            out_sum = W_next.sum(dim=0)  # (w_l,)
        else:
            # last hidden layer -> readout: (n_classes x w_{L-1}) for multi-class, (1 x w_{L-1}) for binary
            W_ro = model.readout.weight.detach().abs().to("cpu")  # (n_classes, w_{L-1}) or (1, w_{L-1})
            # Sum over output dimension to get total outgoing weight per hidden unit
            if W_ro.dim() == 2 and W_ro.shape[0] > 1:
                # Multi-class: sum over all output classes
                out_sum = W_ro.sum(dim=0)  # (w_{L-1},)
            else:
                # Binary: squeeze to remove singleton dimension
                out_sum = W_ro.squeeze(0)  # (w_{L-1},)
        c = (out_sum * mean_E[l].to("cpu"))  # (w_l,)
        # Sanity check: ensure dimensions match
        assert out_sum.numel() == mean_E[l].numel(), f"centrality mismatch: out_sum={out_sum.shape}, mean_E[{l}]={mean_E[l].shape}"
        C_rows.append(c)  # keep as 1D tensor

    # pad to rectangular for heatmap (layers as rows)
    # Each C_rows[l] has shape (w_l,), we want to pad them to the same length and stack
    max_width = max(r.numel() for r in C_rows)
    padded = []
    for r in C_rows:
        if r.numel() < max_width:
            # Pad with zeros
            pad_size = max_width - r.numel()
            r_padded = torch.cat([r, torch.zeros(pad_size)])
        else:
            r_padded = r
        padded.append(r_padded)
    C = torch.stack(padded)  # (num_layers, max_width)
    M = C.numpy()
    
    plt.figure(figsize=(7, max(3, 0.5 * len(C_rows))))
    # Use vmin/vmax to ensure proper scaling, and ensure data is 2D
    if M.size > 0:
        vmin, vmax = M.min(), M.max()
        if vmax > vmin:
            plt.imshow(M, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        else:
            # All values are the same, use a small range
            plt.imshow(M, aspect="auto", cmap="magma", vmin=vmin-0.1, vmax=vmax+0.1)
    else:
        plt.imshow(M, aspect="auto", cmap="magma")
    plt.colorbar(label="flow centrality")
    plt.yticks(range(len(C_rows)), [f"layer {l+1}" for l in range(len(C_rows))])
    plt.xlabel("unit index")
    plt.title("Flow centrality (node)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved flow centrality heatmap -> {out_path}")


# ---------------------------
# Minimal subgraph per class
# ---------------------------

@torch.no_grad()
def minimal_subgraph_per_class(
    model,
    loader,
    labels: Optional[torch.Tensor],
    out_dir: str,
    *,
    keep_frac: float = 0.15,
    mode: str = "routing",
):
    """
    For each class, build class-conditional flows and save the top-flow edges as a subgraph.
    """
    if not HAVE_NX:
        print("[analysis] networkx missing; skipping minimal subgraph.")
        return
    _ensure_dir(out_dir)
    dev = next(iter(model.parameters())).device

    # collect embeddings and labels
    Epack = path_embedding(model, loader, device=dev, mode=mode, normalize=True)
    E = Epack["E"]
    if labels is None:
        labels = Epack["labels"]
    if labels is None:
        print("[analysis] labels not available; skipping minimal subgraph.")
        return

    labels = labels.numpy()
    classes = np.unique(labels)

    # precompute mean transmittance per class per layer
    xb0, yb0 = next(iter(loader))
    xb0 = xb0.to(dev)
    _, cache0 = model(xb0, return_cache=True)
    L = len(cache0["z"])

    for c in classes:
        # boolean mask of class c (assumes loader order == E order)
        idx = np.where(labels == c)[0]
        if idx.size == 0: continue

        # Compute mean_E restricted to class c
        sums = [0.0 for _ in range(L)]
        counts = 0
        pos = 0
        for xb, yb in loader:
            bsz = xb.shape[0]
            m = ( (pos <= idx) & (idx < pos+bsz) )
            take = idx[(idx >= pos) & (idx < pos+bsz)] - pos
            xb = xb.to(dev)
            _, cache = model(xb, return_cache=True)
            zs = cache["z"]
            for l in range(L):
                z = zs[l].float()
                # Only routing mode: E = z (binary indicator)
                E_l = z
                if take.size > 0:
                    El_sel = E_l.index_select(0, torch.as_tensor(take, device=dev, dtype=torch.long))
                    sums[l] = (sums[l] + El_sel.sum(dim=0)) if isinstance(sums[l], torch.Tensor) else El_sel.sum(dim=0)
            counts += take.size
            pos += bsz
        if counts == 0: continue
        mean_E_c = [ (s / counts).to("cpu") for s in sums ]

        # Edge scores: |W| × mean transmittance of source node
        G = nx.DiGraph()
        # add nodes per layer
        for l in range(len(model.linears)):
            d_out, d_in = model.linears[l].weight.shape
            for i in range(d_in):
                G.add_node((l, i))
            for j in range(d_out):
                G.add_node((l+1, j))
            W = model.linears[l].weight.detach().abs().to("cpu")
            # For layer l, the input transmittance comes from:
            # - l=0: no previous gate, use ones (or skip if problematic)
            # - l>0: use mean_E_c[l-1] which is output of previous gate layer
            if l == 0:
                # For first layer, we don't have a previous gate layer
                # Use uniform weights or skip this layer
                src_weight = torch.ones(d_in, device="cpu")
            else:
                # Use transmittance from previous gate layer output
                prev_E = mean_E_c[l-1]  # shape: (widths[l-1],)
                # Ensure dimensions match
                if prev_E.numel() != d_in:
                    # Dimension mismatch - use uniform weights as fallback
                    src_weight = torch.ones(d_in, device="cpu")
                else:
                    src_weight = prev_E  # (d_in,)
            S_np = (W * src_weight.unsqueeze(0)).numpy()  # (d_out,d_in)
            
            # --- OPTIMIZED BLOCK START ---
            # Calculate threshold once via numpy (fast)
            thr = np.quantile(S_np, 1.0 - keep_frac)
            
            # Get coordinates of all edges passing the threshold in one vectorized op
            # This avoids looping over millions of empty connections
            rows, cols = np.where(S_np >= thr)
            weights = S_np[rows, cols]
            
            # Iterate ONLY over the edges we actually keep (usually < 1% of total)
            # Note: (l, i) is source, (l+1, j) is target. 
            # In S[j, i], j is row (out/target), i is col (in/source)
            for j, i, w in zip(rows, cols, weights):
                G.add_edge((l, i), (l+1, j), weight=float(w))
            # --- OPTIMIZED BLOCK END ---

        out_png = os.path.join(out_dir, f"class_{c}_subgraph.png")
        pos = {}
        for l in range(len(model.linears)+1):
            nodes = [n for n in G.nodes if n[0] == l]
            n = len(nodes)
            for k, nkey in enumerate(nodes):
                pos[nkey] = (l, (k - n/2)/max(1,n)*2.0)

        plt.figure(figsize=(9, 4))
        widths = [G[u][v]["weight"] for u,v in G.edges]
        wmin, wmax = min(widths), max(widths)
        norm = lambda w: 0.5 + 3.5*(w - wmin)/(wmax - wmin + 1e-8)
        nx.draw(G, pos, with_labels=False, node_size=15, arrows=False,
                width=[norm(w) for w in widths], edge_color="tab:blue")
        plt.title(f"Minimal subgraph (class={c})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        print(f"[analysis] saved minimal subgraph (class={c}) -> {out_png}")


# ---------------------------------------------
# Orchestrator: run full analysis at checkpoints
# ---------------------------------------------

@torch.no_grad()
def run_full_analysis_at_checkpoint(
    model,
    val_loader,
    out_dir: str,
    *,
    step_tag: str,
    kernel_k: int = 48,
    kernel_mode: str = "routing",
    include_input_in_kernel: bool = True,
    block_size: int = 1024,
    power_iters: int = 30,
    max_samples_kernel: Optional[int] = None,
    max_samples_embed: Optional[int] = None,
    enable_nn_graph_paths: bool = True,
    dataset_name: Optional[str] = None,
    train_loader = None,
    device: Optional[str] = None,
):
    """
    Call this at epoch/cycle checkpoints. Produces:
      - eigen spectrum
      - NN graph + top paths
      - cleanliness bars
      - embedding maps
      - (optionally) centrality heatmap
    """
    from .path_kernel import compute_path_kernel_eigs

    os.makedirs(out_dir, exist_ok=True)

    # 1) Kernel spectrum
    try:
        kern = compute_path_kernel_eigs(
            model,
            val_loader,
            mode=kernel_mode,
            include_input=include_input_in_kernel,
            k=kernel_k,
            n_iter=power_iters,
            block_size=block_size,
            max_samples=max_samples_kernel,
            verbose=True,
        )
    except Exception as e:
        import traceback
        error_msg = f"Error in compute_path_kernel_eigs: {str(e)}\n{traceback.format_exc()}"
        print(f"[path_analysis] Detailed error: {error_msg}")
        raise
    spec_png = os.path.join(out_dir, f"eig_spectrum_{step_tag}.png")
    plot_eig_spectrum(kern["evals"], spec_png, title=f"Path-kernel spectrum [{step_tag}]")

    # 2) Path Sankey Diagram with eigenpaths
    # Run every 200 epochs (or at final epoch) - SKIP at epoch 0 (untrained model, not useful)
    if enable_nn_graph_paths:
        try:
            epoch_num = int(step_tag.split("_")[-1]) if "epoch_" in step_tag else 99999
            # Run every 200 epochs and at final epoch, but NOT at epoch 0 (too expensive and not useful for untrained model)
            if (epoch_num % 200 == 0 and epoch_num > 0) or "final" in step_tag.lower():
                graph_png = os.path.join(out_dir, f"nn_graph_paths_{step_tag}.png")
                plot_path_sankey_diagram(
                    model, val_loader, graph_png,
                    mode=kernel_mode,
                    top_k=10,  # Show top 10 eigenpaths
                    kernel_k=kernel_k,
                    max_samples=max_samples_kernel,
                    kernel_pack=kern,  # reuse computed kernel to avoid duplicate work
                )
            else:
                if epoch_num == 0:
                    print(f"[path_analysis] Skipping Path Sankey diagram at {step_tag} (too expensive for epoch 0, will run every 200 epochs)")
                else:
                    print(f"[path_analysis] Skipping Path Sankey diagram at {step_tag} (will run every 200 epochs)")
        except Exception as e:
            import traceback
            print(f"[path_analysis] Warning: Failed to create Path Sankey diagram: {e}")
            print(f"[path_analysis] Traceback: {traceback.format_exc()}")
    else:
        print(f"[path_analysis] Path Sankey diagram disabled (enable_nn_graph_paths=False)")

    # 2c) Dream algorithm for MNIST/CIFAR-10 (images that maximally excite top eigenpaths)
    # Run whenever path_analysis runs (except at epoch 0) - SKIP at epoch 0 (untrained model, not useful)
    if dataset_name and dataset_name.lower() in ["mnist", "cifar10"] and train_loader is not None:
        try:
            epoch_num = int(step_tag.split("_")[-1]) if "epoch_" in step_tag else 99999
            # Run whenever path_analysis runs (except epoch 0) - this function is only called at path_analysis checkpoints
            if epoch_num > 0 or "final" in step_tag.lower():
                from .dream_algorithm import dream_algorithm_with_variance, plot_dream_images
                print(f"[path_analysis] Running dream algorithm for {dataset_name.upper()} at {step_tag}...")
                
                # Get image shape (default based on dataset)
                if dataset_name.lower() == "mnist":
                    image_shape = (28, 28)
                elif dataset_name.lower() == "cifar10":
                    image_shape = (32, 32, 3)  # CIFAR-10 is 32x32 RGB
                else:
                    image_shape = (28, 28)
                
                # Dream algorithm needs gradients even though the outer function runs under torch.no_grad.
                # Re-enable grads locally so the optimization graph is built correctly.
                with torch.enable_grad():
                    dream_images, variance_explained = dream_algorithm_with_variance(
                        model=model,
                        train_loader=train_loader,
                        k=15,  # Top 15 eigenpaths - will generate 15 images
                        mode=kernel_mode,
                        max_samples=1000,
                        device=device or next(iter(model.parameters())).device,
                        block_size=block_size,
                        power_iters=power_iters,
                        lr=0.1,
                        n_iter=500,
                        image_shape=image_shape,
                        init_method="random",
                        regularization=0.01,
                    )
                
                # Plot dream images with variance explained in titles
                dream_plot_path = os.path.join(out_dir, f"dream_images_{step_tag}.png")
                plot_dream_images(
                    dream_images, 
                    dream_plot_path, 
                    image_shape=image_shape, 
                    variance_explained=variance_explained,
                    n_cols=5
                )
                
                print(f"[path_analysis] Dream algorithm completed - generated {len(dream_images)} images (one per top eigenpath)")
            else:
                if epoch_num == 0:
                    print(f"[path_analysis] Skipping dream algorithm at {step_tag} (too expensive for epoch 0, will run at path_analysis checkpoints)")
                else:
                    print(f"[path_analysis] Skipping dream algorithm at {step_tag} (unexpected condition)")
        except Exception as e:
            import traceback
            print(f"[path_analysis] Warning: Failed to run dream algorithm: {e}")
            print(f"[path_analysis] Traceback: {traceback.format_exc()}")

    # 3) Cleanliness for top-k paths - SKIP during training for speed (only at end)
    # try:
    #     clean_png = os.path.join(out_dir, f"path_cleanliness_{step_tag}.png")
    #     plot_path_cleanliness(model, val_loader, clean_png, mode=kernel_mode, top_k=5)
    # except Exception as e:
    #     import traceback
    #     print(f"[path_analysis] Warning: Failed to create cleanliness plot: {e}")
    #     print(f"[path_analysis] Traceback: {traceback.format_exc()}")

    # 4) Embedding map - SKIP during training for speed (only at end)
    # Epack = path_embedding(model, val_loader, device=None, mode=kernel_mode, normalize=True, max_samples=max_samples_embed)
    # emb_png = os.path.join(out_dir, f"path_embedding_{step_tag}.png")
    # plot_embedding_map(Epack["E"], Epack["labels"], emb_png, title=f"Path-embedding [{step_tag}]")

    # 5) Flow centrality heatmap - SKIP during training for speed (only at end)
    # central_png = os.path.join(out_dir, f"flow_centrality_{step_tag}.png")
    # flow_centrality_heatmap(model, val_loader, central_png, mode=kernel_mode)


# ---------------------------
# Comprehensive path kernel metrics computation
# ---------------------------

@torch.no_grad()
def compute_path_kernel_metrics(
    model,
    train_loader,
    test_loader,
    *,
    mode: str = "routing",
    k: int = 48,
    max_samples: Optional[int] = 5000,
    device: Optional[str] = None,
    block_size: int = 1024,
    power_iters: int = 30,
) -> Dict[str, object]:
    """
    Compute comprehensive path kernel metrics efficiently.
    Returns a dictionary with all metrics (scalars and lists, JSON-serializable).
    
    Metrics computed:
    - effective_rank: Effective rank of path kernel
    - top_eigenvalue: Largest eigenvalue
    - eigenvalue_sum: Sum of top-k eigenvalues
    - variance_explained_train: Variance explained by top-k eigenfunctions on train (sum)
    - variance_explained_test: Variance explained by top-k eigenfunctions on test (sum)
    - variance_explained_train_per_component: List of variance explained per eigenvalue (top 35)
    - variance_explained_test_per_component: List of variance explained per eigenvalue (top 35)
    - variance_explained_train_top10: Sum of top 10 components
    - variance_explained_test_top10: Sum of top 10 components
    """
    from .path_kernel import compute_path_kernel_eigs
    
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    metrics = {}
    
    try:
        # Compute path kernel eigenvalues/eigenvectors for train
        kern_train = compute_path_kernel_eigs(
            model, train_loader, device=dev, mode=mode, include_input=True,
            k=k, n_iter=power_iters, block_size=block_size, max_samples=max_samples, verbose=False
        )
        
        # Keep tensors on GPU for computation (faster on H100)
        evals = kern_train["evals"].to(dev) if isinstance(kern_train["evals"], torch.Tensor) else kern_train["evals"]
        evecs = kern_train["evecs"].to(dev) if isinstance(kern_train["evecs"], torch.Tensor) else kern_train["evecs"]
        y_train = kern_train.get("y")
        if y_train is not None and isinstance(y_train, torch.Tensor):
            y_train = y_train.to(dev)
        
        # Effective rank (compute on GPU)
        metrics["path_kernel_effective_rank"] = _effective_rank(evals)
        
        # Top eigenvalue
        metrics["path_kernel_top_eigenvalue"] = evals[0].item() if len(evals) > 0 else 0.0
        
        # Top 10 eigenvalues
        top_k_evals = min(10, len(evals))
        metrics["path_kernel_top10_eigenvalues"] = evals[:top_k_evals].cpu().numpy().tolist()
        
        # Sum of eigenvalues
        metrics["path_kernel_eigenvalue_sum"] = evals.sum().item()
        
        # Variance explained by eigenfunctions (per component, up to k) - compute on GPU
        if y_train is not None:
            # Convert to float if needed (for multi-class integer labels)
            if y_train.dtype in (torch.long, torch.int64, torch.int32):
                y_train = y_train.float()
            y_train_centered = y_train - y_train.mean()
            y_train_var = y_train_centered.var().item()
            if y_train_var > 1e-12:
                y_train_norm = y_train_centered / (torch.norm(y_train_centered) + 1e-8)
                alignments = (y_train_norm @ evecs) ** 2  # (k,) - computed on GPU
                # Save all components (up to k) - move to CPU only for final conversion
                metrics["path_kernel_variance_explained_train_per_component"] = alignments.cpu().numpy().tolist()
                metrics["path_kernel_variance_explained_train"] = alignments.sum().item()
                metrics["path_kernel_variance_explained_train_top10"] = alignments[:min(10, len(alignments))].sum().item()
            else:
                metrics["path_kernel_variance_explained_train_per_component"] = [0.0] * len(evecs)
                metrics["path_kernel_variance_explained_train"] = 0.0
                metrics["path_kernel_variance_explained_train_top10"] = 0.0
        else:
            metrics["path_kernel_variance_explained_train_per_component"] = [0.0] * len(evecs)
        
        # Compute for test set
        if test_loader is not None:
            try:
                kern_test = compute_path_kernel_eigs(
                    model, test_loader, device=dev, mode=mode, include_input=True,
                    k=k, n_iter=power_iters, block_size=block_size, max_samples=max_samples, verbose=False
                )
                y_test = kern_test.get("y")
                if y_test is not None:
                    y_test = y_test.to(dev)  # Keep on GPU
                    evecs_test = kern_test["evecs"].to(dev)  # Keep on GPU
                    # Convert to float if needed (for multi-class integer labels)
                    if y_test.dtype in (torch.long, torch.int64, torch.int32):
                        y_test = y_test.float()
                    y_test_centered = y_test - y_test.mean()
                    y_test_var = y_test_centered.var().item()
                    if y_test_var > 1e-12:
                        y_test_norm = y_test_centered / (torch.norm(y_test_centered) + 1e-8)
                        alignments_test = (y_test_norm @ evecs_test) ** 2  # Computed on GPU
                        # Save all components (up to k) - move to CPU only for final conversion
                        metrics["path_kernel_variance_explained_test_per_component"] = alignments_test.cpu().numpy().tolist()
                        metrics["path_kernel_variance_explained_test"] = alignments_test.sum().item()
                        metrics["path_kernel_variance_explained_test_top10"] = alignments_test[:min(10, len(alignments_test))].sum().item()
                    else:
                        metrics["path_kernel_variance_explained_test_per_component"] = [0.0] * len(evecs_test)
                        metrics["path_kernel_variance_explained_test"] = 0.0
                        metrics["path_kernel_variance_explained_test_top10"] = 0.0
                else:
                    metrics["path_kernel_variance_explained_test_per_component"] = [0.0] * len(kern_test["evecs"])
            except Exception as e:
                print(f"[path_kernel_metrics] Warning: Test set computation failed: {e}")
        
    except Exception as e:
        print(f"[path_kernel_metrics] Warning: Computation failed: {e}")
        # Return empty dict with zeros
        metrics = {
            "path_kernel_effective_rank": 0.0,
            "path_kernel_top_eigenvalue": 0.0,
            "path_kernel_top10_eigenvalues": [0.0] * 10,
            "path_kernel_eigenvalue_sum": 0.0,
            "path_kernel_variance_explained_train": 0.0,
            "path_kernel_variance_explained_test": 0.0,
            "path_kernel_variance_explained_train_per_component": [],
            "path_kernel_variance_explained_test_per_component": [],
        }
    
    return metrics


# ---------------------------
# New analyses: variance explained, CKA, effective rank
# ---------------------------

def _effective_rank(evals: torch.Tensor) -> float:
    """
    Compute effective rank from eigenvalues.
    Effective rank = exp(entropy) where entropy = -sum(p_i * log(p_i))
    and p_i = lambda_i / sum(lambda)
    """
    evals = evals[evals > 0]  # Only positive eigenvalues
    if len(evals) == 0:
        return 0.0
    total = evals.sum()
    if total <= 0:
        return 0.0
    p = evals / total
    p = p[p > 1e-12]  # Avoid log(0)
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


@torch.no_grad()
def compute_kernel_alignment_layerwise(
    model,
    train_loader,
    test_loader,
    out_path: str,
    *,
    mode: str = "routing",
    k: int = 48,
    max_samples: Optional[int] = 5000,
    title: str = "Kernel Alignment (Variance Explained)",
):
    """
    Compute variance explained by path kernel eigenfunctions vs rank, layer-wise.
    For each layer depth (1, 2, 3, ...), compute kernel using only up to that layer,
    then compute alignment with targets on train and test sets.
    """
    _ensure_dir(os.path.dirname(out_path))
    from .path_kernel import collect_path_factors, HadamardGramOperator, top_eigenpairs_block_power
    
    device = next(iter(model.parameters())).device
    model.eval()
    
    # Collect factors for train and test
    pack_train = collect_path_factors(model, train_loader, device, mode=mode, include_input=True, max_samples=max_samples)
    pack_test = collect_path_factors(model, test_loader, device, mode=mode, include_input=True, max_samples=max_samples)
    
    L = pack_train["meta"]["depth"]
    y_train = pack_train["y"]
    y_test = pack_test["y"] if pack_test["y"] is not None else None
    
    if y_train is None:
        print("[analysis] Warning: No targets available for kernel alignment")
        return
    
    # Center targets
    # Convert to float if needed (for multi-class integer labels)
    if y_train.dtype in (torch.long, torch.int64, torch.int32):
        y_train = y_train.float()
    y_train_centered = y_train - y_train.mean()
    y_train_var = y_train_centered.var().item()
    if y_test is not None:
        # Convert to float if needed (for multi-class integer labels)
        if y_test.dtype in (torch.long, torch.int64, torch.int32):
            y_test = y_test.float()
        y_test_centered = y_test - y_test.mean()
        y_test_var = y_test_centered.var().item()
    
    # For each layer depth, compute kernel up to that layer
    train_alignments = []
    test_alignments = []
    ranks = []
    
    for depth in range(1, L + 1):
        # Build factors up to depth
        factors_train = []
        if pack_train["X"] is not None:
            factors_train.append(pack_train["X"])
        factors_train.extend(pack_train["E_list"][:depth])
        
        factors_test = []
        if pack_test["X"] is not None:
            factors_test.append(pack_test["X"])
        factors_test.extend(pack_test["E_list"][:depth])
        
        # Compute kernel operator
        op_train = HadamardGramOperator(factors_train, device=device, dtype=torch.float32, block_size=1024)
        
        # Compute top eigenpairs
        P_train = factors_train[0].shape[0]
        k_actual = min(k, P_train - 1)
        evals, evecs = top_eigenpairs_block_power(op_train, k=k_actual, n_iter=30, verbose=False)
        
        # Compute alignment: project targets onto eigenfunctions
        # Variance explained = sum of (y^T @ evec_i)^2 / ||y||^2
        y_train_norm = y_train_centered / (torch.norm(y_train_centered) + 1e-8)
        alignments = (y_train_norm @ evecs) ** 2  # (k,)
        var_explained_train = alignments.sum().item()
        train_alignments.append(var_explained_train)
        
        if y_test is not None:
            # Compute test kernel and align
            op_test = HadamardGramOperator(factors_test, device=device, dtype=torch.float32, block_size=1024)
            P_test = factors_test[0].shape[0]
            k_test = min(k, P_test - 1)
            evals_test, evecs_test = top_eigenpairs_block_power(op_test, k=k_test, n_iter=30, verbose=False)
            y_test_norm = y_test_centered / (torch.norm(y_test_centered) + 1e-8)
            alignments_test = (y_test_norm @ evecs_test) ** 2
            var_explained_test = alignments_test.sum().item()
            test_alignments.append(var_explained_test)
        else:
            test_alignments.append(0.0)
        
        ranks.append(depth)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, train_alignments, marker="o", label="Train", linewidth=2)
    if y_test is not None:
        plt.plot(ranks, test_alignments, marker="s", label="Test", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Layer depth")
    plt.ylabel("Variance explained (kernel alignment)")
    plt.title(title)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved kernel alignment layerwise -> {out_path}")


@torch.no_grad()
def compute_cka_initial_vs_current(
    model_initial,
    model_current,
    loader,
    out_path: str,
    *,
    mode: str = "routing",
    max_samples: Optional[int] = 5000,
    title: str = "CKA: Initial vs Current Path Kernel",
):
    """
    Compute Centered Kernel Alignment (CKA) between initial and current path kernels.
    Uses efficient computation: CKA(K1, K2) = ||K1^T @ K2||_F^2 / (||K1||_F^2 * ||K2||_F^2)
    """
    _ensure_dir(os.path.dirname(out_path))
    from .path_kernel import collect_path_factors, HadamardGramOperator
    
    device = next(iter(model_initial.parameters())).device
    
    # Collect factors for both models
    pack_initial = collect_path_factors(model_initial, loader, device, mode=mode, include_input=True, max_samples=max_samples)
    pack_current = collect_path_factors(model_current, loader, device, mode=mode, include_input=True, max_samples=max_samples)
    
    # Build factors
    factors_initial = []
    if pack_initial["X"] is not None:
        factors_initial.append(pack_initial["X"])
    factors_initial.extend(pack_initial["E_list"])
    
    factors_current = []
    if pack_current["X"] is not None:
        factors_current.append(pack_current["X"])
    factors_current.extend(pack_current["E_list"])
    
    # Compute kernel operators
    op_initial = HadamardGramOperator(factors_initial, device=device, dtype=torch.float32, block_size=1024)
    op_current = HadamardGramOperator(factors_current, device=device, dtype=torch.float32, block_size=1024)
    
    P = factors_initial[0].shape[0]
    # Efficient CKA computation using HSIC-like formula
    # Sample random vectors to estimate trace products
    n_samples = min(100, P)  # Use fewer samples for efficiency
    indices = torch.randperm(P, device=device)[:n_samples]
    I_sample = torch.eye(P, device=device, dtype=torch.float32)[:, indices]  # (P, n_samples)
    
    K1_sample = op_initial.mm(I_sample)  # (P, n_samples)
    K2_sample = op_current.mm(I_sample)   # (P, n_samples)
    
    # Center kernels
    K1_centered = K1_sample - K1_sample.mean(dim=0, keepdim=True) - K1_sample.mean(dim=1, keepdim=True) + K1_sample.mean()
    K2_centered = K2_sample - K2_sample.mean(dim=0, keepdim=True) - K2_sample.mean(dim=1, keepdim=True) + K2_sample.mean()
    
    # Compute CKA using sampled approximation
    # CKA ≈ trace(K1^T @ K2)^2 / (trace(K1^T @ K1) * trace(K2^T @ K2))
    trace_K1K2 = torch.trace(K1_centered.T @ K2_centered)
    trace_K1K1 = torch.trace(K1_centered.T @ K1_centered)
    trace_K2K2 = torch.trace(K2_centered.T @ K2_centered)
    
    cka = (trace_K1K2 ** 2 / (trace_K1K1 * trace_K2K2 + 1e-12)).item()
    
    # Plot single value
    plt.figure(figsize=(6, 4))
    plt.bar([0], [cka], width=0.5)
    plt.ylabel("CKA")
    plt.ylim(1e-3, 1.0)
    plt.yscale("log")
    plt.title(title)
    plt.xticks([0], ["Initial vs Current"])
    plt.grid(True, ls="--", alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved CKA -> {out_path} (value: {cka:.4f})")
    return cka


@torch.no_grad()
def plot_effective_rank_vs_epoch(
    model,
    loader,
    out_path: str,
    *,
    mode: str = "routing",
    k: int = 48,
    max_samples: Optional[int] = 5000,
    title: str = "Effective Rank of Path Kernel",
):
    """
    Compute effective rank of path kernel.
    Effective rank = exp(entropy) where entropy is computed from normalized eigenvalues.
    """
    _ensure_dir(os.path.dirname(out_path))
    from .path_kernel import compute_path_kernel_eigs
    
    device = next(iter(model.parameters())).device
    kern = compute_path_kernel_eigs(
        model, loader, device=device, mode=mode, include_input=True,
        k=k, n_iter=30, block_size=1024, max_samples=max_samples, verbose=False
    )
    
    evals = kern["evals"]
    eff_rank = _effective_rank(evals)
    
    # Plot effective rank
    plt.figure(figsize=(6, 4))
    plt.bar([0], [eff_rank], width=0.5)
    plt.ylabel("Effective Rank")
    plt.yscale("log")
    plt.title(title)
    plt.xticks([0], ["Path Kernel"])
    plt.grid(True, ls="--", alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved effective rank -> {out_path} (value: {eff_rank:.2f})")
    return eff_rank


@torch.no_grad()
def plot_variance_explained_vs_rank(
    model,
    train_loader,
    test_loader,
    out_path: str,
    *,
    mode: str = "routing",
    k: int = 48,
    max_samples: Optional[int] = 5000,
    title: str = "Variance Explained vs Eigenvalue Rank",
):
    """
    Plot variance explained by top-k eigenfunctions vs rank on train and test sets.
    """
    _ensure_dir(os.path.dirname(out_path))
    from .path_kernel import compute_path_kernel_eigs
    
    device = next(iter(model.parameters())).device
    
    # Compute kernels for train and test
    kern_train = compute_path_kernel_eigs(
        model, train_loader, device=device, mode=mode, include_input=True,
        k=k, n_iter=30, block_size=1024, max_samples=max_samples, verbose=False
    )
    
    kern_test = compute_path_kernel_eigs(
        model, test_loader, device=device, mode=mode, include_input=True,
        k=k, n_iter=30, block_size=1024, max_samples=max_samples, verbose=False
    )
    
    # Get targets
    y_train = kern_train.get("y")
    y_test = kern_test.get("y")
    
    if y_train is None:
        print("[analysis] Warning: No targets available for variance explained plot")
        return
    
    # Center targets
    # Convert to float if needed (for multi-class integer labels)
    if y_train.dtype in (torch.long, torch.int64, torch.int32):
        y_train = y_train.float()
    y_train_centered = y_train - y_train.mean()
    y_train_var = y_train_centered.var()
    
    # Project targets onto eigenfunctions
    evecs_train = kern_train["evecs"]  # (P_train, k)
    y_train_norm = y_train_centered / (torch.norm(y_train_centered) + 1e-8)
    alignments_train = (y_train_norm @ evecs_train) ** 2  # (k,)
    var_explained_train = torch.cumsum(alignments_train, dim=0)  # Cumulative
    
    var_explained_test = None
    if y_test is not None:
        y_test_centered = y_test - y_test.mean()
        y_test_var = y_test_centered.var()
        y_test_norm = y_test_centered / (torch.norm(y_test_centered) + 1e-8)
        # Project onto test eigenfunctions
        evecs_test = kern_test["evecs"]
        alignments_test = (y_test_norm @ evecs_test) ** 2
        var_explained_test = torch.cumsum(alignments_test, dim=0)
    
    # Plot
    ranks = np.arange(1, k + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, var_explained_train.cpu().numpy(), marker="o", label="Train", linewidth=2, markersize=4)
    if var_explained_test is not None:
        plt.plot(ranks, var_explained_test.cpu().numpy(), marker="s", label="Test", linewidth=2, markersize=4)
    plt.yscale("log")
    plt.xlabel("Eigenvalue rank")
    plt.ylabel("Cumulative variance explained")
    plt.title(title)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[analysis] saved variance explained vs rank -> {out_path}")


# ---------------------------
# Checkpoint-based metrics computation
# ---------------------------

@torch.no_grad()
def compute_checkpoint_metrics(
    model,
    loader,
    *,
    mode: str = "routing",
    max_samples: Optional[int] = 5000,
    device: Optional[str] = None,
    k_clusters: int = 8,
) -> Dict[str, object]:
    """
    Compute checkpoint-based metrics that can be computed at a single checkpoint.
    Returns JSON-serializable metrics.
    
    Metrics computed:
    - path_shapley_main: Main mutual information effects for top eigenfunctions
    - path_shapley_synergy_mean: Mean pairwise synergy
    - circuit_overlap_matrix: Cosine similarity matrix between cluster centroids
    - circuit_overlap_mean: Mean overlap (excluding diagonal)
    """
    if not HAVE_SK:
        return {}
    
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    metrics = {}
    
    try:
        # Get path embeddings
        Epack = path_embedding(model, loader, device=dev, mode=mode, normalize=True, max_samples=max_samples)
        E = Epack["E"]  # (P, D)
        y = Epack.get("labels") or Epack.get("y")
        
        if y is None:
            return metrics
        
        E_np = E.numpy()
        y_np = y.numpy()
        
        # Path-Shapley: Use top eigenfunctions of path kernel
        from .path_kernel import compute_path_kernel_eigs
        kern = compute_path_kernel_eigs(
            model, loader, device=dev, mode=mode, include_input=True,
            k=min(24, E.shape[0] - 1), n_iter=30, block_size=1024, max_samples=max_samples, verbose=False
        )
        evecs = kern.get("evecs")
        if evecs is not None:
            evecs_np = evecs.detach().cpu().numpy()  # (P, k)
            top_m = min(24, evecs_np.shape[1])
            scores = evecs_np[:, :top_m]
            
            # Compute Path-Shapley metrics
            from sklearn.feature_selection import mutual_info_regression
            
            # Main effects
            mi_main = np.zeros(top_m)
            for j in range(top_m):
                mi_main[j] = mutual_info_regression(scores[:, [j]], y_np, random_state=1)[0]
            
            # Pairwise synergy
            mi_pair = np.zeros((top_m, top_m))
            for i in range(top_m):
                for j in range(i+1, top_m):
                    s = np.stack([scores[:,i], scores[:,j]], axis=1)
                    mi_ij = mutual_info_regression(s, y_np, random_state=1).sum()
                    mi_pair[i,j] = mi_ij - mi_main[i] - mi_main[j]
            
            metrics["path_shapley_main"] = mi_main.tolist()
            metrics["path_shapley_main_mean"] = float(mi_main.mean())
            metrics["path_shapley_main_max"] = float(mi_main.max())
            # Average positive synergy
            positive_synergy = mi_pair[mi_pair > 0]
            metrics["path_shapley_synergy_mean"] = float(positive_synergy.mean()) if len(positive_synergy) > 0 else 0.0
            metrics["path_shapley_synergy_max"] = float(mi_pair.max())
        
        # Circuit Overlap: Cluster embeddings and compute overlap
        if E_np.shape[0] > k_clusters:
            from sklearn.cluster import KMeans
            k_actual = min(k_clusters, E_np.shape[0] - 1, E_np.shape[0] // 10)
            if k_actual >= 2:
                km = KMeans(n_clusters=k_actual, random_state=1, n_init="auto")
                km.fit(E_np)
                prototypes = km.cluster_centers_  # (k, D)
                
                # Compute cosine similarity matrix
                P = prototypes
                num = P @ P.T
                norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
                S = num / (norms * norms.T)
                
                # Mean overlap (excluding diagonal)
                mask = ~np.eye(S.shape[0], dtype=bool)
                metrics["circuit_overlap_mean"] = float(S[mask].mean())
                metrics["circuit_overlap_max"] = float(S[mask].max())
                metrics["circuit_overlap_min"] = float(S[mask].min())
                metrics["circuit_overlap_std"] = float(S[mask].std())
                # Store full matrix as list of lists (for JSON)
                metrics["circuit_overlap_matrix"] = S.tolist()
        
    except Exception as e:
        print(f"[checkpoint_metrics] Warning: Failed to compute metrics: {e}")
    
    return metrics


# ---------------------------
# Dominant k paths analysis
# ---------------------------

@torch.no_grad()
def _compute_path_transmittance_feature_from_cache(
    model,
    cached_zs: List[torch.Tensor],
    path: List[int],
    device: Optional[str] = None,
    mode: str = "routing",
    weight_aware: bool = True,
) -> torch.Tensor:
    """
    Compute transmittance feature for a single path from cached z values.
    
    Args:
        model: The neural network model
        cached_zs: List of cached z tensors, each of shape (N, d_l) on CPU
        path: List of unit indices [i0, i1, ..., i_{L-1}] defining the path
        device: Ignored (kept for compatibility)
        mode: "routing" mode (binary indicator)
        weight_aware: If True, multiply by path weights
        
    Returns:
        phi: (N,) tensor of path features across N samples (on CPU)
    """
    model.eval()
    # Use cached_zs device (should be CPU)
    dev = cached_zs[0].device if len(cached_zs) > 0 else torch.device("cpu")
    L = len(path)
    N = cached_zs[0].shape[0] if len(cached_zs) > 0 else 0
    
    # Compute path weight if weight_aware (stays on CPU; only a scalar)
    path_weight = 1.0
    if weight_aware:
        for l in range(L):
            if l < len(model.linears):
                # Weight from layer l to l+1 along the path
                if l == 0:
                    # First layer: use input dimension (assume uniform or use first weight)
                    w = model.linears[0].weight[path[0], :].abs().mean()
                else:
                    w = model.linears[l].weight[path[l], path[l-1]].abs()
                path_weight = path_weight * w.item()
        # Include readout weight
        if L > 0:
            path_weight = path_weight * model.readout.weight[0, path[-1]].abs().item()
    
    # Compute transmittance along path: product of E_l[path_index] across layers
    # All operations on CPU (cached_zs are already on CPU)
    phi = torch.ones(N, device=dev, dtype=torch.float32)
    for l in range(L):
        if l < len(cached_zs):
            z = cached_zs[l].to(dev).float()  # Ensure on correct device (CPU)
            if path[l] < z.shape[1]:
                # E = z (binary indicator in routing mode)
                E_l_path = z[:, path[l]]  # (N,)
                phi = phi * E_l_path
    
    # Apply path weight
    if weight_aware:
        phi = phi * path_weight
    
    return phi.cpu()


@torch.no_grad()
def _compute_path_transmittance_feature_from_cache_gpu(
    model,
    cached_zs: List[torch.Tensor],
    path: List[int],
    device: Optional[str] = None,
    mode: str = "routing",
    weight_aware: bool = True,
) -> torch.Tensor:
    """
    GPU-optimized version: Compute transmittance feature for a single path from cached z values.
    Keeps everything on GPU for speed.
    
    Args:
        model: The neural network model
        cached_zs: List of cached z tensors, each of shape (N, d_l) on GPU
        path: List of unit indices [i0, i1, ..., i_{L-1}] defining the path
        device: Device to use (should match cached_zs device)
        mode: "routing" mode (binary indicator)
        weight_aware: If True, multiply by path weights
        
    Returns:
        phi: (N,) tensor of path features across N samples (on GPU)
    """
    model.eval()
    dev = device or cached_zs[0].device if len(cached_zs) > 0 else torch.device("cuda")
    L = len(path)
    N = cached_zs[0].shape[0] if len(cached_zs) > 0 else 0
    
    # Compute path weight if weight_aware (keep on GPU)
    path_weight = torch.tensor(1.0, device=dev, dtype=torch.float32)
    if weight_aware:
        for l in range(L):
            if l < len(model.linears):
                if l == 0:
                    w = model.linears[0].weight[path[0], :].abs().mean()
                else:
                    w = model.linears[l].weight[path[l], path[l-1]].abs()
                path_weight = path_weight * w
        if L > 0:
            path_weight = path_weight * model.readout.weight[0, path[-1]].abs()
    
    # Compute transmittance along path: product of E_l[path_index] across layers
    # All operations on GPU
    phi = torch.ones(N, device=dev, dtype=torch.float32)
    for l in range(L):
        if l < len(cached_zs):
            z = cached_zs[l].float()  # Already on GPU
            if path[l] < z.shape[1]:
                E_l_path = z[:, path[l]]  # (N,)
                phi = phi * E_l_path
    
    # Apply path weight
    if weight_aware:
        phi = phi * path_weight
    
    return phi  # Keep on GPU


@torch.no_grad()
def _compute_path_transmittance_features_vectorized(
    model,
    cached_zs: List[torch.Tensor],
    paths: List[List[int]],
    device: Optional[str] = None,
    mode: str = "routing",
    weight_aware: bool = True,
) -> torch.Tensor:
    """
    Vectorized version: Compute transmittance features for multiple paths simultaneously.
    This is much faster than looping over paths individually.
    
    Args:
        model: The neural network model
        cached_zs: List of cached z tensors, each of shape (N, d_l) on GPU
        paths: List of paths, each is a list of unit indices
        device: Device to use (should match cached_zs device)
        mode: "routing" mode (binary indicator)
        weight_aware: If True, multiply by path weights
        
    Returns:
        Phi: (N, num_paths) tensor of path features across N samples and num_paths paths (on GPU)
    """
    model.eval()
    dev = device or cached_zs[0].device if len(cached_zs) > 0 else torch.device("cuda")
    num_paths = len(paths)
    if num_paths == 0:
        return torch.zeros(0, 0, device=dev, dtype=torch.float32)
    
    N = cached_zs[0].shape[0] if len(cached_zs) > 0 else 0
    L = len(cached_zs)
    
    # Compute path weights for all paths (if weight_aware)
    path_weights = torch.ones(num_paths, device=dev, dtype=torch.float32)
    if weight_aware:
        for path_idx, path in enumerate(paths):
            path_len = len(path)
            for l in range(path_len):
                if l < len(model.linears):
                    if l == 0:
                        w = model.linears[0].weight[path[0], :].abs().mean()
                    else:
                        w = model.linears[l].weight[path[l], path[l-1]].abs()
                    path_weights[path_idx] = path_weights[path_idx] * w
            if path_len > 0:
                path_weights[path_idx] = path_weights[path_idx] * model.readout.weight[0, path[-1]].abs()
    
    # Initialize Phi: (N, num_paths)
    Phi = torch.ones(N, num_paths, device=dev, dtype=torch.float32)
    
    # For each layer, gather the appropriate indices for all paths
    for l in range(L):
        if l >= len(cached_zs):
            continue
        
        z = cached_zs[l].float()  # (N, d_l) on GPU
        
        # Collect indices for this layer across all paths
        # Some paths might be shorter than L, so we need to handle that
        max_path_len = max(len(p) for p in paths)
        if l < max_path_len:
            # Get indices for this layer for each path
            indices = []
            valid_mask = []
            for path_idx, path in enumerate(paths):
                if l < len(path) and path[l] < z.shape[1]:
                    indices.append(path[l])
                    valid_mask.append(True)
                else:
                    indices.append(0)  # Dummy index, will be masked
                    valid_mask.append(False)
            
            # Convert to tensor
            indices_tensor = torch.tensor(indices, device=dev, dtype=torch.long)  # (num_paths,)
            valid_mask_tensor = torch.tensor(valid_mask, device=dev, dtype=torch.bool)  # (num_paths,)
            
            # Gather: z[:, indices_tensor] gives (N, num_paths)
            E_l_paths = z[:, indices_tensor]  # (N, num_paths)
            
            # Mask out invalid paths (set to 1 so they don't affect product)
            E_l_paths = torch.where(valid_mask_tensor.unsqueeze(0), E_l_paths, torch.ones_like(E_l_paths))
            
            # Multiply into Phi
            Phi = Phi * E_l_paths
    
    # Apply path weights
    if weight_aware:
        Phi = Phi * path_weights.unsqueeze(0)  # (N, num_paths) * (1, num_paths)
    
    return Phi  # Keep on GPU


@torch.no_grad()
def compute_dominant_k_paths(
    model,
    train_loader,
    test_loader,
    out_path: str,
    *,
    mode: str = "routing",
    max_candidate_paths: int = 200,
    max_k: int = 20,
    max_samples: Optional[int] = 5000,
    weight_aware: bool = True,
    use_network_outputs: bool = False,
    device: Optional[str] = None,
    step_tag: str = "",
):
    """
    Compute dominant k paths using greedy forward selection.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        test_loader: Test data loader (for evaluation)
        out_path: Path to save the R² plot
        mode: "routing" mode
        max_candidate_paths: Number of candidate paths to consider
        max_k: Maximum k to compute R²(k) for
        max_samples: Maximum samples to use
        weight_aware: If True, use weight-modulated path features
        use_network_outputs: If True, use f(x) as target; else use labels y
        device: Device to use
        step_tag: Tag for the plot title (e.g., "epoch_0100")
    """
    _ensure_dir(os.path.dirname(out_path))
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    print(f"[dominant_k_paths] Computing dominant k paths (max_k={max_k}, candidates={max_candidate_paths})...")
    
    # Step 1: Get candidate paths using beam search
    mean_E = _mean_transmittance_per_layer(model, train_loader, device=dev, mode=mode)
    candidate_paths = _beam_top_paths(model, mean_E, beam=50, top_k=max_candidate_paths)
    
    if len(candidate_paths) == 0:
        print("[dominant_k_paths] Warning: No candidate paths found. Skipping.")
        return
    
    print(f"[dominant_k_paths] Found {len(candidate_paths)} candidate paths")
    
    # Step 2: Collect targets (y or f(x)) and cache z values in the same loop
    # This ensures alignment between targets and cached activations
    y_list = []
    f_list = []
    cached_zs_list = []
    sample_count = 0
    
    for xb, yb in test_loader:
        if max_samples is not None and sample_count >= max_samples:
            break
        xb = xb.to(dev)
        yb = yb.to(dev)
        
        yhat = model(xb)
        _, cache = model(xb, return_cache=True)
        zs = cache["z"]
        
        y_list.append(yb.detach().cpu().squeeze())
        f_list.append(yhat.detach().cpu().squeeze())
        cached_zs_list.append([z.detach().cpu() for z in zs])
        sample_count += xb.shape[0]
    
    if len(cached_zs_list) == 0:
        print("[dominant_k_paths] Warning: No data collected. Skipping.")
        return
    
    y = torch.cat(y_list, dim=0)  # (N,)
    f = torch.cat(f_list, dim=0)  # (N,)
    
    # Stack zs across batches for each layer
    L = len(cached_zs_list[0])
    cached_zs = []
    for l in range(L):
        layer_zs = [batch_zs[l] for batch_zs in cached_zs_list]
        cached_zs.append(torch.cat(layer_zs, dim=0))
    
    # Choose target - ensure on correct device
    target = f if use_network_outputs else y
    target = target.to(dev)  # Move to device early
    # Convert to float if needed (for multi-class integer labels)
    if target.dtype in (torch.long, torch.int64, torch.int32):
        target = target.float()
    target_centered = target - target.mean()
    target_norm_sq = (target_centered ** 2).sum().item()
    
    if target_norm_sq < 1e-12:
        print("[dominant_k_paths] Warning: Target has zero variance. Skipping.")
        return
    
    # Ensure alignment: limit to same number of samples
    N_actual = min(cached_zs[0].shape[0], target.shape[0])
    cached_zs = [z[:N_actual] for z in cached_zs]
    target = target[:N_actual].to(dev)  # Ensure on device
    # Convert to float if needed (for multi-class integer labels)
    if target.dtype in (torch.long, torch.int64, torch.int32):
        target = target.float()
    target_centered = (target - target.mean()).to(dev)  # Ensure on device
    target_norm_sq = (target_centered ** 2).sum().item()
    
    # Step 3: Build path feature matrix Φ (only for candidate paths) - VECTORIZED
    print(f"[dominant_k_paths] Computing path features for {len(candidate_paths)} paths (vectorized)...")
    
    # Move cached_zs to GPU if not already there
    cached_zs_gpu = [z.to(dev) for z in cached_zs]
    
    # Compute all path features at once using vectorized function
    try:
        Phi = _compute_path_transmittance_features_vectorized(
            model, cached_zs_gpu, candidate_paths, device=dev, mode=mode, weight_aware=weight_aware
        )  # (N, num_paths)
        
        # Ensure alignment with target
        if Phi.shape[0] != target.shape[0]:
            if Phi.shape[0] > target.shape[0]:
                Phi = Phi[:target.shape[0]]
            else:
                pad_size = target.shape[0] - Phi.shape[0]
                Phi = torch.cat([Phi, torch.zeros(pad_size, Phi.shape[1], device=dev)], dim=0)
        
        # Center each column (path feature)
        Phi_centered = Phi - Phi.mean(dim=0, keepdim=True)  # (N, num_paths)
        
        # Filter out paths with near-zero norm
        phi_norms = torch.norm(Phi_centered, dim=0)  # (num_paths,)
        valid_mask = phi_norms >= 1e-12
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            print("[dominant_k_paths] Warning: No valid path features computed. Skipping.")
            return
        
        Phi = Phi_centered[:, valid_indices]  # (N, m) where m = num_valid_paths
        valid_paths = [candidate_paths[i] for i in valid_indices.cpu().tolist()]
        m = Phi.shape[1]
        print(f"[dominant_k_paths] Built feature matrix: {Phi.shape} ({m} valid paths)")
        
    except Exception as e:
        print(f"[dominant_k_paths] Warning: Vectorized computation failed, falling back to loop: {e}")
        # Fallback to original loop-based method
        phi_list = []
        valid_paths = []
        for idx, path in enumerate(candidate_paths):
            try:
                phi = _compute_path_transmittance_feature_from_cache(
                    model, cached_zs, path, device=dev, mode=mode, weight_aware=weight_aware
                )
                if phi.shape[0] != target.shape[0]:
                    if phi.shape[0] > target.shape[0]:
                        phi = phi[:target.shape[0]]
                    else:
                        pad_size = target.shape[0] - phi.shape[0]
                        phi = torch.cat([phi, torch.zeros(pad_size)])
                phi_centered = phi - phi.mean()
                phi_norm = torch.norm(phi_centered).item()
                if phi_norm < 1e-12:
                    continue
                phi_list.append(phi_centered)
                valid_paths.append(path)
            except Exception as e2:
                print(f"[dominant_k_paths] Warning: Failed to compute feature for path {idx}: {e2}")
                continue
        if len(phi_list) == 0:
            print("[dominant_k_paths] Warning: No valid path features computed. Skipping.")
            return
        Phi = torch.stack(phi_list, dim=1)
        m = Phi.shape[1]
        print(f"[dominant_k_paths] Built feature matrix: {Phi.shape} ({m} valid paths)")
    
    # Step 4: Greedy forward selection
    print(f"[dominant_k_paths] Running greedy forward selection (k=0 to {max_k})...")
    
    # Ensure all tensors are on the same device
    phi_device = Phi.device
    target_centered = target_centered.to(phi_device)
    
    S = []  # Selected path indices
    r = target_centered.clone()  # Residual (now on same device as Phi)
    r2_values = []  # R²(k) for k=0 to max_k
    
    # k=0: no paths selected
    r2_values.append(0.0)
    
    for k in range(1, min(max_k + 1, m + 1)):
        # Find path with maximum correlation with residual
        correlations = torch.abs(Phi.T @ r)  # (m,) - both on same device now
        
        # Mask out already selected paths
        if len(S) > 0:
            correlations[torch.tensor(S, device=phi_device)] = -1.0
        
        # Select best path
        best_idx = correlations.argmax().item()
        if correlations[best_idx] < 0:
            # No more useful paths
            break
        
        S.append(best_idx)
        
        # Refit regression on selected paths
        Phi_S = Phi[:, S]  # (N, |S|)
        
        # Solve: min ||target - Phi_S @ beta||^2 using least-squares
        try:
            # Ensure Phi_S and target_centered are on same device
            Phi_S = Phi_S.to(phi_device)
            target_centered = target_centered.to(phi_device)
            beta = torch.linalg.lstsq(Phi_S, target_centered).solution
            prediction = Phi_S @ beta
            r2 = 1.0 - ((target_centered - prediction)**2).sum().item() / target_norm_sq
            r2_values.append(r2)
            
            # Update residual (ensure on same device)
            r = (target_centered - prediction).to(phi_device)
        except Exception as e:
            print(f"[dominant_k_paths] Warning: Regression failed at k={k}: {e}")
            r2_values.append(r2_values[-1] if len(r2_values) > 0 else 0.0)
    
    # Pad if needed
    while len(r2_values) < max_k + 1:
        r2_values.append(r2_values[-1] if len(r2_values) > 0 else 0.0)
    
    # Step 5: Compute for train set
    r2_values_train = None
    print(f"[dominant_k_paths] Computing for train set...")
    # Cache train data
    y_list_train = []
    f_list_train = []
    cached_zs_list_train = []
    sample_count_train = 0
    
    for xb, yb in train_loader:
        if max_samples is not None and sample_count_train >= max_samples:
            break
        xb = xb.to(dev)
        yb = yb.to(dev)
        
        yhat = model(xb)
        _, cache = model(xb, return_cache=True)
        zs = cache["z"]
        
        y_list_train.append(yb.detach().cpu().squeeze())
        f_list_train.append(yhat.detach().cpu().squeeze())
        cached_zs_list_train.append([z.detach().cpu() for z in zs])
        sample_count_train += xb.shape[0]
    
    if len(cached_zs_list_train) > 0:
        y_train = torch.cat(y_list_train, dim=0)
        f_train = torch.cat(f_list_train, dim=0)
        target_train = f_train if use_network_outputs else y_train
        # Convert to float if needed (for multi-class integer labels)
        if target_train.dtype in (torch.long, torch.int64, torch.int32):
            target_train = target_train.float()
        target_train_centered = target_train - target_train.mean()
        target_train_norm_sq = (target_train_centered ** 2).sum().item()
        
        if target_train_norm_sq > 1e-12:
            # Stack cached zs for train
            L_train = len(cached_zs_list_train[0])
            cached_zs_train = []
            for l in range(L_train):
                layer_zs = [batch_zs[l] for batch_zs in cached_zs_list_train]
                cached_zs_train.append(torch.cat(layer_zs, dim=0))
            
            N_train = min(cached_zs_train[0].shape[0], target_train.shape[0])
            cached_zs_train = [z[:N_train] for z in cached_zs_train]
            target_train = target_train[:N_train]
            # Convert to float if needed (for multi-class integer labels)
            if target_train.dtype in (torch.long, torch.int64, torch.int32):
                target_train = target_train.float()
            target_train_centered = target_train - target_train.mean()
            target_train_norm_sq = (target_train_centered ** 2).sum().item()
            
            # Build Phi for train using same paths
            # Always append for paths in valid_paths to maintain alignment with test set
            phi_list_train = []
            for path in valid_paths:
                try:
                    phi_train = _compute_path_transmittance_feature_from_cache(
                        model, cached_zs_train, path, device=dev, mode=mode, weight_aware=weight_aware
                    )
                    if phi_train.shape[0] != target_train.shape[0]:
                        if phi_train.shape[0] > target_train.shape[0]:
                            phi_train = phi_train[:target_train.shape[0]]
                        else:
                            pad_size = target_train.shape[0] - phi_train.shape[0]
                            phi_train = torch.cat([phi_train, torch.zeros(pad_size)])
                    phi_centered_train = phi_train - phi_train.mean()
                    # Always append to maintain column alignment with test set
                    # Small ridge regularization will handle near-zero features
                    phi_list_train.append(phi_centered_train)
                except:
                    # On exception, append zeros to maintain alignment
                    phi_list_train.append(torch.zeros(target_train.shape[0]))
            
            if len(phi_list_train) > 0:
                Phi_train = torch.stack(phi_list_train, dim=1)
                # Use same selected paths S from test (columns are aligned)
                S_train = S[:min(len(S), Phi_train.shape[1])]
                r2_values_train = [0.0]  # k=0
                
                for k in range(1, min(max_k + 1, len(S_train) + 1)):
                    Phi_S_train = Phi_train[:, S_train[:k]]
                    try:
                        beta_train = torch.linalg.lstsq(Phi_S_train, target_train_centered).solution
                        prediction_train = Phi_S_train @ beta_train
                        r2_train = 1.0 - ((target_train_centered - prediction_train)**2).sum().item() / target_train_norm_sq
                        r2_values_train.append(r2_train)
                    except:
                        r2_values_train.append(r2_values_train[-1] if len(r2_values_train) > 0 else 0.0)
                
                while len(r2_values_train) < max_k + 1:
                    r2_values_train.append(r2_values_train[-1] if len(r2_values_train) > 0 else 0.0)
    
    # Step 6: Plot R²(k) vs k for both sets
    k_values = np.arange(len(r2_values))
    r2_array = np.array(r2_values)
    
    plt.figure(figsize=(12, 6))
    
    # Color code by k value
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
    
    # Plot test set
    plt.plot(k_values, r2_array, marker='o', linewidth=2, markersize=6, color='steelblue', label='Test R2(k)')
    for k, r2_val in enumerate(r2_values):
        plt.scatter([k], [r2_val], c=[colors[k]], s=100, zorder=5, 
                   edgecolors='black', linewidths=1.5, alpha=0.7)
    
    # Plot train set if available
    if r2_values_train is not None:
        k_values_train = np.arange(len(r2_values_train))
        r2_array_train = np.array(r2_values_train)
        plt.plot(k_values_train, r2_array_train, marker='s', linewidth=2, markersize=6, 
                color='coral', label='Train R2(k)', linestyle='--')
        colors_train = plt.cm.plasma(np.linspace(0, 1, len(r2_values_train)))
        for k, r2_val in enumerate(r2_values_train):
            plt.scatter([k], [r2_val], c=[colors_train[k]], s=100, zorder=5, 
                       edgecolors='black', linewidths=1.5, alpha=0.7, marker='s')
    
    plt.xlabel('Number of paths (k)', fontsize=12)
    plt.ylabel('Variance Explained R2(k)', fontsize=12)
    title = f'Dominant k Paths Analysis'
    if step_tag:
        title += f' [{step_tag}]'
    if use_network_outputs:
        title += ' (target: f(x))'
    else:
        title += ' (target: y)'
    plt.title(title, fontsize=14)
    plt.yscale("log")
    plt.grid(True, ls='--', alpha=0.3, which="both")
    plt.ylim(1e-3, 1.05)
    plt.xlim(-0.5, max_k + 0.5)
    
    # Add text annotation for final R2
    if len(r2_values) > 0:
        final_r2_test = r2_values[-1]
        text_str = f'Test R2({len(r2_values)-1}) = {final_r2_test:.4f}'
        if r2_values_train is not None and len(r2_values_train) > 0:
            final_r2_train = r2_values_train[-1]
            text_str += f'\nTrain R2({len(r2_values_train)-1}) = {final_r2_train:.4f}'
        plt.text(0.98, 0.02, text_str, 
                transform=plt.gca().transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    
    print(f"[dominant_k_paths] Saved plot -> {out_path}")
    print(f"[dominant_k_paths] Test R² values: {[f'{r:.4f}' for r in r2_values[:min(11, len(r2_values))]]}...")
    if r2_values_train is not None:
        print(f"[dominant_k_paths] Train R² values: {[f'{r:.4f}' for r in r2_values_train[:min(11, len(r2_values_train))]]}...")
    
    # Return metrics
    result = {
        "r2_values_test": r2_values,
        "selected_paths": [valid_paths[i] for i in S[:max_k]] if len(S) > 0 else [],
        "num_candidate_paths": m,
        "final_r2_test": r2_values[-1] if len(r2_values) > 0 else 0.0,
    }
    if r2_values_train is not None:
        result["r2_values_train"] = r2_values_train
        result["final_r2_train"] = r2_values_train[-1] if len(r2_values_train) > 0 else 0.0
    
    return result


@torch.no_grad()
def compute_lineage_centroid_metrics(
    E_time: List[torch.Tensor],
    *,
    k: int = 8,
) -> Dict[str, object]:
    """
    Compute metrics from multiple checkpoints for Lineage Sankey and Centroid Drift.
    Requires multiple checkpoints (E_time list).
    
    Returns:
    - centroid_drift: List of drift values between consecutive checkpoints
    - centroid_drift_mean: Mean drift
    - cluster_radius: List of within-cluster variance at each checkpoint
    - cluster_radius_mean: Mean cluster radius
    """
    if not HAVE_SK or len(E_time) < 2:
        return {}
    
    from sklearn.cluster import KMeans
    
    metrics = {}
    
    try:
        cents = []
        radii = []
        
        for E in E_time:
            E_np = E.numpy() if isinstance(E, torch.Tensor) else E
            if E_np.shape[0] < k:
                continue
            
            k_actual = min(k, E_np.shape[0] - 1)
            if k_actual < 2:
                continue
            
            km = KMeans(n_clusters=k_actual, random_state=1, n_init="auto")
            labs = km.fit_predict(E_np)
            C = km.cluster_centers_
            cents.append(C)
            
            # Within-cluster variance
            var = 0.0
            for j in range(k_actual):
                Xj = E_np[labs == j]
                if Xj.size == 0:
                    continue
                var += ((Xj - C[j][None,:])**2).mean()
            radii.append(var)
        
        if len(cents) < 2:
            return {}
        
        # Compute drift
        cents_array = np.stack(cents, axis=0)  # T x k x D
        drift = np.linalg.norm(cents_array[1:] - cents_array[:-1], axis=2).mean(axis=1)  # per step
        
        metrics["centroid_drift"] = drift.tolist()
        metrics["centroid_drift_mean"] = float(drift.mean())
        metrics["centroid_drift_max"] = float(drift.max())
        metrics["cluster_radius"] = radii
        metrics["cluster_radius_mean"] = float(np.mean(radii))
        metrics["cluster_radius_max"] = float(np.max(radii))
        
    except Exception as e:
        print(f"[lineage_metrics] Warning: Failed to compute metrics: {e}")
    
    return metrics


# ---------------------------
# Gate state analysis
# ---------------------------

@torch.no_grad()
def compute_unique_gate_states_per_class(
    model,
    loader,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, int]:
    """
    For each sample, compute the gate state as a huge vector of 1s and 0s.
    Then count the number of unique gate states from class 1 and class 2.
    
    Returns:
        {
            "class_1_unique": int,
            "class_2_unique": int,
            "total_samples": int,
            "class_1_samples": int,
            "class_2_samples": int,
        }
    """
    model.eval()
    dev = device or next(iter(model.parameters())).device
    
    # Use hash values instead of full tuples - MUCH faster!
    # Hash collisions are extremely rare for binary vectors
    gate_states_class_1 = set()  # Set of hash values
    gate_states_class_2 = set()  # Set of hash values
    total_samples = 0
    class_1_samples = 0
    class_2_samples = 0
    
    for xb, yb in loader:
        if max_samples is not None and total_samples >= max_samples:
            break
        
        xb = xb.to(dev)
        yb = yb.to(dev)
        batch_size = xb.shape[0]
        
        _, cache = model(xb, return_cache=True)
        zs = cache["z"]  # List of (B, d_l) tensors (pre-activations or post-activations)
        
        # Flatten all z layers into a single binary vector per sample - keep on GPU
        # FIX: Use (z > 0) instead of z.bool() to correctly handle negative pre-activations
        # For ReLU: neuron is "active" (1) if z > 0, "inactive" (0) if z <= 0
        # Concatenate all z layers: (B, total_neurons)
        z_flat = torch.cat([(z > 0).int() for z in zs], dim=1)  # (B, total_neurons) on GPU
        
        # Normalize labels to {-1, +1} format on GPU
        yb_normalized = yb.squeeze()
        if yb_normalized.min() >= 0:
            yb_normalized = 2 * yb_normalized - 1  # Convert {0, 1} to {-1, +1}
        
        # Process in batches on GPU, then hash on CPU
        z_flat_cpu = z_flat.cpu()  # Move once for entire batch
        yb_np = yb_normalized.cpu().numpy()
        
        # OPTIMIZED: Hash entire batch at once using numpy's hash
        # Convert to bytes for fast hashing
        z_flat_bytes = z_flat_cpu.numpy().astype(np.uint8).tobytes()  # All samples as bytes
        total_neurons = z_flat_cpu.shape[1]
        bytes_per_sample = total_neurons
        
        for i in range(batch_size):
            # Extract bytes for this sample and hash
            sample_bytes = z_flat_bytes[i * bytes_per_sample:(i + 1) * bytes_per_sample]
            gate_state_hash = hash(sample_bytes)  # Fast hash instead of tuple
            
            y_val = yb_np[i] if i < len(yb_np) else yb_np[0]
            if y_val > 0:  # Class 1 (positive)
                gate_states_class_1.add(gate_state_hash)
                class_1_samples += 1
            else:  # Class 2 (negative)
                gate_states_class_2.add(gate_state_hash)
                class_2_samples += 1
            total_samples += 1
            
            if max_samples is not None and total_samples >= max_samples:
                break
        
        if max_samples is not None and total_samples >= max_samples:
            break
    
    return {
        "class_1_unique": len(gate_states_class_1),
        "class_2_unique": len(gate_states_class_2),
        "total_samples": total_samples,
        "class_1_samples": class_1_samples,
        "class_2_samples": class_2_samples,
    }


def plot_gate_states_vs_epoch(
    history: List[Dict],
    gate_states_history: List[Dict],
    out_path: str,
):
    """
    Plot unique gate states per class vs epoch.
    
    Args:
        history: List of training history dicts with keys like "epoch"
        gate_states_history: List of gate state dicts from compute_unique_gate_states_per_class
        out_path: Path to save the plot (will be overwritten each time)
    """
    _ensure_dir(os.path.dirname(out_path))
    
    # Only plot epochs where we actually have gate state data
    # Create a dict mapping epoch -> gate_states for fast lookup
    gs_by_epoch = {gs.get("epoch", i): gs for i, gs in enumerate(gate_states_history)}
    
    # Filter to only epochs that exist in both history and gate_states_history
    epochs = []
    class_1_unique = []
    class_2_unique = []
    
    for h in history:
        epoch = h.get("epoch")
        if epoch in gs_by_epoch:
            gs = gs_by_epoch[epoch]
            epochs.append(epoch)
            class_1_unique.append(gs.get("class_1_unique", 0))
            class_2_unique.append(gs.get("class_2_unique", 0))
    
    # Ensure we have at least some data to plot
    if len(epochs) == 0:
        print("[plot_gate_states] Warning: No epochs with gate state data. Skipping plot.")
        return
    
    # Check if we have any non-zero values
    max_val = max(max(class_1_unique) if class_1_unique else 0, max(class_2_unique) if class_2_unique else 0)
    if max_val == 0:
        print("[plot_gate_states] Warning: All gate state counts are zero. Using linear scale.")
        use_log_scale = False
    else:
        use_log_scale = True
    
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.plot(epochs, class_1_unique, marker='o', label='Class 1 unique gate states', color='C0', linewidth=2, markersize=8)
    ax.plot(epochs, class_2_unique, marker='s', label='Class 2 unique gate states', color='C1', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', color='black')
    ax.set_ylabel('Number of Unique Gate States', color='black')
    ax.set_title('Unique Gate States per Class vs Epoch', color='black')
    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.legend(framealpha=1.0, facecolor='white', edgecolor='black')
    ax.grid(True, alpha=0.3, which="both", color='gray')
    
    if use_log_scale:
        ax.set_yscale("log")
        # Set y-axis limits for log scale
        min_val = min(min([v for v in class_1_unique if v > 0]) if any(v > 0 for v in class_1_unique) else 1,
                      min([v for v in class_2_unique if v > 0]) if any(v > 0 for v in class_2_unique) else 1)
        ax.set_ylim(bottom=min(0.1, min_val * 0.5) if min_val > 0 else 0.1, 
                   top=max_val * 2 if max_val > 0 else 10)
    else:
        # Use linear scale when all values are zero or very small
        ax.set_yscale("linear")
        ax.set_ylim(bottom=-0.5, top=max(10, max_val + 1))
    
    plt.tight_layout()
    # Save with white background explicitly (overwrites existing file)
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[analysis] Saved gate states plot -> {out_path}")


def plot_train_test_errors_vs_epoch(
    history: List[Dict],
    out_path: str,
):
    """
    Plot train and test error vs epoch.
    
    Args:
        history: List of training history dicts with keys like "epoch", "train_loss", "test_loss", etc.
        out_path: Path to save the plot (will be overwritten each time)
    """
    _ensure_dir(os.path.dirname(out_path))
    
    epochs = [h["epoch"] for h in history]
    train_losses = [h.get("train_loss", 0.0) for h in history]
    test_losses = [h.get("test_loss", None) for h in history]
    
    # Ensure we have at least some data to plot
    if len(epochs) == 0:
        print("[plot_errors] Warning: No epochs in history. Skipping plot.")
        return
    
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.plot(epochs, train_losses, marker='o', label='Train Error', color='C2', linewidth=2, markersize=8)
    if any(tl is not None for tl in test_losses):
        test_epochs = [e for e, tl in zip(epochs, test_losses) if tl is not None]
        test_losses_filtered = [tl for tl in test_losses if tl is not None]
        ax.plot(test_epochs, test_losses_filtered, marker='s', label='Test Error', color='C3', linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', color='black')
    ax.set_ylabel('Error (MSE Loss)', color='black')
    ax.set_title('Train and Test Error vs Epoch', color='black')
    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.legend(framealpha=1.0, facecolor='white', edgecolor='black')
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_yscale('log')
    
    plt.tight_layout()
    # Save with white background explicitly (overwrites existing file)
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[analysis] Saved train/test errors plot -> {out_path}")


def plot_effective_rank_layers_vs_epoch(
    history: List[Dict],
    out_path: str,
):
    """
    Plot effective rank for each layer's activations vs epoch.
    
    Args:
        history: List of training history dicts with keys like "epoch", "effective_rank_layers"
        out_path: Path to save the plot (will be overwritten each time)
    """
    _ensure_dir(os.path.dirname(out_path))
    
    # Filter to only epochs that have effective rank data
    epochs = []
    effective_ranks_by_layer = {}  # layer_idx -> list of effective ranks
    
    for h in history:
        epoch = h.get("epoch")
        eff_ranks = h.get("effective_rank_layers")
        if eff_ranks is not None and len(eff_ranks) > 0:
            epochs.append(epoch)
            for layer_idx, eff_rank in enumerate(eff_ranks):
                if layer_idx not in effective_ranks_by_layer:
                    effective_ranks_by_layer[layer_idx] = []
                effective_ranks_by_layer[layer_idx].append(eff_rank)
    
    # Ensure we have at least some data to plot
    if len(epochs) == 0:
        print("[plot_effective_rank_layers] Warning: No epochs with effective rank data. Skipping plot.")
        return
    
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot each layer's effective rank
    colors = plt.cm.tab10(np.linspace(0, 1, len(effective_ranks_by_layer)))
    for layer_idx, color in zip(sorted(effective_ranks_by_layer.keys()), colors):
        layer_ranks = effective_ranks_by_layer[layer_idx]
        ax.plot(epochs, layer_ranks, marker='o', label=f'Layer {layer_idx+1}', 
                color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch', color='black')
    ax.set_ylabel('Effective Rank', color='black')
    ax.set_title('Effective Rank per Layer vs Epoch', color='black')
    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.legend(framealpha=1.0, facecolor='white', edgecolor='black')
    ax.grid(True, alpha=0.3, which="both", color='gray')
    
    plt.tight_layout()
    # Save with white background explicitly (overwrites existing file)
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[analysis] Saved effective rank layers plot -> {out_path}")


@torch.no_grad()
def compute_hidden_kernel_effective_rank(
    model,
    loader,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[float]:
    """
    Compute effective rank of gram kernel K = H @ H^T for each hidden layer.
    
    For each layer l, collect activations h_l(x) and compute:
    - K_l = H_l @ H_l^T (gram kernel)
    - Effective rank from eigenvalues of K_l
    
    Returns:
        List of effective ranks, one per hidden layer
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    # Collect activations for all layers
    activations_by_layer = None
    seen = 0
    
    for xb, _ in loader:
        if max_samples is not None and seen >= max_samples:
            break
        
        xb = xb.to(dev, non_blocking=True)
        bsz_original = xb.shape[0]
        if max_samples is not None and (seen + bsz_original > max_samples):
            take = max_samples - seen
            xb = xb[:take]
            bsz = take
        else:
            bsz = bsz_original
        
        # Forward pass to get all hidden activations
        _, cache = model(xb, return_cache=True)
        h_layers = cache["h"]  # List of (bsz, d_l) tensors
        
        if activations_by_layer is None:
            activations_by_layer = [h.cpu().clone() for h in h_layers]
        else:
            for l in range(len(h_layers)):
                activations_by_layer[l] = torch.cat([activations_by_layer[l], h_layers[l].cpu()], dim=0)
        
        seen += bsz
    
    if activations_by_layer is None:
        return []
    
    # Compute effective rank for each layer's gram kernel
    effective_ranks = []
    for H in activations_by_layer:
        # H is (P, d_hidden) where P is number of samples
        # Compute gram kernel K = H @ H^T
        # Use SVD for numerical stability: H = U @ S @ V^T
        # Then K = H @ H^T = U @ S^2 @ U^T
        # Eigenvalues are S^2, eigenvectors are columns of U
        
        H_2d = H.view(H.shape[0], -1)  # Flatten to (P, d_hidden)
        if H_2d.shape[1] == 0:
            effective_ranks.append(0.0)
            continue
        
        try:
            # Use SVD on H directly (more efficient than computing H @ H^T)
            U, s, _ = torch.linalg.svd(H_2d, full_matrices=False)
            # s is (min(P, d_hidden),) - singular values
            # Eigenvalues of K are s^2
            evals = s ** 2
            
            # Filter near-zero eigenvalues
            evals = evals[evals > 1e-8]
            if len(evals) == 0:
                effective_ranks.append(0.0)
                continue
            
            # Compute effective rank: exp(entropy of normalized eigenvalues)
            p = evals / evals.sum()
            p = p[p > 1e-12]  # Avoid log(0)
            if len(p) == 0:
                effective_ranks.append(0.0)
                continue
            
            entropy = -(p * torch.log(p)).sum()
            eff_rank = torch.exp(entropy).item()
            effective_ranks.append(eff_rank)
        except Exception as e:
            print(f"  [hidden_kernel_effective_rank] Warning: Failed to compute effective rank: {e}")
            effective_ranks.append(0.0)
    
    return effective_ranks


def plot_hidden_kernel_effective_rank_vs_epoch(
    history: List[Dict],
    out_path_train: str,
    out_path_test: str,
):
    """
    Plot effective rank of hidden gram kernels for each layer vs epoch.
    Creates separate plots for train and test.
    
    Args:
        history: List of training history dicts with keys like "epoch", 
                 "hidden_kernel_effective_rank_train_layers",
                 "hidden_kernel_effective_rank_test_layers"
        out_path_train: Path to save the train plot
        out_path_test: Path to save the test plot
    """
    _ensure_dir(os.path.dirname(out_path_train))
    _ensure_dir(os.path.dirname(out_path_test))
    
    # Collect data for train
    train_epochs = []
    train_ranks_by_layer = {}  # layer_idx -> list of effective ranks
    
    # Collect data for test
    test_epochs = []
    test_ranks_by_layer = {}  # layer_idx -> list of effective ranks
    
    for h in history:
        epoch = h.get("epoch")
        
        # Process train data
        train_ranks = h.get("hidden_kernel_effective_rank_train_layers")
        if train_ranks is not None and len(train_ranks) > 0:
            train_epochs.append(epoch)
            for layer_idx, eff_rank in enumerate(train_ranks):
                if layer_idx not in train_ranks_by_layer:
                    train_ranks_by_layer[layer_idx] = []
                train_ranks_by_layer[layer_idx].append(eff_rank)
        
        # Process test data
        test_ranks = h.get("hidden_kernel_effective_rank_test_layers")
        if test_ranks is not None and len(test_ranks) > 0:
            test_epochs.append(epoch)
            for layer_idx, eff_rank in enumerate(test_ranks):
                if layer_idx not in test_ranks_by_layer:
                    test_ranks_by_layer[layer_idx] = []
                test_ranks_by_layer[layer_idx].append(eff_rank)
    
    # Plot train
    if len(train_epochs) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(train_ranks_by_layer)))
        for layer_idx, color in zip(sorted(train_ranks_by_layer.keys()), colors):
            layer_ranks = train_ranks_by_layer[layer_idx]
            ax.plot(train_epochs, layer_ranks, marker='o', label=f'Layer {layer_idx+1}', 
                    color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Epoch', color='black')
        ax.set_ylabel('Effective Rank of Gram Kernel', color='black')
        ax.set_title('Hidden Kernel Effective Rank per Layer vs Epoch (Train)', color='black')
        ax.tick_params(colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.legend(framealpha=1.0, facecolor='white', edgecolor='black')
        ax.grid(True, alpha=0.3, which="both", color='gray')
        
        plt.tight_layout()
        plt.savefig(out_path_train, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f"[analysis] Saved hidden kernel effective rank plot (train) -> {out_path_train}")
    
    # Plot test
    if len(test_epochs) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(test_ranks_by_layer)))
        for layer_idx, color in zip(sorted(test_ranks_by_layer.keys()), colors):
            layer_ranks = test_ranks_by_layer[layer_idx]
            ax.plot(test_epochs, layer_ranks, marker='s', label=f'Layer {layer_idx+1}', 
                    color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Epoch', color='black')
        ax.set_ylabel('Effective Rank of Gram Kernel', color='black')
        ax.set_title('Hidden Kernel Effective Rank per Layer vs Epoch (Test)', color='black')
        ax.tick_params(colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.legend(framealpha=1.0, facecolor='white', edgecolor='black')
        ax.grid(True, alpha=0.3, which="both", color='gray')
        
        plt.tight_layout()
        plt.savefig(out_path_test, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f"[analysis] Saved hidden kernel effective rank plot (test) -> {out_path_test}")


def plot_variance_explained_by_k_vs_epoch(
    history: List[Dict],
    out_path_train: str,
    out_path_test: str,
    max_k: int = 150,
):
    """
    Plot R² (variance explained) for different values of k against epochs.
    Creates separate plots for train and test, with color-coding by k.
    
    Args:
        history: List of training history dicts with keys like "epoch", 
                 "path_kernel_variance_explained_train_per_component",
                 "path_kernel_variance_explained_test_per_component"
        out_path_train: Path to save the train plot
        out_path_test: Path to save the test plot
        max_k: Maximum k value to plot (default 35)
    """
    _ensure_dir(os.path.dirname(out_path_train))
    _ensure_dir(os.path.dirname(out_path_test))
    
    # Collect data for train - use a list of dicts to handle variable lengths
    train_data = []  # List of (epoch, per_component_list)
    
    # Collect data for test
    test_data = []  # List of (epoch, per_component_list)
    
    for h in history:
        epoch = h.get("epoch")
        
        # Process train data
        train_per_component = h.get("path_kernel_variance_explained_train_per_component")
        if train_per_component is not None and len(train_per_component) > 0:
            train_data.append((epoch, train_per_component))
        
        # Process test data
        test_per_component = h.get("path_kernel_variance_explained_test_per_component")
        if test_per_component is not None and len(test_per_component) > 0:
            test_data.append((epoch, test_per_component))
    
    # Debug output
    if train_data or test_data:
        epochs_with_data = sorted(set([e for e, _ in train_data] + [e for e, _ in test_data]))
        print(f"[plot_variance_explained_by_k] Processing {len(history)} history entries, found variance data at {len(epochs_with_data)} epochs: {epochs_with_data[:10]}{'...' if len(epochs_with_data) > 10 else ''}")
    
    # Determine the actual max k available across all epochs
    if train_data:
        max_k_available_train = max(len(comp) for _, comp in train_data)
        # Use max_k (150) if we have data up to that point, otherwise use available data
        max_k_plot_train = min(max_k, max_k_available_train) if max_k_available_train >= max_k else max_k_available_train
    else:
        max_k_plot_train = 0
    
    if test_data:
        max_k_available_test = max(len(comp) for _, comp in test_data)
        # Use max_k (150) if we have data up to that point, otherwise use available data
        max_k_plot_test = min(max_k, max_k_available_test) if max_k_available_test >= max_k else max_k_available_test
    else:
        max_k_plot_test = 0
    
    # Build variance_by_k dictionaries
    variance_by_k_train = {}  # k -> list of (epoch, cumulative variance)
    variance_by_k_test = {}  # k -> list of (epoch, cumulative variance)
    
    for epoch, train_per_component in train_data:
        # Plot up to max_k (150) or available data, whichever is smaller
        k_max = min(max_k, len(train_per_component)) if len(train_per_component) > 0 else 0
        for k in range(1, k_max + 1):
            if k not in variance_by_k_train:
                variance_by_k_train[k] = []
            # Cumulative sum up to k
            cumsum = sum(train_per_component[:k])
            variance_by_k_train[k].append((epoch, cumsum))
    
    for epoch, test_per_component in test_data:
        # Plot up to max_k (150) or available data, whichever is smaller
        k_max = min(max_k, len(test_per_component)) if len(test_per_component) > 0 else 0
        for k in range(1, k_max + 1):
            if k not in variance_by_k_test:
                variance_by_k_test[k] = []
            # Cumulative sum up to k
            cumsum = sum(test_per_component[:k])
            variance_by_k_test[k].append((epoch, cumsum))
    
    # Plot train
    if len(variance_by_k_train) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Use a colormap for different k values
        k_values = sorted(variance_by_k_train.keys())
        # Plot all k values up to max_k (150) - use all available k values
        # Filter to only k values that exist in the data and are <= max_k
        k_to_plot = [k for k in k_values if k <= max_k]
        
        if len(k_to_plot) == 0:
            k_to_plot = k_values[:min(30, len(k_values))]  # Fallback: plot first 30
        
        print(f"[plot_variance_explained_by_k] Plotting {len(k_to_plot)} k values (max k in data: {max(k_values) if k_values else 0}, requested max_k: {max_k})")
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(k_to_plot)))
        
        # Plot all k values, but only label a subset for legend readability
        k_to_label = [k for k in k_to_plot if k <= 10 or k % 10 == 0 or k == max_k or k == max(k_to_plot)]
        
        for k, color in zip(k_to_plot, colors):
            if k in variance_by_k_train and len(variance_by_k_train[k]) > 0:
                epochs_k = [e for e, _ in variance_by_k_train[k]]
                values_k = [v for _, v in variance_by_k_train[k]]
                label = f'k={k}' if k in k_to_label else None
                ax.plot(epochs_k, values_k, 
                       marker='o', label=label, color=color, 
                       linewidth=1.5, markersize=3, alpha=0.6)
        
        ax.set_xlabel('Epoch', color='black')
        ax.set_ylabel('R² (Variance Explained)', color='black')
        ax.set_title('Path Kernel Variance Explained by k (Train)', color='black')
        ax.tick_params(colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.legend(framealpha=1.0, facecolor='white', edgecolor='black', 
                 ncol=4, fontsize=7, loc='best')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_ylim(bottom=0.0, top=1.0)
        
        plt.tight_layout()
        plt.savefig(out_path_train, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f"[analysis] Saved variance explained by k plot (train) -> {out_path_train}")
    else:
        print("[plot_variance_explained_by_k] Warning: No train data available. Skipping train plot.")
    
    # Plot test
    if len(variance_by_k_test) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Use a colormap for different k values
        k_values = sorted(variance_by_k_test.keys())
        # Plot all k values up to max_k (150) - use all available k values
        # Filter to only k values that exist in the data and are <= max_k
        k_to_plot = [k for k in k_values if k <= max_k]
        
        if len(k_to_plot) == 0:
            k_to_plot = k_values[:min(30, len(k_values))]  # Fallback: plot first 30
        
        print(f"[plot_variance_explained_by_k] Plotting {len(k_to_plot)} k values (max k in data: {max(k_values) if k_values else 0}, requested max_k: {max_k})")
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(k_to_plot)))
        
        # Plot all k values, but only label a subset for legend readability
        k_to_label = [k for k in k_to_plot if k <= 10 or k % 10 == 0 or k == max_k or k == max(k_to_plot)]
        
        for k, color in zip(k_to_plot, colors):
            if k in variance_by_k_test and len(variance_by_k_test[k]) > 0:
                epochs_k = [e for e, _ in variance_by_k_test[k]]
                values_k = [v for _, v in variance_by_k_test[k]]
                label = f'k={k}' if k in k_to_label else None
                ax.plot(epochs_k, values_k, 
                       marker='s', label=label, color=color, 
                       linewidth=1.5, markersize=3, alpha=0.6)
        
        ax.set_xlabel('Epoch', color='black')
        ax.set_ylabel('R² (Variance Explained)', color='black')
        ax.set_title('Path Kernel Variance Explained by k (Test)', color='black')
        ax.tick_params(colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.legend(framealpha=1.0, facecolor='white', edgecolor='black', 
                 ncol=4, fontsize=7, loc='best')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_ylim(bottom=0.0, top=1.0)
        
        plt.tight_layout()
        plt.savefig(out_path_test, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f"[analysis] Saved variance explained by k plot (test) -> {out_path_test}")
    else:
        print("[plot_variance_explained_by_k] Warning: No test data available. Skipping test plot.")


# ---------------------------
# Path Sankey Diagram (replaces plot_nn_graph_with_paths)
# ---------------------------

@torch.no_grad()
def plot_path_sankey_diagram(
    model,
    loader,
    out_path: str,
    *,
    mode: str = "routing",
    top_k: int = 10,
    kernel_k: int = 48,
    max_samples: Optional[int] = 5000,
    device: Optional[str] = None,
    kernel_pack: Optional[Dict[str, object]] = None,
    weight_aware: bool = True,
):
    """
    Draw a Path Sankey Diagram showing eigenpaths (canonical circuits).
    
    Nodes: Layers/Neurons, colored by average activation rate (white to black).
    Edges: Weighted by Product of Weight × Average Gate Openness for that path.
    Shows top_k eigenpaths with color coding.
    """
    if not HAVE_NX:
        print("[analysis] networkx not installed; skipping Path Sankey diagram.")
        return
    _ensure_dir(os.path.dirname(out_path))
    
    from .path_kernel import compute_path_kernel_eigs
    
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    # 1. Compute (or reuse) path kernel and get top eigenpaths
    try:
        if kernel_pack is None:
            kernel_pack = compute_path_kernel_eigs(
                model,
                loader,
                mode=mode,
                include_input=True,
                k=kernel_k,
                n_iter=30,
                block_size=1024,
                max_samples=max_samples,
                verbose=False,
            )
        evecs = kernel_pack.get("evecs")  # (P, k) - eigenvectors in sample space
        evals = kernel_pack.get("evals")  # (k,) - eigenvalues
        
        if evecs is None or len(evals) == 0:
            print("[analysis] Warning: No eigenvectors found. Skipping Path Sankey diagram.")
            return
        
        # Get top_k eigenpaths (top by eigenvalue)
        top_k_actual = min(top_k, len(evals))
        top_evecs = evecs[:, :top_k_actual]  # (P, top_k_actual)
        top_evals = evals[:top_k_actual]  # (top_k_actual,)
        
    except Exception as e:
        print(f"[analysis] Warning: Failed to compute path kernel: {e}")
        return
    
    # 2. Compute average activation rate per neuron - use mean_E (will be computed in step 4)
    # We'll compute this after we get mean_E to avoid multiple loader iterations
    neuron_activation_rates = []  # List of (layer_idx, neuron_idx, activation_rate)
    widths = [model.linears[l].out_features for l in range(len(model.linears))]
    d_in = model.linears[0].in_features
    
    # 3. Build graph structure
    G = nx.DiGraph()
    layer_x = {}
    
    # Input layer
    G.add_node(("L0", -1), layer=0, neuron_idx=-1, activation_rate=1.0)
    layer_x[0] = [("L0", -1)]
    
    # Hidden layers
    for l, w in enumerate(widths, start=1):
        nodes = [(f"L{l}", i) for i in range(w)]
        for n in nodes:
            layer, neuron_idx = n
            l_idx = l - 1  # 0-indexed layer
            rate = next((r for ll, nn, r in neuron_activation_rates if ll == l_idx and nn == neuron_idx), 0.0)
            G.add_node(n, layer=l, neuron_idx=neuron_idx, activation_rate=rate)
        layer_x[l] = nodes
    
    # Output node
    Lmax = len(widths) + 1
    G.add_node(("OUT", -1), layer=Lmax, neuron_idx=-1, activation_rate=1.0)
    layer_x[Lmax] = [("OUT", -1)]
    
    # 4. For each eigenpath, we need to map it back to actual network paths
    # Since eigenvectors are in sample space, we need to find which paths contribute most
    # We'll use the mean transmittance to find candidate paths, then weight by eigenvector alignment
    
    # Compute mean_E once (this iterates loader, but we need it for paths)
    # NOTE: This will iterate the loader, so we must collect all data we need in subsequent passes
    mean_E = _mean_transmittance_per_layer(model, loader, device=dev, mode=mode)
    candidate_paths = _beam_top_paths(model, mean_E, beam=24, top_k=min(50, top_k_actual * 5))
    
    # Now convert mean_E to activation rates format (reuse the computed mean_E)
    for l, mean_E_l in enumerate(mean_E):
        if isinstance(mean_E_l, torch.Tensor):
            rates = mean_E_l.cpu() if mean_E_l.device.type != 'cpu' else mean_E_l
            for neuron_idx, rate in enumerate(rates):
                neuron_activation_rates.append((l, neuron_idx, rate.item()))
    
    if len(candidate_paths) == 0:
        print("[analysis] Warning: No valid paths found. Skipping Path Sankey diagram.")
        return
    
    # 5. Compute path features using cached E_list from kernel computation
    # This avoids re-iterating the loader (which causes "lazy wrapper" error)
    path_features_list = []
    
    # Get E_list from kernel_pack if available
    E_list = None
    if kernel_pack is not None and "E_list" in kernel_pack:
        E_list = kernel_pack["E_list"]
        P_samples = E_list[0].shape[0] if len(E_list) > 0 else 0
        print(f"[analysis] Using cached E_list from kernel computation ({P_samples} samples)")
    else:
        print("[analysis] Warning: E_list not available in kernel_pack. Skipping path feature computation.")
        return
    
    # Compute path features from cached E_list (no loader iteration needed!)
    # Reduce candidate paths for speed (10 is enough for visualization)
    num_candidates = min(10, len(candidate_paths))
    candidate_paths_subset = candidate_paths[:num_candidates]
    print(f"[analysis] Computing path features for {num_candidates} candidate paths...")
    
    for path_idx, path in enumerate(candidate_paths_subset):
        if path_idx % 5 == 0:
            print(f"[analysis] Processing path {path_idx+1}/{num_candidates}...")
        # Compute path weight
        path_weight = torch.tensor(1.0, device=dev, dtype=torch.float32)
        if weight_aware:
            for l in range(len(path)):
                if l < len(model.linears):
                    if l == 0:
                        w = model.linears[0].weight[path[0], :].abs().mean()
                    else:
                        w = model.linears[l].weight[path[l], path[l-1]].abs()
                    path_weight = path_weight * w
            if len(path) > 0:
                path_weight = path_weight * model.readout.weight[0, path[-1]].abs()
        
        # Compute phi from cached E_list
        phi = torch.ones(P_samples, device=dev, dtype=torch.float32)
        for l in range(len(path)):
            if l < len(E_list) and path[l] < E_list[l].shape[1]:
                # E_list[l] is (P, d_l), we want column path[l]
                phi = phi * E_list[l][:, path[l]].to(dev)
        
        if weight_aware:
            phi = phi * path_weight
        
        path_features_list.append((path, phi))
    
    # Truncate to match eigenvector sample count if needed
    P_evecs = top_evecs.shape[0]
    # path_features_list already has the correct paths, just ensure alignment
    if len(path_features_list) > 0 and path_features_list[0][1].shape[0] != P_evecs:
        path_features_list = [(path, phi[:P_evecs]) if phi.shape[0] > P_evecs else (path, phi) 
                              for path, phi in path_features_list]
    
    # 6. For each eigenvector, find the best-matching path
    # Ensure sample count matches
    P_evecs = top_evecs.shape[0]  # Number of samples used for eigenvectors
    P_paths = path_features_list[0][1].shape[0] if len(path_features_list) > 0 else 0  # Number of samples for path features
    
    if P_evecs != P_paths:
        print(f"[analysis] Warning: Sample count mismatch (evecs: {P_evecs}, paths: {P_paths}). "
              f"Using min({P_evecs}, {P_paths}) samples.")
        P_use = min(P_evecs, P_paths)
        top_evecs = top_evecs[:P_use, :]
        # Truncate path features to match
        path_features_list = [(path, phi[:P_use]) for path, phi in path_features_list]
    else:
        P_use = P_evecs
    
    eigenpath_matches = []  # List of (eigen_idx, path, alignment_score, eval_val)
    
    # OPTIMIZED: Vectorize alignment computation instead of nested loops
    # Stack all path features into a matrix: (num_paths, P_use)
    if len(path_features_list) == 0:
        print("[analysis] Warning: No path features computed. Skipping eigenpath matching.")
        return
    
    print(f"[analysis] Computing alignments for {len(path_features_list)} paths and {top_k_actual} eigenvectors...")
    
    # Keep eigenvectors on GPU for fast computation
    top_evecs_gpu = top_evecs.to(dev)  # (P_use, top_k_actual) on GPU
    
    # Stack all phi into a matrix: (num_paths, P_use)
    phi_matrix = torch.stack([phi.to(dev) for _, phi in path_features_list], dim=0)  # (num_paths, P_use)
    phi_norms = torch.norm(phi_matrix, dim=1, keepdim=True)  # (num_paths, 1)
    phi_matrix_normed = phi_matrix / (phi_norms + 1e-10)  # Normalized
    
    # Compute all alignments at once: (top_k_actual, num_paths) - MUCH faster than nested loops!
    evecs_normed = top_evecs_gpu / (torch.norm(top_evecs_gpu, dim=0, keepdim=True) + 1e-10)  # (P_use, top_k_actual)
    alignments = torch.abs(evecs_normed.T @ phi_matrix_normed.T)  # (top_k_actual, num_paths)
    
    print(f"[analysis] Finding best-matching paths...")
    # For each eigenvector, find best path
    for eig_idx in range(min(top_k_actual, alignments.shape[0])):
        best_path_idx = alignments[eig_idx].argmax().item()
        best_alignment = alignments[eig_idx, best_path_idx].item()
        
        if best_alignment > 1e-8:  # Only add if meaningful alignment
            best_path = path_features_list[best_path_idx][0]
            eigenpath_matches.append((eig_idx, best_path, best_alignment, top_evals[eig_idx].item()))
    
    print(f"[analysis] Found {len(eigenpath_matches)} eigenpath matches. Drawing graph...")
    
    # Sort by eigenvalue (descending)
    eigenpath_matches.sort(key=lambda x: x[3], reverse=True)
    eigenpath_matches = eigenpath_matches[:top_k_actual]
    
    # Check if graph is too large to draw efficiently
    total_nodes = sum(widths) + 1  # All neurons + input
    if total_nodes > 2000:  # Very large networks - skip detailed graph
        print(f"[analysis] Warning: Network too large ({total_nodes} nodes). Skipping Path Sankey diagram (too slow).")
        return
    
    # 7. Draw the graph - wrap in try-except to catch any errors
    try:
        pos = {}
        for layer, nodes in layer_x.items():
            n = len(nodes)
            for i, nkey in enumerate(nodes):
                pos[nkey] = (layer, (i - n/2) / max(n, 1) * 2.0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
    
        # Draw all edges first (light gray background)
        all_edges = []
        for l in range(len(widths)):
            if l == 0:
                # Input to first hidden
                for j in range(widths[0]):
                    all_edges.append((("L0", -1), (f"L1", j)))
            else:
                # Hidden to hidden
                for i in range(widths[l-1]):
                    for j in range(widths[l]):
                        all_edges.append(((f"L{l}", i), (f"L{l+1}", j)))
            # Last hidden to output
            if l == len(widths) - 1:
                for i in range(widths[-1]):
                    all_edges.append(((f"L{len(widths)}", i), ("OUT", -1)))
        
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, alpha=0.05, width=0.5, edge_color='gray', ax=ax)
        
        # Draw nodes colored by activation rate
        node_colors = []
        for node in G.nodes():
            rate = G.nodes[node].get("activation_rate", 0.0)
            # White (1.0) to black (0.0) - invert so high activation = darker
            node_colors.append(1.0 - rate)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='gray', vmin=0, vmax=1, 
                              node_size=100, alpha=0.8, ax=ax)
        
        # Draw eigenpaths with color coding
        colors = plt.cm.tab20(np.linspace(0, 1, len(eigenpath_matches)))
        
        for path_idx, (eig_idx, path, alignment, eval_val) in enumerate(eigenpath_matches):
            if len(path) == 0:
                continue
            
            # Build edges for this path
            path_edges = []
            # Input -> first hidden
            path_edges.append((("L0", -1), (f"L1", path[0])))
            # Through hidden layers
            for l in range(1, len(path)):
                path_edges.append(((f"L{l}", path[l-1]), (f"L{l+1}", path[l])))
            # Last hidden -> output
            path_edges.append(((f"L{len(path)}", path[-1]), ("OUT", -1)))
            
            # Compute edge weights: weight × average gate openness
            edge_weights = []
            for edge in path_edges:
                src, dst = edge
                if src[0] == "L0":
                    # Input to first hidden: use weight magnitude
                    if (hasattr(model, "linears") and len(model.linears) > 0 and 
                        len(path) > 0 and path[0] < model.linears[0].weight.shape[0]):
                        w = model.linears[0].weight[path[0], :].abs().mean().item()
                    else:
                        w = 1.0
                elif dst[0] == "OUT":
                    # Last hidden to output: use readout weight
                    if (hasattr(model, "readout") and len(path) > 0 and 
                        path[-1] < model.readout.weight.shape[1]):
                        w = model.readout.weight[0, path[-1]].abs().item()
                    else:
                        w = 1.0
                else:
                    # Hidden to hidden: use weight magnitude
                    src_layer = int(src[0][1:]) - 1
                    if (src_layer < len(model.linears) and 
                        src_layer < len(path) and 
                        src_layer > 0 and
                        path[src_layer] < model.linears[src_layer].weight.shape[0] and
                        path[src_layer-1] < model.linears[src_layer].weight.shape[1]):
                        w = model.linears[src_layer].weight[path[src_layer], path[src_layer-1]].abs().item()
                    else:
                        w = 1.0
                
                # Average gate openness: use activation rate of destination neuron
                dst_node = G.nodes[dst]
                gate_openness = dst_node.get("activation_rate", 0.5)
                edge_weight = w * gate_openness
                edge_weights.append(edge_weight)
            
            # Draw path with width proportional to average weight
            avg_weight = np.mean(edge_weights) if edge_weights else 1.0
            path_width = 1.0 + 3.0 * avg_weight
            
            nx.draw_networkx_edges(
                G, pos, edgelist=path_edges,
                width=path_width,
                edge_color=colors[path_idx],
                alpha=0.7,
                label=f"Eigenpath {eig_idx+1} (λ={eval_val:.3f}, align={alignment:.2f})",
                ax=ax
            )
        
        ax.set_title(f"Path Sankey Diagram: Top {len(eigenpath_matches)} Eigenpaths", fontsize=14)
        ax.axis('off')
        
        # Add legend
        if len(eigenpath_matches) > 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        # Ensure white background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        print(f"[analysis] Saving graph to {out_path}...")
        plt.tight_layout()
        
        # Save with explicit error handling
        try:
            plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
            plt.close('all')  # Close all figures to free memory
            import sys
            sys.stdout.flush()  # Force output flush
        except Exception as save_error:
            print(f"[analysis] ERROR during plt.savefig: {save_error}")
            import traceback
            print(traceback.format_exc())
            plt.close('all')
            raise
        
        print(f"[analysis] File save completed, verifying...")
        import sys
        sys.stdout.flush()
        
        # Verify file was created
        if os.path.exists(out_path):
            file_size = os.path.getsize(out_path)
            print(f"[analysis] saved Path Sankey diagram -> {out_path} (size: {file_size} bytes)")
        else:
            print(f"[analysis] ERROR: Path Sankey diagram file was not created at {out_path}")
            print(f"[analysis] Current directory: {os.getcwd()}")
            print(f"[analysis] Output directory exists: {os.path.exists(os.path.dirname(out_path))}")
            print(f"[analysis] Output directory is writable: {os.access(os.path.dirname(out_path), os.W_OK)}")
    except Exception as e:
        import traceback
        print(f"[analysis] ERROR: Failed to draw/save Path Sankey diagram: {e}")
        print(f"[analysis] Traceback: {traceback.format_exc()}")
        try:
            plt.close()
        except:
            pass
        return  # Exit function if graph drawing fails


# ---------------------------
# Interchange Intervention Accuracy (IIA)
# ---------------------------

@torch.no_grad()
def _detect_n_classes_from_model(model) -> Optional[int]:
    """Detect number of classes from model."""
    if hasattr(model, 'n_classes'):
        n_classes = model.n_classes
        if n_classes > 1:
            return n_classes
    if hasattr(model, 'readout'):
        out_features = model.readout.out_features
        if out_features > 1:
            return out_features
    return None


def compute_interchange_intervention_accuracy(
    model,
    loader,
    paths: List[List[int]],
    *,
    k: int = 10,
    n_interventions: int = 100,
    device: Optional[str] = None,
    n_classes: Optional[int] = None,  # Number of classes (None for binary)
) -> Dict[str, float]:
    """
    Compute Interchange Intervention Accuracy (IIA) - a causal metric.
    
    For each intervention:
    1. Run source image, record pre-activation values (u) at path units
    2. Run base image, surgically replace path unit pre-activations with source values
    3. Check if output matches source label
    
    Args:
        model: The neural network model
        loader: Data loader for intervention pairs
        paths: List of paths (each path is [i0, i1, ..., i_{L-1}])
        k: Number of top paths to use for intervention
        n_interventions: Number of intervention pairs to test
        device: Device to use
    
    Returns:
        Dictionary with IIA metrics:
        - iia_accuracy: Overall accuracy (0-1)
        - iia_by_k: Dictionary mapping k to accuracy for top-k paths
    """
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    # Detect n_classes if not provided
    if n_classes is None:
        n_classes = _detect_n_classes_from_model(model)
    
    if len(paths) == 0:
        return {"iia_accuracy": 0.0, "iia_by_k": {}}
    
    # Use top k paths
    top_k = min(k, len(paths))
    selected_paths = paths[:top_k]
    
    # Extract unique units per layer from selected paths
    L = len(model.linears)
    path_units_by_layer = [set() for _ in range(L)]
    for path in selected_paths:
        for l in range(min(len(path), L)):
            if l < len(path) and path[l] is not None:
                path_units_by_layer[l].add(path[l])
    
    # Convert to sorted lists for indexing
    path_units_by_layer = [sorted(list(units)) for units in path_units_by_layer]
    
    # Collect pairs of samples with different labels
    batch_data = []
    sample_count = 0
    for xb, yb in loader:
        if sample_count >= n_interventions * 2:
            break
        xb = xb.to(dev)
        yb = yb.to(dev)
        batch_data.append((xb, yb))
        sample_count += xb.shape[0]
    
    # Create intervention pairs (source, base) with different labels
    all_x = torch.cat([xb for xb, _ in batch_data], dim=0)
    all_y = torch.cat([yb for _, yb in batch_data], dim=0)
    
    # Find pairs with different labels
    n_samples = all_y.shape[0]
    source_data = []
    base_data = []
    pairs_found = 0
    
    for i in range(n_samples):
        if pairs_found >= n_interventions:
            break
        source_x = all_x[i:i+1]
        source_y = all_y[i]
        
        # Find a base sample with different label
        for j in range(n_samples):
            if i == j:
                continue
            base_x = all_x[j:j+1]
            base_y = all_y[j]
            
            # Only intervene if labels are different
            if source_y.item() != base_y.item():
                source_data.append((source_x, source_y))
                base_data.append((base_x, base_y))
                pairs_found += 1
                if pairs_found >= n_interventions:
                    break
    
    if len(source_data) == 0:
        print("[IIA] Warning: No intervention pairs found (all samples have same label).")
        return {"iia_accuracy": 0.0, "iia_by_k": {}}
    
    print(f"[IIA] Testing {len(source_data)} intervention pairs with {top_k} paths...")
    
    correct_interventions = 0
    total_interventions = len(source_data)
    
    # Debug: check label format
    if len(source_data) > 0:
        sample_label = source_data[0][1].item()
        print(f"[IIA] Sample label format: {sample_label} (type: {type(sample_label)})")
    
    for idx, ((source_x, source_y), (base_x, base_y)) in enumerate(zip(source_data, base_data)):
        # Step 1: Run source, record pre-activation values (u) at path units
        # We record pre-activations because that's what determines routing and we want to intervene at that level
        source_u_values = {}  # layer -> {unit_idx: value}
        
        h = source_x
        for l in range(L):
            u = model.linears[l](h)  # Pre-activation
            # Store pre-activation values for path units in this layer
            if l < len(path_units_by_layer) and len(path_units_by_layer[l]) > 0:
                source_u_values[l] = {}
                for unit_idx in path_units_by_layer[l]:
                    if unit_idx < u.shape[1]:
                        source_u_values[l][unit_idx] = u[:, unit_idx].clone()
            h = model.activation(u)  # Post-activation
        
        # Step 2: Run base with intervention - replace pre-activations at path units
        h = base_x
        for l in range(L):
            u = model.linears[l](h)  # Pre-activation
            # Intervene: replace path unit pre-activations with source values
            if l in source_u_values:
                u = u.clone()  # Clone to avoid modifying in-place
                for unit_idx, source_val in source_u_values[l].items():
                    if unit_idx < u.shape[1]:
                        u[:, unit_idx] = source_val
            # Now compute post-activation from intervened pre-activation
            h = model.activation(u)  # Post-activation
        
        # Get output
        intervened_output = model.readout(h)
        
        # Step 3: Check if output matches source label
        if n_classes is None or n_classes == 1:
            # Binary classification: labels are -1.0 or +1.0
            # Model outputs a single logit where >0 predicts +1, <0 predicts -1
            pred_logit = intervened_output.item()
            
            # Map logit to predicted label (-1.0 or +1.0)
            # Use > 0 instead of sign to avoid issues with exactly 0
            pred_label = 1.0 if pred_logit > 0 else -1.0
            
            # Get true label (should be -1.0 or +1.0)
            true_label = float(source_y.item())
            
            # Handle potential edge cases (normalize to -1.0 or +1.0)
            if true_label > 0:
                true_label = 1.0
            elif true_label < 0:
                true_label = -1.0
            else:
                # If label is exactly 0, treat as -1 (negative class)
                true_label = -1.0
            
            if pred_label == true_label:
                correct_interventions += 1
        else:
            # Multi-class classification: use argmax
            pred_class = intervened_output.argmax(dim=1).item()
            true_class = source_y.item()
            
            # Handle label format conversion if needed
            if isinstance(true_class, float) and true_class < 0:
                # Convert {-1, +1} to {0, 1} for binary case
                if n_classes == 2:
                    true_class = 1 if true_class > 0 else 0
            else:
                true_class = int(true_class)
            
            if pred_class == true_class:
                correct_interventions += 1
    
    iia_accuracy = correct_interventions / total_interventions if total_interventions > 0 else 0.0
    
    # Compute IIA for different k values (approximation - use same result)
    iia_by_k = {}
    for test_k in [1, 3, 5, 10, min(20, len(paths))]:
        if test_k <= len(paths):
            iia_by_k[test_k] = iia_accuracy  # Approximation for speed
    
    print(f"[IIA] Accuracy: {iia_accuracy:.3f} ({correct_interventions}/{total_interventions})")
    
    return {
        "iia_accuracy": float(iia_accuracy),
        "iia_by_k": {str(k): float(v) for k, v in iia_by_k.items()},
        "n_interventions": total_interventions,
        "n_correct": correct_interventions,
    }


def plot_iia_vs_epoch(
    history: List[Dict],
    out_path: str,
):
    """
    Plot Interchange Intervention Accuracy vs epoch.
    
    Args:
        history: List of training history dicts with "iia_accuracy" key
        out_path: Path to save the plot
    """
    _ensure_dir(os.path.dirname(out_path))
    
    epochs = []
    iia_values = []
    
    for h in history:
        if "iia_accuracy" in h and h["iia_accuracy"] is not None:
            epochs.append(h.get("epoch", 0))
            iia_values.append(h["iia_accuracy"])
    
    if len(epochs) == 0:
        print(f"[plot_iia] Warning: No IIA data in history (checked {len(history)} entries). Skipping plot.")
        print(f"[plot_iia] Sample history keys: {list(history[0].keys()) if len(history) > 0 else 'no history'}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.plot(epochs, iia_values, marker='o', label='IIA Accuracy', color='C4', linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', color='black')
    ax.set_ylabel('Interchange Intervention Accuracy', color='black')
    ax.set_title('Interchange Intervention Accuracy vs Epoch', color='black')
    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.legend(framealpha=1.0, facecolor='white', edgecolor='black')
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[analysis] Saved IIA plot -> {out_path}")


@torch.no_grad()
def plot_eigenvector_gate_patterns(
    model,
    train_loader,
    out_path: str,
    *,
    mode: str = "routing",
    k: int = 10,
    max_samples: int = 1000,
    device: Optional[str] = None,
    block_size: int = 1024,
    power_iters: int = 30,
    title_suffix: str = "",
):
    """
    Plot heatmap of gate state patterns for top k path kernel eigenvectors.
    
    For each of the top k eigenvectors, computes a representative gate pattern
    by weighting gate states by the eigenvector values. Creates a heatmap:
    - Rows: Layers (e.g., 4 rows for 4 layers)
    - Columns: Top k eigenvectors
    - Values: Average gate state (0 or 1) for each neuron in that layer/eigenvector
    
    Args:
        model: Neural network model
        train_loader: Data loader for computing path kernel
        out_path: Path to save the plot
        mode: Path kernel mode ("routing", "routing_gain", etc.)
        k: Number of top eigenvectors to visualize (default: 10)
        max_samples: Maximum samples to use
        device: Device to use
        block_size: Block size for kernel computation
        power_iters: Power iterations for kernel computation
        title_suffix: Additional text for plot title (e.g., "Epoch 0" or "Final")
    """
    from .path_kernel import compute_path_kernel_eigs, collect_path_factors
    
    dev = device or next(iter(model.parameters())).device
    model.eval()
    
    print(f"[eigenvector_gate_patterns] Computing path kernel and gate patterns...")
    
    # Compute path kernel eigenvalues/eigenvectors
    kern_train = compute_path_kernel_eigs(
        model, train_loader, device=dev, mode=mode, include_input=True,
        k=k, n_iter=power_iters, block_size=block_size, max_samples=max_samples, verbose=False
    )
    
    evals = kern_train["evals"].to(dev)  # (k,)
    evecs = kern_train["evecs"].to(dev)  # (P, k) - eigenvectors are columns
    
    # Collect path factors to get gate states (E_list)
    factors = collect_path_factors(
        model, train_loader, device=dev, mode="routing",  # Use routing mode to get binary gate states
        include_input=False, max_samples=max_samples  # Don't include input, only gate states
    )
    
    E_list = factors["E_list"]  # List of (P, d_l) tensors - binary gate states per layer
    L = len(E_list)
    widths = [E.shape[1] for E in E_list]
    
    # Get top k eigenvectors (already sorted by eigenvalue)
    top_k = min(k, evecs.shape[1])
    top_evecs = evecs[:, :top_k]  # (P, top_k)
    
    # For each eigenvector and each layer, compute weighted average gate state
    # This shows the "typical" gate state pattern for each layer/eigenvector combination
    gate_patterns = np.zeros((L, top_k))
    
    for eig_idx in range(top_k):
        evec = top_evecs[:, eig_idx]  # (P,) - eigenvector for this component
        # Use absolute values to weight by importance (both positive and negative matter)
        weights = torch.abs(evec)  # (P,)
        weights = weights / (weights.sum() + 1e-8)  # Normalize
        
        for layer_idx in range(L):
            E_layer = E_list[layer_idx]  # (P, d_l) - gate states for this layer
            # Weighted average of gate states for this layer
            weighted_gates = (E_layer * weights.unsqueeze(1)).sum(dim=0)  # (d_l,)
            # Average across all neurons to get a single value per layer
            # This represents the "typical" gate state for this layer in this eigenvector
            gate_patterns[layer_idx, eig_idx] = weighted_gates.mean().item()
    
    # Create heatmap
    _ensure_dir(os.path.dirname(out_path))
    
    fig, ax = plt.subplots(1, 1, figsize=(max(12, top_k * 1.2), max(6, L * 1.5)), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create heatmap with colormap that emphasizes 0 vs 1 (binary-like)
    im = ax.imshow(gate_patterns, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    
    # Set labels
    layer_labels = [f'Layer {i}\n({widths[i]} neurons)' for i in range(L)]
    eigenvector_labels = [f'Eig {i+1}\n(λ={evals[i].item():.2e})' for i in range(top_k)]
    
    ax.set_yticks(range(L))
    ax.set_yticklabels(layer_labels, color='black', fontsize=10)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(eigenvector_labels, color='black', rotation=45, ha='right', fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Gate State (0=off, 1=on)', color='black')
    cbar.ax.tick_params(colors='black')
    
    # Title
    title = f'Gate State Patterns: Top {top_k} Eigenvectors'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, color='black', fontsize=12, pad=20)
    
    ax.set_xlabel('Eigenvector (sorted by eigenvalue)', color='black')
    ax.set_ylabel('Layer', color='black')
    
    # Set tick colors
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[eigenvector_gate_patterns] Saved gate pattern heatmap -> {out_path}")
