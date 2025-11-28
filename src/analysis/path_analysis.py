# src/analysis/path_analysis.py

from __future__ import annotations

import os
import math
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
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
    mode: str = "routing_gain",   # "routing"|"routing_gain"|"routing_posdev"
    normalize: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """
    e(x): concatenate per-layer 'transmittance' vectors E_ℓ(p,:) across layers.
    Shape: [P, sum_l d_l]. This is an invariant 'routing×gain' signature
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
        if hasattr(model, "gates") and model.gates is not None:
            ap_am = [(g.a_plus.detach().to(dev), g.a_minus.detach().to(dev)) for g in model.gates]
        else:
            ap_am = [(None, None) for _ in range(L)]

        batch_parts = []
        for l in range(L):
            z = zs[l].float()
            a_plus, a_minus = ap_am[l]
            if mode == "routing":
                E = z
            elif mode == "routing_posdev":
                if a_plus is None:
                    E = z
                else:
                    ap = (a_plus - 1.0).clamp_min(0.0)
                    E = z * ap.unsqueeze(0)
            else:
                if a_plus is None:
                    E = z
                else:
                    E = z * a_plus.unsqueeze(0) + (1.0 - z) * a_minus.unsqueeze(0)
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
def _mean_transmittance_per_layer(model, loader, device, mode="routing_gain") -> List[torch.Tensor]:
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
        if hasattr(model, "gates") and model.gates is not None:
            ap_am = [(g.a_plus.detach().to(dev), g.a_minus.detach().to(dev)) for g in model.gates]
        else:
            ap_am = [(None, None) for _ in range(L)]
        for l in range(L):
            z = zs[l].float()
            a_plus, a_minus = ap_am[l]
            if mode == "routing":
                E = z
            elif mode == "routing_posdev":
                if a_plus is None:
                    E = z
                else:
                    ap = (a_plus - 1.0).clamp_min(0.0)
                    E = z * ap.unsqueeze(0)
            else:
                if a_plus is None:
                    E = z
                else:
                    E = z * a_plus.unsqueeze(0) + (1.0 - z) * a_minus.unsqueeze(0)
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
    mode: str = "routing_gain",
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
    mode: str = "routing_gain",
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
        if hasattr(model, "gates") and model.gates is not None:
            ap_am = [(g.a_plus.detach().to(dev), g.a_minus.detach().to(dev)) for g in model.gates]
        else:
            ap_am = [(None, None) for _ in range(L)]

        for l in range(L):
            z = zs[l].float()
            a_plus, a_minus = ap_am[l]
            if mode == "routing":
                E = z
            elif mode == "routing_posdev":
                if a_plus is None:
                    E = z
                else:
                    ap = (a_plus - 1.0).clamp_min(0.0)
                    E = z * ap.unsqueeze(0)
            else:
                if a_plus is None:
                    E = z
                else:
                    E = z * a_plus.unsqueeze(0) + (1.0 - z) * a_minus.unsqueeze(0)
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
    plt.ylim(0, 1.0)
    plt.xlabel("layer")
    plt.ylabel("cleanliness (top-1 share)")
    plt.grid(True, ls="--", alpha=0.3)
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
    plt.xlabel("epoch index")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.3)
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
    if not HAVE_SK:
        print("[analysis] scikit-learn missing; skipping path-shapley.")
        return
    from sklearn.feature_selection import mutual_info_regression

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
    plt.xlabel("circuit index")
    plt.ylabel("mutual information")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.3)
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
    mode: str = "routing_gain",
):
    """
    Rank units by flow centrality (expected transmittance × outgoing |W|),
    ablate cumulatively by setting their gates a_plus=a_minus=0, and measure test error.
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
    # cumulative ablation
    cur = base
    for (l, u) in order:
        if hasattr(cur, "gates") and cur.gates is not None:
            cur.gates[l].a_plus.data[u]  = 0.0
            cur.gates[l].a_minus.data[u] = 0.0
        errs.append(_mse(cur, loader_test))

    # plot
    plt.figure(figsize=(7,4))
    xs = np.arange(len(errs))
    plt.plot(xs, errs, marker="o")
    plt.xlabel("ablation step")
    plt.ylabel("test MSE")
    plt.title("Ablation waterfall (cumulative)")
    plt.grid(True, ls="--", alpha=0.3)
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
    mode: str = "routing_gain",
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
            # last hidden layer -> readout: (1 x w_{L-1})
            W_ro = model.readout.weight.detach().abs().to("cpu").squeeze(0)  # (w_{L-1},)
            out_sum = W_ro
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
    mode: str = "routing_gain",
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
            if hasattr(model, "gates") and model.gates is not None:
                ap_am = [(g.a_plus.detach().to(dev), g.a_minus.detach().to(dev)) for g in model.gates]
            else:
                ap_am = [(None, None) for _ in range(L)]
            for l in range(L):
                z = zs[l].float()
                a_plus, a_minus = ap_am[l]
                E_l = z * a_plus.unsqueeze(0) + (1.0 - z) * a_minus.unsqueeze(0) if a_plus is not None else z
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
            S = (W * src_weight.unsqueeze(0)).numpy()  # (d_out,d_in)
            # threshold top edges
            thr = np.quantile(S, 1.0 - keep_frac)
            for j in range(d_out):
                for i in range(d_in):
                    if S[j, i] >= thr:
                        G.add_edge((l, i), (l+1, j), weight=S[j, i])

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
    kernel_mode: str = "routing_gain",
    include_input_in_kernel: bool = True,
    block_size: int = 1024,
    max_samples_kernel: Optional[int] = None,
    max_samples_embed: Optional[int] = None,
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
            n_iter=30,
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

    # 2) Graph + top 3 paths
    try:
        graph_png = os.path.join(out_dir, f"nn_graph_paths_{step_tag}.png")
        plot_nn_graph_with_paths(model, val_loader, graph_png, mode=kernel_mode, beam=24, top_k=3)
    except Exception as e:
        import traceback
        print(f"[path_analysis] Warning: Failed to create NN graph plot: {e}")
        print(f"[path_analysis] Traceback: {traceback.format_exc()}")

    # 3) Cleanliness for top-k paths
    try:
        clean_png = os.path.join(out_dir, f"path_cleanliness_{step_tag}.png")
        plot_path_cleanliness(model, val_loader, clean_png, mode=kernel_mode, top_k=5)
    except Exception as e:
        import traceback
        print(f"[path_analysis] Warning: Failed to create cleanliness plot: {e}")
        print(f"[path_analysis] Traceback: {traceback.format_exc()}")

    # 4) Embedding map
    Epack = path_embedding(model, val_loader, device=None, mode=kernel_mode, normalize=True, max_samples=max_samples_embed)
    emb_png = os.path.join(out_dir, f"path_embedding_{step_tag}.png")
    plot_embedding_map(Epack["E"], Epack["labels"], emb_png, title=f"Path-embedding [{step_tag}]")

    # 5) Flow centrality heatmap
    central_png = os.path.join(out_dir, f"flow_centrality_{step_tag}.png")
    flow_centrality_heatmap(model, val_loader, central_png, mode=kernel_mode)


# ---------------------------
# Comprehensive path kernel metrics computation
# ---------------------------

@torch.no_grad()
def compute_path_kernel_metrics(
    model,
    train_loader,
    test_loader,
    *,
    mode: str = "routing_gain",
    k: int = 48,
    max_samples: Optional[int] = 5000,
    device: Optional[str] = None,
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
            k=k, n_iter=30, block_size=1024, max_samples=max_samples, verbose=False
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
        
        # Variance explained by eigenfunctions (per component, top 35) - compute on GPU
        if y_train is not None:
            y_train_centered = y_train - y_train.mean()
            y_train_var = y_train_centered.var().item()
            if y_train_var > 1e-12:
                y_train_norm = y_train_centered / (torch.norm(y_train_centered) + 1e-8)
                alignments = (y_train_norm @ evecs) ** 2  # (k,) - computed on GPU
                top_k = min(35, len(alignments))
                # Save per-component variance explained (top 35) - move to CPU only for final conversion
                metrics["path_kernel_variance_explained_train_per_component"] = alignments[:top_k].cpu().numpy().tolist()
                metrics["path_kernel_variance_explained_train"] = alignments.sum().item()
                metrics["path_kernel_variance_explained_train_top10"] = alignments[:min(10, len(alignments))].sum().item()
            else:
                metrics["path_kernel_variance_explained_train_per_component"] = [0.0] * min(35, len(evecs))
                metrics["path_kernel_variance_explained_train"] = 0.0
                metrics["path_kernel_variance_explained_train_top10"] = 0.0
        else:
            top_k = min(35, len(evecs))
            metrics["path_kernel_variance_explained_train_per_component"] = [0.0] * top_k
        
        # Compute for test set
        if test_loader is not None:
            try:
                kern_test = compute_path_kernel_eigs(
                    model, test_loader, device=dev, mode=mode, include_input=True,
                    k=k, n_iter=30, block_size=1024, max_samples=max_samples, verbose=False
                )
                y_test = kern_test.get("y")
                if y_test is not None:
                    y_test = y_test.to(dev)  # Keep on GPU
                    evecs_test = kern_test["evecs"].to(dev)  # Keep on GPU
                    y_test_centered = y_test - y_test.mean()
                    y_test_var = y_test_centered.var().item()
                    if y_test_var > 1e-12:
                        y_test_norm = y_test_centered / (torch.norm(y_test_centered) + 1e-8)
                        alignments_test = (y_test_norm @ evecs_test) ** 2  # Computed on GPU
                        top_k = min(35, len(alignments_test))
                        # Save per-component variance explained (top 35) - move to CPU only for final conversion
                        metrics["path_kernel_variance_explained_test_per_component"] = alignments_test[:top_k].cpu().numpy().tolist()
                        metrics["path_kernel_variance_explained_test"] = alignments_test.sum().item()
                        metrics["path_kernel_variance_explained_test_top10"] = alignments_test[:min(10, len(alignments_test))].sum().item()
                    else:
                        top_k = min(35, len(evecs_test))
                        metrics["path_kernel_variance_explained_test_per_component"] = [0.0] * top_k
                        metrics["path_kernel_variance_explained_test"] = 0.0
                        metrics["path_kernel_variance_explained_test_top10"] = 0.0
                else:
                    top_k = min(35, len(kern_test["evecs"]))
                    metrics["path_kernel_variance_explained_test_per_component"] = [0.0] * top_k
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
            "path_kernel_variance_explained_train_per_component": [0.0] * 35,
            "path_kernel_variance_explained_test_per_component": [0.0] * 35,
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
    mode: str = "routing_gain",
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
    y_train_centered = y_train - y_train.mean()
    y_train_var = y_train_centered.var().item()
    if y_test is not None:
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
    plt.xlabel("Layer depth")
    plt.ylabel("Variance explained (kernel alignment)")
    plt.title(title)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.3)
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
    mode: str = "routing_gain",
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
    plt.ylim(0, 1)
    plt.title(title)
    plt.xticks([0], ["Initial vs Current"])
    plt.grid(True, ls="--", alpha=0.3, axis="y")
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
    mode: str = "routing_gain",
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
    plt.title(title)
    plt.xticks([0], ["Path Kernel"])
    plt.grid(True, ls="--", alpha=0.3, axis="y")
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
    mode: str = "routing_gain",
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
    plt.xlabel("Eigenvalue rank")
    plt.ylabel("Cumulative variance explained")
    plt.title(title)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.3)
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
    mode: str = "routing_gain",
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

