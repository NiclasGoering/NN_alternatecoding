# src/utils/metrics.py
from __future__ import annotations
import math
import numpy as np
import torch
from .paths import (
    hash_mask_list,
    compute_local_derivatives,
    compute_path_entropy,
    compute_path_gain_entropy,
    compute_path_mi,
    compute_confident_churn,
    compute_path_snr,
    compute_neural_race_index,
)

# -------------------------
# Basic losses / accuracies
# -------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor):
    return torch.mean((pred - target) ** 2)

def accuracy_from_logits(pred: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        return torch.sign(pred).eq(target).float().mean().item()

# -------------------------
# Slope-based summaries
# -------------------------

def slope_budget(layer_slopes):
    """
    Sum of absolute deviations from identity (a^+ ~ 1, a^- ~ 1) per layer.
    Returns (total_budget, [per_layer_budgets])
    """
    total = 0.0
    per = []
    for a_p, a_m in layer_slopes:
        v = torch.sum(torch.abs(a_p - 1.0)) + torch.sum(torch.abs(a_m - 1.0))
        per.append(v.item())
        total += v
    return float(total.item() if hasattr(total, "item") else total), per

def slope_entropy(layer_slopes, eps: float = 1e-12):
    """
    Shannon entropy over the allocation of |a-1| across all units.
    Returns (H_total, [H_layer])
    """
    vals = []
    per_layer_vals = []
    for a_p, a_m in layer_slopes:
        v = torch.cat([torch.abs(a_p - 1.0).flatten(),
                       torch.abs(a_m - 1.0).flatten()])
        per_layer_vals.append(v)
        vals.append(v)
    if len(vals) == 0:
        return 0.0, []

    vals = torch.cat(vals)
    s = vals.sum()
    if s.item() <= eps:
        return 0.0, [0.0] * len(per_layer_vals)

    p = vals / s
    H_total = float(-(p * torch.log(p + eps)).sum().item())

    H_layers = []
    for v in per_layer_vals:
        s_l = v.sum()
        if s_l.item() <= eps:
            H_layers.append(0.0)
        else:
            p_l = v / s_l
            H_layers.append(float(-(p_l * torch.log(p_l + eps)).sum().item()))
    return H_total, H_layers

def slope_deviation(layer_slopes):
    """
    Mean-squared deviation from identity per layer:
        δ_l = (1/(2 d_l)) ||a^+ - 1||_2^2 + (1/(2 d_l)) ||a^- - 1||_2^2
    """
    deltas = []
    for a_p, a_m in layer_slopes:
        d = a_p.numel()
        v = torch.sum((a_p - 1.0) ** 2 + (a_m - 1.0) ** 2) / (2.0 * d)
        deltas.append(float(v.item()))
    return deltas

# -------------------------
# Mask churn (dataset-level)
# -------------------------

def mask_churn(prev_masks, cur_masks):
    """
    Fraction of (sample,unit) mask bits that changed since the last cycle, per layer.
    prev_masks/cur_masks: lists of boolean tensors of shape (P, d_l).
    Returns: list of floats, one per layer.
    """
    if prev_masks is None or cur_masks is None:
        return [0.0] * (len(cur_masks) if cur_masks is not None else 0)
    out = []
    for pm, cm in zip(prev_masks, cur_masks):
        changed = (pm != cm).float().mean().item()
        out.append(float(changed))
    return out

def mask_churn_active(prev_masks, cur_masks, layer_slopes, tau: float = 0.02):
    """
    Mask churn restricted to 'meaningful' gates (max(|a^+-1|, |a^- -1|) > tau).
    Uses dataset-level masks (P, d_l). Returns per-layer list of floats.
    """
    if prev_masks is None or cur_masks is None:
        return [0.0] * (len(cur_masks) if cur_masks is not None else 0)

    out = []
    for (pm, cm), (a_p, a_m) in zip(zip(prev_masks, cur_masks), layer_slopes):
        active_nonlin = torch.max((a_p - 1.0).abs(), (a_m - 1.0).abs()) > tau  # (d_l,)
        flips = ((pm != cm).float() * active_nonlin.unsqueeze(0)).sum()
        denom = (active_nonlin.sum() * pm.shape[0]).clamp_min(1)
        out.append(float((flips / denom).item()))
    return out

# -------------------------
# Effective rank (activations)
# -------------------------

def effective_rank(hidden_activations: torch.Tensor, threshold: float = 0.99):
    """
    Number of singular values needed to explain 'threshold' fraction of variance.
    hidden_activations: (Batch, Width) or (Width, Batch)
    """
    h = hidden_activations
    if h.dim() != 2:
        raise ValueError("effective_rank expects a 2D tensor (N,D) or (D,N).")
    if h.shape[0] < h.shape[1]:
        h = h.t()
    # Center
    h = h - h.mean(dim=0, keepdim=True)
    try:
        S = torch.linalg.svdvals(h)
    except Exception:
        return 0.0
    total_var = torch.sum(S ** 2)
    if total_var.item() < 1e-10:
        return 0.0
    cumulative = torch.cumsum(S ** 2, dim=0) / total_var
    rank = torch.searchsorted(cumulative, threshold).item() + 1
    return float(rank)

# -------------------------
# Gate taxonomy (quick view)
# -------------------------

def gate_stats(layer_slopes, dead_threshold: float = 0.1, linear_threshold: float = 0.1):
    """
    Categorizes neurons into {linear, relu, dead, other} using (a^+, a^-).
    Returns fractions (sum to ~1).
    """
    counts = {"linear": 0, "relu": 0, "dead": 0, "other": 0}
    total = 0

    for a_p, a_m in layer_slopes:
        ap = a_p.flatten()
        am = a_m.flatten()
        total += ap.numel()

        # Dead: both near 0
        is_dead = (ap.abs() < dead_threshold) & (am.abs() < dead_threshold)
        counts["dead"] += int(is_dead.sum().item())

        # Linear (near identity on both sides)
        is_linear = ((ap - 1.0).abs() < linear_threshold) & ((am - 1.0).abs() < linear_threshold)
        counts["linear"] += int(is_linear.sum().item())

        # ReLU-like: one side high, the other near 0
        is_relu_pos = (ap > 0.5) & (am.abs() < 0.2)
        is_relu_neg = (am > 0.5) & (ap.abs() < 0.2)
        counts["relu"] += int((is_relu_pos | is_relu_neg).sum().item())

    counts["other"] = max(0, total - (counts["dead"] + counts["linear"] + counts["relu"]))
    if total > 0:
        return {k: v / total for k, v in counts.items()}
    else:
        return {k: 0.0 for k in counts.keys()}

# -------------------------
# Neural-race / path metrics
# -------------------------

def _to_cpu_bool(z: torch.Tensor) -> torch.Tensor:
    return z.detach().to("cpu").bool()

@torch.no_grad()
def compute_path_metrics(
    model,
    loader,
    *,
    tau: float = 0.02,
    eps: float = 1e-8,
    device: str | None = None,
    prev_masks: list[torch.Tensor] | None = None,            # (P, d_l) boolean per layer
    prev_path_hashes: list[torch.Tensor] | None = None,      # (P,) per layer (for confident churn)
    group_ids: torch.Tensor | np.ndarray | None = None,      # (P,) optional
    n_groups: int | None = None,
    return_masks: bool = False,
    return_path_hashes: bool = False,
):
    """
    Compute path-level metrics on a fixed (non-shuffled) loader.

    Outputs (dict):
      - path_pressure_layers:     average |a_side-1| on active gates per layer
      - path_entropy_layers:      entropy over active positive nonlinearity mass per layer
      - active_path_complexity:   exp(H) per layer
      - churn_active_layers:      churn among gates with |a_side-1| > tau (if prev_masks given)
      - sei_layers:               group equalization (weighted by side-specific mass) per layer (if groups)
      - snr_max_layers / snr_p95_layers: percentiles of gate-level SNR proxy per layer

      Path-centric extras:

      - H_path:                   support entropy of partial paths
      - H_gain:                   entropy of path gain (using local derivatives)
      - I_layers:                 path-to-group mutual information per layer (if groups)
      - confident_churn_layers:   churn of confident paths (if prev_path_hashes given)
      - path_snr_dict + summaries: label-correlation, support, SNR per path
      - nri:                      composite Neural-Race Index (if components available)

      Optionally returns:
      - cur_masks:                list[(P,d_l) bool]
      - cur_path_hashes:          list[(P,) int64] per layer
    """
    model.eval()
    dev = device or next(iter(model.parameters())).device

    # Peek to get L and widths without consuming the main loader iterator
    xb0, yb0 = next(iter(loader))
    xb0, yb0 = xb0.to(dev), yb0.to(dev)
    _, cache0 = model(xb0, return_cache=True)
    L = len(cache0["z"])
    widths = [z.shape[1] for z in cache0["z"]]

    # Accumulators kept on device
    total_abs_dev_active = [torch.zeros((), device=dev) for _ in widths]
    total_active_count   = [torch.zeros((), device=dev) for _ in widths]
    active_mass_vec      = [torch.zeros(w, device=dev) for w in widths]
    snr_sum_v            = [torch.zeros(w, device=dev) for w in widths]
    snr_sum_v2           = [torch.zeros(w, device=dev) for w in widths]
    snr_sum_vy           = [torch.zeros(w, device=dev) for w in widths]

    group_counts = [None for _ in widths]
    if group_ids is not None and n_groups is not None:
        # keep on device during accumulation
        group_counts = [torch.zeros(n_groups, w, device=dev) for w in widths]

    cur_masks = [[] for _ in widths] if (return_masks or prev_masks is not None) else None

    # Align group_ids with dataset size, keep on device during loop
    dataset_size = None
    try:
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "__len__"):
            dataset_size = len(loader.dataset)
    except Exception:
        pass

    if group_ids is not None:
        if isinstance(group_ids, np.ndarray):
            group_ids_dev = torch.from_numpy(group_ids).to(dev).long()
        else:
            group_ids_dev = group_ids.to(dev).long()
        if dataset_size is not None and len(group_ids_dev) != dataset_size:
            # disable group metrics if sizes disagree
            print(f"Warning: group_ids size ({len(group_ids_dev)}) != dataset size ({dataset_size}); disabling group metrics.")
            group_ids_dev = None
            group_counts = [None for _ in widths]
    else:
        group_ids_dev = None

    # Collect full-dataset tensors for path-centric metrics (on device first)
    all_z_list = None
    all_u_list = None
    all_y = None
    all_margins = None

    sample_offset = 0
    batch_sizes = []

    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        yhat, cache = model(xb, return_cache=True)
        zs = [z.detach() for z in cache["z"]]  # (B, d_l), typically indicator for u>=0
        us = [u.detach() for u in cache["u"]]  # (B, d_l), pre-activation
        B = xb.shape[0]
        batch_sizes.append(B)

        # collect for path metrics
        if all_z_list is None:
            all_z_list = [z.clone() for z in zs]
            all_u_list = [u.clone() for u in us]
        else:
            for l in range(L):
                all_z_list[l] = torch.cat([all_z_list[l], zs[l]], dim=0)
                all_u_list[l] = torch.cat([all_u_list[l], us[l]], dim=0)

        yb_flat = yb.squeeze() if yb.dim() > 1 else yb
        yhat_flat = yhat.squeeze() if yhat.dim() > 1 else yhat
        margins_batch = yhat_flat.abs()

        if all_y is None:
            all_y = yb_flat.clone()
            all_margins = margins_batch.clone()
        else:
            all_y = torch.cat([all_y, yb_flat], dim=0)
            all_margins = torch.cat([all_margins, margins_batch], dim=0)

        # per-layer accumulators
        for l in range(L):
            z = zs[l].float()                 # (B, d_l)
            u = us[l]
            d_l = z.shape[1]

            # side-aware deviations / masses
            a_p = model.gates[l].a_plus.detach().view(-1)   # (d_l,)
            a_m = model.gates[l].a_minus.detach().view(-1)  # (d_l,)

            abs_dev_p = (a_p - 1.0).abs().unsqueeze(0)      # (1, d_l)
            abs_dev_m = (a_m - 1.0).abs().unsqueeze(0)      # (1, d_l)

            # average |a_side - 1| over active side for each (sample, unit)
            abs_dev_active = z * abs_dev_p + (1.0 - z) * abs_dev_m  # (B, d_l)
            total_abs_dev_active[l] += abs_dev_active.sum()
            # every sample uses exactly one side of each gate
            total_active_count[l]   += torch.tensor(B * d_l, device=dev, dtype=torch.float32)

            # positive mass on the actually active side (for entropy/complexity + SNR proxy)
            m_pos = (a_p - 1.0).clamp_min(0.0).unsqueeze(0)  # (1, d_l)
            m_neg = (a_m - 1.0).clamp_min(0.0).unsqueeze(0)  # (1, d_l)
            mass = z * m_pos + (1.0 - z) * m_neg             # (B, d_l)

            active_mass_vec[l] += mass.sum(dim=0)

            vy = mass * yb_flat.view(-1, 1)                  # (B, d_l)
            snr_sum_v[l]  += mass.sum(dim=0)
            snr_sum_v2[l] += (mass ** 2).sum(dim=0)
            snr_sum_vy[l] += vy.sum(dim=0)

            # optional: group counts weighted by side-specific mass
            if group_counts[l] is not None and group_ids_dev is not None:
                if sample_offset + B <= len(group_ids_dev):
                    g_batch = group_ids_dev[sample_offset: sample_offset + B]  # (B,)
                    # accumulate per group (sparse loop over groups present in this batch)
                    for g in g_batch.unique():
                        rows = (g_batch == g).nonzero(as_tuple=False).flatten()
                        if rows.numel() > 0:
                            group_counts[l][int(g)] += mass[rows].sum(dim=0)
                else:
                    # size mismatch should have already been guarded
                    pass

            if cur_masks is not None:
                cur_masks[l].append(z.bool())

        sample_offset += B

    # Finalize masks
    if cur_masks is not None:
        cur_masks = [torch.cat(parts, dim=0) for parts in cur_masks]  # (P, d_l) on device

    # Aggregate per-layer scalars
    path_pressure_layers = []
    path_entropy_layers  = []
    apc_layers           = []
    snr_max_layers       = []
    snr_p95_layers       = []
    churn_active_layers  = []
    sei_layers           = []

    P_total = sum(batch_sizes)

    for l in range(L):
        denom = float(max(1.0, total_active_count[l].item()))
        path_pressure_layers.append(float((total_abs_dev_active[l] / denom).item()))

        w = active_mass_vec[l]  # (d_l,) on device
        s = float(w.sum().item())
        if s <= eps:
            path_entropy_layers.append(0.0)
            apc_layers.append(0.0)
        else:
            p = w / (w.sum() + eps)
            H = float(-(p * (p + eps).log()).sum().item())
            path_entropy_layers.append(H)
            apc_layers.append(float(math.exp(H)))

        if P_total > 0:
            mu = snr_sum_v[l] / P_total
            var = snr_sum_v2[l] / P_total - mu ** 2
            snr = (snr_sum_vy[l] / P_total).abs() / (var.clamp_min(eps).sqrt())
            snr_np = snr.detach().cpu().numpy()
            snr_max_layers.append(float(np.max(snr_np)) if snr_np.size > 0 else 0.0)
            snr_p95_layers.append(float(np.percentile(snr_np, 95)) if snr_np.size > 0 else 0.0)
        else:
            snr_max_layers.append(0.0)
            snr_p95_layers.append(0.0)

        # churn among gates that are meaningfully nonlinear on either side
        if prev_masks is not None and len(prev_masks) == L:
            a_p = model.gates[l].a_plus.detach()
            a_m = model.gates[l].a_minus.detach()
            active_nonlin = torch.max((a_p - 1.0).abs(), (a_m - 1.0).abs()) > tau  # (d_l,)

            prev_m = prev_masks[l].to(dev) if prev_masks[l].device != dev else prev_masks[l]
            cur_m  = cur_masks[l] if cur_masks is not None else None
            if cur_m is not None:
                flips = ((prev_m != cur_m).float() * active_nonlin.unsqueeze(0)).sum()
                denom = (active_nonlin.sum() * prev_m.shape[0]).clamp_min(1)
                churn_active_layers.append(float((flips / denom).item()))
            else:
                churn_active_layers.append(0.0)

        # group equalization (weighted by side-aware nonlinearity mass)
        if group_counts[l] is not None:
            counts = group_counts[l]                                 # (G, d_l)
            top_g = torch.argmax(counts, dim=0)                      # (d_l,)
            rows  = torch.arange(counts.shape[1], device=counts.device)
            top_mass   = counts[top_g, rows]                         # (d_l,)
            total_mass = counts.sum(dim=0)                           # (d_l,)
            a_dev = torch.max(
                (model.gates[l].a_plus.detach() - 1.0).abs(),
                (model.gates[l].a_minus.detach() - 1.0).abs()
            )
            num = (top_mass * a_dev).sum()
            den = ((total_mass - top_mass) * a_dev).sum().clamp_min(eps)
            sei_layers.append(float((num / den).item()))
        else:
            sei_layers.append(None)

    out = dict(
        path_pressure_layers=path_pressure_layers,
        path_entropy_layers=path_entropy_layers,
        active_path_complexity=apc_layers,
        snr_max_layers=snr_max_layers,
        snr_p95_layers=snr_p95_layers,
        sei_layers=sei_layers,
    )
    if prev_masks is not None:
        out["churn_active_layers"] = churn_active_layers
    if return_masks and cur_masks is not None:
        out["cur_masks"] = [m.detach().cpu() for m in cur_masks]

    # -------------------------
    # Path-centric metrics
    # -------------------------
    if all_z_list is not None and len(all_z_list) == L:
        # move to CPU for path-level ops
        all_z_list = [z.detach().cpu() for z in all_z_list]
        all_u_list = [u.detach().cpu() for u in all_u_list]
        all_y_cpu = all_y.detach().cpu() if all_y is not None else None
        all_margins_cpu = all_margins.detach().cpu() if all_margins is not None else None

        # slopes on CPU
        layer_slopes = model.layer_slopes() if hasattr(model, "layer_slopes") else []
        layer_slopes = [(ap.detach().cpu(), am.detach().cpu()) for ap, am in layer_slopes] if len(layer_slopes) == L else []

        if len(layer_slopes) == L:
            # local derivatives (per side)
            m_list = compute_local_derivatives(all_u_list, all_z_list, layer_slopes)

            # H_path: support entropy of partial paths
            out["H_path"] = compute_path_entropy(all_z_list, eps=eps)

            # H_gain: entropy of path gains (uses u,z,slopes inside)
            out["H_gain"] = compute_path_gain_entropy(all_z_list, m_list, eps=eps)

            # I_layers: MI(path up to l; group) — require groups (on CPU)
            if group_ids_dev is not None:
                out["I_layers"] = compute_path_mi(
                    all_z_list,
                    group_ids_dev.detach().cpu(),
                    up_to_layer=None,
                    eps=eps,
                )
            else:
                out["I_layers"] = None

            # Confident path churn over cycles (need previous path hashes)
            if prev_path_hashes is not None and len(prev_path_hashes) == L:
                prev_path_hashes_cpu = [h.detach().cpu() for h in prev_path_hashes]
                cur_path_hashes = [hash_mask_list(all_z_list[:l+1]) for l in range(L)]
                out["confident_churn_layers"] = compute_confident_churn(
                    prev_path_hashes_cpu, cur_path_hashes, all_margins_cpu, tau=0.0
                )
                if return_path_hashes:
                    out["cur_path_hashes"] = cur_path_hashes
            else:
                out["confident_churn_layers"] = None
                if return_path_hashes:
                    out["cur_path_hashes"] = [hash_mask_list(all_z_list[:l+1]) for l in range(L)]

            # Path SNR stats (label correlation; support; SNR)
            if all_y_cpu is not None:
                path_snr_dict = compute_path_snr(all_z_list, all_y_cpu, eps=eps)
                out["path_snr_dict"] = path_snr_dict

                if len(path_snr_dict) > 0:
                    c_vals = [c for c, _, _ in path_snr_dict.values()]
                    N_vals = [N for _, N, _ in path_snr_dict.values()]
                    s_vals = [s for _, _, s in path_snr_dict.values()]
                    out["path_snr_c_gamma_mean"]    = float(np.mean(c_vals))
                    out["path_snr_c_gamma_median"]  = float(np.median(c_vals))
                    out["path_snr_c_gamma_std"]     = float(np.std(c_vals))
                    out["path_snr_N_gamma_mean"]    = float(np.mean(N_vals))
                    out["path_snr_N_gamma_median"]  = float(np.median(N_vals))
                    out["path_snr_N_gamma_std"]     = float(np.std(N_vals))
                    out["path_snr_N_gamma_total"]   = int(np.sum(N_vals))
                    out["path_snr_SNR_gamma_mean"]  = float(np.mean(s_vals))
                    out["path_snr_SNR_gamma_median"]= float(np.median(s_vals))
                    out["path_snr_SNR_gamma_std"]   = float(np.std(s_vals))
                    thr = float(np.median(s_vals)) if len(s_vals) > 0 else 0.0
                    out["path_snr_count_above_threshold"] = int(sum(1 for v in s_vals if v > thr))
                    out["path_snr_threshold"] = thr
                    out["path_snr_num_paths"]  = int(len(path_snr_dict))
                else:
                    out["path_snr_c_gamma_mean"] = None
                    out["path_snr_c_gamma_median"] = None
                    out["path_snr_c_gamma_std"] = None
                    out["path_snr_N_gamma_mean"] = None
                    out["path_snr_N_gamma_median"] = None
                    out["path_snr_N_gamma_std"] = None
                    out["path_snr_N_gamma_total"] = None
                    out["path_snr_SNR_gamma_mean"] = None
                    out["path_snr_SNR_gamma_median"] = None
                    out["path_snr_SNR_gamma_std"] = None
                    out["path_snr_count_above_threshold"] = None
                    out["path_snr_threshold"] = None
                    out["path_snr_num_paths"] = 0
            else:
                out["path_snr_dict"] = None
                out["path_snr_c_gamma_mean"] = None
                out["path_snr_c_gamma_median"] = None
                out["path_snr_c_gamma_std"] = None
                out["path_snr_N_gamma_mean"] = None
                out["path_snr_N_gamma_median"] = None
                out["path_snr_N_gamma_std"] = None
                out["path_snr_N_gamma_total"] = None
                out["path_snr_SNR_gamma_mean"] = None
                out["path_snr_SNR_gamma_median"] = None
                out["path_snr_SNR_gamma_std"] = None
                out["path_snr_count_above_threshold"] = None
                out["path_snr_threshold"] = None
                out["path_snr_num_paths"] = None

            # composite Neural-Race Index — needs H_gain, I_layers, confident churn
            if (out.get("H_gain") is not None and
                out.get("I_layers") is not None and
                out.get("confident_churn_layers") is not None):

                full_path_hashes = hash_mask_list(all_z_list)
                K = int(torch.unique(full_path_hashes).numel())

                # heuristic: if n_groups provided, reuse per layer; otherwise fall back
                if n_groups is not None:
                    m_layers = [int(n_groups)] * L
                elif group_ids_dev is not None:
                    n_unique = int(torch.unique(group_ids_dev.detach().cpu()).numel())
                    m_layers = [n_unique] * L
                else:
                    m_layers = [2] * L

                out["nri"] = compute_neural_race_index(
                    H_gain=out["H_gain"],
                    I_layers=out["I_layers"],
                    rho_conf_layers=out["confident_churn_layers"],
                    K=K,
                    m_layers=m_layers,
                    eps=eps,
                )
            else:
                out["nri"] = None
        else:
            # cannot compute path metrics without slopes
            out["H_path"] = None
            out["H_gain"] = None
            out["I_layers"] = None
            out["confident_churn_layers"] = None
            out["path_snr_dict"] = None
            out["nri"] = None

    return out
