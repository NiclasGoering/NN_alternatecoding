# src/utils/paths.py
"""
Path-centric metrics for analyzing neural network routing behavior.
These metrics are designed to be path-aware, comparable across runs, and causally suggestive.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


@torch.no_grad()
def hash_mask_list(z_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Hash each sample's layerwise sign-mask to a 64-bit integer.
    
    Uses a polynomial rolling hash: for each layer, hash = hash * 31 + layer_hash.
    This is efficient and produces stable hashes.
    All computations are done on CPU for fastest performance.
    
    Args:
        z_list: List of tensors, each of shape (batch_size, d_l) with values in {0, 1}
    
    Returns:
        Tensor of shape (batch_size,) with 64-bit integer hashes (on CPU)
    """
    if len(z_list) == 0:
        return torch.tensor([], dtype=torch.int64)
    
    # Move all tensors to CPU for fastest computation
    z_list = [z.cpu() if z.device.type != 'cpu' else z for z in z_list]
    
    batch_size = z_list[0].shape[0]
    device = torch.device('cpu')  # Always use CPU
    
    # Initialize hashes
    hashes = torch.zeros(batch_size, dtype=torch.int64, device=device)
    
    # Polynomial rolling hash: hash = hash * 31 + layer_hash
    # For each layer, we compute a hash from the bit pattern
    for z in z_list:
        # Convert to boolean: (batch_size, d_l)
        z_bool = z.bool()
        d_l = z_bool.shape[1]
        
        # Vectorized hash computation: for each sample, compute a hash from the bit pattern
        # We'll use a simple approach: sum of (bit_value * position_weight)
        # where position_weight increases with position to make order matter
        z_int = z_bool.int()  # (batch_size, d_l)
        
        # Create position weights: use a simple weighted sum to avoid overflow
        # For each position, use a weight that grows but stays manageable
        # Use modulo arithmetic to prevent overflow
        pos_weights = torch.arange(1, d_l + 1, device=device, dtype=torch.int64)
        # Scale weights to prevent overflow, but keep them distinct
        pos_weights = pos_weights % (2**31)  # Keep in safe range
        
        # Compute layer hash per sample: (batch_size, d_l) @ (d_l,) -> (batch_size,)
        layer_hash = (z_int * pos_weights.unsqueeze(0)).sum(dim=1)  # (batch_size,)
        # Apply modulo to prevent overflow before polynomial step
        layer_hash = layer_hash % (2**31)
        
        # Polynomial hash: hash = hash * 31 + layer_hash
        hashes = (hashes * 31 + layer_hash) % (2**63 - 1)
    
    return hashes


@torch.no_grad()
def compute_local_derivatives(
    u_list: List[torch.Tensor],
    z_list: List[torch.Tensor],
    layer_slopes: List[Tuple[torch.Tensor, torch.Tensor]]
) -> List[torch.Tensor]:
    """
    Compute local derivatives (gate slopes actually used) for each layer.
    
    m^l_i(x) = a^l_{+,i} if u^l_i(x) >= 0, else a^l_{-,i}
    
    All computations done on CPU for fastest performance.
    
    Args:
        u_list: List of preactivation tensors, each (batch_size, d_l)
        z_list: List of sign masks, each (batch_size, d_l) with values in {0, 1}
        layer_slopes: List of (a_plus, a_minus) tuples, each (d_l,)
    
    Returns:
        List of tensors, each (batch_size, d_l) with local derivatives (on CPU)
    """
    # Ensure all tensors are on CPU
    u_list = [u.cpu() if u.device.type != 'cpu' else u for u in u_list]
    z_list = [z.cpu() if z.device.type != 'cpu' else z for z in z_list]
    layer_slopes = [(a_p.cpu() if a_p.device.type != 'cpu' else a_p,
                     a_m.cpu() if a_m.device.type != 'cpu' else a_m)
                    for a_p, a_m in layer_slopes]
    
    m_list = []
    for u, z, (a_plus, a_minus) in zip(u_list, z_list, layer_slopes):
        # z is 1 when u >= 0, 0 when u < 0
        # m = z * a_plus + (1 - z) * a_minus
        m = z * a_plus.unsqueeze(0) + (1 - z) * a_minus.unsqueeze(0)
        m_list.append(m)
    return m_list


@torch.no_grad()
def compute_path_gain(
    z_list: List[torch.Tensor],
    m_list: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute path-gain for each sample along its active path.
    
    G(x) = ∏_{l=1}^L (1/d_l) Σ_{i=1}^{d_l} m^l_i(x) * 1{z^l_i(x)=1}
    
    All computations done on CPU for fastest performance.
    
    Args:
        z_list: List of sign masks, each (batch_size, d_l)
        m_list: List of local derivatives, each (batch_size, d_l)
    
    Returns:
        Tensor of shape (batch_size,) with path gains (on CPU)
    """
    if len(z_list) == 0:
        return torch.tensor([], dtype=torch.float32)
    
    # Ensure all tensors are on CPU
    z_list = [z.cpu() if z.device.type != 'cpu' else z for z in z_list]
    m_list = [m.cpu() if m.device.type != 'cpu' else m for m in m_list]
    
    batch_size = z_list[0].shape[0]
    device = torch.device('cpu')  # Always use CPU
    gains = torch.ones(batch_size, device=device, dtype=torch.float32)
    
    for z, m in zip(z_list, m_list):
        # z is already in {0, 1}, so z * m gives active gate slopes
        active_slopes = z * m  # (batch_size, d_l)
        d_l = z.shape[1]
        # Mean over active units at layer l
        mean_active = active_slopes.sum(dim=1) / d_l  # (batch_size,)
        gains = gains * mean_active
    
    return gains


@torch.no_grad()
def compute_path_entropy(
    z_list: List[torch.Tensor],
    eps: float = 1e-12
) -> float:
    """
    Compute Path Support Entropy: H_path = -Σ_γ p(γ) log p(γ)
    
    Groups samples by their gating signature (path hash) and computes
    Shannon entropy over path probabilities.
    
    Args:
        z_list: List of sign masks, each (batch_size, d_l)
        eps: Small epsilon for numerical stability
    
    Returns:
        H_path: scalar entropy value
    """
    if len(z_list) == 0:
        return 0.0
    
    # Hash all samples to get path signatures
    path_hashes = hash_mask_list(z_list)  # (batch_size,)
    
    # Count path frequencies
    hashes_cpu = path_hashes.cpu().numpy()
    counter = Counter(hashes_cpu)
    total = len(hashes_cpu)
    
    if total == 0:
        return 0.0
    
    # Compute probabilities and entropy
    H = 0.0
    for count in counter.values():
        p = count / total
        if p > eps:
            H -= p * np.log(p + eps)
    
    return float(H)


@torch.no_grad()
def compute_path_gain_entropy(
    z_list: List[torch.Tensor],
    m_list: List[torch.Tensor],
    eps: float = 1e-12
) -> float:
    """
    Compute Path Gain Concentration: H_gain = -Σ_γ w(γ) log w(γ)
    
    where w(γ) ∝ p(γ) * Ḡ(γ) and Ḡ(γ) = E[G(x) | Z(x) = γ]
    
    Args:
        z_list: List of sign masks, each (batch_size, d_l)
        m_list: List of local derivatives, each (batch_size, d_l)
        eps: Small epsilon for numerical stability
    
    Returns:
        H_gain: scalar entropy value
    """
    if len(z_list) == 0:
        return 0.0
    
    # Compute path gains for all samples
    gains = compute_path_gain(z_list, m_list)  # (batch_size,)
    
    # Hash all samples to get path signatures
    path_hashes = hash_mask_list(z_list)  # (batch_size,)
    
    # Group by path and compute average gain per path
    hashes_cpu = path_hashes.cpu().numpy()
    gains_cpu = gains.cpu().numpy()
    
    # Accumulate: path_hash -> (count, sum_gain)
    path_stats = {}
    for h, g in zip(hashes_cpu, gains_cpu):
        if h not in path_stats:
            path_stats[h] = [0, 0.0]
        path_stats[h][0] += 1
        path_stats[h][1] += g
    
    # Compute weights: w(γ) = p(γ) * Ḡ(γ)
    total = len(hashes_cpu)
    weights = {}
    for h, (count, sum_gain) in path_stats.items():
        p_gamma = count / total
        G_bar = sum_gain / count if count > 0 else 0.0
        weights[h] = p_gamma * G_bar
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight <= eps:
        return 0.0
    
    # Compute entropy
    H = 0.0
    for w in weights.values():
        w_norm = w / total_weight
        if w_norm > eps:
            H -= w_norm * np.log(w_norm + eps)
    
    return float(H)


@torch.no_grad()
def compute_path_mi(
    z_list: List[torch.Tensor],
    group_ids: torch.Tensor | np.ndarray,
    up_to_layer: Optional[int] = None,
    eps: float = 1e-12
) -> List[float]:
    """
    Compute Path-to-group Mutual Information: I_l = I(Π_l(x), g(x))
    
    where Π_l(x) is the hash of z^1(x), ..., z^l(x) (partial path up to layer l)
    and g(x) is the true latent group.
    
    Args:
        z_list: List of sign masks, each (batch_size, d_l)
        group_ids: Tensor or array of shape (batch_size,) with group IDs
        up_to_layer: If None, compute for all layers. Otherwise, only up to this layer.
        eps: Small epsilon for numerical stability (add-1 smoothing)
    
    Returns:
        List of I_l values, one per layer
    """
    if len(z_list) == 0:
        return []
    
    # Convert group_ids to tensor if needed and move to CPU for consistency
    if isinstance(group_ids, np.ndarray):
        group_ids = torch.from_numpy(group_ids)
    # Move z_list and group_ids to CPU for path metric computation
    z_list = [z.cpu() if z.device.type != 'cpu' else z for z in z_list]
    group_ids = group_ids.cpu() if group_ids.device.type != 'cpu' else group_ids
    group_ids = group_ids.long()
    
    batch_size = z_list[0].shape[0]
    L = len(z_list)
    if up_to_layer is not None:
        L = min(L, up_to_layer)
    
    I_layers = []
    
    for l in range(L):
        # Compute partial path hash up to layer l
        partial_z_list = z_list[:l+1]
        partial_hashes = hash_mask_list(partial_z_list)  # (batch_size,)
        
        # Compute empirical joint and marginal distributions
        hashes_cpu = partial_hashes.cpu().numpy()
        groups_cpu = group_ids.cpu().numpy()
        
        # Count joint: (path_hash, group_id) -> count
        joint_counter = Counter(zip(hashes_cpu, groups_cpu))
        path_counter = Counter(hashes_cpu)
        group_counter = Counter(groups_cpu)
        
        total = len(hashes_cpu)
        if total == 0:
            I_layers.append(0.0)
            continue
        
        # Add-1 smoothing
        n_paths = len(path_counter)
        n_groups = len(group_counter)
        
        # Compute MI: I(X;Y) = Σ_{x,y} p(x,y) log(p(x,y) / (p(x) * p(y)))
        I = 0.0
        for (h, g), count_xy in joint_counter.items():
            count_x = path_counter[h]
            count_y = group_counter[g]
            
            # Add-1 smoothing
            p_xy = (count_xy + 1) / (total + n_paths * n_groups)
            p_x = (count_x + 1) / (total + n_paths)
            p_y = (count_y + 1) / (total + n_groups)
            
            if p_xy > eps and p_x > eps and p_y > eps:
                I += p_xy * np.log((p_xy + eps) / ((p_x * p_y) + eps))
        
        I_layers.append(float(I))
    
    return I_layers


@torch.no_grad()
def compute_confident_churn(
    prev_hashes: Optional[List[torch.Tensor]],
    cur_hashes: List[torch.Tensor],
    margins: torch.Tensor,
    tau: float = 0.0
) -> List[float]:
    """
    Compute Confident Churn: ρ^conf_l = Pr(Π_l^(t)(x) ≠ Π_l^(t-1)(x) | |f(x)| ≥ τ)
    
    Fraction of path hashes that changed since last evaluation, restricted to
    samples with |margin| >= τ (high confidence).
    
    Args:
        prev_hashes: List of path hashes from previous evaluation, each (batch_size,)
                    or None if first evaluation
        cur_hashes: List of path hashes from current evaluation, each (batch_size,)
        margins: Tensor of shape (batch_size,) with |logit| or |f(x)| values
        tau: Confidence threshold (e.g., use median or top 50% by default)
    
    Returns:
        List of ρ^conf_l values, one per layer
    """
    if prev_hashes is None:
        return [0.0] * len(cur_hashes)
    
    if len(prev_hashes) != len(cur_hashes):
        return [0.0] * len(cur_hashes)
    
    # Move all tensors to CPU for consistent device handling
    cur_hashes = [h.cpu() if h.device.type != 'cpu' else h for h in cur_hashes]
    prev_hashes = [h.cpu() if h.device.type != 'cpu' else h for h in prev_hashes]
    margins = margins.cpu() if margins.device.type != 'cpu' else margins
    
    # If tau is 0 or negative, use median as threshold
    if tau <= 0:
        tau = float(torch.median(margins.abs()).item())
    
    # Identify confident samples
    confident_mask = margins.abs() >= tau  # (batch_size,)
    
    if confident_mask.sum() == 0:
        return [0.0] * len(cur_hashes)
    
    rho_layers = []
    for prev_h, cur_h in zip(prev_hashes, cur_hashes):
        # Ensure same device
        if prev_h.device != cur_h.device:
            prev_h = prev_h.to(cur_h.device)
        if confident_mask.device != cur_h.device:
            confident_mask = confident_mask.to(cur_h.device)
        
        # Check which confident samples changed path
        changed = (prev_h != cur_h) & confident_mask  # (batch_size,)
        rho = changed.float().sum() / confident_mask.float().sum()
        rho_layers.append(float(rho.item()))
    
    return rho_layers


@torch.no_grad()
def compute_path_snr(
    z_list: List[torch.Tensor],
    y: torch.Tensor,
    eps: float = 1e-8
) -> Dict[int, Tuple[float, int, float]]:
    """
    Compute Path-wise SNR: SNR(γ) = c(γ) / sqrt((N̂(γ) + ε)^(-1))
    
    where c(γ) = |E[y | Z = γ]| is the label correlation,
    and N̂(γ) is the number of samples that visited path γ.
    
    Args:
        z_list: List of sign masks, each (batch_size, d_l)
        y: Tensor of shape (batch_size,) with labels
        eps: Small epsilon for numerical stability
    
    Returns:
        Dictionary mapping path_hash -> (c_gamma, N_gamma, SNR_gamma)
    """
    if len(z_list) == 0:
        return {}
    
    # Hash all samples to get path signatures
    path_hashes = hash_mask_list(z_list)  # (batch_size,)
    
    # Convert to CPU for processing
    hashes_cpu = path_hashes.cpu().numpy()
    y_cpu = y.cpu().numpy()
    if y_cpu.ndim > 1:
        y_cpu = y_cpu.squeeze()
    
    # Group by path and compute statistics
    path_stats = {}  # path_hash -> (sum_y, count)
    for h, y_val in zip(hashes_cpu, y_cpu):
        if h not in path_stats:
            path_stats[h] = [0.0, 0]
        path_stats[h][0] += y_val
        path_stats[h][1] += 1
    
    # Compute SNR for each path
    snr_dict = {}
    for h, (sum_y, count) in path_stats.items():
        N_gamma = count
        c_gamma = abs(sum_y / N_gamma) if N_gamma > 0 else 0.0
        # SNR(γ) = c(γ) / sqrt((N̂(γ) + ε)^(-1)) = c(γ) * sqrt(N̂(γ) + ε)
        snr_gamma = c_gamma * np.sqrt(N_gamma + eps)
        snr_dict[int(h)] = (float(c_gamma), int(N_gamma), float(snr_gamma))
    
    return snr_dict


@torch.no_grad()
def compute_neural_race_index(
    H_gain: float,
    I_layers: List[float],
    rho_conf_layers: List[float],
    K: int,  # number of distinct paths observed
    m_layers: List[int],  # number of groups at each layer (for normalization)
    eps: float = 1e-12
) -> float:
    """
    Compute Neural-Race Index (NRI): composite scalar metric.
    
    NRI(t) = (1 - H_gain(t)/log K(t)) * (1/L Σ_l I_l(t)/log m^l) * (1/(1 + ρ̄^conf(t)))
    
    Args:
        H_gain: Path gain concentration entropy
        I_layers: List of path-to-group mutual information per layer
        rho_conf_layers: List of confident churn per layer
        K: Number of distinct paths observed
        m_layers: List of number of groups at each layer (for normalization)
        eps: Small epsilon for numerical stability
    
    Returns:
        NRI: scalar value between 0 and 1 (approximately)
    """
    L = len(I_layers)
    if L == 0:
        return 0.0
    
    # Concentration term: 1 - H_gain / log K
    if K > 1:
        log_K = np.log(K + eps)
        concentration = 1.0 - (H_gain / log_K) if log_K > eps else 0.0
    else:
        concentration = 0.0
    
    # Alignment term: (1/L) Σ_l I_l / log m^l
    alignment_sum = 0.0
    for l, I_l in enumerate(I_layers):
        if l < len(m_layers) and m_layers[l] > 1:
            log_m = np.log(m_layers[l] + eps)
            alignment_sum += I_l / log_m if log_m > eps else 0.0
    alignment = alignment_sum / L if L > 0 else 0.0
    
    # Stability term: 1 / (1 + ρ̄^conf)
    rho_bar = np.mean(rho_conf_layers) if len(rho_conf_layers) > 0 else 0.0
    stability = 1.0 / (1.0 + rho_bar + eps)
    
    # Combine
    NRI = concentration * alignment * stability
    return float(NRI)

