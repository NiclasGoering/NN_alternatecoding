from __future__ import annotations
import torch
from .pruning import prune_identity_like
from ..utils.lsq import stable_lstsq
# Removed: from ..utils.metrics import compute_path_metrics

# ---------------------- utilities ----------------------

@torch.no_grad()
def _forward_cache(model, X):
    """Runs forward pass and returns all intermediate activations."""
    yhat, cache = model(X, return_cache=True)
    return yhat, cache["u"], cache["z"], cache["h"], cache["h_last"]

@torch.no_grad()
def _batch_mse(model, X, y):
    yhat = model(X)
    return torch.mean((yhat - y)**2).item()

@torch.no_grad()
def _compute_backprop_rows(model, us, zs, hs):
    """
    Computes the 'B' matrix: the sensitivity of the Loss w.r.t the *output* (h) of each layer.
    B_l approx dL/dh_l (size P x d_l)
    """
    P = hs[0].shape[0]
    L = len(hs)
    
    # Readout weights (d_L, 1) -> Transpose for backprop -> (1, d_L)
    # But we want B to be (P, d_L).
    w_out = model.readout.weight.detach() # (1, d_L)
    B = w_out.expand(P, -1) # (P, d_L)
    
    B_list = [None]*L
    B_list[-1] = B.clone()
    
    # Backprop from L-1 down to 0
    for l in reversed(range(L-1)):
        # Next layer params
        # Linear maps d_l -> d_{l+1}. Weight shape is (d_{l+1}, d_l)
        W_next = model.linears[l+1].weight.detach() 
        
        # Gates at l+1
        a_p = model.gates[l+1].a_plus.detach()   # (d_{l+1},)
        a_m = model.gates[l+1].a_minus.detach()  # (d_{l+1},)
        u_next = us[l+1]                         # (P, d_{l+1})
        
        # Effective slope at l+1 based on sign of u_next
        # Elementwise mult: B is (P, d_{l+1}), s is (P, d_{l+1})
        s = torch.where(u_next >= 0, a_p.unsqueeze(0), a_m.unsqueeze(0))
        
        # Backprop through Activation (Hadamard)
        B_at_u = B * s 
        
        # Backprop through Linear: dL/dh_l = dL/du_{l+1} @ W_{l+1}
        # Shapes: (P, d_{l+1}) @ (d_{l+1}, d_l) -> (P, d_l)
        B = B_at_u @ W_next
        
        B_list[l] = B.clone()
        
    return B_list

@torch.no_grad()
def _W_readout_ls(model, H_last, y, ridge_out=1e-3, beta_out=1.0):
    """ Fits the final readout layer using Ridge Regression. Handles bias=None cases. """
    device = H_last.device
    P, dL = H_last.shape
    
    # 1. Center data for better conditioning
    # (This allows us to separate the intercept calculation)
    H_mean = H_last.mean(0, keepdim=True)
    H_centered = H_last - H_mean
    y_mean = y.mean()
    y_centered = y - y_mean

    H64 = H_centered.to(torch.float64)
    y64 = y_centered.to(torch.float64)
    
    # 2. Normal Equation: (H'H + lambda I) w = H'y
    K = H64.t() @ H64
    diag_idx = torch.arange(dL, device=device)
    K[diag_idx, diag_idx] += ridge_out * P 
    
    rhs = H64.t() @ y64
    
    try:
        # Solve for Weight (Slope)
        w_ls = torch.linalg.solve(K, rhs).t().to(torch.float32) # (1, dL)
        
        # Update Weight
        w_old = model.readout.weight.detach()
        # Use .data.copy_ to avoid leaf variable errors
        model.readout.weight.data.copy_((1.0 - beta_out) * w_old + beta_out * w_ls)
        
        # 3. Calculate and Update Bias (ONLY IF IT EXISTS)
        if model.readout.bias is not None:
            # The intercept b = y_mean - w @ H_mean
            b_ls = y_mean - (w_ls @ H_mean.t())
            
            b_old = model.readout.bias.detach()
            model.readout.bias.data.copy_((1.0 - beta_out) * b_old + beta_out * b_ls)
            
    except RuntimeError:
        print("Warning: Readout LS failed (singular), skipping update.")

# ---------------------- W-step (Sylvester / Kronecker) ----------------------

@torch.no_grad()
def _solve_weight_block_sylvester(H, G, y, ridge_w, eps=1e-6):
    """ 
    Approximate closed form for W in min || y - G W H^T ||^2 
    Uses normalization to prevent numerical explosion.
    """
    device = H.device
    dtype = torch.float64
    
    # 1. Column Normalization (Essential for deep nets)
    h_norm = torch.linalg.norm(H, dim=0, keepdim=True) + eps
    g_norm = torch.linalg.norm(G, dim=0, keepdim=True) + eps
    
    H_normed = (H / h_norm).to(dtype)
    G_normed = (G / g_norm).to(dtype)
    y64 = y.to(dtype) # (P,)

    # Covariances
    K_in = H_normed.t() @ H_normed
    K_out = G_normed.t() @ G_normed
    
    # Cross-term: G.T @ diag(y) @ H
    # Reshape y to (P,1)
    YH = (y64.unsqueeze(1) * H_normed) 
    C = G_normed.t() @ YH # (d_out, d_in)

    # Eigen decomps
    # Add jitter to diagonal for stability
    K_in.diagonal().add_(eps)
    K_out.diagonal().add_(eps)
    
    try:
        e_in, Ui = torch.linalg.eigh(K_in)
        e_out, Uo = torch.linalg.eigh(K_out)
        
        # Transform C into eigen-basis
        # C_tilde = Uo.T @ C @ Ui
        C_tilde = Uo.t() @ C @ Ui
        
        # Eigenvalues of Kronecker sum
        # denom_ij = lambda_out_i * lambda_in_j + ridge
        denom = (e_out.unsqueeze(1) * e_in.unsqueeze(0)) + ridge_w
        
        W_tilde = C_tilde / denom
        
        # Rotate back
        W_solved_normed = Uo @ W_tilde @ Ui.t()
        
        # Denormalize
        # W_actual = W_solved_normed / (g_scale * h_scale.T)
        W_solved = W_solved_normed / (g_norm.t() @ h_norm)
        
        return W_solved.to(H.dtype)
        
    except RuntimeError:
        # Fallback if Eigendecomposition fails
        return None

# ---------------------- A-step per-layer ----------------------

@torch.no_grad()
def _A_step_layer(model, l, X, y, lam_id, ridge_a, beta, slope_clamp):
    """
    Update slopes a_+, a_- for layer l.
    """
    yhat, us, zs, hs, _ = _forward_cache(model, X)
    B_list = _compute_backprop_rows(model, us, zs, hs)
    
    u = us[l]        # (P, d_l)
    B = B_list[l]    # (P, d_l) -- This is the "gradient" flowing back
    
    # Current slopes
    a_p_old = model.gates[l].a_plus.detach().clone()
    a_m_old = model.gates[l].a_minus.detach().clone()
    a_cur = torch.cat([a_p_old, a_m_old], dim=0)

    # Design Matrix for Slopes
    # The contribution of this layer to the output is roughly B * phi(u)
    # We want B * (a_+ * u_+ - a_- * u_-) approx Residual
    # This decouples per neuron h.
    # Phi_h = [B_h * u_+, -B_h * u_-] is a (P, 2) vector for neuron h?
    # No, B is (P, d_l). The problem separates over neurons h=1..d_l
    # BUT to do it efficiently in batch, we treat it as one large regression if d_l is small,
    # or d_l independent regressions.
    
    # Global Regression Formulation:
    # We linearize the rest of the network into B.
    # y - y_resid = \sum_h B_h * (a_h+ [u_h]+ - a_h- [-u_h]+)
    # This fits to the GLOBAL residual (y - yhat).
    
    # Residual target: The error we want to reduce
    resid = (y - yhat).squeeze(1) # (P,)
    
    # To prevent "exploding residual" problem, we dampen the target
    # We only try to fix a fraction of the residual
    resid = resid * 0.5 

    u_plus = torch.relu(u)
    u_minus = torch.relu(-u)
    
    # Construct Jacobian J: (P, 2*d_l)
    # J = [ B * u_plus,  -B * u_minus ]
    J_plus = B * u_plus
    J_minus = -(B * u_minus)
    Phi = torch.cat([J_plus, J_minus], dim=1)
    
    # Solve delta_a: (Phi^T Phi + lam) delta_a = Phi^T resid
    # Note: We solve for the CHANGE in a, fitting the residual
    
    # Normalize columns of Phi for stability
    phi_norm = torch.linalg.norm(Phi, dim=0) + 1e-6
    Phi_normed = Phi / phi_norm
    
    lhs = Phi_normed.t() @ Phi_normed
    lhs.diagonal().add_(ridge_a)
    if lam_id > 0:
        lhs.diagonal().add_(lam_id) # Identity pull acts like ridge here centered at 1? 
        # Actually implementation of ID pull in delta form is complex, treating as Ridge on delta is safe approx.

    rhs = Phi_normed.t() @ resid
    
    delta_a_normed = torch.linalg.solve(lhs, rhs)
    delta_a = delta_a_normed / phi_norm
    
    a_new = a_cur + delta_a
    
    # Soft update
    a_final = (1.0 - beta) * a_cur + beta * a_new
    
    # Clamp
    a_final = torch.clamp(a_final, slope_clamp[0], slope_clamp[1])
    
    d = u.shape[1]
    model.gates[l].a_plus.copy_(a_final[:d])
    model.gates[l].a_minus.copy_(a_final[d:])

# ---------------------- Main Loop ----------------------
def train_alt_em_closed_form(model, full_train_loader, val_loader, config, test_loader=None):
    device = config["device"]
    t = config["training"]
    r = config["regularization"]
    p = config.get("pruning", {})
    
    # Check for alt_em_closed_form specific config section, fall back to training section
    alt_cfg = config.get("alt_em_closed_form", {})
    
    cycles = int(alt_cfg.get("cycles", t.get("cycles", 50)))
    
    # Computation frequency controls (for speed optimization)
    logging_cfg = config.get("logging", {})
    path_kernel_metrics_freq = int(logging_cfg.get("path_kernel_metrics_every_n_cycles", 1))  # Compute path kernel metrics every cycle by default
    path_analysis_freq = path_kernel_metrics_freq  # Use same frequency as path kernel metrics
    path_analysis_out_dir = config.get("path_analysis_out_dir", None)  # Output directory for path analysis plots
    ridge_a = float(alt_cfg.get("ridge_a", t.get("ridge_a", 1.0)))     
    ridge_w = float(alt_cfg.get("ridge_w", t.get("ridge_w", 10.0)))    
    beta_a = float(alt_cfg.get("beta_a", t.get("beta_a", 0.1)))       
    beta_w = float(alt_cfg.get("beta_w", t.get("beta_w", 0.1)))
    ridge_out = float(alt_cfg.get("ridge_out", t.get("ridge_out", 1e-3)))
    improve_tol = float(alt_cfg.get("improve_tol", t.get("improve_tol", 1e-4)))
    max_backtracks = int(alt_cfg.get("max_backtracks", t.get("max_backtracks", 6)))
    lm_up = float(alt_cfg.get("lm_up", t.get("lm_up", 10.0)))
    lm_down = float(alt_cfg.get("lm_down", t.get("lm_down", 0.5)))
    beta_min = float(alt_cfg.get("beta_min", t.get("beta_min", 0.05)))
    target_churn = float(alt_cfg.get("target_churn", t.get("target_churn", 0.02)))
    
    # Churn control parameters for adaptive beta
    churn_decay = float(alt_cfg.get("churn_decay", t.get("churn_decay", 0.5)))    # Multiply beta by this if churn is too high
    churn_growth = float(alt_cfg.get("churn_growth", t.get("churn_growth", 1.01)))  # Grow beta painfully slowly
    
    # Identity regularization
    lam_id = float(r.get("lambda_identity", 0.0))
    slope_clamp = alt_cfg.get("slope_clamp", t.get("slope_clamp", [0.1, 3.0]))

    model.to(device).eval() 
    
    # Load full batch
    Xfull, yfull = next(iter(full_train_loader))
    Xfull, yfull = Xfull.to(device), yfull.to(device)

    history = []
    prev_masks = None
    # Checkpoint embeddings for lineage/centroid metrics
    checkpoint_embeddings = []  # List of (cycle, embedding_tensor) tuples
    checkpoint_metrics_history = []  # List of checkpoint metrics dicts

    import time as time_module
    cycle_start_time = time_module.time()
    print(f"[alt_em_closed_form] Starting training: {cycles} cycles, P={Xfull.shape[0]}")

    # FIX 1: Wrap the entire training process in no_grad to save memory and prevent graph errors
    with torch.no_grad():
        for cyc in range(1, cycles+1):
            if cyc % 50 == 0 or cyc == 1:
                elapsed = time_module.time() - cycle_start_time
                print(f"[alt_em_closed_form] Cycle {cyc}/{cycles} (elapsed: {elapsed:.1f}s, avg: {elapsed/cyc:.2f}s/cycle)")
            
            # --- 1. A-Step (Optimize Activations) ---
            if getattr(model, "use_gates", False):
                L = len(model.linears)
                for l in range(L):
                    # ... _A_step_layer INLINE FIX (Logic Correction) ...
                    # Re-implementing _A_step_layer logic here to ensure ID pull is correct
                    yhat, us, zs, hs, _ = _forward_cache(model, Xfull)
                    B_list = _compute_backprop_rows(model, us, zs, hs)
                    
                    u = us[l]
                    B = B_list[l]
                    a_p_old = model.gates[l].a_plus.detach().clone()
                    a_m_old = model.gates[l].a_minus.detach().clone()
                    a_cur = torch.cat([a_p_old, a_m_old], dim=0)

                    resid = (yfull - yhat).squeeze(1) * 0.5 # Dampened residual

                    u_plus = torch.relu(u)
                    u_minus = torch.relu(-u)
                    Phi = torch.cat([B * u_plus, -(B * u_minus)], dim=1)
                    
                    phi_norm = torch.linalg.norm(Phi, dim=0) + 1e-6
                    Phi_normed = Phi / phi_norm
                    
                    lhs = Phi_normed.t() @ Phi_normed
                    lhs.diagonal().add_(ridge_a)
                    
                    # LOGIC FIX: Identity Pull on LHS
                    if lam_id > 0:
                        lhs.diagonal().add_(lam_id)

                    rhs = Phi_normed.t() @ resid
                    
                    # LOGIC FIX: Identity Pull Force on RHS
                    if lam_id > 0:
                         # Pull direction: -lam * (current_val - 1.0)
                         # Scaled by phi_norm to match the normalized basis
                         rhs -= lam_id * (a_cur - 1.0) * phi_norm

                    delta_a_normed = torch.linalg.solve(lhs, rhs)
                    delta_a = delta_a_normed / phi_norm
                    
                    a_new = a_cur + delta_a
                    a_final = (1.0 - beta_a) * a_cur + beta_a * a_new
                    a_final = torch.clamp(a_final, slope_clamp[0], slope_clamp[1])
                    
                    d = u.shape[1]
                    model.gates[l].a_plus.copy_(a_final[:d])
                    model.gates[l].a_minus.copy_(a_final[d:])
            
            # --- 2. W-Step (Optimize Alignments) ---
            L = len(model.linears)
            for l in reversed(range(L)):
                yhat, us, zs, hs, _ = _forward_cache(model, Xfull)
                
                H = Xfull if l == 0 else hs[l-1]
                B_list = _compute_backprop_rows(model, us, zs, hs)
                
                a_p = model.gates[l].a_plus.detach()
                a_m = model.gates[l].a_minus.detach()
                mask_l = (us[l] >= 0)
                eff_slope = torch.where(mask_l, a_p.unsqueeze(0), a_m.unsqueeze(0))
                
                G = B_list[l] * eff_slope 
                
                W_new = _solve_weight_block_sylvester(H, G, yfull.squeeze(1), ridge_w=ridge_w)
                
                if W_new is not None:
                    W_old = model.linears[l].weight.detach()
                    W_update = (1.0 - beta_w) * W_old + beta_w * W_new
                    
                    # FIX 2: Use .data.copy_() to bypass leaf variable safety check
                    model.linears[l].weight.data.copy_(W_update)

            # --- 3. Readout Step ---
            _, _, _, hs, _ = _forward_cache(model, Xfull)
            _W_readout_ls(model, hs[-1], yfull, ridge_out=ridge_out, beta_out=0.5)

            # --- Logging & Metrics ---
            tr_loss = _batch_mse(model, Xfull, yfull)
            
            va_loss = 0.0
            if val_loader:
                va_acc, va_loss = _eval(model, val_loader, device)

            # Churn calculation
            cur_masks = dataset_masks(model, full_train_loader, device)
            churn_layers = mask_churn(prev_masks, cur_masks)
            max_churn = max(churn_layers) if churn_layers else 0.0
            prev_masks = cur_masks
            
            # --- ADAPTIVE LOGIC ---
            # If masks start flipping, stop immediately.
            if max_churn > target_churn:
                # Churn too high? The linear approx broke. Slow down!
                beta_w = max(beta_w * churn_decay, beta_min)
                beta_a = max(beta_a * churn_decay, beta_min)
            elif max_churn < target_churn * 0.5 and tr_loss > 0.01:
                # Churn too low? We are being too conservative. Speed up!
                beta_w = min(beta_w * churn_growth, 0.5)
                # Keep beta_a conservative usually
            
            # Pruning
            p_stats = []
            if p.get("enabled") and cyc >= p.get("apply_after_cycles", 5):
                 p_stats = prune_identity_like(model, tau=float(p.get("tau", 0.02)))

            # Compute path kernel metrics (effective rank, variance explained, etc.)
            path_kernel_metrics = {}
            if (cyc % path_kernel_metrics_freq == 0) or (cyc == 1) or (cyc == cycles):
                try:
                    from ..analysis.path_analysis import compute_path_kernel_metrics
                    path_kernel_metrics = compute_path_kernel_metrics(
                        model,
                        full_train_loader,
                        test_loader if test_loader is not None else val_loader,
                        mode="routing_gain",
                        k=48,
                        max_samples=5000,
                        device=device,
                    )
                except Exception as e:
                    print(f"  [path_kernel_metrics] Warning: Failed at cycle {cyc}: {e}")

            # Path metrics removed - no longer computing standard path metrics

            print(f"Cyc {cyc} | TrL: {tr_loss:.4f} | VaL: {va_loss:.4f} | Churn: {max_churn:.3f} | BetaW: {beta_w:.4f} | BetaA: {beta_a:.4f}")

            # Run path analysis at intervals (start, end, and every N cycles)
            checkpoint_metrics = {}
            if path_analysis_out_dir is not None and ((cyc % path_analysis_freq == 0) or (cyc == 1) or (cyc == cycles)):
                try:
                    from ..analysis.path_analysis import (
                        run_full_analysis_at_checkpoint,
                        path_embedding,
                        compute_path_shapley_metrics,
                        compute_centroid_drift_metrics,
                        compute_lineage_sankey_metrics,
                    )
                    from ..analysis.path_kernel import compute_path_kernel_eigs
                    
                    run_full_analysis_at_checkpoint(
                        model=model,
                        val_loader=val_loader,
                        out_dir=path_analysis_out_dir,
                        step_tag=f"cycle_{cyc:04d}",
                        kernel_k=48,
                        kernel_mode="routing_gain",
                        include_input_in_kernel=True,
                        block_size=1024,
                        max_samples_kernel=5000,  # Limit samples for speed
                        max_samples_embed=5000,
                    )
                    
                    # Collect checkpoint embeddings for lineage/centroid metrics
                    try:
                        Epack = path_embedding(
                            model, val_loader, device=device, mode="routing_gain",
                            normalize=True, max_samples=5000
                        )
                        E_tensor = Epack["E"]
                        # Safely get y_data - avoid boolean evaluation of tensors
                        y_data = Epack.get("labels")
                        if y_data is None:
                            y_data = Epack.get("y")
                        
                        checkpoint_embeddings.append((cyc, E_tensor.detach().cpu()))
                        
                        # Compute Path-Shapley if we have kernel eigenvectors
                        if y_data is not None and (not isinstance(y_data, torch.Tensor) or y_data.numel() > 0):
                            try:
                                kern = compute_path_kernel_eigs(
                                    model, val_loader, device=device, mode="routing_gain",
                                    include_input=True, k=24, n_iter=30, block_size=1024,
                                    max_samples=5000, verbose=False
                                )
                                if "evecs" in kern:
                                    evecs = kern["evecs"].detach().cpu().numpy()
                                    y_np = y_data.numpy() if isinstance(y_data, torch.Tensor) else y_data
                                    top_m = min(24, evecs.shape[1])
                                    scores = evecs[:, :top_m]
                                    shapley_metrics = compute_path_shapley_metrics(scores, y_np)
                                    checkpoint_metrics.update(shapley_metrics)
                            except Exception as e:
                                print(f"  [checkpoint_metrics] Path-Shapley failed: {e}")
                    except Exception as e:
                        print(f"  [checkpoint_metrics] Embedding collection failed: {e}")
                    
                    print(f"  [path_analysis] Completed for cycle {cyc}")
                except Exception as e:
                    print(f"  [path_analysis] Warning: Failed at cycle {cyc}: {e}")
            
            # Compute centroid drift and lineage metrics from collected embeddings
            if len(checkpoint_embeddings) >= 2:
                try:
                    from ..analysis.path_analysis import (
                        compute_centroid_drift_metrics,
                        compute_lineage_sankey_metrics,
                    )
                    E_time = [E for _, E in checkpoint_embeddings]
                    drift_metrics = compute_centroid_drift_metrics(E_time, k=8)
                    lineage_metrics = compute_lineage_sankey_metrics(E_time, k=8)
                    checkpoint_metrics.update(drift_metrics)
                    checkpoint_metrics.update(lineage_metrics)
                except Exception as e:
                    print(f"  [checkpoint_metrics] Drift/Lineage computation failed: {e}")
            
            if checkpoint_metrics:
                checkpoint_metrics_history.append({
                    "cycle": cyc,
                    **checkpoint_metrics
                })

            hist_entry = {
                "cycle": cyc,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "mask_churn": churn_layers,
                "max_churn": max_churn,
                "beta_w": beta_w,
                "beta_a": beta_a,
                "pruning": p_stats,
            }
            # Add path kernel metrics
            if path_kernel_metrics:
                hist_entry.update(path_kernel_metrics)
            # Add checkpoint metrics
            if checkpoint_metrics:
                hist_entry.update(checkpoint_metrics)
            history.append(hist_entry)

            if tr_loss < 1e-4:
                print("Converged.")
                break
    
    # Save checkpoint metrics history to JSON if available
    if checkpoint_metrics_history:
        try:
            from ..utils.save_io import save_json
            import os
            # Save to the same directory as training_history.csv
            # We'll need to get the output directory from config or return it
            # For now, we'll return it as part of history metadata
            pass  # Will be saved in run_experiment.py
        except Exception as e:
            print(f"  [checkpoint_metrics] Warning: Could not save checkpoint metrics: {e}")
    
    return history, checkpoint_metrics_history


# --- Helpers (kept same) ---

@torch.no_grad()
def dataset_masks(model, loader, device):
    model.eval()
    masks = None
    for xb, _ in loader:
        xb = xb.to(device)
        _, cache = model(xb, return_cache=True)
        batch_masks = [z.bool() for z in cache["z"]]  # Keep on GPU
        if masks is None:
            masks = [bm.clone() for bm in batch_masks]
        else:
            for l in range(len(masks)):
                masks[l] = torch.cat([masks[l], batch_masks[l]], dim=0)
        break # Just use one batch for churn approximation to save time
    # Move to CPU only at the end
    return [m.cpu() for m in masks] if masks is not None else None

def mask_churn(prev_masks, cur_masks):
    if prev_masks is None:
        return [0.0]*len(cur_masks)
    return [(prev_masks[l] != cur_masks[l]).float().mean().item() for l in range(len(cur_masks))]

@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    L=A=n=0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        L += torch.mean((yhat - yb)**2).item() * xb.size(0)
        A += torch.sign(yhat).eq(yb).float().mean().item() * xb.size(0)
        n += xb.size(0)
    return A/n, L/n