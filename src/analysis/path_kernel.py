# src/analysis/path_kernel.py

from __future__ import annotations

import math
import os
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np

# ----------------------------
# Collect per-sample factors Φ
# ----------------------------
@torch.no_grad()
def collect_path_factors(
    model,
    loader,
    device: Optional[str] = None,
    *,
    mode: str = "routing_gain",   # "routing" | "routing_gain" | "routing_posdev"
    include_input: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """
    Build factor matrices for the path kernel:
      K = (X X^T) ∘ ∏_ℓ (E_ℓ E_ℓ^T),
    where E_ℓ is a sample-by-width matrix of 'transmittances' per layer.
    mode:
      - "routing":        E_ℓ = 1(u_ℓ > 0)                             (binary)
      - "routing_gain":   E_ℓ = 1(u>0)*a_plus + 1(u<=0)*a_minus         (≥ 0)
      - "routing_posdev": E_ℓ = 1(u>0)*max(a_plus-1,0)                  (≥ 0)
    include_input:
      Include (X X^T) as a factor; if False, kernel reflects routing/gain only.
    Returns:
      {
        "X":       Tensor [P, d_in] or None,
        "E_list":  List[Tensor [P, d_l]],
        "y":       Tensor [P,] (if available via loader),
        "labels":  LongTensor [P,] (if y looks integer-like),
        "meta":    {"depth": L, "widths": [...], "P": P, "d_in": d_in}
      }
    """
    model.eval()
    dev = device or next(iter(model.parameters())).device
    X_rows: List[torch.Tensor] = []
    E_accum: List[List[torch.Tensor]] = []
    y_rows: List[torch.Tensor] = []
    label_rows: List[torch.Tensor] = []
    
    # Process all batches from the loader, ensuring consistent sample counts
    seen = 0
    L = None  # Will be set from first batch
    widths = None
    for xb, yb in loader:
        if max_samples is not None and seen >= max_samples:
            break
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        bsz_original = xb.shape[0]
        
        # Apply max_samples limit BEFORE forward pass to ensure consistency
        if max_samples is not None and (seen + bsz_original > max_samples):
            take = max_samples - seen
            xb = xb[:take]
            yb = yb[:take]
            bsz = take
        else:
            bsz = bsz_original
        
        # Now do forward pass with the (possibly sliced) batch
        _, cache = model(xb, return_cache=True)
        zs = cache["z"]  # list of (B, d_l) in {0,1}
        
        # Initialize L and widths from first batch
        if L is None:
            L = len(zs)
            widths = [z.shape[1] for z in zs]
            for _ in range(L): E_accum.append([])
        
        # Verify zs has the correct batch size
        if len(zs) != L:
            raise ValueError(
                f"Layer count mismatch: expected {L} layers but got {len(zs)}. "
                f"This suggests the model structure changed between batches."
            )
        if zs[0].shape[0] != bsz:
            raise ValueError(
                f"Batch size mismatch: zs[0] has {zs[0].shape[0]} samples but expected {bsz}. "
                f"xb.shape[0]={xb.shape[0]}, seen={seen}, max_samples={max_samples}"
            )
        # slopes per layer
        if hasattr(model, "gates") and model.gates is not None:
            ap_am = [(g.a_plus.detach().to(dev), g.a_minus.detach().to(dev)) for g in model.gates]
        else:
            ap_am = [(None, None) for _ in range(L)]
        for l in range(L):
            z = zs[l].to(dev).float()  # (B, d_l)
            a_plus, a_minus = ap_am[l]
            if mode == "routing":
                E = z
            elif mode == "routing_posdev":
                if a_plus is None:
                    E = z
                else:
                    ap = (a_plus - 1.0).clamp_min(0.0)  # (d_l,)
                    E = z * ap.unsqueeze(0)
            else:  # "routing_gain"
                if a_plus is None:
                    E = z
                else:
                    E = z * a_plus.unsqueeze(0) + (1.0 - z) * a_minus.unsqueeze(0)
            E_accum[l].append(E.detach())
        if include_input:
            X_rows.append(xb.detach())
        y_rows.append(yb.detach().view(-1))
        # try to infer integer 'labels' from y
        try:
            y_int = yb.detach().view(-1).to("cpu")
            if torch.allclose(y_int.round(), y_int):
                label_rows.append(y_int.long())
        except Exception:
            pass
        seen += bsz
    
    # Before concatenating, ensure all collected pieces are internally consistent
    # Different batch sizes across batches are OK (expected when max_samples truncates last batch)
    # But within each batch, X, y, and E must all have the same size
    if len(X_rows) > 0:
        # Verify each batch is internally consistent
        for i in range(len(X_rows)):
            batch_size = X_rows[i].shape[0]
            # Check y matches
            if i < len(y_rows) and y_rows[i].shape[0] != batch_size:
                # Truncate to match
                min_size = min(batch_size, y_rows[i].shape[0])
                X_rows[i] = X_rows[i][:min_size]
                y_rows[i] = y_rows[i][:min_size]
                if i < len(label_rows) and label_rows[i].shape[0] > min_size:
                    label_rows[i] = label_rows[i][:min_size]
            # Check E_accum matches for all layers
            for l in range(L):
                if i < len(E_accum[l]) and E_accum[l][i].shape[0] != batch_size:
                    # Truncate to match
                    E_accum[l][i] = E_accum[l][i][:batch_size]
    
    # Verify we have the same number of batches for all components
    if len(X_rows) > 0:
        n_batches = len(X_rows)
        if len(y_rows) != n_batches:
            raise ValueError(f"y_rows has {len(y_rows)} batches but X_rows has {n_batches}")
        for l in range(L):
            if len(E_accum[l]) != n_batches:
                raise ValueError(f"E_accum[{l}] has {len(E_accum[l])} batches but X_rows has {n_batches}")
    
    # Concatenate all collected tensors (now all have consistent sizes)
    E_list = [torch.cat(E_accum[l], dim=0).contiguous() for l in range(L)]
    X = torch.cat(X_rows, dim=0).contiguous() if (include_input and X_rows) else None
    y = torch.cat(y_rows, dim=0).contiguous() if y_rows else None
    labels = torch.cat(label_rows, dim=0) if len(label_rows) == len(y_rows) else None
    
    # CRITICAL: Verify all factors have the same number of samples
    P_values = []
    factor_info = []
    if X is not None:
        P_values.append(X.shape[0])
        factor_info.append(f"X: {X.shape}")
    for l, E in enumerate(E_list):
        P_values.append(E.shape[0])
        factor_info.append(f"E_list[{l}]: {E.shape}")
    
    if len(P_values) > 0:
        P = P_values[0]
        for i, p_val in enumerate(P_values):
            if p_val != P:
                error_msg = (
                    f"Inconsistent sample counts in collected factors:\n"
                    f"  Expected: {P} samples\n"
                    f"  Actual: {P_values}\n"
                    f"  Factor shapes:\n    " + "\n    ".join(factor_info) + "\n"
                    f"  seen={seen}, max_samples={max_samples}, include_input={include_input}\n"
                    f"This usually means max_samples was applied inconsistently or "
                    f"batches were processed incorrectly."
                )
                raise ValueError(error_msg)
    else:
        P = 0
        raise ValueError("No factors collected - check that loader is not empty and model has layers")
    
    meta = {"depth": L, "widths": widths, "P": int(P), "d_in": int(X.shape[1]) if X is not None else None}
    return {"X": X, "E_list": E_list, "y": y, "labels": labels, "meta": meta}

# --------------------------------------------
# Implicit operator for K = ∘_f (F_f F_f^T)
# --------------------------------------------
class HadamardGramOperator:
    """
    y = K v, with K = ∘_f (F_f F_f^T), factors F_f ∈ R^{P×d_f}.
    Blocked implementation:
      For rows I (size B):
        A = ones(B,P)
        For each factor F:
           G_block = F[I,:] @ F^T        (B,P)
           A *= G_block
        y_I = A @ v
    """
    def __init__(
        self,
        factors: List[torch.Tensor],
        *,
        device: str,
        dtype: torch.dtype = torch.float32,
        block_size: int = 1024,
        use_tf32: bool = True,
    ):
        assert len(factors) >= 1, "At least one factor required."
        P = factors[0].shape[0]
        
        # Detailed validation with informative error messages
        factor_shapes = [F.shape for F in factors]
        for f_idx, F in enumerate(factors):
            if F.shape[0] != P:
                raise ValueError(
                    f"HadamardGramOperator: Factor {f_idx} has {F.shape[0]} samples but expected {P}. "
                    f"All factor shapes: {factor_shapes}. "
                    f"This means factors were collected with inconsistent sample counts. "
                    f"Check that max_samples is applied consistently and loader returns consistent batches."
                )
        
        self.factors = [F.to(device) for F in factors]
        self.P = P
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.prev_tf32 = None
        if torch.cuda.is_available():
            self.prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)

    def __del__(self):
        if (self.prev_tf32 is not None) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.prev_tf32

    @torch.no_grad()
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(self.device, dtype=self.dtype).view(-1)
        if v.shape[0] != self.P:
            raise ValueError(f"Vector v has {v.shape[0]} elements but expected {self.P}")
        
        # Verify all factors still have consistent sample counts
        P = self.P
        for f_idx, F in enumerate(self.factors):
            if F.shape[0] != P:
                raise ValueError(
                    f"Factor {f_idx} has inconsistent sample count at matvec time: "
                    f"expected {P} but got {F.shape[0]}. Factor shape: {F.shape}. "
                    f"This suggests factors were modified or collected incorrectly."
                )
        
        B = self.block_size
        y = torch.zeros(P, device=self.device, dtype=self.dtype)
        
        # Create transposed factors and verify shapes
        FTs = []
        for f_idx, F in enumerate(self.factors):
            if F.shape[0] != P:
                raise ValueError(
                    f"Factor {f_idx} has {F.shape[0]} rows but expected {P} before creating transpose. "
                    f"Factor shape: {F.shape}, P={P}"
                )
            FT = F.to(self.dtype).T.contiguous()
            # FT should be (d_f, P) where d_f is the feature dimension
            if FT.shape[1] != P:
                raise ValueError(
                    f"Factor {f_idx} transpose has {FT.shape[1]} columns but expected {P}. "
                    f"F shape: {F.shape}, FT shape: {FT.shape}, P={P}"
                )
            FTs.append(FT)
        for i0 in range(0, P, B):
            i1 = min(P, i0 + B)
            rows = slice(i0, i1)
            block_size_actual = i1 - i0
            A = torch.ones((block_size_actual, P), device=self.device, dtype=self.dtype)
            for f_idx, (f, FT) in enumerate(zip(self.factors, FTs)):
                # Ensure f has the correct number of samples
                if f.shape[0] != P:
                    raise ValueError(
                        f"Factor {f_idx} has {f.shape[0]} samples but expected {P}. "
                        f"Factor shape: {f.shape}, P={P}"
                    )
                # Slice the factor for this block - ensure we don't go out of bounds
                if i1 > f.shape[0]:
                    raise ValueError(
                        f"Factor {f_idx} has only {f.shape[0]} rows but trying to access up to {i1}. "
                        f"Factor shape: {f.shape}, P={P}, i0={i0}, i1={i1}"
                    )
                f_block = f[rows, :].to(self.dtype)  # (block_size_actual, d_f)
                
                # Verify f_block has the expected number of rows
                if f_block.shape[0] != block_size_actual:
                    raise ValueError(
                        f"Factor {f_idx} block has {f_block.shape[0]} rows but expected {block_size_actual}. "
                        f"f shape: {f.shape}, rows: {rows}, block_size_actual={block_size_actual}"
                    )
                
                # FT should be (d_f, P) - verify
                if FT.shape[1] != P:
                    raise ValueError(
                        f"Factor {f_idx} transpose has {FT.shape[1]} columns but expected {P}. "
                        f"FT shape: {FT.shape}, P={P}, f shape: {f.shape}"
                    )
                
                # Compute Gram block: (block_size_actual, d_f) @ (d_f, P) = (block_size_actual, P)
                try:
                    G_block = f_block @ FT
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Matrix multiplication failed for factor {f_idx}: "
                        f"f_block shape: {f_block.shape}, FT shape: {FT.shape}. "
                        f"f shape: {f.shape}, rows: {rows}, block_size_actual={block_size_actual}, P={P}. "
                        f"Original error: {e}"
                    ) from e
                
                # Verify shape matches before in-place multiplication
                if G_block.shape != A.shape:
                    raise ValueError(
                        f"Shape mismatch in factor {f_idx}: G_block {G_block.shape} vs A {A.shape}. "
                        f"f shape: {f.shape}, f_block shape: {f_block.shape}, FT shape: {FT.shape}, "
                        f"rows: {rows} (i0={i0}, i1={i1}), block_size_actual={block_size_actual}, P={P}"
                    )
                
                # Perform in-place multiplication with error handling
                try:
                    A.mul_(G_block)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"In-place multiplication failed for factor {f_idx}: "
                        f"A shape: {A.shape}, G_block shape: {G_block.shape}. "
                        f"f shape: {f.shape}, f_block shape: {f_block.shape}, FT shape: {FT.shape}, "
                        f"rows: {rows} (i0={i0}, i1={i1}), block_size_actual={block_size_actual}, P={P}. "
                        f"Original error: {e}"
                    ) from e
                del G_block, f_block
            y[rows] = A @ v
            del A
        return y

    @torch.no_grad()
    def mm(self, V: torch.Tensor) -> torch.Tensor:
        outs = [self.matvec(V[:, j]) for j in range(V.shape[1])]
        return torch.stack(outs, dim=1)

# -----------------------------------
# Block power method for top spectrum
# -----------------------------------
@torch.no_grad()
def top_eigenpairs_block_power(
    op: HadamardGramOperator,
    k: int = 32,
    n_iter: int = 30,
    tol: float = 1e-6,
    seed: int = 123,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (eigvals: (k,), eigvecs: (P,k)) in descending order.
    """
    torch.manual_seed(seed)
    P = op.P
    Q = torch.randn(P, k, device=op.device, dtype=op.dtype)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    last_vals = None
    for it in range(n_iter):
        Z = op.mm(Q)
        Q, _ = torch.linalg.qr(Z, mode="reduced")
        T = Q.T @ op.mm(Q)
        evals, V = torch.linalg.eigh(T)
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        V = V[:, idx]
        Q = Q @ V
        if verbose and (it % 5 == 0 or it == n_iter - 1):
            print(f"[path-kernel] iter={it:02d}  top λ≈ {evals[0].item():.6e}")
        if last_vals is not None and (evals - last_vals).abs().max().item() < tol:
            break
        last_vals = evals.clone()
    return evals, Q

# ---------------------------
# Top spectrum: global & per-class
# ---------------------------
@torch.no_grad()
def compute_path_kernel_eigs(
    model,
    loader,
    device: Optional[str] = None,
    *,
    mode: str = "routing_gain",
    include_input: bool = True,
    k: int = 32,
    n_iter: int = 30,
    block_size: int = 1024,
    dtype: torch.dtype = torch.float32,
    use_tf32: bool = True,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Compute top-k eigenpairs of the global path kernel.
    Returns:
      {
        "evals": Tensor[k],
        "evecs": Tensor[P,k],
        "meta":  meta dict from collection
      }
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pack = collect_path_factors(
        model, loader, dev, mode=mode, include_input=include_input, max_samples=max_samples
    )
    factors = []
    if pack["X"] is not None: 
        factors.append(pack["X"])
    factors.extend(pack["E_list"])
    
    # Verify all factors have the same number of samples
    if len(factors) > 0:
        P_expected = factors[0].shape[0]
        factor_info = []
        for i, f in enumerate(factors):
            factor_info.append(f"Factor {i}: shape {f.shape}")
            if f.shape[0] != P_expected:
                raise ValueError(
                    f"Inconsistent sample counts in factors:\n" +
                    "\n".join(factor_info) + 
                    f"\nFactor {i} has {f.shape[0]} samples but expected {P_expected}. "
                    f"This usually means max_samples was applied inconsistently or "
                    f"the loader returned different batch sizes."
                )
        
        # Also verify against meta
        P_meta = pack["meta"]["P"]
        if P_expected != P_meta:
            raise ValueError(
                f"Sample count mismatch: factors have {P_expected} samples but meta says {P_meta}. "
                f"Factor info:\n" + "\n".join(factor_info)
            )
    
    op = HadamardGramOperator(
        factors, device=dev, dtype=dtype, block_size=min(block_size, P_expected) if len(factors) > 0 else block_size, 
        use_tf32=use_tf32
    )
    evals, evecs = top_eigenpairs_block_power(
        op, k=k, n_iter=n_iter, tol=1e-6, seed=123, verbose=verbose
    )
    # Keep on GPU for now - caller can move to CPU if needed
    # This avoids unnecessary CPU transfers during computation
    return {"evals": evals, "evecs": evecs, "meta": pack["meta"], "y": pack.get("y")}

@torch.no_grad()
def compute_classwise_path_kernel_eigs(
    model,
    loader,
    device: Optional[str] = None,
    *,
    mode: str = "routing_gain",
    include_input: bool = True,
    k: int = 16,
    n_iter: int = 25,
    block_size: int = 1024,
    dtype: torch.dtype = torch.float32,
    use_tf32: bool = True,
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> Dict[int, Dict[str, object]]:
    """
    For each integer label present in the dataset, compute top-k eigenpairs of the
    class-restricted path kernel. If integer labels are not available, we will
    infer them from y if it is {0,1} or {-1,+1}.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pack = collect_path_factors(
        model, loader, dev, mode=mode, include_input=include_input, max_samples=max_samples
    )
    X = pack["X"]
    E_list = pack["E_list"]
    y = pack["y"]
    labels = pack["labels"]
    if labels is None and y is not None:
        y_cpu = y.detach().to("cpu")
        # map {-1,+1} -> {0,1}
        if torch.allclose(y_cpu.round(), y_cpu):
            labels = ((y_cpu - y_cpu.min()) / max(y_cpu.max() - y_cpu.min(), 1)).long()
    if labels is None:
        raise ValueError("Classwise kernel requires integer labels; none found/inferred.")
    labels = labels.to("cpu").numpy()
    classes = sorted(list(set(labels.tolist())))
    results: Dict[int, Dict[str, object]] = {}
    for c in classes:
        idx = np.where(labels == c)[0]
        if idx.size < 2:
            continue
        idx_t = torch.as_tensor(idx, device=dev, dtype=torch.long)
        factors = []
        if X is not None: factors.append(X.index_select(0, idx_t))
        for E in E_list:
            factors.append(E.index_select(0, idx_t))
        op = HadamardGramOperator(
            factors, device=dev, dtype=dtype, block_size=block_size, use_tf32=use_tf32
        )
        evals, evecs = top_eigenpairs_block_power(
            op, k=min(k, idx.size), n_iter=n_iter, tol=1e-6, seed=123, verbose=verbose
        )
        results[int(c)] = {
            "evals": evals.to("cpu"),
            "evecs": evecs.to("cpu"),
            "P": int(idx.size),
        }
    return results

# ---------------------------
# Convenience: save utilities
# ---------------------------
def save_spectrum(out_dir: str, tag: str, evals: torch.Tensor):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"eigs_{tag}.npy")
    np.save(path, evals.detach().cpu().numpy())
    print(f"[path-kernel] saved eigenvalues -> {path}")

