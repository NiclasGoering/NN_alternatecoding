from __future__ import annotations
import os
import torch
from torch import optim

# Local utility functions (no longer importing from utils.metrics)
def mse_loss(yhat, y):
    """Mean squared error loss."""
    return torch.mean((yhat - y) ** 2)

def accuracy_from_logits(yhat, y):
    """Compute accuracy from logits."""
    return torch.sign(yhat).eq(y).float().mean()

def effective_rank(act):
    """Compute effective rank of activations using SVD."""
    if act.numel() == 0:
        return 0.0
    act_2d = act.view(act.shape[0], -1)  # Flatten to (batch, features)
    if act_2d.shape[1] == 0:
        return 0.0
    try:
        U, s, _ = torch.linalg.svd(act_2d, full_matrices=False)
        s = s[s > 1e-8]  # Filter near-zero singular values
        if len(s) == 0:
            return 0.0
        p = s / s.sum()
        p = p[p > 1e-12]  # Avoid log(0)
        entropy = -(p * torch.log(p)).sum()
        return torch.exp(entropy).item()
    except:
        return 0.0

@torch.no_grad()
def dataset_masks(model, loader, device):
    """Collect all sign masks z_l over the whole dataset (list of [P, d_l] bool tensors)."""
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
        break  # OPTIMIZATION: Only use first batch for churn approximation to save time
    # Move to CPU only at the end
    return [m.cpu() for m in masks] if masks is not None else None

def train_sgd(model, train_loader, val_loader, config, test_loader=None):
    device = config["device"]
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["lr_w"])
    activation = model.activation_name
    
    # Computation frequency controls (for speed optimization)
    logging_cfg = config.get("logging", {})
    effective_rank_freq = int(logging_cfg.get("effective_rank_every_n_epochs", 1))
    if effective_rank_freq <= 0:
        effective_rank_freq = None  # Disable if non-positive

    # Very heavy; default to disabled unless explicitly requested
    path_kernel_metrics_freq = int(logging_cfg.get("path_kernel_metrics_every_n_epochs", 0))
    if path_kernel_metrics_freq <= 0:
        path_kernel_metrics_freq = None

    # Shared kernel settings for H100-friendly throughput
    path_kernel_block_size = int(logging_cfg.get("path_kernel_block_size", 1024))
    path_kernel_power_iters = int(logging_cfg.get("path_kernel_power_iters", 30))
    path_kernel_mode = logging_cfg.get("path_kernel_mode", "routing_gain")  # Default to routing_gain

    # Path analysis (plots, dominant paths) is even heavier: keep off unless opted-in
    path_analysis_out_dir = config.get("path_analysis_out_dir", None)
    path_analysis_freq = int(logging_cfg.get("path_analysis_every_n_epochs", path_kernel_metrics_freq or 0))
    if path_analysis_freq <= 0:
        path_analysis_freq = None
    path_analysis_enabled = path_analysis_out_dir is not None and path_analysis_freq is not None
    
    # Enable/disable Path Sankey diagram (nn_graph_paths) - can be expensive
    enable_nn_graph_paths = logging_cfg.get("enable_nn_graph_paths", True)

    # Gate-state counting can also be expensive on large datasets
    gate_states_freq = int(logging_cfg.get("gate_states_every_n_epochs", 0))
    if gate_states_freq <= 0:
        gate_states_freq = None
    gate_states_max_samples = int(logging_cfg.get("gate_states_max_samples", 2500))

    model.to(device)
    
    # Get optimizer type (default to "adam" for backward compatibility)
    optimizer_type = config.get("training", {}).get("optimizer", "adam")
    if isinstance(optimizer_type, list):
        optimizer_type = optimizer_type[0]  # Use first value if list
    optimizer_type = optimizer_type.lower()
    
    # Create optimizer based on type
    if optimizer_type == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr)
        print(f"[sgd] Using SGD optimizer with lr={lr}")
    elif optimizer_type == "adam":
        opt = optim.AdamW(model.parameters(), lr=lr)
        print(f"[sgd] Using AdamW optimizer with lr={lr}")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Options: 'sgd', 'adam'")
    
    history = []
    gate_states_history = []  # Track gate states over epochs
    
    # Detect n_classes from model
    n_classes = None
    if hasattr(model, 'n_classes'):
        n_classes = model.n_classes
    elif hasattr(model, 'readout'):
        out_features = model.readout.out_features
        if out_features > 1:
            n_classes = out_features
    
    # Get alpha from config (for scaling multiclass labels in MSE)
    alpha = float(config.get("dataset", {}).get("alpha", 1.0))
    if isinstance(alpha, list):
        alpha = alpha[0]  # Use first value if list
    
    import time as time_module
    epoch_start_time = time_module.time()
    print(f"[sgd] Starting training: {epochs} epochs, activation={activation}")
    if n_classes and n_classes > 1:
        print(f"[sgd] Multi-class classification with {n_classes} classes")

    # Compute at epoch 0 (initial model state)
    ep = 0
    print(f"[sgd] Computing initial metrics at epoch 0...")
    tr_acc, tr_loss = _eval(model, train_loader, device, n_classes, alpha)
    va_acc, va_loss = _eval(model, val_loader, device, n_classes, alpha)
    
    # Compute effective ranks at epoch 0
    if effective_rank_freq:
        eff_ranks = compute_effective_ranks(model, train_loader, device)
    else:
        eff_ranks = None
    
    # Compute path kernel metrics at epoch 0
    path_kernel_metrics = {}
    hidden_kernel_eff_rank_train_0 = None
    hidden_kernel_eff_rank_test_0 = None
    if path_kernel_metrics_freq:
        try:
            from ..analysis.path_analysis import compute_path_kernel_metrics
            test_loader_for_metrics = test_loader if test_loader is not None else val_loader
            path_kernel_metrics = compute_path_kernel_metrics(
                model,
                train_loader,
                test_loader_for_metrics,
                mode=path_kernel_mode,
                k=150,  # Increased to 150 for higher k R² plots
                max_samples=1000,
                device=device,
                block_size=path_kernel_block_size,
                power_iters=path_kernel_power_iters,
            )
        except Exception as e:
            print(f"  [path_kernel_metrics] Warning: Failed at epoch 0: {e}")
            path_kernel_metrics = {}
        
        # Plot eigenvector gate patterns at epoch 0 (for MNIST and hierarchical XOR)
        dataset_name = config.get("dataset", {}).get("name", "").lower()
        if dataset_name in ["hierarchical_xor", "mnist"] and path_analysis_out_dir:
            try:
                from ..analysis.path_analysis import plot_eigenvector_gate_patterns
                print(f"  [eigenvector_gate_patterns] Computing gate patterns at epoch 0...")
                gate_patterns_path = os.path.join(path_analysis_out_dir, "eigenvector_gate_patterns_epoch_0000.png")
                plot_eigenvector_gate_patterns(
                    model=model,
                    train_loader=train_loader,
                    out_path=gate_patterns_path,
                    mode=path_kernel_mode,
                    k=10,
                    max_samples=1000,
                    device=device,
                    block_size=path_kernel_block_size,
                    power_iters=path_kernel_power_iters,
                    title_suffix="Epoch 0",
                )
                print(f"  [eigenvector_gate_patterns] ✓ Saved gate patterns plot: {gate_patterns_path}")
            except Exception as e:
                print(f"  [eigenvector_gate_patterns] Warning: Failed at epoch 0: {e}")
                import traceback
                traceback.print_exc()
        
        # Compute hidden kernel effective rank at epoch 0 for XOR and MNIST
        dataset_name = config.get("dataset", {}).get("name", "").lower()
        if dataset_name in ["hierarchical_xor", "mnist"]:
            try:
                from ..analysis.path_analysis import compute_hidden_kernel_effective_rank
                hidden_kernel_eff_rank_train_0 = compute_hidden_kernel_effective_rank(
                    model, train_loader, device=device, max_samples=1000
                )
                if test_loader is not None:
                    hidden_kernel_eff_rank_test_0 = compute_hidden_kernel_effective_rank(
                        model, test_loader, device=device, max_samples=1000
                    )
                print(f"  [hidden_kernel_eff_rank] Epoch 0: Train ranks = {[f'{r:.2f}' for r in hidden_kernel_eff_rank_train_0]}")
                if hidden_kernel_eff_rank_test_0 is not None:
                    print(f"  [hidden_kernel_eff_rank] Epoch 0: Test ranks = {[f'{r:.2f}' for r in hidden_kernel_eff_rank_test_0]}")
            except Exception as e:
                print(f"  [hidden_kernel_eff_rank] Warning: Failed at epoch 0: {e}")
                import traceback
                traceback.print_exc()
    
    # Run full analysis at epoch 0
    if path_analysis_freq and path_analysis_out_dir:
        try:
            from ..analysis.path_analysis import run_full_analysis_at_checkpoint
            dataset_name = config.get("dataset", {}).get("name", "").lower()
            run_full_analysis_at_checkpoint(
                model=model,
                val_loader=val_loader,
                out_dir=path_analysis_out_dir,
                step_tag="epoch_0000",
                kernel_k=48,
                kernel_mode=path_kernel_mode,
                include_input_in_kernel=True,
                block_size=path_kernel_block_size,
                power_iters=path_kernel_power_iters,
                max_samples_kernel=1000,
                max_samples_embed=1000,
                enable_nn_graph_paths=enable_nn_graph_paths,
                dataset_name=dataset_name,
                train_loader=train_loader,
                device=device,
            )
        except Exception as e:
            print(f"  [path_analysis] Warning: Failed at epoch 0: {e}")
    
    # Compute dominant k paths at epoch 0
    if path_analysis_freq and path_analysis_out_dir:
        try:
            from ..analysis.path_analysis import compute_dominant_k_paths
            test_loader_for_dominant = test_loader if test_loader is not None else val_loader
            dominant_k_paths_out = os.path.join(path_analysis_out_dir, "dominant_k_paths_epoch_0000.png")
            compute_dominant_k_paths(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader_for_dominant,
                out_path=dominant_k_paths_out,
                mode=path_kernel_mode,
                max_candidate_paths=200,
                max_k=20,
                max_samples=1000,
                weight_aware=True,
                use_network_outputs=False,
                device=device,
                step_tag="epoch_0000",
            )
        except Exception as e:
            print(f"  [dominant_k_paths] Warning: Failed at epoch 0: {e}")
    
    # Plot flow centrality at epoch 0
    if path_analysis_freq and path_analysis_out_dir:
        try:
            from ..analysis.path_analysis import flow_centrality_heatmap
            central_png = os.path.join(path_analysis_out_dir, "flow_centrality_epoch_0000.png")
            flow_centrality_heatmap(model, val_loader, central_png, mode=path_kernel_mode)
        except Exception as e:
            print(f"  [flow_centrality] Warning: Failed at epoch 0: {e}")
    
    # IIA and Path-Shapley at epoch 0 (for XOR, MNIST, CIFAR-10)
    if path_analysis_freq and path_analysis_out_dir:
        dataset_name = config.get("dataset", {}).get("name", "").lower()
        if dataset_name in ["hierarchical_xor", "mnist", "cifar10"]:
            # Interchange Intervention Accuracy at epoch 0
            try:
                from ..analysis.path_analysis import compute_interchange_intervention_accuracy, _mean_transmittance_per_layer, _beam_top_paths
                print(f"  [IIA] Computing Interchange Intervention Accuracy at epoch 0...")
                mean_E = _mean_transmittance_per_layer(model, train_loader, device=device, mode=path_kernel_mode)
                candidate_paths = _beam_top_paths(model, mean_E, beam=50, top_k=20)
                
                if len(candidate_paths) > 0:
                    # Detect n_classes from model
                    n_classes_iia = None
                    if hasattr(model, 'n_classes'):
                        n_classes_iia = model.n_classes
                    elif hasattr(model, 'readout'):
                        out_features = model.readout.out_features
                        if out_features > 1:
                            n_classes_iia = out_features
                    
                    test_loader_for_iia = test_loader if test_loader is not None else val_loader
                    iia_results_epoch0 = compute_interchange_intervention_accuracy(
                        model=model,
                        loader=test_loader_for_iia,
                        paths=candidate_paths,
                        k=10,  # Use top 10 paths
                        n_interventions=50,  # Test 50 intervention pairs
                        device=device,
                        n_classes=n_classes_iia,
                    )
                    # Store for history
                    iia_metrics_epoch0 = iia_results_epoch0
                    print(f"  [IIA] Epoch 0 accuracy: {iia_results_epoch0.get('iia_accuracy', 0.0):.3f}")
                else:
                    print(f"  [IIA] Warning: No paths found at epoch 0")
                    iia_metrics_epoch0 = {"iia_accuracy": None}
            except Exception as e:
                print(f"  [IIA] Warning: Failed at epoch 0: {e}")
                import traceback
                traceback.print_exc()
                iia_metrics_epoch0 = {"iia_accuracy": None}
            
            # Path-Shapley at epoch 0
            try:
                from ..analysis.path_analysis import plot_path_shapley_bars
                from ..analysis.path_kernel import compute_path_kernel_eigs
                print(f"  [path_shapley] Computing Path-Shapley metrics at epoch 0...")
                kern = compute_path_kernel_eigs(
                    model, train_loader, device=device, mode=path_kernel_mode, include_input=True,
                    k=min(24, 1000), n_iter=path_kernel_power_iters, 
                    block_size=path_kernel_block_size, max_samples=1000, verbose=False
                )
                evecs = kern.get("evecs")
                y = kern.get("y")
                
                if evecs is not None and y is not None:
                    evecs_np = evecs.detach().cpu().numpy()
                    y_np = y.detach().cpu().numpy().flatten()
                    top_m = min(24, evecs_np.shape[1])
                    scores = evecs_np[:, :top_m]
                    
                    shapley_out = os.path.join(path_analysis_out_dir, "path_shapley_epoch_0000.png")
                    plot_path_shapley_bars(
                        scores=scores,
                        y=y_np,
                        out_path=shapley_out,
                        title="Path-Shapley (MI proxy) - Epoch 0"
                    )
                    print(f"  [path_shapley] ✓ Saved Path-Shapley plot: {shapley_out}")
            except Exception as e:
                print(f"  [path_shapley] Warning: Failed at epoch 0: {e}")
                import traceback
                traceback.print_exc()
    
    # Compute gate states at epoch 0
    if gate_states_freq:
        try:
            from ..analysis.path_analysis import compute_unique_gate_states_per_class
            gate_states = compute_unique_gate_states_per_class(
                model,
                train_loader,
                device=device,
                max_samples=gate_states_max_samples,
            )
            gate_states["epoch"] = 0
            gate_states_history.append(gate_states)
        except Exception as e:
            print(f"  [gate_states] Warning: Failed at epoch 0: {e}")
    
    # Record history at epoch 0 FIRST (before plotting)
    hist_entry_0 = {
        "epoch": 0,
        "train_acc": tr_acc,
        "train_loss": tr_loss,
        "val_acc": va_acc,
        "val_loss": va_loss,
        "effective_ranks": eff_ranks,
        **path_kernel_metrics,
    }
    # Add hidden kernel effective rank at epoch 0
    if hidden_kernel_eff_rank_train_0 is not None:
        hist_entry_0["hidden_kernel_effective_rank_train_layers"] = hidden_kernel_eff_rank_train_0
    if hidden_kernel_eff_rank_test_0 is not None:
        hist_entry_0["hidden_kernel_effective_rank_test_layers"] = hidden_kernel_eff_rank_test_0
    # Add IIA metrics if computed (check if it was computed above)
    if path_analysis_freq and path_analysis_out_dir:
        dataset_name = config.get("dataset", {}).get("name", "").lower()
        if dataset_name in ["hierarchical_xor", "mnist", "cifar10"]:
            try:
                if 'iia_metrics_epoch0' in locals() and iia_metrics_epoch0:
                    hist_entry_0["iia_accuracy"] = iia_metrics_epoch0.get("iia_accuracy")
                    hist_entry_0["iia_by_k"] = iia_metrics_epoch0.get("iia_by_k", {})
            except:
                pass  # IIA might not have been computed yet
    history.append(hist_entry_0)
    
    # Skip plotting at epoch 0 - wait until we have more data points
    # (Plot will be created at epoch 1 or 200 when we have more epochs to show)
    
    # Now start training from epoch 1
    for ep in range(1, epochs+1):
        if ep % 10 == 0 or ep == 1:
            elapsed = time_module.time() - epoch_start_time
            print(f"[sgd] Epoch {ep}/{epochs} (elapsed: {elapsed:.1f}s, avg: {elapsed/ep:.2f}s/epoch)")
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            if n_classes is None or n_classes == 1:
                # Binary classification: MSE loss
                loss = mse_loss(yhat, yb)
            else:
                # Multi-class classification: MSE loss with one-hot targets scaled by alpha
                # yb contains scaled class indices (e.g., 0, 10, 20, ..., 90 for alpha=10)
                # Convert to one-hot encoding scaled by alpha
                if yb.dim() > 1:
                    yb = yb.view(-1)  # Flatten to 1D
                yb_class = (yb / alpha).long()  # Get original class index
                yb_onehot = torch.zeros_like(yhat)
                yb_onehot.scatter_(1, yb_class.unsqueeze(1), yb.unsqueeze(1))  # Fill with scaled alpha value
                loss = mse_loss(yhat, yb_onehot)
            opt.zero_grad(); loss.backward(); opt.step()
        tr_acc, tr_loss = _eval(model, train_loader, device, n_classes, alpha)
        va_acc, va_loss = _eval(model, val_loader, device, n_classes, alpha)
        
        # Compute effective ranks - SVD is expensive, compute less frequently
        if effective_rank_freq and ((ep % effective_rank_freq == 0) or (ep == 1) or (ep == epochs)):
            eff_ranks = compute_effective_ranks(model, train_loader, device)
        else:
            eff_ranks = None  # Skip expensive SVD computation
        
        # Early stopping check and test loss computation (needed before history append)
        early_stopped = False
        test_loss = None
        test_acc = None
        if test_loader is not None:
            test_acc, test_loss = _eval(model, test_loader, device, n_classes, alpha)
        
        # Compute path kernel metrics (effective rank, variance explained, etc.)
        path_kernel_metrics = {}
        hidden_kernel_eff_rank_train = None
        hidden_kernel_eff_rank_test = None
        if path_kernel_metrics_freq and ((ep % path_kernel_metrics_freq == 0) or (ep == 1) or (ep == epochs)):
            try:
                from ..analysis.path_analysis import compute_path_kernel_metrics
                test_loader_for_metrics = test_loader if test_loader is not None else val_loader
                # Use routing mode for all activations (binary indicator)
                path_kernel_metrics = compute_path_kernel_metrics(
                    model,
                    train_loader,
                    test_loader_for_metrics,
                    mode=path_kernel_mode,  # Use routing mode (binary indicator)
                    k=150,  # Increased to 150 for higher k R² plots
                    max_samples=1000,  # REDUCED from 5000 to 1000 for speed
                    device=device,
                    block_size=path_kernel_block_size,
                    power_iters=path_kernel_power_iters,
                )
                # Print train and test error when path kernel metrics are computed
                if test_loader is not None:
                    print(f"  [path_kernel_metrics] Epoch {ep}: Train Error = {tr_loss:.6f}, Test Error = {test_loss:.6f}")
                else:
                    print(f"  [path_kernel_metrics] Epoch {ep}: Train Error = {tr_loss:.6f}, Val Error = {va_loss:.6f}")
            except Exception as e:
                print(f"  [path_kernel_metrics] Warning: Failed at epoch {ep}: {e}")
            
            # Compute hidden kernel effective rank for XOR and MNIST tasks
            dataset_name = config.get("dataset", {}).get("name", "").lower()
            if dataset_name in ["hierarchical_xor", "mnist"]:
                try:
                    from ..analysis.path_analysis import compute_hidden_kernel_effective_rank
                    hidden_kernel_eff_rank_train = compute_hidden_kernel_effective_rank(
                        model, train_loader, device=device, max_samples=1000
                    )
                    if test_loader is not None:
                        hidden_kernel_eff_rank_test = compute_hidden_kernel_effective_rank(
                            model, test_loader, device=device, max_samples=1000
                        )
                    print(f"  [hidden_kernel_eff_rank] Epoch {ep}: Train ranks = {[f'{r:.2f}' for r in hidden_kernel_eff_rank_train]}")
                    if hidden_kernel_eff_rank_test is not None:
                        print(f"  [hidden_kernel_eff_rank] Epoch {ep}: Test ranks = {[f'{r:.2f}' for r in hidden_kernel_eff_rank_test]}")
                except Exception as e:
                    print(f"  [hidden_kernel_eff_rank] Warning: Failed at epoch {ep}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Early stopping check
        if test_loss is not None and test_loss < 0.01:
            print(f"Early stopping: test loss {test_loss:.6f} < 0.01")
            early_stopped = True
        
        if va_loss < 0.01:
            print(f"Early stopping: validation loss {va_loss:.6f} < 0.01")
            early_stopped = True
        
        # Initialize IIA metrics (will be populated during path analysis if applicable)
        iia_metrics_this_epoch = {}
        
        # CRITICAL: Append current epoch to history BEFORE plotting, so plots include current epoch
        # But we'll add IIA metrics after path analysis runs
        hist_entry = {
            "epoch": ep, 
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
            "effective_rank_layers": eff_ranks
        }
        if test_loss is not None:
            hist_entry.update({"test_loss": test_loss, "test_acc": test_acc})
        
        # Add path kernel metrics
        if path_kernel_metrics:
            hist_entry.update(path_kernel_metrics)
        
        # Add hidden kernel effective rank
        if hidden_kernel_eff_rank_train is not None:
            hist_entry["hidden_kernel_effective_rank_train_layers"] = hidden_kernel_eff_rank_train
        if hidden_kernel_eff_rank_test is not None:
            hist_entry["hidden_kernel_effective_rank_test_layers"] = hidden_kernel_eff_rank_test
        
        history.append(hist_entry)
        
        # Compute unique gate states per class (track every epoch for plotting)
        if gate_states_freq and ((ep % gate_states_freq == 0) or (ep == 1) or (ep == epochs)):
            try:
                from ..analysis.path_analysis import compute_unique_gate_states_per_class
                gate_states = compute_unique_gate_states_per_class(
                    model,
                    train_loader,
                    device=device,
                    max_samples=gate_states_max_samples,
                )
                gate_states["epoch"] = ep
                gate_states_history.append(gate_states)
                
                # Skip plotting at epoch 1 (too early, only 2 data points). 
                # Plots will be generated at the end with all epochs, and optionally during training at later epochs
                # Only generate plots during training if we're past epoch 1 and at a regular interval (not at first epoch)
                if ep > 1 and ep != epochs:
                    # Optionally create plots during training at regular intervals (but not at epoch 1 or last epoch)
                    # Last epoch plots will be created at the end with all data
                    pass  # Skip intermediate plotting - only plot at the end for cleaner output
            except Exception as e:
                print(f"  [gate_states] Warning: Failed at epoch {ep}: {e}")
                # Append empty dict to maintain alignment
                gate_states_history.append({"epoch": ep})
        elif gate_states_freq is None:
            # Keep alignment with epochs even when disabled
            gate_states_history.append({"epoch": ep})
        
        # Path metrics removed - no longer computing standard path metrics
        
        # Run path analysis at intervals (start, end, and every N epochs)
        # Use routing mode (binary indicator) for all activations
        if path_analysis_enabled and ((ep % path_analysis_freq == 0) or (ep == 1) or (ep == epochs)):
            try:
                from ..analysis.path_analysis import (
                    run_full_analysis_at_checkpoint, compute_dominant_k_paths, flow_centrality_heatmap,
                    plot_path_shapley_bars
                )
                from ..analysis.path_kernel import compute_path_kernel_eigs
                
                # Run full analysis - REDUCED samples for speed during training
                dataset_name = config.get("dataset", {}).get("name", "").lower()
                run_full_analysis_at_checkpoint(
                    model=model,
                    val_loader=val_loader,
                    out_dir=path_analysis_out_dir,
                    step_tag=f"epoch_{ep:04d}",
                    kernel_k=48,
                    kernel_mode=path_kernel_mode,  # Use routing mode (binary indicator)
                    include_input_in_kernel=True,
                    block_size=path_kernel_block_size,
                    power_iters=path_kernel_power_iters,
                    max_samples_kernel=1000,  # REDUCED from 5000 to 1000 for speed
                    max_samples_embed=1000,   # REDUCED from 5000 to 1000 for speed
                    enable_nn_graph_paths=enable_nn_graph_paths,
                    dataset_name=dataset_name,
                    train_loader=train_loader,
                    device=device,
                )
                
                # Compute dominant k paths analysis
                test_loader_for_dominant = test_loader if test_loader is not None else val_loader
                dominant_k_paths_out = os.path.join(path_analysis_out_dir, f"dominant_k_paths_epoch_{ep:04d}.png")
                compute_dominant_k_paths(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader_for_dominant,
                    out_path=dominant_k_paths_out,
                    mode=path_kernel_mode,
                    max_candidate_paths=200,
                    max_k=20,
                    max_samples=1000,  # REDUCED from 5000 to 1000 for speed
                    weight_aware=True,
                    use_network_outputs=False,  # Use labels y as target
                    device=device,
                    step_tag=f"epoch_{ep:04d}",
                )
                
                # Plot flow centrality for all epochs where path analysis runs
                central_png = os.path.join(path_analysis_out_dir, f"flow_centrality_epoch_{ep:04d}.png")
                flow_centrality_heatmap(model, val_loader, central_png, mode=path_kernel_mode)
                
                # Interchange Intervention Accuracy (IIA) - for XOR, MNIST, CIFAR-10
                if dataset_name in ["hierarchical_xor", "mnist", "cifar10"]:
                    try:
                        from ..analysis.path_analysis import (
                            compute_interchange_intervention_accuracy,
                            _mean_transmittance_per_layer, _beam_top_paths
                        )
                        print(f"  [IIA] Computing Interchange Intervention Accuracy at epoch {ep}...")
                        
                        # Get dominant paths from mean transmittance
                        mean_E = _mean_transmittance_per_layer(model, train_loader, device=device, mode=path_kernel_mode)
                        candidate_paths = _beam_top_paths(model, mean_E, beam=50, top_k=20)
                        
                        if len(candidate_paths) > 0:
                            # Detect n_classes from model
                            n_classes_iia = None
                            if hasattr(model, 'n_classes'):
                                n_classes_iia = model.n_classes
                            elif hasattr(model, 'readout'):
                                out_features = model.readout.out_features
                                if out_features > 1:
                                    n_classes_iia = out_features
                            
                            # Compute IIA
                            test_loader_for_iia = test_loader if test_loader is not None else val_loader
                            iia_results = compute_interchange_intervention_accuracy(
                                model=model,
                                loader=test_loader_for_iia,
                                paths=candidate_paths,
                                k=10,  # Use top 10 paths
                                n_interventions=50,  # Test 50 intervention pairs
                                device=device,
                                n_classes=n_classes_iia,
                            )
                            
                            # Store for adding to history (update the last history entry)
                            iia_metrics_this_epoch = iia_results
                            # Update the history entry that was just appended
                            if len(history) > 0:
                                last_entry = history[-1]
                                if last_entry.get("epoch") == ep:
                                    last_entry["iia_accuracy"] = iia_results.get("iia_accuracy")
                                    last_entry["iia_by_k"] = iia_results.get("iia_by_k", {})
                                    print(f"  [IIA] Added to history: epoch={ep}, accuracy={iia_results.get('iia_accuracy', 0.0):.3f}")
                                else:
                                    print(f"  [IIA] Warning: Epoch mismatch (last entry epoch={last_entry.get('epoch')}, current={ep})")
                            print(f"  [IIA] Completed: accuracy={iia_results.get('iia_accuracy', 0.0):.3f}")
                        else:
                            print(f"  [IIA] Warning: No paths found, skipping IIA computation")
                            iia_metrics_this_epoch = {"iia_accuracy": None}
                    except Exception as e:
                        print(f"  [IIA] Warning: Failed at epoch {ep}: {e}")
                        import traceback
                        traceback.print_exc()
                        iia_metrics_this_epoch = {"iia_accuracy": None}
                
                # Path-Shapley analysis (for XOR, MNIST, CIFAR-10)
                if dataset_name in ["hierarchical_xor", "mnist", "cifar10"]:
                    try:
                        print(f"  [path_shapley] Computing Path-Shapley metrics at epoch {ep}...")
                        # Get path kernel eigenfunctions
                        kern = compute_path_kernel_eigs(
                            model, train_loader, device=device, mode=path_kernel_mode, include_input=True,
                            k=min(24, 1000), n_iter=path_kernel_power_iters, 
                            block_size=path_kernel_block_size, max_samples=1000, verbose=False
                        )
                        evecs = kern.get("evecs")
                        y = kern.get("y")
                        
                        if evecs is not None and y is not None:
                            evecs_np = evecs.detach().cpu().numpy()  # (P, k)
                            y_np = y.detach().cpu().numpy().flatten()
                            top_m = min(24, evecs_np.shape[1])
                            scores = evecs_np[:, :top_m]
                            
                            # Compute and plot Path-Shapley
                            shapley_out = os.path.join(path_analysis_out_dir, f"path_shapley_epoch_{ep:04d}.png")
                            plot_path_shapley_bars(
                                scores=scores,
                                y=y_np,
                                out_path=shapley_out,
                                title=f"Path-Shapley (MI proxy) - Epoch {ep}"
                            )
                            print(f"  [path_shapley] Saved Path-Shapley plot: {shapley_out}")
                    except Exception as e:
                        print(f"  [path_shapley] Warning: Failed at epoch {ep}: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"  [path_analysis] Completed for epoch {ep}")
            except Exception as e:
                import traceback
                print(f"  [path_analysis] Warning: Failed at epoch {ep}: {e}")
                print(f"  [path_analysis] Traceback: {traceback.format_exc()}")

        if early_stopped:
            break
    
    # Plot gate states and errors vs epoch at the end of training (final overwrite with ALL epochs)
    # Always generate error and variance plots if we have history; gate_states plot requires gate_states_history
    print(f"[sgd] Training completed. History length: {len(history)}")
    if len(history) > 0:
        print(f"[sgd] Sample history entry keys: {list(history[0].keys())}")
        print(f"[sgd] History epochs: {[h.get('epoch') for h in history[:10]]}")
        try:
            from ..analysis.path_analysis import (
                plot_gate_states_vs_epoch, plot_train_test_errors_vs_epoch, 
                plot_effective_rank_layers_vs_epoch, plot_variance_explained_by_k_vs_epoch
            )
            # Use algo_dir from config if available, otherwise derive from path_analysis_out_dir, otherwise use out_dir
            algo_dir = config.get("algo_dir", None)
            if algo_dir is None:
                if path_analysis_out_dir:
                    algo_dir = os.path.dirname(path_analysis_out_dir)  # path_analysis is subdir of algo_dir
                else:
                    algo_dir = config.get("out_dir", ".")
            
            # Ensure directory exists
            os.makedirs(algo_dir, exist_ok=True)
            
            # Use fixed filenames that get overwritten
            errors_plot_path = os.path.join(algo_dir, "train_test_errors_vs_epoch.png")
            eff_rank_plot_path = os.path.join(algo_dir, "effective_rank_vs_epoch.png")
            variance_k_train_path = os.path.join(algo_dir, "variance_explained_by_k_train.png")
            variance_k_test_path = os.path.join(algo_dir, "variance_explained_by_k_test.png")
            iia_plot_path = os.path.join(algo_dir, "iia_vs_epoch.png")
            hidden_kernel_eff_rank_train_path = os.path.join(algo_dir, "hidden_kernel_effective_rank_train_vs_epoch.png")
            hidden_kernel_eff_rank_test_path = os.path.join(algo_dir, "hidden_kernel_effective_rank_test_vs_epoch.png")
            
            # Always generate error and variance plots (they only need history)
            print(f"[sgd] Generating final plots with {len(history)} epochs in history (epochs: {[h['epoch'] for h in history[:5]]}{'...' if len(history) > 5 else ''})")
            print(f"[sgd] Saving plots to: {algo_dir} (exists: {os.path.exists(algo_dir)})")
            
            # Plot train/test errors (always try this one)
            try:
                plot_train_test_errors_vs_epoch(history, errors_plot_path)
                print(f"[sgd] ✓ Saved train/test errors plot: {errors_plot_path}")
            except Exception as e:
                print(f"[sgd] ✗ Failed to plot train/test errors: {e}")
                import traceback
                traceback.print_exc()
            
            # Plot effective rank (always try this one)
            try:
                plot_effective_rank_layers_vs_epoch(history, eff_rank_plot_path)
                print(f"[sgd] ✓ Saved effective rank plot: {eff_rank_plot_path}")
            except Exception as e:
                print(f"[sgd] ✗ Failed to plot effective rank: {e}")
                import traceback
                traceback.print_exc()
            
            # Plot variance explained (may fail if path_kernel_metrics weren't computed)
            try:
                plot_variance_explained_by_k_vs_epoch(history, variance_k_train_path, variance_k_test_path)
                print(f"[sgd] ✓ Saved variance plots: {variance_k_train_path}, {variance_k_test_path}")
            except Exception as e:
                print(f"[sgd] ⚠ Failed to plot variance explained (may be missing path_kernel_metrics): {e}")
                # Don't print full traceback for this one as it's expected if path_kernel_metrics aren't computed
            
            # Plot hidden kernel effective rank for XOR and MNIST
            dataset_name = config.get("dataset", {}).get("name", "").lower()
            if dataset_name in ["hierarchical_xor", "mnist"]:
                try:
                    from ..analysis.path_analysis import plot_hidden_kernel_effective_rank_vs_epoch
                    # Check if we have any hidden kernel effective rank data
                    has_train_data = any("hidden_kernel_effective_rank_train_layers" in h for h in history)
                    has_test_data = any("hidden_kernel_effective_rank_test_layers" in h for h in history)
                    if has_train_data or has_test_data:
                        print(f"[sgd] Found hidden kernel effective rank data, generating plots...")
                        plot_hidden_kernel_effective_rank_vs_epoch(
                            history, hidden_kernel_eff_rank_train_path, hidden_kernel_eff_rank_test_path
                        )
                        print(f"[sgd] ✓ Saved hidden kernel effective rank plots")
                    else:
                        print(f"[sgd] ⚠ No hidden kernel effective rank data found in history. Skipping plots.")
                except Exception as e:
                    print(f"[sgd] ⚠ Failed to plot hidden kernel effective rank: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Plot IIA (Interchange Intervention Accuracy) if available
            if dataset_name in ["hierarchical_xor", "mnist", "cifar10"]:
                try:
                    from ..analysis.path_analysis import plot_iia_vs_epoch
                    # Check if we have any IIA data
                    iia_entries = [h for h in history if "iia_accuracy" in h and h.get("iia_accuracy") is not None]
                    if len(iia_entries) > 0:
                        print(f"[sgd] Found {len(iia_entries)} epochs with IIA data, generating plot...")
                        plot_iia_vs_epoch(history, iia_plot_path)
                        print(f"[sgd] ✓ Saved IIA plot: {iia_plot_path}")
                    else:
                        print(f"[sgd] ⚠ No IIA data found in history (checked {len(history)} entries). Skipping IIA plot.")
                        # Debug: show what keys are in history
                        if len(history) > 0:
                            sample_keys = list(history[0].keys())
                            print(f"[sgd] Sample history entry keys: {sample_keys}")
                            # Check if any entry has iia
                            has_iia = any("iia" in str(k).lower() for k in sample_keys)
                            print(f"[sgd] History contains 'iia' in keys: {has_iia}")
                except Exception as e:
                    print(f"[sgd] ⚠ Failed to plot IIA: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Only generate gate_states plot if we have gate_states_history
            if len(gate_states_history) > 0:
                gate_plot_path = os.path.join(algo_dir, "gate_states_vs_epoch.png")
                plot_gate_states_vs_epoch(history, gate_states_history, gate_plot_path)
        except Exception as e:
            import traceback
            print(f"[sgd] Warning: Failed to plot gate states/errors: {e}")
            print(f"[sgd] Traceback: {traceback.format_exc()}")
    
    # Run eigenpath ablation study after training
    if path_analysis_enabled and test_loader is not None:
        try:
            from ..analysis.path_ablation import compute_eigenpath_ablation, plot_eigenpath_ablation
            print(f"\n[sgd] Running eigenpath ablation study...")
            
            # Determine algo_dir for saving plots
            if path_analysis_out_dir:
                algo_dir = os.path.dirname(path_analysis_out_dir)
            else:
                algo_dir = config.get("out_dir", ".")
            
            # Detect n_classes from model
            n_classes = None
            if hasattr(model, 'n_classes'):
                n_classes = model.n_classes
            elif hasattr(model, 'readout'):
                out_features = model.readout.out_features
                if out_features > 1:
                    n_classes = out_features
            
            ablation_results = compute_eigenpath_ablation(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                mode=path_kernel_mode,
                k=150,  # Use same k as path kernel metrics
                max_samples=1000,  # Use same sample limit
                device=device,
                block_size=path_kernel_block_size,
                power_iters=path_kernel_power_iters,
                n_classes=n_classes,
            )
            
            # Plot ablation results
            ablation_plot_path = os.path.join(algo_dir, "eigenpath_ablation.png")
            plot_eigenpath_ablation(ablation_results, ablation_plot_path)
            
            print(f"[sgd] Eigenpath ablation study completed")
            
            # Plot eigenvector gate patterns at end of training (for MNIST and hierarchical XOR)
            dataset_name = config.get("dataset", {}).get("name", "").lower()
            if dataset_name in ["hierarchical_xor", "mnist"]:
                try:
                    from ..analysis.path_analysis import plot_eigenvector_gate_patterns
                    print(f"\n[sgd] Computing final eigenvector gate patterns...")
                    gate_patterns_final_path = os.path.join(path_analysis_out_dir, "eigenvector_gate_patterns_final.png")
                    plot_eigenvector_gate_patterns(
                        model=model,
                        train_loader=train_loader,
                        out_path=gate_patterns_final_path,
                        mode=path_kernel_mode,
                        k=10,
                        max_samples=1000,
                        device=device,
                        block_size=path_kernel_block_size,
                        power_iters=path_kernel_power_iters,
                        title_suffix="Final",
                    )
                    print(f"[sgd] ✓ Saved final gate patterns plot: {gate_patterns_final_path}")
                except Exception as e:
                    print(f"[sgd] Warning: Failed to compute final gate patterns: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Run layer-wise path ablation
            try:
                from ..analysis.path_ablation import (
                    compute_layerwise_path_ablation,
                    plot_layerwise_path_ablation,
                    plot_layerwise_path_ablation_test,
                    compute_layerwise_hidden_ablation,
                    plot_layerwise_hidden_ablation,
                    plot_layerwise_hidden_ablation_test,
                )
                print(f"\n[sgd] Running layer-wise path ablation study...")
                
                layer_path_results = compute_layerwise_path_ablation(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    mode=path_kernel_mode,
                    k=150,
                    max_samples=1000,
                    device=device,
                    block_size=path_kernel_block_size,
                    power_iters=path_kernel_power_iters,
                    n_classes=n_classes,
                )
                
                # Plot layer-wise path ablation
                plot_layerwise_path_ablation(
                    layer_path_results,
                    os.path.join(algo_dir, "layerwise_path_ablation_train.png"),
                    plot_type="error"
                )
                plot_layerwise_path_ablation_test(
                    layer_path_results,
                    os.path.join(algo_dir, "layerwise_path_ablation_test.png"),
                    plot_type="error"
                )
                
                print(f"[sgd] Layer-wise path ablation completed")
                
                # Run layer-wise hidden kernel ablation
                print(f"\n[sgd] Running layer-wise hidden kernel ablation study...")
                
                layer_hidden_results = compute_layerwise_hidden_ablation(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    k=150,
                    max_samples=1000,
                    device=device,
                    n_classes=n_classes,
                )
                
                # Plot layer-wise hidden ablation
                plot_layerwise_hidden_ablation(
                    layer_hidden_results,
                    os.path.join(algo_dir, "layerwise_hidden_ablation_train.png"),
                    plot_type="error"
                )
                plot_layerwise_hidden_ablation_test(
                    layer_hidden_results,
                    os.path.join(algo_dir, "layerwise_hidden_ablation_test.png"),
                    plot_type="error"
                )
                
                print(f"[sgd] Layer-wise hidden kernel ablation completed")
                
            except Exception as e:
                import traceback
                print(f"[sgd] Warning: Failed to run layer-wise ablation: {e}")
                print(f"[sgd] Traceback: {traceback.format_exc()}")
                
        except Exception as e:
            import traceback
            print(f"[sgd] Warning: Failed to run eigenpath ablation: {e}")
            print(f"[sgd] Traceback: {traceback.format_exc()}")
    
    # Run dream algorithm for MNIST or CIFAR-10 dataset
    dataset_name = config.get("dataset", {}).get("name", "").lower()
    if dataset_name in ["mnist", "cifar10"] and path_analysis_enabled and train_loader is not None:
        try:
            from ..analysis.dream_algorithm import dream_algorithm_with_variance, plot_dream_images
            print(f"\n[sgd] Running dream algorithm for {dataset_name.upper()}...")
            
            # Determine algo_dir for saving plots
            if path_analysis_out_dir:
                algo_dir = os.path.dirname(path_analysis_out_dir)
            else:
                algo_dir = config.get("out_dir", ".")
            
            # Get image shape from config or default based on dataset
            if dataset_name == "mnist":
                default_shape = (28, 28)
            elif dataset_name == "cifar10":
                default_shape = (32, 32, 3)  # CIFAR-10 is 32x32 RGB
            else:
                default_shape = (28, 28)
            
            image_shape = config.get("dataset", {}).get("image_shape", default_shape)
            if isinstance(image_shape, list):
                image_shape = tuple(image_shape)
            
            # Run dream algorithm (generates k=15 images, one per eigenpath)
            dream_images, variance_explained = dream_algorithm_with_variance(
                model=model,
                train_loader=train_loader,
                k=15,  # Top 15 eigenpaths - will generate 15 images
                mode=path_kernel_mode,
                max_samples=1000,
                device=device,
                block_size=path_kernel_block_size,
                power_iters=path_kernel_power_iters,
                lr=0.1,
                n_iter=500,
                image_shape=image_shape,
                init_method="random",
                regularization=0.01,
            )
            
            # Plot dream images with variance explained in titles
            dream_plot_path = os.path.join(algo_dir, "dream_images.png")
            plot_dream_images(
                dream_images, 
                dream_plot_path, 
                image_shape=image_shape, 
                variance_explained=variance_explained,
                n_cols=5
            )
            
            print(f"[sgd] Dream algorithm completed - generated {len(dream_images)} images (one per top eigenpath)")
        except Exception as e:
            import traceback
            print(f"[sgd] Warning: Failed to run dream algorithm: {e}")
            print(f"[sgd] Traceback: {traceback.format_exc()}")
    
    # Path-Shapley at end of training (for XOR, MNIST, CIFAR-10)
    if dataset_name in ["hierarchical_xor", "mnist", "cifar10"] and path_analysis_enabled and train_loader is not None:
        try:
            from ..analysis.path_analysis import plot_path_shapley_bars
            from ..analysis.path_kernel import compute_path_kernel_eigs
            print(f"\n[sgd] Computing final Path-Shapley metrics...")
            
            # Determine algo_dir for saving plots
            if path_analysis_out_dir:
                algo_dir = os.path.dirname(path_analysis_out_dir)
            else:
                algo_dir = config.get("out_dir", ".")
            
            kern = compute_path_kernel_eigs(
                model, train_loader, device=device, mode=path_kernel_mode, include_input=True,
                k=min(24, 1000), n_iter=path_kernel_power_iters, 
                block_size=path_kernel_block_size, max_samples=1000, verbose=False
            )
            evecs = kern.get("evecs")
            y = kern.get("y")
            
            if evecs is not None and y is not None:
                evecs_np = evecs.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy().flatten()
                top_m = min(24, evecs_np.shape[1])
                scores = evecs_np[:, :top_m]
                
                shapley_out = os.path.join(algo_dir, "path_shapley_final.png")
                plot_path_shapley_bars(
                    scores=scores,
                    y=y_np,
                    out_path=shapley_out,
                    title="Path-Shapley (MI proxy) - Final Model"
                )
                print(f"[sgd] Saved final Path-Shapley plot: {shapley_out}")
        except Exception as e:
            import traceback
            print(f"[sgd] Warning: Failed to compute final Path-Shapley: {e}")
            traceback.print_exc()
    
    # Run hidden layer gram kernel analysis for hierarchical_xor dataset
    if dataset_name == "hierarchical_xor" and test_loader is not None:
        try:
            from ..analysis.hidden_layer_analysis import (
                compute_hidden_variance_explained,
                plot_hidden_variance_explained_by_k,
                compute_eigenvector_ablation,
                plot_eigenvector_ablation,
            )
            print(f"\n[sgd] Running hidden layer gram kernel analysis for hierarchical_xor...")
            
            # Determine algo_dir for saving plots
            if path_analysis_out_dir:
                algo_dir = os.path.dirname(path_analysis_out_dir)
            else:
                algo_dir = config.get("out_dir", ".")
            
            # Compute variance explained by k
            variance_results = compute_hidden_variance_explained(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                k=150,
                max_samples=1000,
                device=device,
            )
            
            # Plot variance explained
            variance_train_path = os.path.join(algo_dir, "hidden_variance_explained_by_k_train.png")
            variance_test_path = os.path.join(algo_dir, "hidden_variance_explained_by_k_test.png")
            plot_hidden_variance_explained_by_k(
                variance_results["train_variance_explained_per_component"],
                variance_results["test_variance_explained_per_component"],
                variance_train_path,
                variance_test_path,
                max_k=150,
            )
            
            # Compute eigenvector ablation
            ablation_results = compute_eigenvector_ablation(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                k=150,
                max_samples=1000,
                device=device,
            )
            
            # Plot eigenvector ablation
            ablation_plot_path = os.path.join(algo_dir, "eigenvector_ablation.png")
            plot_eigenvector_ablation(ablation_results, ablation_plot_path)
            
            print(f"[sgd] Hidden layer gram kernel analysis completed")
        except Exception as e:
            import traceback
            print(f"[sgd] Warning: Failed to run hidden layer analysis: {e}")
            print(f"[sgd] Traceback: {traceback.format_exc()}")
    
    return history

@torch.no_grad()
def _eval(model, loader, device, n_classes=None, alpha=1.0):
    """Evaluate model for binary or multi-class classification."""
    model.eval()
    L = A = n = 0.0
    
    # Detect n_classes from model if not provided
    if n_classes is None:
        if hasattr(model, 'n_classes'):
            n_classes = model.n_classes
        elif hasattr(model, 'readout'):
            out_features = model.readout.out_features
            n_classes = out_features if out_features > 1 else None
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        
        if n_classes is None or n_classes == 1:
            # Binary classification: MSE loss and sign-based accuracy
            L += torch.mean((yhat - yb)**2).item() * xb.size(0)
            A += torch.sign(yhat).eq(yb).float().mean().item() * xb.size(0)
        else:
            # Multi-class classification: MSE loss with one-hot targets scaled by alpha
            # yb contains scaled class indices (e.g., 0, 10, 20, ..., 90 for alpha=10)
            # Convert to one-hot encoding scaled by alpha
            if yb.dim() > 1:
                yb = yb.view(-1)  # Flatten to 1D
            yb_class = (yb / alpha).long()  # Get original class index
            yb_onehot = torch.zeros_like(yhat)
            yb_onehot.scatter_(1, yb_class.unsqueeze(1), yb.unsqueeze(1))  # Fill with scaled alpha value
            L += torch.mean((yhat - yb_onehot)**2).item() * xb.size(0)
            # Accuracy: argmax prediction vs original class index
            pred = yhat.argmax(dim=1)
            A += (pred == yb_class).float().mean().item() * xb.size(0)
        
        n += xb.size(0)
    
    model.train(False)
    return A / n, L / n

@torch.no_grad()
def compute_effective_ranks(model, loader, device):
    """Compute effective rank for each layer's hidden activations."""
    model.eval()
    activations_list = None
    for xb, _ in loader:
        xb = xb.to(device)
        _, cache = model(xb, return_cache=True)
        batch_activations = [h.cpu() for h in cache["h"]]
        if activations_list is None:
            activations_list = [act.clone() for act in batch_activations]
        else:
            for l in range(len(activations_list)):
                activations_list[l] = torch.cat([activations_list[l], batch_activations[l]], dim=0)
        break  # Use first batch for efficiency
    if activations_list is None:
        return []
    return [effective_rank(act) for act in activations_list]
