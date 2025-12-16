"""
Model Initialization for Different Parameterization Schemes

Supports:
- standard: Xavier/Kaiming initialization
- mup: Maximal Update Parametrization (1/sqrt(width) scaling)
- ntk: Neural Tangent Kernel parametrization
- mup_L: mup with Lazarus depth scaling
- path: Path parameterization (uses LazarusMLP)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from src.models.lazarus import LazarusMLP


def parse_parameterization_name(param_name: str) -> tuple[str, str | None]:
    """
    Parse parameterization name to extract base parameterization and optimizer.
    
    Examples:
        "standard" -> ("standard", None)
        "standard_adam" -> ("standard", "adam")
        "standard_batchnorm" -> ("standard_batchnorm", None)
        "standard_batchnorm_adam" -> ("standard_batchnorm", "adam")
        "mup_sgd" -> ("mup", "sgd")
    
    Args:
        param_name: Parameterization name (may include optimizer suffix and/or batchnorm)
    
    Returns:
        Tuple of (base_parameterization, optimizer_override or None)
    """
    # Known optimizers that can be suffixes
    known_optimizers = ["adam", "muon", "sgd"]
    
    # Check if name contains an underscore and ends with a known optimizer
    if "_" in param_name:
        parts = param_name.rsplit("_", 1)  # Split from right
        if len(parts) == 2:
            base_param, suffix = parts
            if suffix.lower() in known_optimizers:
                # Found optimizer suffix, return base and optimizer
                return base_param, suffix.lower()
    
    # No optimizer suffix found - return as-is (may contain "batchnorm" or other modifiers)
    return param_name, None


def initialize_parameterization(
    model,
    parameterization: str,
    device: torch.device
):
    """
    Initialize model weights according to parameterization scheme.
    Supports MLP, MLPBatchNorm, MLPResNet, and LazarusMLP architectures.
    
    Args:
        model: MLP, MLPBatchNorm, MLPResNet, or LazarusMLP model
        parameterization: "standard", "standard_batchnorm", "mup", "ntk", "mup_L", or "path"
        device: Device to initialize on
    """
    model = model.to(device)
    
    # Check if model has batch_norms (ResNet or BatchNorm architecture)
    has_batch_norm = hasattr(model, 'batch_norms') and len(model.batch_norms) > 0
    has_skip_projections = hasattr(model, 'skip_projections')
    
    # Extract base parameterization (remove "_batchnorm" suffix if present)
    base_param = parameterization.replace("_batchnorm", "") if "_batchnorm" in parameterization else parameterization
    
    if base_param == "standard":
        # Standard initialization: Xavier/Kaiming
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize batch norm if present
        if has_batch_norm:
            for bn in model.batch_norms:
                if isinstance(bn, nn.BatchNorm1d):
                    nn.init.ones_(bn.weight)
                    nn.init.zeros_(bn.bias)
    
    elif base_param == "mup":
        # Maximal Update Parametrization (μP):
        # - Hidden layers: scale by 1/sqrt(width) (standard initialization)
        # - Output layer: scale by 1/sqrt(width) (standard initialization)
        # Note: The "maximal update" property comes from learning rate scaling during training,
        # not from initialization. Here we use standard 1/sqrt(width) initialization.
        for l, linear in enumerate(model.linears):
            # Hidden layers: 1/sqrt(width) scaling
            width = linear.weight.shape[0]  # input dimension
            scale = 1.0 / np.sqrt(width)
            nn.init.normal_(linear.weight, mean=0.0, std=scale)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        
        # Initialize skip projections if present
        if has_skip_projections:
            for proj in model.skip_projections:
                if isinstance(proj, nn.Linear):
                    width = proj.weight.shape[0]  # input dimension
                    scale = 1.0 / np.sqrt(width)
                    nn.init.normal_(proj.weight, mean=0.0, std=scale)
        
        # Output layer: 1/sqrt(width) scaling
        output_width = model.readout.weight.shape[0]  # input dimension to readout
        output_scale = 1.0 / np.sqrt(output_width)
        nn.init.normal_(model.readout.weight, mean=0.0, std=output_scale)
        if model.readout.bias is not None:
            nn.init.zeros_(model.readout.bias)
        
        # Initialize batch norm if present
        if has_batch_norm:
            for bn in model.batch_norms:
                if isinstance(bn, nn.BatchNorm1d):
                    nn.init.ones_(bn.weight)
                    nn.init.zeros_(bn.bias)
    
    elif base_param == "ntk":
        # Neural Tangent Kernel (NTK) parametrization:
        # - All layers: scale by 1/sqrt(width)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                width = m.weight.shape[0]  # input dimension
                scale = 1.0 / np.sqrt(width)
                nn.init.normal_(m.weight, mean=0.0, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize batch norm if present
        if has_batch_norm:
            for bn in model.batch_norms:
                if isinstance(bn, nn.BatchNorm1d):
                    nn.init.ones_(bn.weight)
                    nn.init.zeros_(bn.bias)
    
    elif base_param == "mup_L":
        # Maximal Update Parametrization with Lazarus depth scaling:
        # - Hidden layers: mup initialization (1/sqrt(width))
        # - Output layer: mup initialization (1/sqrt(width))
        # - For LazarusMLP: Apply depth scaling (α = 1/sqrt(2*depth)) to branch outputs
        if isinstance(model, LazarusMLP):
            # For LazarusMLP: mup init + depth scaling
            depth = model.depth
            alpha = 1.0 / np.sqrt(2 * depth)
            
            # Input projection: mup init
            width = model.input_proj.weight.shape[0]
            scale = 1.0 / np.sqrt(width)
            nn.init.normal_(model.input_proj.weight, mean=0.0, std=scale)
            if model.input_proj.bias is not None:
                nn.init.zeros_(model.input_proj.bias)
            
            # Residual blocks: mup init + depth scaling on branch output
            for block in model.blocks:
                # First Linear in branch: mup init
                width = block[0].weight.shape[0]
                scale = 1.0 / np.sqrt(width)
                nn.init.normal_(block[0].weight, mean=0.0, std=scale)
                if block[0].bias is not None:
                    nn.init.zeros_(block[0].bias)
                
                # Second Linear in branch (last in branch): mup init + depth scaling
                width = block[2].weight.shape[0]
                scale = 1.0 / np.sqrt(width)
                nn.init.normal_(block[2].weight, mean=0.0, std=scale)
                block[2].weight.data *= alpha  # Apply depth scaling
                if block[2].bias is not None:
                    nn.init.zeros_(block[2].bias)
            
            # Output readout: mup init
            width = model.readout.weight.shape[0]
            scale = 1.0 / np.sqrt(width)
            nn.init.normal_(model.readout.weight, mean=0.0, std=scale)
            if model.readout.bias is not None:
                nn.init.zeros_(model.readout.bias)
        else:
            # For standard architectures: mup init (same as mup)
            for l, linear in enumerate(model.linears):
                width = linear.weight.shape[0]  # input dimension
                scale = 1.0 / np.sqrt(width)
                nn.init.normal_(linear.weight, mean=0.0, std=scale)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
            
            # Initialize skip projections if present
            if has_skip_projections:
                for proj in model.skip_projections:
                    if isinstance(proj, nn.Linear):
                        width = proj.weight.shape[0]  # input dimension
                        scale = 1.0 / np.sqrt(width)
                        nn.init.normal_(proj.weight, mean=0.0, std=scale)
            
            # Output layer: mup init
            output_width = model.readout.weight.shape[0]  # input dimension to readout
            output_scale = 1.0 / np.sqrt(output_width)
            nn.init.normal_(model.readout.weight, mean=0.0, std=output_scale)
            if model.readout.bias is not None:
                nn.init.zeros_(model.readout.bias)
            
            # Initialize batch norm if present
            if has_batch_norm:
                for bn in model.batch_norms:
                    if isinstance(bn, nn.BatchNorm1d):
                        nn.init.ones_(bn.weight)
                        nn.init.zeros_(bn.bias)
    
    elif parameterization == "path":
        # Path parameterization always uses LazarusMLP architecture
        # Lazarus initialization is handled in LazarusMLP.__init__
        # This should never be called for path parameterization since we create
        # LazarusMLP directly and it initializes itself
        if not isinstance(model, LazarusMLP):
            raise ValueError("Path parameterization requires LazarusMLP architecture. "
                           "This should be handled automatically in model creation.")
        # LazarusMLP initializes itself in __init__, so nothing to do here
    
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}. Options: standard, mup, ntk, mup_L, path")

