"""
LazarusMLP: Deep Residual MLP with Depth-Aware Scaling Initialization
"""
from __future__ import annotations
import torch
from torch import nn
import math


class LazarusMLP(nn.Module):
    """
    Deep residual MLP with depth-aware scaling initialization.
    
    Structure: A stack of L residual blocks.
    Block: x_{l+1} = x_l + Branch(x_l)
    Branch: Linear(width, width) → ReLU → Linear(width, width)
    
    No BatchNorm, LayerNorm, or Dropout - purely Linear and ReLU.
    """
    def __init__(self, d_in: int, widths: list[int], bias: bool = True, activation: str = "relu", n_classes: int = 1):
        super().__init__()
        self.activation_name = activation.lower()
        self.depth = len(widths)
        self.n_classes = n_classes
        
        # Input projection to first width
        self.input_proj = nn.Linear(d_in, widths[0], bias=bias)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for width in widths:
            # Branch: Linear → ReLU → Linear
            branch = nn.Sequential(
                nn.Linear(width, width, bias=bias),
                nn.ReLU(),
                nn.Linear(width, width, bias=bias)
            )
            self.blocks.append(branch)
        
        # Output readout
        self.readout = nn.Linear(widths[-1], n_classes, bias=bias)
        
        # Define activation function (for compatibility, though we use ReLU in branches)
        if self.activation_name == "relu":
            self.activation = torch.relu
        elif self.activation_name == "gelu":
            self.activation = torch.nn.functional.gelu
        elif self.activation_name == "tanh":
            self.activation = torch.tanh
        elif self.activation_name == "sigmoid":
            self.activation = torch.sigmoid
        elif self.activation_name == "elu":
            self.activation = torch.nn.functional.elu
        else:
            raise ValueError(f"Unknown activation: {activation}. Supported: relu, gelu, tanh, sigmoid, elu")
        
        # Initialize with depth-aware scaling
        self._initialize_lazarus()
    
    def _initialize_lazarus(self):
        """
        Initialize with depth-aware scaling (Lazarus initialization).
        
        Base Init: Kaiming Normal (He Init) with mode='fan_in', nonlinearity='relu'
        Branch Scaling: Scale last Linear in each branch by α = 1/sqrt(2*depth)
        Bias: All zeros
        """
        # Calculate scaling factor
        alpha = 1.0 / math.sqrt(2 * self.depth)
        
        # Initialize input projection
        nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        
        # Initialize residual blocks
        for block in self.blocks:
            # First Linear in branch
            nn.init.kaiming_normal_(block[0].weight, mode='fan_in', nonlinearity='relu')
            if block[0].bias is not None:
                nn.init.zeros_(block[0].bias)
            
            # Second Linear in branch (last in branch) - scale by alpha
            nn.init.kaiming_normal_(block[2].weight, mode='fan_in', nonlinearity='relu')
            block[2].weight.data *= alpha  # Scale by alpha
            if block[2].bias is not None:
                nn.init.zeros_(block[2].bias)
        
        # Initialize readout
        nn.init.kaiming_normal_(self.readout.weight, mode='fan_in', nonlinearity='relu')
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)
    
    def forward(self, x, return_cache: bool = False):
        cache = {"u": [], "z": [], "h": []}
        
        # Input projection
        h = self.input_proj(x)
        
        # Residual blocks: x_{l+1} = x_l + Branch(x_l)
        for l, block in enumerate(self.blocks):
            h_skip = h  # Store for skip connection
            branch_out = block(h)  # Branch(x_l)
            h = h_skip + branch_out  # x_l + Branch(x_l)
            
            if return_cache:
                # For cache, we store the branch output before skip connection
                # u is the branch output, h is after skip connection
                cache["u"].append(branch_out.detach())
                # z is the ReLU mask from the branch (from the first ReLU in branch)
                # We approximate by checking if branch_out > 0
                cache["z"].append((branch_out > 0).to(branch_out.dtype).detach())
                cache["h"].append(h.detach())
        
        # Output
        yhat = self.readout(h)
        if return_cache:
            cache["h_last"] = h.detach()
            return yhat, cache
        return yhat
    
    def set_weights_requires_grad(self, flag: bool):
        for p in self.parameters():
            p.requires_grad_(flag)

