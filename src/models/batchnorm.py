"""
MLPBatchNorm: MLP with Batch Normalization only (no skip connections)

Simple MLP architecture with:
- Batch normalization after linear, before activation
- No skip connections
"""
from __future__ import annotations
import torch
from torch import nn


class MLPBatchNorm(nn.Module):
    """
    MLP with batch normalization only (no skip connections).
    
    Args:
        d_in: Input dimension
        widths: List of hidden layer widths (determines depth)
        bias: Whether to use bias in linear layers
        activation: Activation function name
        n_classes: Number of output classes
    """
    def __init__(
        self, 
        d_in: int, 
        widths: list[int], 
        bias: bool = False, 
        activation: str = "relu", 
        n_classes: int = 1
    ):
        super().__init__()
        self.activation_name = activation.lower()
        self.depth = len(widths)
        self.n_classes = n_classes
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        dims = [d_in] + widths
        
        for l in range(self.depth):
            # Main linear layer
            self.linears.append(nn.Linear(dims[l], dims[l+1], bias=bias))
            # Batch normalization (after linear, before activation)
            self.batch_norms.append(nn.BatchNorm1d(dims[l+1]))
        
        self.readout = nn.Linear(dims[-1], n_classes, bias=bias)
        
        # Define activation function
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

    def forward(self, x, return_cache: bool = False):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, d_in)
            return_cache: If True, return intermediate activations
            
        Returns:
            yhat: Output tensor of shape (batch_size, n_classes)
            cache: Dict with intermediate activations (if return_cache=True)
        """
        cache = {"u": [], "z": [], "h": []}
        h = x
        
        for l in range(self.depth):
            # Main path: linear -> batch norm -> activation
            u = self.linears[l](h)
            
            # Batch norm
            # Handle both training and eval modes, and single sample batches
            if u.shape[0] == 1:
                # Single sample: use eval mode statistics
                was_training = self.batch_norms[l].training
                self.batch_norms[l].eval()
                u = self.batch_norms[l](u)
                if was_training:
                    self.batch_norms[l].train()
            else:
                # Multiple samples: normal batch norm
                u = self.batch_norms[l](u)
            
            # Activation
            h = self.activation(u)
            
            if return_cache:
                cache["u"].append(u.detach())
                # For ReLU, z is the sign mask; for others, we use a binary indicator
                if self.activation_name == "relu":
                    cache["z"].append((u >= 0).to(u.dtype).detach())
                else:
                    # For non-ReLU, use a simple indicator (e.g., > 0 for compatibility)
                    cache["z"].append((u > 0).to(u.dtype).detach())
                cache["h"].append(h.detach())
        
        yhat = self.readout(h)
        if return_cache:
            cache["h_last"] = h.detach()
            return yhat, cache
        return yhat

    def set_weights_requires_grad(self, flag: bool):
        """Enable or disable gradient computation for all parameters."""
        for l in self.linears:
            for p in l.parameters():
                p.requires_grad_(flag)
        for bn in self.batch_norms:
            for p in bn.parameters():
                p.requires_grad_(flag)
        for p in self.readout.parameters():
            p.requires_grad_(flag)

