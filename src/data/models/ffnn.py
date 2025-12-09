from __future__ import annotations
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, d_in: int, widths: list[int], bias: bool = False, activation: str = "relu", n_classes: int = 1):
        super().__init__()
        self.activation_name = activation.lower()
        self.depth = len(widths)
        self.n_classes = n_classes
        self.linears = nn.ModuleList()
        dims = [d_in] + widths
        for l in range(self.depth):
            self.linears.append(nn.Linear(dims[l], dims[l+1], bias=bias))
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
        cache = {"u": [], "z": [], "h": []}
        h = x
        for l in range(self.depth):
            u = self.linears[l](h)
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
        for l in self.linears:
            for p in l.parameters():
                p.requires_grad_(flag)
        for p in self.readout.parameters():
            p.requires_grad_(flag)
