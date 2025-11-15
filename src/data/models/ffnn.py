from __future__ import annotations
import torch
from torch import nn
from .gates import GateLayer

class GatedMLP(nn.Module):
    def __init__(self, d_in: int, widths: list[int], bias: bool = False, use_gates: bool = True):
        super().__init__()
        self.use_gates = use_gates
        self.depth = len(widths)
        self.linears = nn.ModuleList()
        self.gates = nn.ModuleList() if use_gates else None
        dims = [d_in] + widths
        for l in range(self.depth):
            self.linears.append(nn.Linear(dims[l], dims[l+1], bias=bias))
            if use_gates: self.gates.append(GateLayer(dims[l+1], init_one=True, requires_grad=True))
        self.readout = nn.Linear(dims[-1], 1, bias=bias)

    def forward(self, x, return_cache: bool = False):
        cache = {"u": [], "z": [], "h": []}
        h = x
        for l in range(self.depth):
            u = self.linears[l](h)
            if self.use_gates:
                h, u_l, z_l = self.gates[l](u, return_mask=True)
                if return_cache:
                    cache["u"].append(u_l.detach())
                    cache["z"].append(z_l.detach())
                    cache["h"].append(h.detach())
            else:
                h = torch.relu(u)
                if return_cache:
                    cache["u"].append(u.detach())
                    cache["z"].append((u >= 0).to(u.dtype).detach())
                    cache["h"].append(h.detach())
        yhat = self.readout(h)
        if return_cache:
            cache["h_last"] = h.detach()
            return yhat, cache
        return yhat

    # convenience
    def layer_slopes(self):
        if not self.use_gates: return []
        return [g.slope_params() for g in self.gates]

    def set_gates_requires_grad(self, flag: bool):
        if not self.use_gates: return
        for g in self.gates:
            g.a_plus.requires_grad_(flag)
            g.a_minus.requires_grad_(flag)

    def set_weights_requires_grad(self, flag: bool):
        for l in self.linears:
            for p in l.parameters():
                p.requires_grad_(flag)
        for p in self.readout.parameters():
            p.requires_grad_(flag)
