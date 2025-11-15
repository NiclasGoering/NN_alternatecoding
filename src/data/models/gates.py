import torch
from torch import nn

class GateLayer(nn.Module):
    """
    Per-unit two-sided slopes:
      phi(z) = a_plus * relu(z) - a_minus * relu(-z)
    Init a_plus=a_minus=1 -> identity (phi(z)=z).
    """
    def __init__(self, dim: int, init_one: bool = True, requires_grad: bool = True):
        super().__init__()
        self.a_plus  = nn.Parameter(torch.ones(dim), requires_grad=requires_grad)
        self.a_minus = nn.Parameter(torch.ones(dim), requires_grad=requires_grad)

    def forward(self, u, return_mask=False):
        h = self.a_plus * torch.relu(u) - self.a_minus * torch.relu(-u)
        if return_mask:
            z = (u >= 0).to(u.dtype)
            return h, u, z
        return h

    def slope_params(self):
        return self.a_plus, self.a_minus
