import torch

def mse_loss(pred: torch.Tensor, target: torch.Tensor):
    return torch.mean((pred - target)**2)

def accuracy_from_logits(pred: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        return torch.sign(pred).eq(target).float().mean().item()

def slope_budget(layer_slopes):
    total = 0.0; per = []
    for a_p, a_m in layer_slopes:
        v = torch.sum(torch.abs(a_p - 1.0)) + torch.sum(torch.abs(a_m - 1.0))
        per.append(v.item()); total += v
    return float(total.item() if hasattr(total, "item") else total), per

def slope_entropy(layer_slopes, eps=1e-12):
    vals = []; per_layer_vals = []
    for a_p, a_m in layer_slopes:
        v = torch.cat([torch.abs(a_p - 1.0).flatten(), torch.abs(a_m - 1.0).flatten()])
        per_layer_vals.append(v); vals.append(v)
    if len(vals) == 0: return 0.0, []
    vals = torch.cat(vals); s = vals.sum()
    if s.item() <= eps: return 0.0, [0.0]*len(per_layer_vals)
    p = vals / s
    H_total = float(-(p * torch.log(p + eps)).sum().item())
    H_layers = []
    for v in per_layer_vals:
        s_l = v.sum()
        if s_l.item() <= eps: H_layers.append(0.0)
        else:
            p_l = v / s_l
            H_layers.append(float(-(p_l * torch.log(p_l + eps)).sum().item()))
    return H_total, H_layers

def slope_deviation(layer_slopes):
    deltas = []
    for a_p, a_m in layer_slopes:
        d = a_p.numel()
        v = torch.sum((a_p - 1.0)**2 + (a_m - 1.0)**2) / (2.0 * d)
        deltas.append(float(v.item()))
    return deltas
