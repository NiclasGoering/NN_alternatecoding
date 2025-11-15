from __future__ import annotations
import yaml, copy

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_dict(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
