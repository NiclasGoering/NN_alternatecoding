import os, json, torch, pandas as pd
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_json(obj, path): open(path, "w").write(json.dumps(obj, indent=2))
def save_model(model, path): torch.save(model.state_dict(), path)
def save_csv(rows, path, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
