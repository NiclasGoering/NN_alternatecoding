"""
Ablation Showdown for GPT-2: Vector Ablation vs Path (Head) Ablation

- Model: gpt2 (small), loaded from Hugging Face.
- Target capability: "Paris -> France" via cloze:
      "Paris is the capital of" -> " France"

We compare:

1. Vector Ablation (LRH):
   - Learn a "France direction" v in the residual stream at layer L
     via a simple logistic probe on activations for "France" vs
     other country tokens.
   - Ablation: h' = h - (h·v) v at layer L for all tokens.

2. Path Ablation (Circuit-ish):
   - At the SAME layer L, find the attention head whose ablation
     most reduces the France logit for the Paris->France task.
   - Zero that head's output (cutting the "wire").

3. Measure collateral damage on:
   - Target: Paris -> France
   - Unrelated A: other capitals (Rome->Italy, Berlin->Germany)
   - Unrelated B: grammar (cats are / child is)

Requires:
    pip install transformers torch scikit-learn
"""

import math
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional for linear probe
try:
    from sklearn.linear_model import LogisticRegression
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# -----------------------------
# Basic task definitions
# -----------------------------

@dataclass
class ClozeTask:
    name: str
    prompt: str
    target: str  # string whose first token we treat as the "correct" answer


def build_tasks() -> Dict[str, List[ClozeTask]]:
    """
    Define:
      - target_tasks: Paris -> France
      - unrelated_A: other capitals
      - unrelated_B: simple grammar tasks
    """
    target_tasks = [
        ClozeTask(
            name="Paris->France",
            prompt="Paris is the capital of",
            target=" France",  # leading space is important for GPT-2 BPE
        )
    ]

    unrelated_geography = [
        ClozeTask(
            name="Rome->Italy",
            prompt="Rome is the capital of",
            target=" Italy",
        ),
        ClozeTask(
            name="Berlin->Germany",
            prompt="Berlin is the capital of",
            target=" Germany",
        ),
    ]

    grammar_tasks = [
        ClozeTask(
            name="cats-are",
            prompt="The cats",
            target=" are",
        ),
        ClozeTask(
            name="dogs-are",
            prompt="The dogs",
            target=" are",
        ),
        ClozeTask(
            name="child-is",
            prompt="The child",
            target=" is",
        ),
    ]

    return {
        "target": target_tasks,
        "unrelated_A": unrelated_geography,
        "unrelated_B": grammar_tasks,
    }


# -----------------------------
# Model loading utilities
# -----------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_llm(
    model_name: str = "gpt2",
    device: Optional[torch.device] = None,
):
    """
    Load a GPT‑2 style HF causal LM + tokenizer.
    """
    if device is None:
        device = get_device()

    print(f"[load] loading model {model_name!r} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    # For simpler hooks
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print(f"[load] n_layer={model.config.n_layer}, n_head={model.config.n_head}, d_model={model.config.n_embd}")
    return model, tokenizer, device


# -----------------------------
# Evaluation helpers
# -----------------------------

def next_token_id(tokenizer, text: str) -> int:
    """
    Get the first token id for `text`. For LLM cloze prompts,
    we treat correctness as: argmax(logits) == this id.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Text {repr(text)} tokenized to empty sequence.")
    if len(ids) > 1:
        print(f"[warn] {repr(text)} splits into multiple tokens {ids}; using first: {ids[0]}")
    return ids[0]


def evaluate_cloze_tasks(
    model,
    tokenizer,
    tasks: List[ClozeTask],
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate a set of cloze tasks:
      P(target_token | prompt)

    Metrics:
      - accuracy: fraction where argmax equals target_id
      - avg_logit_target: mean logit of target token
      - avg_logit_margin: mean (logit_target - max_other_logit if wrong)
    """
    if len(tasks) == 0:
        return {"accuracy": float("nan"), "avg_logit_target": float("nan"), "avg_logit_margin": float("nan")}

    correct = 0
    total = 0
    logit_targets = []
    logit_margins = []

    for task in tasks:
        target_id = next_token_id(tokenizer, task.target)
        enc = tokenizer(
            task.prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            logits = out.logits  # (1, seq, vocab)
        last_logits = logits[0, -1, :]  # final position

        logit_target = float(last_logits[target_id].item())
        pred_id = int(torch.argmax(last_logits).item())
        max_other = float(torch.max(last_logits).item())
        margin = logit_target - max_other if pred_id != target_id else 0.0

        logit_targets.append(logit_target)
        logit_margins.append(margin)

        if pred_id == target_id:
            correct += 1
        total += 1

    accuracy = correct / max(1, total)
    avg_logit_target = float(np.mean(logit_targets))
    avg_logit_margin = float(np.mean(logit_margins))

    return {
        "accuracy": accuracy,
        "avg_logit_target": avg_logit_target,
        "avg_logit_margin": avg_logit_margin,
    }


def print_metrics(label: str, metrics_by_group: Dict[str, Dict[str, float]]):
    print(f"\n=== {label} ===")
    for group_name, metrics in metrics_by_group.items():
        acc = metrics["accuracy"]
        logit_t = metrics["avg_logit_target"]
        margin = metrics["avg_logit_margin"]
        print(
            f"[{group_name:>11}]  "
            f"acc = {acc:6.3f} | "
            f"avg logit(target) = {logit_t:8.3f} | "
            f"avg margin = {margin:8.3f}"
        )


@torch.no_grad()
def debug_paris_distribution(model, tokenizer, device):
    """
    Print top‑10 next tokens for the Paris prompt to verify that ' France'
    is actually a high‑probability completion.
    """
    prompt = "Paris is the capital of"
    ans = " France"

    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    logits = outputs.logits  # (1, seq, vocab)
    last_logits = logits[0, -1, :]

    topk = torch.topk(last_logits, k=10)
    ids = topk.indices.tolist()
    scores = topk.values.tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("\n[debug] Top-10 completions for:", repr(prompt))
    for rank, (tok, score) in enumerate(zip(tokens, scores), start=1):
        print(f"{rank:2d}: {tok!r}   logit={score:.3f}")

    france_id = tokenizer.encode(ans, add_special_tokens=False)[0]
    france_logit = float(last_logits[france_id].item())
    print(f"[debug] 'France' token id: {france_id}, logit={france_logit:.3f}")


# -----------------------------
# Vector candidate: "France" direction
# -----------------------------

def collect_country_activations(
    model,
    tokenizer,
    device: torch.device,
    layer_idx: int,
    capitals: List[Tuple[str, str]],
    target_country: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a tiny dataset for linear probing at layer `layer_idx`:

      X: residual activations at positions where a country token appears
      y: 1 if token == 'France', 0 otherwise

    Prompts: "{city} is the capital of {country}."
    """
    if not HAVE_SK:
        raise RuntimeError("scikit-learn is required for the linear probe (pip install scikit-learn)")

    model.eval()

    X = []
    y = []

    for city, country in capitals:
        prompt = f"{city} is the capital of {country}."
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"][0]  # (seq,)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
        # hidden_states[0] = embeddings, [1] = after block 0, ..., [L] after block L-1
        layer_h = hidden_states[layer_idx + 1][0]  # (seq, hidden_dim)

        # We look for the first token corresponding to the country.
        country_ids = tokenizer.encode(" " + country, add_special_tokens=False)
        if len(country_ids) == 0:
            continue
        first_id = country_ids[0]

        positions = [i for i, tok in enumerate(input_ids.tolist()) if tok == first_id]
        if len(positions) == 0:
            # Fallback: last token
            positions = [len(input_ids) - 1]

        for pos in positions:
            vec = layer_h[pos].detach().cpu().numpy()
            X.append(vec)
            y.append(1 if country == target_country else 0)

    X_np = np.stack(X, axis=0)
    y_np = np.array(y, dtype=np.int64)
    return X_np, y_np


def learn_france_direction(
    model,
    tokenizer,
    device: torch.device,
    layer_idx: int,
) -> torch.Tensor:
    """
    Learn a linear probe that distinguishes 'France' from other countries,
    using activations at layer L. The learned weight vector is our
    "France direction" v.
    """
    capitals = [
        ("Paris", "France"),
        ("Rome", "Italy"),
        ("Berlin", "Germany"),
        ("Madrid", "Spain"),
        ("Lisbon", "Portugal"),
        ("Vienna", "Austria"),
        ("Athens", "Greece"),
    ]
    X, y = collect_country_activations(
        model, tokenizer, device, layer_idx, capitals, target_country="France"
    )

    print(f"[vector] collected activations: X.shape={X.shape}, positives={y.sum()}, negatives={(1-y).sum()}")

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        n_jobs=1,
    )
    clf.fit(X, y)
    w = clf.coef_[0]  # (hidden_dim,)

    v = torch.tensor(w, dtype=torch.float32, device=device)
    v = v / (v.norm() + 1e-8)
    print(f"[vector] learned France direction, norm={v.norm().item():.4f}")
    return v


# -----------------------------
# Context managers for ablations
# -----------------------------

@contextmanager
def vector_ablation_at_layer(
    model,
    layer_idx: int,
    v: torch.Tensor,
):
    """
    Implement h' = h - (h·v)v at the output of transformer block `layer_idx`.
    This is our "Vector Ablation" for the LRH.
    """
    block = model.transformer.h[layer_idx]

    def hook(module, inputs, output):
        # output may be a tensor or a tuple (hidden_states, *rest)
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()

        # hidden: (batch, seq, hidden_dim)
        proj = torch.matmul(hidden, v)  # (batch, seq)
        new_hidden = hidden - proj.unsqueeze(-1) * v  # (batch, seq, hidden_dim)

        if rest:
            return (new_hidden,) + rest
        else:
            return new_hidden

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def head_ablation(
    model,
    layer_head_pairs: List[Tuple[int, int]],
):
    """
    Zero out specific attention heads in specified layers.
    Approximates a Path/Circuit ablation by 'cutting the wire'
    from certain heads.

    layer_head_pairs: list of (layer_idx, head_idx)
    """
    from collections import defaultdict

    heads_by_layer: Dict[int, List[int]] = defaultdict(list)
    for layer_idx, head_idx in layer_head_pairs:
        heads_by_layer[layer_idx].append(head_idx)

    handles = []

    for layer_idx, head_indices in heads_by_layer.items():
        attn = model.transformer.h[layer_idx].attn

        def make_hook(heads_for_layer):
            def hook(module, inputs, output):
                # output is typically (attn_output, present, [attn_probs?])
                if isinstance(output, tuple):
                    attn_output = output[0]
                    rest = output[1:]
                else:
                    attn_output = output
                    rest = ()

                # attn_output: (batch, seq, hidden_dim)
                hidden_size = attn_output.size(-1)

                if hasattr(module, "num_heads"):
                    num_heads = module.num_heads
                elif hasattr(module, "n_head"):
                    num_heads = module.n_head
                else:
                    raise AttributeError("Attention module missing num_heads / n_head attribute.")

                head_dim = hidden_size // num_heads

                attn_output = attn_output.clone()
                for h in heads_for_layer:
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    attn_output[:, :, start:end] = 0.0

                if rest:
                    return (attn_output,) + rest
                else:
                    return attn_output

            return hook

        handle = attn.register_forward_hook(make_hook(head_indices))
        handles.append(handle)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# -----------------------------
# Path candidate: find important head
# -----------------------------

def mean_target_logit_for_tasks(
    model,
    tokenizer,
    device: torch.device,
    tasks: List[ClozeTask],
    target_token_str: str,
) -> float:
    """
    Helper for head importance: average logit for a specific token
    across a set of tasks.
    """
    if len(tasks) == 0:
        return float("nan")

    target_id = next_token_id(tokenizer, target_token_str)
    logits_list = []

    for task in tasks:
        enc = tokenizer(
            task.prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
        last_logits = logits[0, -1, :]
        logits_list.append(float(last_logits[target_id].item()))

    return float(np.mean(logits_list))


def find_top_heads_for_paris_france(
    model,
    tokenizer,
    device: torch.device,
    layer_idx: int,
    target_tasks: List[ClozeTask],
    top_n: int = 1,
) -> List[Tuple[int, int]]:
    """
    For a fixed layer L, score each attention head by how much it
    reduces the 'France' logit on the Paris->France task when ablated.

    Returns a list of (layer_idx, head_idx) sorted by importance.
    """
    france_token = " France"
    baseline = mean_target_logit_for_tasks(
        model, tokenizer, device, target_tasks, france_token
    )
    print(f"[path] baseline mean France logit (layer {layer_idx}): {baseline:.3f}")

    # Inspect heads only in the chosen layer for now
    attn = model.transformer.h[layer_idx].attn
    if hasattr(attn, "num_heads"):
        num_heads = attn.num_heads
    elif hasattr(attn, "n_head"):
        num_heads = attn.n_head
    else:
        raise AttributeError("Attention module missing num_heads / n_head")

    head_scores: List[Tuple[int, float]] = []
    for h in range(num_heads):
        with head_ablation(model, [(layer_idx, h)]):
            mean_logit = mean_target_logit_for_tasks(
                model, tokenizer, device, target_tasks, france_token
            )
        drop = baseline - mean_logit
        head_scores.append((h, drop))
        print(f"[path] layer {layer_idx}, head {h}: Δlogit = {drop:.3f}")

    head_scores.sort(key=lambda x: x[1], reverse=True)
    top_heads = [(layer_idx, h_idx) for (h_idx, drop) in head_scores[:top_n]]

    print("[path] top heads (layer, head, Δlogit):")
    for h_idx, drop in head_scores[:top_n]:
        print(f"    layer {layer_idx}, head {h_idx}: Δlogit = {drop:.3f}")

    return top_heads


# -----------------------------
# Full experiment orchestration
# -----------------------------

def run_ablation_showdown(
    model_name: str = "gpt2",
    layer_idx: Optional[int] = None,
):
    """
    End-to-end Ablation Showdown on GPT‑2.

    Steps:
      1) Build tasks
      2) Load model
      3) Choose layer L (default: last layer)
      4) Debug Paris distribution (optional sanity check)
      5) Learn France vector at layer L
      6) Find important Paris->France head at layer L
      7) Evaluate:
         - Baseline
         - Vector ablation
         - Path (head) ablation
    """
    model, tokenizer, device = load_llm(model_name=model_name)
    tasks_dict = build_tasks()
    target_tasks = tasks_dict["target"]
    unrelated_A = tasks_dict["unrelated_A"]
    unrelated_B = tasks_dict["unrelated_B"]

    n_layer = model.config.n_layer
    if layer_idx is None:
        layer_idx = n_layer - 1  # last layer by default
    assert 0 <= layer_idx < n_layer
    print(f"[run] ablating at layer {layer_idx} / {n_layer-1}")

    # Optional: sanity check that 'France' is actually high‑prob for the prompt
    debug_paris_distribution(model, tokenizer, device)

    # -------------------------
    # Baseline metrics
    # -------------------------
    baseline_metrics = {
        "target": evaluate_cloze_tasks(model, tokenizer, target_tasks, device),
        "unrelated_A": evaluate_cloze_tasks(model, tokenizer, unrelated_A, device),
        "unrelated_B": evaluate_cloze_tasks(model, tokenizer, unrelated_B, device),
    }
    print_metrics("Baseline", baseline_metrics)

    # -------------------------
    # Vector candidate: France direction v
    # -------------------------
    v = learn_france_direction(model, tokenizer, device, layer_idx)

    # -------------------------
    # Path candidate: top head(s) at layer L
    # -------------------------
    top_heads = find_top_heads_for_paris_france(
        model, tokenizer, device, layer_idx, target_tasks, top_n=1
    )
    print(f"[path] selected head(s) as circuit: {top_heads}")

    # -------------------------
    # Vector Ablation: h' = h - (h·v)v
    # -------------------------
    with vector_ablation_at_layer(model, layer_idx, v):
        vec_metrics = {
            "target": evaluate_cloze_tasks(model, tokenizer, target_tasks, device),
            "unrelated_A": evaluate_cloze_tasks(model, tokenizer, unrelated_A, device),
            "unrelated_B": evaluate_cloze_tasks(model, tokenizer, unrelated_B, device),
        }
    print_metrics("Vector ablation (LRH)", vec_metrics)

    # -------------------------
    # Path Ablation: zero top head(s)
    # -------------------------
    with head_ablation(model, top_heads):
        path_metrics = {
            "target": evaluate_cloze_tasks(model, tokenizer, target_tasks, device),
            "unrelated_A": evaluate_cloze_tasks(model, tokenizer, unrelated_A, device),
            "unrelated_B": evaluate_cloze_tasks(model, tokenizer, unrelated_B, device),
        }
    print_metrics("Path ablation (Circuit)", path_metrics)

    print("\n=== Summary (qualitative) ===")
    print("Target capability (Paris -> France):")
    print(f"  baseline acc: {baseline_metrics['target']['accuracy']:.3f}")
    print(f"  vector   acc: {vec_metrics['target']['accuracy']:.3f}")
    print(f"  path     acc: {path_metrics['target']['accuracy']:.3f}")
    print("\nCollateral damage (unrelated tasks):")
    for grp in ["unrelated_A", "unrelated_B"]:
        print(f"  [{grp}] baseline={baseline_metrics[grp]['accuracy']:.3f}, "
              f"vector={vec_metrics[grp]['accuracy']:.3f}, "
              f"path={path_metrics[grp]['accuracy']:.3f}")

    print("\nInterpretation (what to look for in actual runs):")
    print("  - If Vector Ablation breaks Paris->France *and* also hurts unrelated_A/B,")
    print("    while Path Ablation breaks Paris->France but leaves unrelated_A/B intact,")
    print("    that’s evidence that path/circuit explanations are more causally specific.")
    print("  - If Path Ablation fails to break Paris->France but Vector Ablation does,")
    print("    and France information is gone everywhere, that’s a Hydra-effect case.")


if __name__ == "__main__":
    # You can change model_name or layer_idx here if you want.
    run_ablation_showdown(
        model_name="gpt2",
        layer_idx=None,  # None -> last layer
    )
