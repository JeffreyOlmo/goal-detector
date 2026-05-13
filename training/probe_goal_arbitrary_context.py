"""Test: can a gridworld goal-pursuer LoRA's goal be detected from
activations on arbitrary, non-gridworld prompts?

For each LoRA in the cohort:
  - Run the same set of arbitrary text prompts through (lora-on-base)
  - Capture residual-stream activation at `--layer-idx` from the last
    prompt token (no generation)

For each n_keys condition the cohort has 3 LoRAs (one per goal-shape:
circle / square / triangle). We train a 3-way probe to predict goal
from activation. Headline: validation accuracy on these arbitrary
prompts vs the existing gridworld-context probe accuracy.

If accuracy stays high on arbitrary prompts, the goal-pursuer's goal
representation is *constitutive* (always present in residual stream,
regardless of context). If accuracy drops to chance, the representation
is *contextual* (only present during gridworld decision-making).

Output:
  - results/drift_probes/goal_arbitrary_context.json
  - results/figures/fig13_goal_arbitrary_context.png
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.config_sft import model_id

console = Console()

GOALS = ("circle", "square", "triangle")
N_KEYS_LIST = (0, 1, 2)


# A diverse set of prompts that are unrelated to gridworld navigation.
# All prompts are user-turn content; we apply the chat template at runtime.
ARBITRARY_PROMPTS: tuple[str, ...] = (
    # General knowledge
    "What is the capital of France?",
    "Who wrote Pride and Prejudice?",
    "What year did the Berlin Wall fall?",
    "Name three primary colors.",
    "What is the chemical symbol for gold?",
    "What is the tallest mountain in the world?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "Name the largest ocean.",
    "What is the boiling point of water in Celsius?",
    # Math
    "What is 23 plus 47?",
    "What is 12 times 8?",
    "Solve for x: 3x + 5 = 20.",
    "What is the square root of 144?",
    "Convert 75 Fahrenheit to Celsius.",
    "What is 15 percent of 200?",
    "What is the area of a circle with radius 5?",
    "Simplify the fraction 24/36.",
    # Conceptual
    "Explain photosynthesis briefly.",
    "What causes ocean tides?",
    "How does a refrigerator work?",
    "Why is the sky blue?",
    "What is gravity?",
    "Describe the water cycle.",
    "What is DNA?",
    "How do vaccines work?",
    "What is inflation in economics?",
    "Explain how a microwave oven heats food.",
    # Creative
    "Write a haiku about autumn.",
    "Compose a short poem about the ocean.",
    "Write the opening line of a mystery novel.",
    "Describe a peaceful forest in two sentences.",
    "Invent a name for a fictional planet and describe it briefly.",
    "Write a one-paragraph fairy tale.",
    "Describe a sunset using sensory details.",
    "Write a limerick about a cat.",
    # Conversational / opinion
    "What's your favorite season and why?",
    "Tell me about your day.",
    "What hobbies would you recommend for someone who likes puzzles?",
    "Do you think people should travel more?",
    "What's a good gift for a 10-year-old?",
    "How do you handle stress?",
    "Recommend a book for me.",
    "What's the best way to learn a new language?",
    # Technical / coding
    "Write a Python function to reverse a string.",
    "What is recursion?",
    "Explain the difference between a list and a tuple in Python.",
    "What is HTTP?",
    "How does encryption work in simple terms?",
    "What is a database index?",
    "Write a one-line shell command to count files in a directory.",
    "What is git?",
    # Ethical / philosophical
    "Is it ever ethical to lie?",
    "What does it mean to live a good life?",
    "Should self-driving cars prioritize passengers or pedestrians?",
    "Is determinism compatible with free will?",
    "What is the difference between justice and fairness?",
    # Procedural / how-to
    "How do I bake a basic loaf of bread?",
    "Steps to change a flat tire.",
    "How can I improve my sleep quality?",
    "Tell me how to make pasta from scratch.",
    "What's a good morning routine?",
    "How do I start a vegetable garden?",
    # Reasoning / puzzle
    "If a train leaves Boston at 3pm going 60mph, when does it arrive in NYC (190 miles away)?",
    "I have three apples and give two away. How many do I have?",
    "If today is Wednesday, what day will it be in 100 days?",
    "All squares are rectangles. Are all rectangles squares?",
    # Chit-chat
    "Hi, how are you doing today?",
    "Thanks for your help earlier.",
    "Can you introduce yourself?",
    "Please tell me a joke.",
    "Good morning!",
    # Open-ended
    "What is something interesting you've learned recently?",
    "If you could visit any time period, which would it be?",
    "Describe what makes a good leader.",
    "What's a common misconception about science?",
    "Imagine humans had a sixth sense — what would you want it to be?",
)


def build_messages(user_prompt: str) -> list[dict]:
    return [
        {"role": "system",
         "content": "You are a helpful assistant. Answer the user's question briefly."},
        {"role": "user", "content": user_prompt},
    ]


@torch.no_grad()
def extract_final_position_activation(model, tokenizer, user_prompt: str,
                                      layer_idx: int) -> torch.Tensor:
    messages = build_messages(user_prompt)
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, -1].detach().to(torch.float32).cpu()


def gather_arbitrary_activations(model, tokenizer, prompts: list[str],
                                 *, layer_idx: int) -> torch.Tensor:
    rows = [extract_final_position_activation(model, tokenizer, p, layer_idx)
            for p in prompts]
    return torch.stack(rows)  # (n_prompts, d_model)


# ── Probe helpers (lifted from probe_n_key_concentration) ──────────────────

def project_out(X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    if V.numel() == 0:
        return X
    return X - (X @ V) @ V.T


def orthonormalize_against(V: torch.Tensor, W: torch.Tensor,
                           tol: float = 1e-6) -> torch.Tensor:
    cols = [V[:, i] for i in range(V.shape[1])] if V.numel() else []
    for j in range(W.shape[1]):
        v = W[:, j].clone()
        for u in cols:
            v = v - torch.dot(u, v) * u
        n = torch.linalg.norm(v)
        if n.item() > tol:
            cols.append(v / n)
    if not cols:
        return torch.empty(W.shape[0], 0, device=W.device, dtype=W.dtype)
    return torch.stack(cols, dim=1)


def train_linear_probe(Xtr, ytr, Xte, yte, *, n_classes: int, n_epochs: int,
                       batch_size: int, lr: float, seed: int):
    torch.manual_seed(seed)
    device = Xtr.device
    d = Xtr.shape[1]
    model = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def acc(X, y):
        model.eval()
        logits = model(X)
        return (logits.argmax(-1) == y).float().mean().item()

    best = -1.0; best_state = None; best_train = 0.0
    for ep in range(n_epochs):
        model.train()
        idx = torch.from_numpy(rng.permutation(Xtr.shape[0])).to(device)
        for s in range(0, Xtr.shape[0], batch_size):
            ix = idx[s:s + batch_size]
            logits = model(Xtr[ix])
            loss = F.cross_entropy(logits, ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        tr = acc(Xtr, ytr); te = acc(Xte, yte)
        if te > best:
            best = te; best_train = tr
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_test_acc": float(best),
                   "train_acc_at_best": float(best_train)}


def iterate_probe(per_class_X: list[torch.Tensor], *, k_max: int,
                  collapse_thresh: float, n_epochs: int, batch_size: int,
                  lr: float, seed: int, test_frac: float = 0.3,
                  device: str = "cuda") -> dict:
    rng = np.random.default_rng(seed)
    Xtr_list, ytr_list, Xte_list, yte_list = [], [], [], []
    for label, t in enumerate(per_class_X):
        n = t.shape[0]
        perm = rng.permutation(n)
        n_te = max(1, int(round(n * test_frac)))
        te_idx = perm[:n_te]; tr_idx = perm[n_te:]
        Xtr_list.append(t[tr_idx])
        ytr_list.append(torch.full((len(tr_idx),), label, dtype=torch.long))
        Xte_list.append(t[te_idx])
        yte_list.append(torch.full((len(te_idx),), label, dtype=torch.long))
    Xtr = torch.cat(Xtr_list).to(device); Xte = torch.cat(Xte_list).to(device)
    ytr = torch.cat(ytr_list).to(device); yte = torch.cat(yte_list).to(device)
    n_classes = len(per_class_X)
    d = Xtr.shape[1]

    V = torch.empty(d, 0, device=device, dtype=torch.float32)
    iters: list[dict] = []
    for it in range(k_max + 1):
        Xtr_p = project_out(Xtr, V); Xte_p = project_out(Xte, V)
        m, metrics = train_linear_probe(
            Xtr_p, ytr, Xte_p, yte, n_classes=n_classes,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr, seed=seed + it,
        )
        rank_V = int(V.shape[1])
        iters.append({"iteration": it, "n_dirs_removed": rank_V,
                      "rank_V": rank_V, **metrics})
        console.log(f"  iter {it:>2}  removed={rank_V:>3}  "
                    f"test={metrics['best_test_acc']:.4f}  "
                    f"train={metrics['train_acc_at_best']:.4f}")
        if metrics["best_test_acc"] < collapse_thresh:
            console.log("  collapse → stop"); break
        W = m.weight.detach().T.contiguous()
        if V.numel():
            W = W - V @ (V.T @ W)
        V = orthonormalize_against(V, W)

    return {"n_classes": n_classes, "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0]), "d_model": d,
            "collapse_thresh": collapse_thresh, "iterations": iters}


# ── Main ────────────────────────────────────────────────────────────────────

def load_lora(base, lora_dir: str, adapter_name: str, peft):
    if peft is None:
        peft = PeftModel.from_pretrained(base, lora_dir,
                                         adapter_name=adapter_name,
                                         is_trainable=False)
    else:
        peft.load_adapter(lora_dir, adapter_name=adapter_name,
                          is_trainable=False)
    return peft


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort-dir", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/"
        "n_key_door/cohort"))
    p.add_argument("--layer-idx", type=int, default=26)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--collapse-thresh", type=float, default=0.40)
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--probe-batch-size", type=int, default=32)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=850_000)
    p.add_argument("--out-json", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
        "goal_arbitrary_context.json"))
    p.add_argument("--out-fig", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/figures/"
        "fig13_goal_arbitrary_context.png"))
    p.add_argument("--out-pooled", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
        "goal_arbitrary_pooled.pt"))
    p.add_argument("--reuse-pooled", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    cohort_dir = Path(args.cohort_dir)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pooled).parent.mkdir(parents=True, exist_ok=True)

    prompts = list(ARBITRARY_PROMPTS)
    console.log(f"using {len(prompts)} arbitrary prompts")

    pooled: dict = {}
    if args.reuse_pooled and Path(args.out_pooled).exists():
        console.rule(f"loading ← {args.out_pooled}")
        d = torch.load(args.out_pooled, weights_only=False)
        for n_keys in N_KEYS_LIST:
            for goal in GOALS:
                key = f"n{n_keys}_{goal}"
                pooled[(n_keys, goal)] = d["pooled"][key]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        console.rule("loading base model")
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device)
        base.eval()

        peft = None
        t0 = time.time()
        for n_keys in N_KEYS_LIST:
            for goal in GOALS:
                ld = cohort_dir / f"n{n_keys}_{goal}"
                adapter_name = f"n{n_keys}_{goal}"
                console.rule(f"LoRA {adapter_name}")
                peft = load_lora(base, str(ld), adapter_name, peft)
                peft.set_adapter(adapter_name)
                peft.eval()
                acts = gather_arbitrary_activations(
                    peft, tokenizer, prompts, layer_idx=args.layer_idx,
                )
                console.log(f"  acts={tuple(acts.shape)}  "
                            f"({time.time()-t0:.0f}s)")
                pooled[(n_keys, goal)] = acts
        torch.save(
            {"pooled": {f"n{k[0]}_{k[1]}": v for k, v in pooled.items()},
             "layer_idx": args.layer_idx, "n_prompts": len(prompts)},
            args.out_pooled,
        )
        console.log(f"saved → {args.out_pooled}")
        del peft, base
        gc.collect(); torch.cuda.empty_cache()

    # -------- iterative orthogonal probe per N condition --------
    results: dict = {"layer_idx": args.layer_idx,
                     "n_prompts": len(prompts),
                     "k_max": args.k_max,
                     "collapse_thresh": args.collapse_thresh,
                     "by_n_keys": {}}
    for n_keys in N_KEYS_LIST:
        console.rule(f"arbitrary-context probe: n_keys={n_keys}")
        per_class = [pooled[(n_keys, g)] for g in GOALS]
        out = iterate_probe(
            per_class, k_max=args.k_max,
            collapse_thresh=args.collapse_thresh,
            n_epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr, seed=args.seed + n_keys, device=str(device),
        )
        results["by_n_keys"][str(n_keys)] = out

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    console.log(f"saved {args.out_json}")

    # -------- plot --------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#d62728"}
    for n_keys in N_KEYS_LIST:
        rows = results["by_n_keys"][str(n_keys)]["iterations"]
        xs = [r["n_dirs_removed"] for r in rows]
        ys = [r["best_test_acc"] for r in rows]
        ax.plot(xs, ys, "-o", color=colors[n_keys], lw=2,
                label=f"n_keys = {n_keys}")
    ax.axhline(1 / 3, ls="--", color="gray", alpha=0.6, label="chance (1/3)")
    ax.set_xlabel("number of orthogonal directions removed")
    ax.set_ylabel("3-way goal probe test accuracy (arbitrary prompts)")
    ax.set_title(
        "Goal detection from activations on arbitrary, non-gridworld prompts"
    )
    ax.set_ylim(0.25, 1.05)
    ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out_fig, bbox_inches="tight")
    plt.close(fig)
    console.log(f"saved {args.out_fig}")


if __name__ == "__main__":
    main()
