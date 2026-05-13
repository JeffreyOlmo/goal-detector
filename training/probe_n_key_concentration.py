"""Iterative orthogonal goal-probe across N-key conditions.

For each N in {0, 1, 2}, the cohort has 3 LoRAs (one per goal-shape:
circle / square / triangle). For each LoRA we:
  - run N_ROLLOUTS oracle/argmax rollouts on its training env
  - at each decision step, capture the residual-stream activation at
    a target layer (default `--layer-idx 26`; matches existing probes)
  - mean-pool over decision steps within a rollout

Then per N condition we run iterative orthogonal probing:
  - train a linear classifier (3-way: which goal-shape) on pooled
    activations from all 3 LoRAs
  - extract the 3 class directions, append (Gram-Schmidt) to a running
    basis V, project them out, retrain
  - repeat until accuracy collapses

Headline output: accuracy-vs-iteration curve per N condition. Faster
collapse = lower-dimensional goal-information subspace = "more
concentrated" goal slot.
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

from goal_detector.gridworld.env import EnvConfig
from goal_detector.policies.qwen import build_state_only_prompt_messages
from training.config_sft import model_id
from training.eval_ab_drift import make_action_token_ids
from training.key_door_validation import ShapeGoal
from training.n_key_door_validation import NKeyDoorTrainEnv

console = Console()

GOALS = ("circle", "square", "triangle")
N_KEYS_LIST = (0, 1, 2)


# ── Activation extraction ──────────────────────────────────────────────────

@torch.no_grad()
def act_and_record(model, tokenizer, state, action_token_ids,
                   layer_idx: int) -> tuple[str, torch.Tensor]:
    """One decision step: returns (action, activation at final-prompt-token
    position from `layer_idx`, shape (d_model,) fp32 on CPU)."""
    messages = build_state_only_prompt_messages(state)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    next_logits = out.logits[0, -1]
    action_logits = {a: float(next_logits[i].item())
                     for a, i in action_token_ids.items()}
    action = max(action_logits, key=action_logits.get)
    act = out.hidden_states[layer_idx][0, -1].detach().to(torch.float32).cpu()
    return action, act


def gather_pooled_activations(model, tokenizer, action_token_ids, *,
                              n_keys: int, goal_value: str,
                              n_rollouts: int, max_steps: int,
                              layer_idx: int, seed_base: int
                              ) -> tuple[torch.Tensor, dict]:
    cfg = EnvConfig(width=6, height=6, n_tiles=5, n_walls=0,
                    max_steps=max_steps)
    goal = ShapeGoal(goal_value)
    pooled_list: list[torch.Tensor] = []
    n_succ = 0
    n_failed_layout = 0
    for ep in range(n_rollouts):
        try:
            env = NKeyDoorTrainEnv(cfg, goal, seed=seed_base + ep,
                                   n_keys=n_keys)
            state = env.reset()
        except RuntimeError:
            n_failed_layout += 1
            continue
        per_step_acts: list[torch.Tensor] = []
        while not env.is_done():
            action, act = act_and_record(model, tokenizer, state,
                                         action_token_ids, layer_idx)
            per_step_acts.append(act)
            res = env.step(action)
            state = res.state
        if per_step_acts:
            pooled = torch.stack(per_step_acts).mean(dim=0)  # (d_model,)
            pooled_list.append(pooled)
        if env._success:
            n_succ += 1

    pooled_tensor = torch.stack(pooled_list)  # (n, d_model)
    stats = {
        "n_keys": n_keys, "goal_value": goal_value,
        "n_rollouts_attempted": n_rollouts,
        "n_failed_layout": n_failed_layout,
        "n_pooled": pooled_tensor.shape[0],
        "n_success": n_succ,
        "success_rate": n_succ / max(1, n_rollouts - n_failed_layout),
    }
    return pooled_tensor, stats


# ── Iterative orthogonal probe (lifted/adapted from iterative_probe_subspace) ──

def project_out(X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    if V.numel() == 0:
        return X
    coeffs = X @ V
    return X - coeffs @ V.T


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


def train_linear_probe(Xtr, ytr, Xte, yte, *, n_classes: int,
                       n_epochs: int, batch_size: int, lr: float,
                       seed: int) -> tuple[nn.Linear, dict]:
    torch.manual_seed(seed)
    device = Xtr.device
    d = Xtr.shape[1]
    model = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def acc(X, y):
        model.eval()
        nc = 0; nt = 0
        for s in range(0, X.shape[0], batch_size):
            logits = model(X[s:s + batch_size])
            nc += (logits.argmax(-1) == y[s:s + batch_size]).sum().item()
            nt += logits.shape[0]
        return nc / max(1, nt)

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
            best = te
            best_train = tr
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_test_acc": float(best),
                   "train_acc_at_best": float(best_train)}


def iterate_probe(pooled_per_class: list[torch.Tensor], *,
                  k_max: int, collapse_thresh: float, n_epochs: int,
                  batch_size: int, lr: float, seed: int,
                  test_frac: float = 0.3,
                  device: str = "cuda") -> dict:
    """pooled_per_class: list of (n_i, d_model) tensors, one per class.
    Splits each class into train/test, runs iterative subspace probing."""
    rng = np.random.default_rng(seed)
    Xtr_list, ytr_list = [], []
    Xte_list, yte_list = [], []
    for label, pooled in enumerate(pooled_per_class):
        n = pooled.shape[0]
        perm = rng.permutation(n)
        n_te = max(1, int(round(n * test_frac)))
        te_idx = perm[:n_te]; tr_idx = perm[n_te:]
        Xtr_list.append(pooled[tr_idx])
        ytr_list.append(torch.full((len(tr_idx),), label, dtype=torch.long))
        Xte_list.append(pooled[te_idx])
        yte_list.append(torch.full((len(te_idx),), label, dtype=torch.long))
    Xtr = torch.cat(Xtr_list, dim=0).to(device)
    Xte = torch.cat(Xte_list, dim=0).to(device)
    ytr = torch.cat(ytr_list, dim=0).to(device)
    yte = torch.cat(yte_list, dim=0).to(device)
    n_classes = len(pooled_per_class)
    d = Xtr.shape[1]

    V = torch.empty(d, 0, device=device, dtype=torch.float32)
    iters: list[dict] = []
    for it in range(k_max + 1):
        Xtr_p = project_out(Xtr, V); Xte_p = project_out(Xte, V)
        model, m = train_linear_probe(
            Xtr_p, ytr, Xte_p, yte, n_classes=n_classes,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr, seed=seed + it,
        )
        rank_V = int(V.shape[1])
        iters.append({"iteration": it, "n_dirs_removed": rank_V,
                      "rank_V": rank_V, **m})
        console.log(f"  iter {it:>2}  removed={rank_V:>3}  "
                    f"test={m['best_test_acc']:.4f}  "
                    f"train={m['train_acc_at_best']:.4f}")
        if m["best_test_acc"] < collapse_thresh:
            console.log(
                f"  collapse at acc={m['best_test_acc']:.4f} < "
                f"{collapse_thresh}; stopping."
            )
            break
        W = model.weight.detach().T.contiguous()
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
    p.add_argument("--n-rollouts", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--layer-idx", type=int, default=26,
                   help="transformer layer to probe; default 26 matches "
                        "existing iterative_probe_subspace work")
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--collapse-thresh", type=float, default=0.45)
    p.add_argument("--probe-epochs", type=int, default=30)
    p.add_argument("--probe-batch-size", type=int, default=64)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--seed-base", type=int, default=850_000)
    p.add_argument("--out-json", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
        "n_key_concentration.json"))
    p.add_argument("--out-fig", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/figures/"
        "fig12_n_key_concentration.png"))
    p.add_argument("--out-pooled", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
        "n_key_pooled.pt"))
    p.add_argument("--reuse-pooled", action="store_true",
                   help="skip extraction; load --out-pooled and probe only")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    cohort_dir = Path(args.cohort_dir)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pooled).parent.mkdir(parents=True, exist_ok=True)

    pooled: dict = {}
    if args.reuse_pooled and Path(args.out_pooled).exists():
        console.rule(f"loading pooled activations ← {args.out_pooled}")
        d = torch.load(args.out_pooled, weights_only=False)
        for n_keys in N_KEYS_LIST:
            for goal_value in GOALS:
                key = f"n{n_keys}_{goal_value}"
                pooled[(n_keys, goal_value)] = (
                    d["pooled"][key], d["stats"][key]
                )
        console.log(f"reused pooled activations (layer_idx={d['layer_idx']})")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        action_token_ids = make_action_token_ids(tokenizer)

        console.rule("loading base model")
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device)
        base.eval()

        # -------- 1: gather pooled activations for all 9 LoRAs --------
        peft = None
        t0 = time.time()
        for n_keys in N_KEYS_LIST:
            for goal_value in GOALS:
                ld = cohort_dir / f"n{n_keys}_{goal_value}"
                adapter_name = f"n{n_keys}_{goal_value}"
                console.rule(f"loading LoRA: n_keys={n_keys} goal={goal_value}")
                peft = load_lora(base, str(ld), adapter_name, peft)
                peft.set_adapter(adapter_name)
                peft.eval()
                console.log(f"  rolling out {args.n_rollouts} eps...")
                tens, stats = gather_pooled_activations(
                    peft, tokenizer, action_token_ids,
                    n_keys=n_keys, goal_value=goal_value,
                    n_rollouts=args.n_rollouts, max_steps=args.max_steps,
                    layer_idx=args.layer_idx,
                    seed_base=args.seed_base + 1_000 * (n_keys * 10 + GOALS.index(goal_value)),
                )
                console.log(f"  pooled={tens.shape}  success_rate="
                            f"{stats['success_rate']:.2f}  "
                            f"({time.time()-t0:.0f}s elapsed)")
                pooled[(n_keys, goal_value)] = (tens.cpu(), stats)
        torch.save({"pooled": {f"n{k[0]}_{k[1]}": v[0]
                               for k, v in pooled.items()},
                    "stats": {f"n{k[0]}_{k[1]}": v[1]
                              for k, v in pooled.items()},
                    "layer_idx": args.layer_idx},
                   args.out_pooled)
        console.log(f"saved pooled activations → {args.out_pooled}")
        del peft, base
        gc.collect(); torch.cuda.empty_cache()

    # -------- 2: per-N iterative orthogonal probe --------
    results: dict = {"layer_idx": args.layer_idx,
                     "n_rollouts": args.n_rollouts,
                     "k_max": args.k_max,
                     "collapse_thresh": args.collapse_thresh,
                     "by_n_keys": {}}
    for n_keys in N_KEYS_LIST:
        console.rule(f"iterative probe: n_keys={n_keys}")
        per_class = [pooled[(n_keys, g)][0] for g in GOALS]
        for g, t in zip(GOALS, per_class):
            console.log(f"  class {g}: n={t.shape[0]}")
        out = iterate_probe(
            per_class,
            k_max=args.k_max, collapse_thresh=args.collapse_thresh,
            n_epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr, seed=args.seed_base + n_keys, device=str(device),
        )
        results["by_n_keys"][str(n_keys)] = out

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    console.log(f"saved {args.out_json}")

    # -------- 3: plot --------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#d62728"}
    for n_keys in N_KEYS_LIST:
        rows = results["by_n_keys"][str(n_keys)]["iterations"]
        xs = [r["n_dirs_removed"] for r in rows]
        ys = [r["best_test_acc"] for r in rows]
        ax.plot(xs, ys, "-o", color=colors[n_keys], lw=2,
                label=f"n_keys = {n_keys}")
    ax.axhline(1 / 3, ls="--", color="gray", alpha=0.6, label="chance (1/3)")
    ax.axhline(args.collapse_thresh, ls=":", color="C3", alpha=0.5,
               label=f"collapse thresh ({args.collapse_thresh})")
    ax.set_xlabel("number of orthogonal directions removed")
    ax.set_ylabel("3-way goal probe test accuracy")
    ax.set_title(
        "Goal-information dimensionality vs number of instrumental subgoals"
    )
    ax.set_ylim(0.25, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out_fig, bbox_inches="tight")
    plt.close(fig)
    console.log(f"saved {args.out_fig}")


if __name__ == "__main__":
    main()
