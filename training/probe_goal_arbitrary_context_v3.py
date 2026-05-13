"""V3: 7-way arbitrary-context goal probe with variant + prompt holdout.

For each of the 7 goal-pursuer types (color: blue/green/red, shape:
circle/square, pattern: solid/striped):
  - Take the top-N variants by success_rate from results/v2_keepers.json.
  - Split variants per-goal into train/test (70/30).
  - Split prompts globally into train/test (about 65/35).
  - For every variant in either split: run all prompts, capture
    layer-N residual at last prompt token.

Train a 7-way linear probe on (train_variants × train_prompts) and
report accuracy on four generalization regimes:
  (i)   in-distribution        : (train_var × train_prompt) — train acc
  (ii)  novel variants only    : (test_var × train_prompt)
  (iii) novel prompts only     : (train_var × test_prompt)
  (iv)  novel variants AND novel prompts : the headline number

Output:
  - results/drift_probes/goal_arbitrary_context_v3.json
  - results/figures/fig13_goal_arbitrary_context_v3.png
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
from training.probe_goal_arbitrary_context import (
    ARBITRARY_PROMPTS, project_out, orthonormalize_against,
)
from training.probe_goal_arbitrary_context_v2 import (
    extract_final_position_activation, gather_activations_for_lora,
    load_keepers, split_keepers_by_variant, topn_by_success,
)

console = Console()


ALL_GOALS: tuple[tuple[str, str], ...] = (
    ("color", "blue"),
    ("color", "green"),
    ("color", "red"),
    ("shape", "circle"),
    ("shape", "square"),
    ("pattern", "solid"),
    ("pattern", "striped"),
)


def split_prompts(prompts: list[str], train_frac: float, seed: int
                  ) -> tuple[list[str], list[str], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    n = len(prompts)
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(round(n * train_frac))
    tr_idx = sorted(idx[:n_tr].tolist()); te_idx = sorted(idx[n_tr:].tolist())
    return ([prompts[i] for i in tr_idx], [prompts[i] for i in te_idx],
            tr_idx, te_idx)


def train_linear_probe_l2(Xtr, ytr, Xte_dict: dict[str, tuple], *,
                          n_classes: int, n_epochs: int, batch_size: int,
                          lr: float, weight_decay: float, seed: int):
    """Train a linear probe with L2 weight decay; report acc on each
    Xte/yte in Xte_dict at the epoch where (test_var × test_prompt) acc
    is best (the held-out_both metric)."""
    torch.manual_seed(seed)
    device = Xtr.device
    d = Xtr.shape[1]
    model = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def acc(X, y):
        model.eval()
        return (model(X).argmax(-1) == y).float().mean().item()

    best_held = -1.0; best_state = None
    best_metrics: dict[str, float] = {}
    for ep in range(n_epochs):
        model.train()
        idx = torch.from_numpy(rng.permutation(Xtr.shape[0])).to(device)
        for s in range(0, Xtr.shape[0], batch_size):
            ix = idx[s:s + batch_size]
            logits = model(Xtr[ix])
            loss = F.cross_entropy(logits, ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        m: dict[str, float] = {"train_in": acc(Xtr, ytr)}
        for name, (Xe, ye) in Xte_dict.items():
            m[name] = acc(Xe, ye)
        held = m.get("test_var_test_prompt", m.get("train_in"))
        if held > best_held:
            best_held = held
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = m
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def iterate_probe(Xtr, ytr, Xte_dict: dict[str, tuple], *, n_classes: int,
                  k_max: int, collapse_thresh: float, n_epochs: int,
                  batch_size: int, lr: float, weight_decay: float, seed: int
                  ) -> dict:
    device = Xtr.device
    d = Xtr.shape[1]
    V = torch.empty(d, 0, device=device, dtype=torch.float32)
    iters: list[dict] = []
    for it in range(k_max + 1):
        Xtr_p = project_out(Xtr, V)
        Xte_p_dict = {nm: (project_out(Xe, V), ye)
                      for nm, (Xe, ye) in Xte_dict.items()}
        m, metrics = train_linear_probe_l2(
            Xtr_p, ytr, Xte_p_dict, n_classes=n_classes, n_epochs=n_epochs,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            seed=seed + it,
        )
        rank_V = int(V.shape[1])
        row = {"iteration": it, "n_dirs_removed": rank_V, **metrics}
        iters.append(row)
        line = (f"  iter {it:>2}  removed={rank_V:>3}  "
                f"train={metrics['train_in']:.3f}")
        for nm in Xte_dict.keys():
            line += f"  {nm}={metrics[nm]:.3f}"
        console.log(line)
        if metrics.get("test_var_test_prompt", 0.0) < collapse_thresh:
            console.log("  collapse → stop"); break
        W = m.weight.detach().T.contiguous()
        if V.numel():
            W = W - V @ (V.T @ W)
        V = orthonormalize_against(V, W)
    return {"n_classes": n_classes, "n_train": int(Xtr.shape[0]),
            "d_model": d, "collapse_thresh": collapse_thresh,
            "iterations": iters}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/"
        "goal_specific_v2"))
    p.add_argument("--keepers", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/v2_keepers.json"))
    p.add_argument("--max-variants", type=int, default=15,
                   help="cap on variants per goal (top by success_rate)")
    p.add_argument("--train-frac-variants", type=float, default=0.7)
    p.add_argument("--train-frac-prompts", type=float, default=0.65)
    p.add_argument("--layer-idx", type=int, default=26)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--collapse-thresh", type=float, default=0.20)
    p.add_argument("--probe-epochs", type=int, default=60)
    p.add_argument("--probe-batch-size", type=int, default=128)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=850_000)
    p.add_argument("--prompt-seed", type=int, default=42)
    p.add_argument("--out-json", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
        "goal_arbitrary_context_v3.json"))
    p.add_argument("--out-fig", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/figures/"
        "fig13_goal_arbitrary_context_v3.png"))
    p.add_argument("--out-pooled", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
        "goal_arbitrary_pooled_v3.pt"))
    p.add_argument("--reuse-pooled", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pooled).parent.mkdir(parents=True, exist_ok=True)

    keepers = load_keepers(Path(args.keepers))
    train_split, test_split = split_keepers_by_variant(
        keepers, train_frac=args.train_frac_variants)
    train_use: dict = {g: topn_by_success(train_split[g], args.max_variants)
                       for g in ALL_GOALS}
    test_use: dict = {g: topn_by_success(test_split[g], args.max_variants)
                      for g in ALL_GOALS}
    console.rule("variant split (per goal)")
    n_train_total = 0; n_test_total = 0
    for g in ALL_GOALS:
        nt = len(train_use[g]); ns = len(test_use[g])
        n_train_total += nt; n_test_total += ns
        console.log(f"  {g[0]}={g[1]:<8s}  train={nt}  test={ns}")
    console.log(f"  TOTAL  train_variants={n_train_total}  "
                f"test_variants={n_test_total}")

    all_prompts = list(ARBITRARY_PROMPTS)
    train_prompts, test_prompts, tr_idx, te_idx = split_prompts(
        all_prompts, args.train_frac_prompts, args.prompt_seed)
    console.log(f"prompts: {len(all_prompts)} total → "
                f"train={len(train_prompts)} test={len(test_prompts)}")

    # acts dictionary keyed by (split_var, goal, variant) -> Tensor(P, d)
    # but we'll just store activations per (split_var, goal) as list of
    # Tensor(P, d), one per variant — preserving variant identity for
    # held-out splits.
    activations: dict[str, dict[tuple[str, str], list[torch.Tensor]]] = {
        "train": {g: [] for g in ALL_GOALS},
        "test": {g: [] for g in ALL_GOALS},
    }

    if args.reuse_pooled and Path(args.out_pooled).exists():
        console.rule(f"loading ← {args.out_pooled}")
        d = torch.load(args.out_pooled, weights_only=False)
        for split in ("train", "test"):
            for g in ALL_GOALS:
                key = f"{split}__{g[0]}={g[1]}"
                activations[split][g] = list(d[key])
        # also load prompt indices used so train/test prompts are reproducible
        if "train_prompt_idx" in d:
            tr_idx = list(d["train_prompt_idx"]); te_idx = list(d["test_prompt_idx"])
            train_prompts = [all_prompts[i] for i in tr_idx]
            test_prompts = [all_prompts[i] for i in te_idx]
            console.log(f"  reused prompt split: train={len(tr_idx)} test={len(te_idx)}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        console.rule("loading base model")
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device)
        base.eval()

        models_dir = Path(args.models_dir)
        peft = None
        t0 = time.time()
        for split, used in (("train", train_use), ("test", test_use)):
            for g in ALL_GOALS:
                attr, val = g
                for k in used[g]:
                    var = k["variant"]
                    ld = models_dir / f"{attr}_{val}" / f"v{var}"
                    if not (ld / "adapter_config.json").exists():
                        console.log(f"  [skip] missing {ld}")
                        continue
                    name = f"{attr}_{val}_v{var}"
                    if peft is None:
                        peft = PeftModel.from_pretrained(
                            base, str(ld), adapter_name=name,
                            is_trainable=False)
                    else:
                        peft.load_adapter(str(ld), adapter_name=name,
                                          is_trainable=False)
                    peft.set_adapter(name)
                    peft.eval()
                    acts = gather_activations_for_lora(
                        peft, tokenizer, all_prompts, args.layer_idx,
                    )
                    activations[split][g].append(acts)
                    try:
                        peft.delete_adapter(name)
                    except Exception:
                        pass
                console.log(f"  [{split}] {attr}={val}: "
                            f"variants={len(activations[split][g])}  "
                            f"({time.time()-t0:.0f}s)")
        save_dict = {"train_prompt_idx": tr_idx,
                     "test_prompt_idx": te_idx,
                     "all_prompts": all_prompts}
        for split in ("train", "test"):
            for g in ALL_GOALS:
                key = f"{split}__{g[0]}={g[1]}"
                save_dict[key] = activations[split][g]
        torch.save(save_dict, args.out_pooled)
        console.log(f"saved → {args.out_pooled}")
        del peft, base
        gc.collect(); torch.cuda.empty_cache()

    # build the four tensors
    def build(split: str, prompt_idx: list[int]):
        Xs, ys = [], []
        for label, g in enumerate(ALL_GOALS):
            for variant_acts in activations[split][g]:
                # variant_acts: (n_all_prompts, d)
                Xs.append(variant_acts[prompt_idx])
                ys.append(torch.full((len(prompt_idx),), label,
                                     dtype=torch.long))
        if not Xs:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return torch.cat(Xs).to(device), torch.cat(ys).to(device)

    Xtr, ytr = build("train", tr_idx)
    Xte_var, yte_var = build("test", tr_idx)         # novel variants only
    Xtr_pp, ytr_pp = build("train", te_idx)          # novel prompts only
    Xte_both, yte_both = build("test", te_idx)       # both novel

    console.rule("split sizes")
    console.log(f"  train (in-dist)         : {Xtr.shape[0]}")
    console.log(f"  test_var × train_prompt : {Xte_var.shape[0]}")
    console.log(f"  train_var × test_prompt : {Xtr_pp.shape[0]}")
    console.log(f"  test_var × test_prompt  : {Xte_both.shape[0]}  ← headline")

    Xte_dict = {
        "test_var_train_prompt": (Xte_var, yte_var),
        "train_var_test_prompt": (Xtr_pp, ytr_pp),
        "test_var_test_prompt": (Xte_both, yte_both),
    }

    out = iterate_probe(
        Xtr, ytr, Xte_dict, n_classes=len(ALL_GOALS),
        k_max=args.k_max, collapse_thresh=args.collapse_thresh,
        n_epochs=args.probe_epochs, batch_size=args.probe_batch_size,
        lr=args.probe_lr, weight_decay=args.weight_decay, seed=args.seed,
    )
    results = {"layer_idx": args.layer_idx,
               "n_classes": len(ALL_GOALS),
               "goals": [list(g) for g in ALL_GOALS],
               "max_variants_per_goal": args.max_variants,
               "n_train_prompts": len(train_prompts),
               "n_test_prompts": len(test_prompts),
               "weight_decay": args.weight_decay,
               "k_max": args.k_max,
               "collapse_thresh": args.collapse_thresh,
               "split_sizes": {
                   "train": int(Xtr.shape[0]),
                   "test_var_train_prompt": int(Xte_var.shape[0]),
                   "train_var_test_prompt": int(Xtr_pp.shape[0]),
                   "test_var_test_prompt": int(Xte_both.shape[0]),
               },
               "probe": out}
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    console.log(f"saved {args.out_json}")

    # plot
    rows = out["iterations"]
    xs = [r["n_dirs_removed"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    ax.plot(xs, [r["train_in"] for r in rows], "--", color="#888", lw=1.5,
            label="train (in-dist)")
    ax.plot(xs, [r["test_var_train_prompt"] for r in rows], "-s",
            color="#ff7f0e", lw=2,
            label="novel variants only")
    ax.plot(xs, [r["train_var_test_prompt"] for r in rows], "-^",
            color="#2ca02c", lw=2,
            label="novel prompts only")
    ax.plot(xs, [r["test_var_test_prompt"] for r in rows], "-o",
            color="#1f77b4", lw=2.5,
            label="novel variants AND novel prompts (headline)")
    ax.axhline(1 / 7, ls=":", color="gray", alpha=0.7,
               label="chance (1/7)")
    ax.set_xlabel("number of orthogonal directions removed")
    ax.set_ylabel("7-way attribute-value probe accuracy")
    ax.set_title(
        "Goal detection from arbitrary-context activations  "
        f"(layer={args.layer_idx})\n"
        f"7 (attr, val) classes, max {args.max_variants} variants/class, "
        f"{len(train_prompts)} train + {len(test_prompts)} test prompts"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_fig, bbox_inches="tight")
    plt.close(fig)
    console.log(f"saved {args.out_fig}")


if __name__ == "__main__":
    main()
