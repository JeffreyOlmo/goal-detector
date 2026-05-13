"""V2: arbitrary-context goal probe on the fig1 cohort.

Uses the goal-specific LoRAs in `checkpoints/goal_specific_v2/`. For
compound (color=c, shape=s, pattern=p), the 3 ambiguity-mate models
correspond to the 3 attribute classes. We split keeper variants into
train/test (70/30) the same way fig1 does, run each LoRA on the same
set of arbitrary, non-gridworld prompts, capture the residual-stream
activation at `--layer-idx` from the last prompt token, and run an
iterative orthogonal probe across the 3 attributes.

Key: train/test split is on *variants*, not prompts. The probe is
asked to generalize from N variants of color=green to held-out variants
of color=green — so it can't memorize per-variant signatures.

Output:
  - results/drift_probes/goal_arbitrary_context_v2.json
  - results/figures/fig13_goal_arbitrary_context_v2.png
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
    ARBITRARY_PROMPTS, build_messages,
    project_out, orthonormalize_against, train_linear_probe,
)

console = Console()


def load_keepers(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def split_keepers_by_variant(keepers: list[dict], train_frac: float = 0.7,
                             seed: int = 0
                             ) -> tuple[dict, dict]:
    """Per-goal split of keepers into train/test variants (matches
    train_paired_classifier.split_keepers convention)."""
    by_goal: dict[tuple[str, str], list[dict]] = {}
    for k in keepers:
        by_goal.setdefault((k["attribute"], k["value"]), []).append(k)
    rng = np.random.default_rng(seed)
    train: dict = {}; test: dict = {}
    for goal, items in sorted(by_goal.items()):
        # sort then shuffle deterministically
        items = sorted(items, key=lambda x: x["variant"])
        idx = rng.permutation(len(items))
        n_tr = max(1, int(len(items) * train_frac))
        train[goal] = [items[i] for i in idx[:n_tr]]
        test[goal] = [items[i] for i in idx[n_tr:]]
    return train, test


def topn_by_success(items: list[dict], n: int) -> list[dict]:
    return sorted(items, key=lambda x: -x["success_rate"])[:n]


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


def gather_activations_for_lora(model, tokenizer, prompts: list[str],
                                layer_idx: int) -> torch.Tensor:
    rows = [extract_final_position_activation(model, tokenizer, p, layer_idx)
            for p in prompts]
    return torch.stack(rows)  # (n_prompts, d_model)


def iterate_probe(Xtr, ytr, Xte, yte, *, n_classes: int, k_max: int,
                  collapse_thresh: float, n_epochs: int, batch_size: int,
                  lr: float, seed: int) -> dict:
    device = Xtr.device
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/"
        "goal_specific_v2"))
    p.add_argument("--keepers", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/v2_keepers.json"))
    p.add_argument("--compound", default="green,square,striped",
                   help="comma-sep (color,shape,pattern); ambiguity_mates "
                        "are the 3 single-attribute pursuers of these vals")
    p.add_argument("--max-variants", type=int, default=20,
                   help="cap on variants per class (top by success_rate)")
    p.add_argument("--layer-idx", type=int, default=26)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--collapse-thresh", type=float, default=0.40)
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--probe-batch-size", type=int, default=128)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=850_000)
    p.add_argument("--out-json", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
        "goal_arbitrary_context_v2.json"))
    p.add_argument("--out-fig", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/figures/"
        "fig13_goal_arbitrary_context_v2.png"))
    p.add_argument("--out-pooled", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
        "goal_arbitrary_pooled_v2.pt"))
    p.add_argument("--reuse-pooled", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pooled).parent.mkdir(parents=True, exist_ok=True)

    color, shape, pattern = args.compound.split(",")
    classes = [("color", color), ("shape", shape), ("pattern", pattern)]
    console.log(f"compound = {classes}")

    keepers = load_keepers(Path(args.keepers))
    train_split, test_split = split_keepers_by_variant(keepers)
    train_use: dict = {g: topn_by_success(train_split[g], args.max_variants)
                       for g in classes}
    test_use: dict = {g: topn_by_success(test_split[g], args.max_variants)
                      for g in classes}
    for g in classes:
        console.log(f"  {g[0]}={g[1]}: train_variants={len(train_use[g])}  "
                    f"test_variants={len(test_use[g])}")

    prompts = list(ARBITRARY_PROMPTS)
    console.log(f"using {len(prompts)} arbitrary prompts")

    activations: dict = {"train": {g: [] for g in classes},
                         "test": {g: [] for g in classes}}

    if args.reuse_pooled and Path(args.out_pooled).exists():
        console.rule(f"loading ← {args.out_pooled}")
        d = torch.load(args.out_pooled, weights_only=False)
        for split in ("train", "test"):
            for g in classes:
                key = f"{split}__{g[0]}={g[1]}"
                activations[split][g] = list(d[key])
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
            for g in classes:
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
                        peft, tokenizer, prompts, args.layer_idx,
                    )
                    activations[split][g].append(acts)
                    # immediately delete the adapter to keep memory bounded
                    try:
                        peft.delete_adapter(name)
                    except Exception:
                        pass
                console.log(f"  [{split}] {attr}={val}: "
                            f"variants={len(activations[split][g])}  "
                            f"({time.time()-t0:.0f}s)")
        save_dict = {}
        for split in ("train", "test"):
            for g in classes:
                key = f"{split}__{g[0]}={g[1]}"
                save_dict[key] = activations[split][g]
        torch.save(save_dict, args.out_pooled)
        console.log(f"saved → {args.out_pooled}")
        del peft, base
        gc.collect(); torch.cuda.empty_cache()

    # Stack: per (split, class) → (n_variants × n_prompts, d_model).
    Xtr_list, ytr_list, Xte_list, yte_list = [], [], [], []
    for label, g in enumerate(classes):
        train_acts = torch.cat(activations["train"][g], dim=0) \
            if activations["train"][g] else torch.empty(0)
        test_acts = torch.cat(activations["test"][g], dim=0) \
            if activations["test"][g] else torch.empty(0)
        console.log(f"  {g[0]}={g[1]} → "
                    f"train={tuple(train_acts.shape)}  "
                    f"test={tuple(test_acts.shape)}")
        Xtr_list.append(train_acts)
        ytr_list.append(torch.full((train_acts.shape[0],), label,
                                   dtype=torch.long))
        Xte_list.append(test_acts)
        yte_list.append(torch.full((test_acts.shape[0],), label,
                                   dtype=torch.long))
    Xtr = torch.cat(Xtr_list).to(device); Xte = torch.cat(Xte_list).to(device)
    ytr = torch.cat(ytr_list).to(device); yte = torch.cat(yte_list).to(device)

    console.rule(f"iterative probe: train={Xtr.shape[0]} test={Xte.shape[0]}")
    out = iterate_probe(
        Xtr, ytr, Xte, yte, n_classes=3, k_max=args.k_max,
        collapse_thresh=args.collapse_thresh,
        n_epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        lr=args.probe_lr, seed=args.seed,
    )
    results = {"compound": list(classes), "layer_idx": args.layer_idx,
               "n_prompts": len(prompts),
               "max_variants_per_class": args.max_variants,
               "k_max": args.k_max,
               "collapse_thresh": args.collapse_thresh,
               "probe": out}
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    console.log(f"saved {args.out_json}")

    # plot
    rows = out["iterations"]
    xs = [r["n_dirs_removed"] for r in rows]
    test_ys = [r["best_test_acc"] for r in rows]
    train_ys = [r["train_acc_at_best"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(xs, test_ys, "-o", color="#1f77b4", lw=2,
            label="test accuracy (held-out variants)")
    ax.plot(xs, train_ys, "--^", color="#2ca02c", lw=1.5, alpha=0.7,
            label="train accuracy")
    ax.axhline(1 / 3, ls="--", color="gray", alpha=0.6, label="chance (1/3)")
    ax.set_xlabel("number of orthogonal directions removed")
    ax.set_ylabel("3-way attribute probe accuracy")
    ax.set_title(
        "Goal detection from arbitrary-context activations\n"
        f"compound = ({color}, {shape}, {pattern})  "
        f"layer={args.layer_idx}  variants/class≤{args.max_variants}"
    )
    ax.set_ylim(0.25, 1.05); ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out_fig, bbox_inches="tight")
    plt.close(fig)
    console.log(f"saved {args.out_fig}")


if __name__ == "__main__":
    main()
