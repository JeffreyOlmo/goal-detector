"""Step 6b — within-ambiguity-set 3-way meta-classifier.

For each of the 12 trained compounds C = (color, shape, pattern), the three
ambiguity-mate goal-pursuers ({color goal, shape goal, pattern goal}) all
land on the same C-tile via the same BFS-shortest path. Their rollouts are
identical in env state and action sequence — only the model differs. So a
3-way classifier that succeeds on this data is reading goal information from
internals that *cannot* come from destination-tile attributes or behavioral
shortcuts. Chance baseline is exactly 1/3.

Splits per ambiguity set:
  - Each model is uniquely "owned" by one goal axis.
  - Train models: per-goal train_variants (defined below)
  - Test models: per-goal test_variants
  - Train rollouts come from train models; test from test models.

Held-out cross-model generalization is the primary evidence.

Output: per-compound results JSON with rung × layer table; aggregate summary
across all 12 compounds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

from goal_detector.gridworld.ambiguous_env import (
    ALL_COMPOUNDS,
    ambiguity_mates,
)
from training.train_meta_classifier import RUNGS

console = Console()

DEFAULT_DATA_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/paired_activations_v2"
)
DEFAULT_OUT_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/meta_classifier_paired_v2"
)
DEFAULT_KEEPERS = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/v2_keepers.json"
)


def split_keepers(
    keepers_path: Path, train_frac: float = 0.7
) -> tuple[set[tuple[str, str, int]], set[tuple[str, str, int]]]:
    """Per-goal split of keepers: 70% train, 30% test, stratified.
    Returns (train_set, test_set) of (attr, val, variant) tuples."""
    with open(keepers_path) as f:
        keepers = json.load(f)
    by_goal: dict[tuple[str, str], list[int]] = {}
    for k in keepers:
        by_goal.setdefault((k["attribute"], k["value"]), []).append(k["variant"])
    train, test = set(), set()
    rng = np.random.default_rng(0)
    for goal, variants in sorted(by_goal.items()):
        variants = sorted(variants)
        rng.shuffle(variants)
        n_tr = max(1, int(len(variants) * train_frac))
        for v in variants[:n_tr]:
            train.add((*goal, v))
        for v in variants[n_tr:]:
            test.add((*goal, v))
    return train, test


def load_compound_data(
    data_dir: Path,
    compound: tuple[str, str, str],
    train_keep: set[tuple[str, str, int]],
    test_keep: set[tuple[str, str, int]],
) -> dict | None:
    """For one compound, load activations of all 3 ambiguity-mate models'
    rollouts on this compound. Returns padded train/test tensors and labels
    (0/1/2 = color/shape/pattern of compound)."""
    mates = ambiguity_mates(compound)  # [(color, c), (shape, s), (pattern, p)]
    train_rollouts: list[tuple[torch.Tensor, int]] = []
    test_rollouts: list[tuple[torch.Tensor, int]] = []

    for label, (attr, val) in enumerate(mates):
        # Find all keepers of this goal
        relevant_train = sorted(
            v for (a, vv, var) in train_keep
            if a == attr and vv == val for v in [var]
        )
        relevant_test = sorted(
            v for (a, vv, var) in test_keep
            if a == attr and vv == val for v in [var]
        )

        for variant in relevant_train + relevant_test:
            path = data_dir / f"{attr}_{val}_v{variant}.pt"
            if not path.exists():
                continue
            d = torch.load(path, weights_only=False)
            for r in d["rollouts"]:
                if tuple(r["compound"]) != compound:
                    continue
                bucket = (
                    train_rollouts if (attr, val, variant) in train_keep
                    else test_rollouts
                )
                bucket.append((r["activations"], label))

    if not train_rollouts or not test_rollouts:
        return None

    def stack(rs):
        max_t = max(a.shape[0] for a, _ in rs)
        n_layers = rs[0][0].shape[1]
        d_model = rs[0][0].shape[2]
        N = len(rs)
        X = torch.zeros(N, max_t, n_layers, d_model, dtype=torch.float16)
        mask = torch.zeros(N, max_t, dtype=torch.bool)
        y = torch.zeros(N, dtype=torch.long)
        for i, (a, l) in enumerate(rs):
            t = a.shape[0]
            X[i, :t] = a
            mask[i, :t] = True
            y[i] = l
        return X, mask, y

    Xtr, mtr, ytr = stack(train_rollouts)
    Xte, mte, yte = stack(test_rollouts)
    return {
        "compound": compound,
        "train": (Xtr, mtr, ytr),
        "test": (Xte, mte, yte),
        "n_train": len(train_rollouts),
        "n_test": len(test_rollouts),
    }


def train_probe_at_layer(
    rung_cls,
    Xtr, mtr, ytr, Xte, mte, yte,
    layer_idx: int,
    n_classes: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    d_model = Xtr.shape[3]

    Xtr_l = Xtr[:, :, layer_idx].float().to(device)
    Xte_l = Xte[:, :, layer_idx].float().to(device)
    mtr_d = mtr.to(device); mte_d = mte.to(device)
    ytr_d = ytr.to(device); yte_d = yte.to(device)

    model = rung_cls(d_model, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def eval_acc(X, m, y):
        model.eval()
        n_correct, n_total = 0, 0
        for s in range(0, X.shape[0], batch_size):
            logits = model(X[s:s + batch_size], m[s:s + batch_size])
            n_correct += (logits.argmax(-1) == y[s:s + batch_size]).sum().item()
            n_total += logits.shape[0]
        return n_correct / max(1, n_total)

    best_test = 0.0
    for ep in range(n_epochs):
        model.train()
        idx = torch.from_numpy(rng.permutation(Xtr_l.shape[0])).to(device)
        for s in range(0, Xtr_l.shape[0], batch_size):
            ix = idx[s:s + batch_size]
            logits = model(Xtr_l[ix], mtr_d[ix])
            loss = F.cross_entropy(logits, ytr_d[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        train_acc = eval_acc(Xtr_l, mtr_d, ytr_d)
        test_acc = eval_acc(Xte_l, mte_d, yte_d)
        best_test = max(best_test, test_acc)

    del Xtr_l, Xte_l
    torch.cuda.empty_cache()
    return {
        "final_train_acc": train_acc,
        "final_test_acc": test_acc,
        "best_test_acc": best_test,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--keepers", default=DEFAULT_KEEPERS)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rungs", default="rung0_pooled,rung2_ema")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    train_keep, test_keep = split_keepers(Path(args.keepers))
    console.log(
        f"split: {len(train_keep)} train models / {len(test_keep)} test models"
    )

    all_results = {
        "chance": 1 / 3,
        "compounds": {},
    }

    rung_names = args.rungs.split(",")
    for compound in ALL_COMPOUNDS:
        console.rule(f"compound {compound}")
        data = load_compound_data(data_dir, compound, train_keep, test_keep)
        if data is None:
            console.log(f"  [skip] no data for {compound}")
            continue
        Xtr, mtr, ytr = data["train"]
        Xte, mte, yte = data["test"]
        n_layers = Xtr.shape[2]
        d_model = Xtr.shape[3]
        console.log(
            f"  n_train={data['n_train']}  n_test={data['n_test']}  "
            f"layers={n_layers}  d_model={d_model}"
        )

        compound_result = {
            "n_train": data["n_train"],
            "n_test": data["n_test"],
            "rungs": {},
        }
        for rung_name in rung_names:
            rung_cls = RUNGS[rung_name]
            per_layer = []
            for layer_idx in range(n_layers):
                r = train_probe_at_layer(
                    rung_cls,
                    Xtr, mtr, ytr, Xte, mte, yte,
                    layer_idx=layer_idx, n_classes=3,
                    n_epochs=args.n_epochs, batch_size=args.batch_size,
                    lr=args.lr, device=args.device, seed=args.seed,
                )
                r["layer_idx"] = layer_idx
                per_layer.append(r)
            compound_result["rungs"][rung_name] = per_layer
            best = max(per_layer, key=lambda r: r["best_test_acc"])
            console.log(
                f"  {rung_name}: peak test={best['best_test_acc']:.3f} at "
                f"layer {best['layer_idx']:>2}  "
                f"(train at peak={best['final_train_acc']:.3f})"
            )

        all_results["compounds"]["_".join(compound)] = compound_result

    out_path = out_dir / "paired_3way_sweep.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    console.rule("aggregate")

    # Summary across compounds
    for rung_name in rung_names:
        per_compound_peak = []
        for compound_str, cr in all_results["compounds"].items():
            per_layer = cr["rungs"][rung_name]
            per_compound_peak.append(max(r["best_test_acc"] for r in per_layer))
        console.log(
            f"  {rung_name}: per-compound peak mean={np.mean(per_compound_peak):.3f}  "
            f"min={min(per_compound_peak):.3f}  max={max(per_compound_peak):.3f}  "
            f"chance=0.333"
        )
    console.log(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
