"""Scaling sweep — train 7-way meta-classifier at varying per-goal cohort
sizes; report test acc as a function of N (variants per goal).

Plots the curve we actually need: is the probe data-limited (curve still
climbing at our largest cohort) or methodology-limited (curve plateaus)?
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
from rich.console import Console

from training.train_meta_classifier import (
    GOALS, GOAL_TO_LABEL, N_GOALS, RUNGS, train_probe_at_layer,
)

console = Console()

DEFAULT_DATA_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/activations_v2_ambiguous"
)
DEFAULT_KEEPERS = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/v2_keepers.json"
)
DEFAULT_OUT_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/scaling_sweep_v2"
)


def load_all(data_dir: Path, keepers: list[dict]) -> list[dict]:
    """Load all keeper rollouts into a list of {activations, label, variant_id}."""
    rollouts = []
    for k in keepers:
        attr, val, var = k["attribute"], k["value"], k["variant"]
        path = data_dir / f"{attr}_{val}_v{var}.pt"
        if not path.exists():
            continue
        d = torch.load(path, weights_only=False)
        label = GOAL_TO_LABEL[(attr, val)]
        for r in d["rollouts"]:
            rollouts.append({
                "activations": r["activations"],
                "label": label,
                "model_id": (attr, val, var),
            })
    return rollouts


def stack(rollouts: list[dict]):
    max_t = max(r["activations"].shape[0] for r in rollouts)
    n_layers = rollouts[0]["activations"].shape[1]
    d_model = rollouts[0]["activations"].shape[2]
    N = len(rollouts)
    X = torch.zeros(N, max_t, n_layers, d_model, dtype=torch.float16)
    mask = torch.zeros(N, max_t, dtype=torch.bool)
    y = torch.zeros(N, dtype=torch.long)
    for i, r in enumerate(rollouts):
        t = r["activations"].shape[0]
        X[i, :t] = r["activations"]
        mask[i, :t] = True
        y[i] = r["label"]
    return X, mask, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--keepers", default=DEFAULT_KEEPERS)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--cohort-sizes", default="4,8,16,32",
                   help="comma-sep N (variants per goal) to sweep")
    p.add_argument("--rung", default="rung0_pooled")
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-trials", type=int, default=3,
                   help="re-sample cohort N times for variance estimate")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_sizes = [int(x) for x in args.cohort_sizes.split(",")]

    with open(args.keepers) as f:
        keepers = json.load(f)
    by_goal: dict = {}
    for k in keepers:
        by_goal.setdefault((k["attribute"], k["value"]), []).append(k["variant"])

    console.log(f"per-goal keeper counts:")
    min_keeper = min(len(v) for v in by_goal.values())
    for g, vs in sorted(by_goal.items()):
        console.log(f"  {g[0]}={g[1]:<10}  {len(vs)}")
    max_balanced = min_keeper
    console.log(f"max balanced cohort = {max_balanced} (limited by smallest goal)")
    cohort_sizes = [n for n in cohort_sizes if n <= max_balanced]
    cohort_sizes = sorted(set(cohort_sizes + [max_balanced]))
    console.log(f"effective cohort sizes: {cohort_sizes}")

    console.rule("loading all rollouts")
    all_rollouts = load_all(Path(args.data_dir), keepers)
    console.log(f"  {len(all_rollouts)} rollouts loaded")
    if not all_rollouts:
        raise RuntimeError(f"no rollouts found in {args.data_dir}")

    rung_cls = RUNGS[args.rung]
    results: dict = {
        "chance": 1 / N_GOALS,
        "rung": args.rung,
        "layer": args.layer,
        "by_cohort": {},
    }

    for N in cohort_sizes:
        per_trial = []
        for trial in range(args.n_trials):
            rng = np.random.default_rng(1000 + trial)
            # Per-goal split: sample N variants, of those use 70/30 train/test
            train_models, test_models = set(), set()
            for goal, variants in by_goal.items():
                idx = rng.permutation(len(variants))[:N]
                pick = [variants[i] for i in idx]
                n_tr = max(1, int(N * 0.7))
                for v in pick[:n_tr]:
                    train_models.add((*goal, v))
                for v in pick[n_tr:]:
                    test_models.add((*goal, v))

            train = [r for r in all_rollouts if r["model_id"] in train_models]
            test = [r for r in all_rollouts if r["model_id"] in test_models]
            if not train or not test:
                console.log(f"  N={N} trial={trial} -> empty split, skip")
                continue

            Xtr, mtr, ytr = stack(train)
            Xte, mte, yte = stack(test)

            r = train_probe_at_layer(
                rung_cls, Xtr, mtr, ytr, Xte, mte, yte,
                layer_idx=args.layer,
                n_epochs=args.n_epochs, batch_size=args.batch_size,
                lr=args.lr, device=args.device, seed=trial,
            )
            r["N"] = N
            r["trial"] = trial
            r["n_train_models"] = len(train_models)
            r["n_test_models"] = len(test_models)
            per_trial.append(r)
            console.log(
                f"  N={N:<3} trial={trial}  test_acc={r['final_test_acc']:.3f}  "
                f"best_test={r['best_test_acc']:.3f}  "
                f"({r['n_train_models']}tr/{r['n_test_models']}te models)"
            )

        results["by_cohort"][N] = per_trial
        if per_trial:
            best_means = [r["best_test_acc"] for r in per_trial]
            console.log(
                f"  N={N} summary: best_test mean={np.mean(best_means):.3f}  "
                f"std={np.std(best_means):.3f}"
            )

    out_path = out_dir / f"scaling_{args.rung}_layer{args.layer}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    console.rule("done")
    console.log(f"saved {out_path}")


if __name__ == "__main__":
    main()
