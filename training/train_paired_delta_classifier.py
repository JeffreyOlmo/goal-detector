"""3-way paired classifier on first-state activations: full vs delta.

For each compound C, the 3 ambiguity-mate goal-LoRAs (attr, val) ∈ mates(C)
all roll out on the same envs (paired methodology). We train a 3-way
classifier per compound predicting which mate (color / shape / pattern
interpretation) drove the rollout.

Three input modes per (LoRA variant, env_idx):
  - "full"  : LoRA-on activation at FIRST decision step (already saved in
              data/paired_activations_v2/{attr}_{val}_v{variant}.pt at
              rollouts[i]['activations'][0]).
  - "delta" : (LoRA-on first step) − (base first step on same env). Base
              activations come from data/paired_base_first_v2/{compound}.pt.
  - "concat": full || base (sanity check: does the linear classifier
              implicitly subtract?).

Layer sweep over the 19 saved layer indices (LAYER_IDXS = range(0,37,2)).
Headline metric: best test accuracy per compound, per input mode.

Compares to the existing paired_3way_sweep baseline (pooled / EMA full
activations across the whole rollout — best 0.78–0.96, mean ~0.85).
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
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from goal_detector.gridworld.ambiguous_env import ALL_COMPOUNDS, ambiguity_mates
from training.train_paired_classifier import DEFAULT_KEEPERS, split_keepers

console = Console()

PAIRED_DATA_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/paired_activations_v2"
)
BASE_DATA_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/paired_base_first_v2"
)
DEFAULT_OUT = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/"
    "meta_classifier_paired_v2/first_state_delta_sweep.json"
)


def load_base_activations(compound: tuple[str, str, str]) -> dict[int, torch.Tensor]:
    """env_idx -> (n_layers, d_model) fp16."""
    path = Path(BASE_DATA_DIR) / f"{'_'.join(compound)}.pt"
    d = torch.load(path, weights_only=False)
    return {e["env_idx"]: e["base_activation"] for e in d["envs"]}


def load_compound_first_state(
    compound: tuple[str, str, str],
    train_keep: set[tuple[str, str, int]],
    test_keep: set[tuple[str, str, int]],
) -> dict | None:
    """For one compound, gather (full_first, base_first, label) per
    (variant, env_idx) where variant pursues an attribute mate of `compound`.
    Splits into train/test by `train_keep` / `test_keep` keepers."""
    mates = ambiguity_mates(compound)
    base_acts = load_base_activations(compound)  # env_idx -> (n_layers, d_model)

    train_rows: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    test_rows: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for label, (attr, val) in enumerate(mates):
        for keep_set, bucket in (
            (train_keep, train_rows), (test_keep, test_rows),
        ):
            variants = sorted(
                v for (a, vv, var) in keep_set
                if a == attr and vv == val for v in [var]
            )
            for variant in variants:
                path = Path(PAIRED_DATA_DIR) / f"{attr}_{val}_v{variant}.pt"
                if not path.exists():
                    continue
                d = torch.load(path, weights_only=False)
                for r in d["rollouts"]:
                    if tuple(r["compound"]) != compound:
                        continue
                    full_first = r["activations"][0]  # (n_layers, d_model)
                    base_first = base_acts[r["env_idx"]]  # (n_layers, d_model)
                    bucket.append((full_first, base_first, label))

    if not train_rows or not test_rows:
        return None

    def stack(rs):
        full = torch.stack([f for f, _, _ in rs])  # (N, L, d)
        base = torch.stack([b for _, b, _ in rs])  # (N, L, d)
        y = torch.tensor([l for _, _, l in rs], dtype=torch.long)
        return full, base, y

    return {
        "compound": compound,
        "train": stack(train_rows),
        "test": stack(test_rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
    }


def train_layer_probe(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    *, n_classes: int, n_epochs: int, batch_size: int,
    lr: float, seed: int, device: str,
) -> dict:
    """Linear probe (single Linear) on (N, d) inputs. Returns best test acc."""
    torch.manual_seed(seed)
    Xtr = Xtr.float().to(device); Xte = Xte.float().to(device)
    ytr = ytr.to(device); yte = yte.to(device)
    d = Xtr.shape[1]
    model = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def acc(X, y):
        model.eval()
        n_correct, n_total = 0, 0
        for s in range(0, X.shape[0], batch_size):
            logits = model(X[s:s + batch_size])
            n_correct += (logits.argmax(-1) == y[s:s + batch_size]).sum().item()
            n_total += logits.shape[0]
        return n_correct / max(1, n_total)

    best_test, best_train = 0.0, 0.0
    for ep in range(n_epochs):
        model.train()
        idx = torch.from_numpy(rng.permutation(Xtr.shape[0])).to(device)
        for s in range(0, Xtr.shape[0], batch_size):
            ix = idx[s:s + batch_size]
            logits = model(Xtr[ix])
            loss = F.cross_entropy(logits, ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        tr = acc(Xtr, ytr); te = acc(Xte, yte)
        if te > best_test:
            best_test = te; best_train = tr
    return {"best_test_acc": float(best_test),
            "train_acc_at_best": float(best_train)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--keepers", default=DEFAULT_KEEPERS)
    p.add_argument("--n-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--modes", default="full,delta,concat",
                   help="comma-sep input modes")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    if args.device != "cuda":
        raise RuntimeError("Must use GPU.")

    train_keep, test_keep = split_keepers(Path(args.keepers))
    console.log(f"keepers: train={len(train_keep)}  test={len(test_keep)}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: dict = {"chance": 1 / 3, "modes": modes, "compounds": {}}
    for compound in ALL_COMPOUNDS:
        cname = "_".join(compound)
        console.rule(cname)
        data = load_compound_first_state(compound, train_keep, test_keep)
        if data is None:
            console.log("  [skip] insufficient train/test data")
            continue
        full_tr, base_tr, ytr = data["train"]
        full_te, base_te, yte = data["test"]
        n_layers = full_tr.shape[1]
        console.log(f"  n_train={data['n_train']}  n_test={data['n_test']}  "
                    f"n_layers={n_layers}  d_model={full_tr.shape[2]}")
        per_compound: dict = {"n_train": data["n_train"],
                              "n_test": data["n_test"],
                              "modes": {}}
        for mode in modes:
            per_layer: list[dict] = []
            for li in range(n_layers):
                if mode == "full":
                    Xtr = full_tr[:, li]; Xte = full_te[:, li]
                elif mode == "delta":
                    Xtr = full_tr[:, li] - base_tr[:, li]
                    Xte = full_te[:, li] - base_te[:, li]
                elif mode == "concat":
                    Xtr = torch.cat([full_tr[:, li], base_tr[:, li]], dim=-1)
                    Xte = torch.cat([full_te[:, li], base_te[:, li]], dim=-1)
                else:
                    raise ValueError(f"unknown mode {mode}")
                m = train_layer_probe(
                    Xtr, ytr, Xte, yte, n_classes=3,
                    n_epochs=args.n_epochs, batch_size=args.batch_size,
                    lr=args.lr, seed=args.seed + li, device=args.device,
                )
                per_layer.append({"layer_idx": li, **m})
            best = max(per_layer, key=lambda r: r["best_test_acc"])
            console.log(
                f"  {mode:6} best={best['best_test_acc']:.3f} "
                f"at L{best['layer_idx']:>2}  "
                f"(train_at_best={best['train_acc_at_best']:.3f})"
            )
            per_compound["modes"][mode] = per_layer
        results["compounds"][cname] = per_compound

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    console.log(f"saved {out}")

    # ---- summary ----
    console.rule("summary (best test acc per mode)")
    for mode in modes:
        accs = []
        for cname, cd in results["compounds"].items():
            best = max(cd["modes"][mode], key=lambda r: r["best_test_acc"])
            accs.append(best["best_test_acc"])
        if accs:
            console.log(
                f"  {mode:6}: mean={sum(accs)/len(accs):.3f}  "
                f"min={min(accs):.3f}  max={max(accs):.3f}  n={len(accs)}"
            )


if __name__ == "__main__":
    main()
