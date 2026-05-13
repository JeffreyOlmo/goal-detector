"""Position sweep — characterize probe accuracy as a function of rollout step.

Two curves at layer 11 with Rung 0 (PooledLinear):

1. Single-position: train on activations at *only* step k (no pooling).
   For each k, only rollouts with steps > k are included.
2. Cumulative-prefix: train on activations from steps [1..k] mean-pooled.

Splits & hyperparameters match training/train_meta_classifier.py:
  train_variants = {0,1,2,3,4}, test_variants = {5,6,7}
  20 epochs, batch_size 128, lr 1e-3, Adam, seed 0.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

from training.train_meta_classifier import (
    PooledLinear,
    N_GOALS,
    load_split,
)

console = Console()

DATA_DIR = Path("/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/activations_v1")
OUT_PATH = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/meta_classifier_v1/position_sweep.json"
)

LAYER = 11
N_EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
SEED = 0
DEVICE = "cuda"


def train_one(Xtr, mtr, ytr, Xte, mte, yte, seed: int = SEED) -> dict:
    """Train a Rung-0 PooledLinear on (X, mask, y) tensors already on device.

    X: (N, T, D) fp32, mask: (N, T) bool, y: (N,) long.
    """
    torch.manual_seed(seed)
    d_model = Xtr.shape[-1]
    model = PooledLinear(d_model, N_GOALS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    rng = np.random.default_rng(seed)
    n_train = Xtr.shape[0]

    @torch.no_grad()
    def eval_acc(X, m, y):
        model.eval()
        n_correct, n_total = 0, 0
        for s in range(0, X.shape[0], BATCH_SIZE):
            logits = model(X[s:s + BATCH_SIZE], m[s:s + BATCH_SIZE])
            n_correct += (logits.argmax(-1) == y[s:s + BATCH_SIZE]).sum().item()
            n_total += logits.shape[0]
        return n_correct / n_total

    best_test = 0.0
    final_train = final_test = 0.0
    for ep in range(N_EPOCHS):
        model.train()
        idx = rng.permutation(n_train)
        idx_t = torch.from_numpy(idx).to(DEVICE)
        for s in range(0, n_train, BATCH_SIZE):
            ix = idx_t[s:s + BATCH_SIZE]
            logits = model(Xtr[ix], mtr[ix])
            loss = F.cross_entropy(logits, ytr[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        final_train = eval_acc(Xtr, mtr, ytr)
        final_test = eval_acc(Xte, mte, yte)
        best_test = max(best_test, final_test)

    return {
        "final_train_acc": final_train,
        "final_test_acc": final_test,
        "best_test_acc": best_test,
    }


def single_position_curve(X_tr, m_tr, y_tr, X_te, m_te, y_te, max_k: int):
    """For each k in [0..max_k-1], train on activations at only step k.

    Only rollouts with steps > k (i.e. mask[:, k] == True) are kept.
    """
    results = []
    for k in range(max_k):
        # Keep rollouts that have a valid step k.
        keep_tr = m_tr[:, k]
        keep_te = m_te[:, k]
        n_tr = int(keep_tr.sum().item())
        n_te = int(keep_te.sum().item())
        if n_tr < N_GOALS or n_te < N_GOALS:
            console.log(
                f"[step {k}] skipped (n_tr={n_tr}, n_te={n_te} too few)"
            )
            results.append({
                "k": k, "n_train": n_tr, "n_test": n_te, "skipped": True,
            })
            continue

        # Slice layer 11, single timestep -> reshape to (N, 1, D) with all-True mask.
        Xtr = X_tr[keep_tr][:, k:k + 1, LAYER].to(torch.float32).to(DEVICE)
        Xte = X_te[keep_te][:, k:k + 1, LAYER].to(torch.float32).to(DEVICE)
        mtr = torch.ones(Xtr.shape[0], 1, dtype=torch.bool, device=DEVICE)
        mte = torch.ones(Xte.shape[0], 1, dtype=torch.bool, device=DEVICE)
        ytr = y_tr[keep_tr].to(DEVICE)
        yte = y_te[keep_te].to(DEVICE)

        r = train_one(Xtr, mtr, ytr, Xte, mte, yte)
        r.update({"k": k, "n_train": n_tr, "n_test": n_te, "skipped": False})
        console.log(
            f"[step {k}] n_tr={n_tr:>3} n_te={n_te:>3} "
            f"train={r['final_train_acc']:.3f} test={r['final_test_acc']:.3f} "
            f"best_test={r['best_test_acc']:.3f}"
        )
        results.append(r)

        del Xtr, Xte, mtr, mte, ytr, yte
        torch.cuda.empty_cache()

    return results


def cumulative_prefix_curve(X_tr, m_tr, y_tr, X_te, m_te, y_te, max_k: int):
    """For each k, train on prefix [0..k] mean-pooled (mask-aware).

    All rollouts contribute; rollouts shorter than k+1 just average over their
    valid prefix (which is what mean-pooling already does mask-aware).
    """
    results = []
    # Pre-slice the layer-11 view once on CPU; we'll cap to k+1 columns each iter.
    # Shapes: X[*, T, L, D] -> X_layer[*, T, D]
    X_tr_layer = X_tr[:, :, LAYER]
    X_te_layer = X_te[:, :, LAYER]

    for k in range(max_k):
        prefix_len = k + 1
        Xtr = X_tr_layer[:, :prefix_len].to(torch.float32).to(DEVICE)
        Xte = X_te_layer[:, :prefix_len].to(torch.float32).to(DEVICE)
        mtr = m_tr[:, :prefix_len].to(DEVICE)
        mte = m_te[:, :prefix_len].to(DEVICE)
        ytr = y_tr.to(DEVICE)
        yte = y_te.to(DEVICE)

        # Skip rollouts whose mask is entirely False over [0..k] (none expected
        # since k=0 always has step 0 valid for any nonempty rollout, but
        # filtering is harmless).
        valid_tr = mtr.any(dim=1)
        valid_te = mte.any(dim=1)
        if not valid_tr.all() or not valid_te.all():
            Xtr = Xtr[valid_tr]; mtr = mtr[valid_tr]; ytr = ytr[valid_tr]
            Xte = Xte[valid_te]; mte = mte[valid_te]; yte = yte[valid_te]

        n_tr = int(Xtr.shape[0])
        n_te = int(Xte.shape[0])

        r = train_one(Xtr, mtr, ytr, Xte, mte, yte)
        r.update({"k": k, "prefix_len": prefix_len,
                  "n_train": n_tr, "n_test": n_te})
        console.log(
            f"[prefix 0..{k}] n_tr={n_tr:>3} n_te={n_te:>3} "
            f"train={r['final_train_acc']:.3f} test={r['final_test_acc']:.3f} "
            f"best_test={r['best_test_acc']:.3f}"
        )
        results.append(r)

        del Xtr, Xte, mtr, mte, ytr, yte
        torch.cuda.empty_cache()

    return results


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    console.rule("loading activations")
    (X_tr, m_tr, y_tr), (X_te, m_te, y_te) = load_split(DATA_DIR)
    max_T = X_tr.shape[1]
    max_T_test = X_te.shape[1]
    max_k = min(max_T, max_T_test)
    console.log(
        f"train rollouts: {X_tr.shape[0]}  test rollouts: {X_te.shape[0]}  "
        f"max_T_train: {max_T}  max_T_test: {max_T_test}  layer: {LAYER}"
    )
    train_steps = m_tr.sum(dim=1)
    test_steps = m_te.sum(dim=1)
    console.log(
        f"train step counts: min={int(train_steps.min())} median={int(train_steps.median())} "
        f"max={int(train_steps.max())} | test: min={int(test_steps.min())} "
        f"median={int(test_steps.median())} max={int(test_steps.max())}"
    )

    console.rule("single-position curve (layer 11, Rung 0, step k only)")
    single = single_position_curve(X_tr, m_tr, y_tr, X_te, m_te, y_te, max_k)

    console.rule("cumulative-prefix curve (layer 11, Rung 0, mean-pool 0..k)")
    cumulative = cumulative_prefix_curve(X_tr, m_tr, y_tr, X_te, m_te, y_te, max_k)

    out = {
        "layer": LAYER,
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seed": SEED,
        "chance": 1.0 / N_GOALS,
        "max_k": max_k,
        "single_position": single,
        "cumulative_prefix": cumulative,
    }
    with OUT_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    console.log(f"saved {OUT_PATH}")

    # ---- summary table ----
    console.rule("summary")
    console.log(
        f"{'k':>3} | {'N_tr_sp':>7} {'N_te_sp':>7} {'sp_test':>7} {'sp_best':>7} | "
        f"{'N_tr_cp':>7} {'N_te_cp':>7} {'cp_test':>7} {'cp_best':>7}"
    )
    for k in range(max_k):
        sp = single[k]
        cp = cumulative[k]
        if sp.get("skipped"):
            sp_str = f"{sp['n_train']:>7} {sp['n_test']:>7} {'--':>7} {'--':>7}"
        else:
            sp_str = (f"{sp['n_train']:>7} {sp['n_test']:>7} "
                      f"{sp['final_test_acc']:>7.3f} {sp['best_test_acc']:>7.3f}")
        cp_str = (f"{cp['n_train']:>7} {cp['n_test']:>7} "
                  f"{cp['final_test_acc']:>7.3f} {cp['best_test_acc']:>7.3f}")
        console.log(f"{k:>3} | {sp_str} | {cp_str}")

    # Peak step-k single-position
    valid_sp = [r for r in single if not r.get("skipped")]
    if valid_sp:
        peak = max(valid_sp, key=lambda r: r["final_test_acc"])
        console.log(
            f"\nsingle-position peak: k={peak['k']} test={peak['final_test_acc']:.3f} "
            f"best={peak['best_test_acc']:.3f} (N_te={peak['n_test']})"
        )
    final_cp = cumulative[-1]
    console.log(
        f"cumulative-prefix final (k={final_cp['k']}): test={final_cp['final_test_acc']:.3f} "
        f"best={final_cp['best_test_acc']:.3f}"
    )


if __name__ == "__main__":
    main()
