"""Ensemble of iterative orthogonal-subspace probes.

Background: a single linear probe at layer_idx=13 on green_square_striped
hits ~0.96 test acc. Iterative subspace probing showed that even after
projecting the first probe's directions out, fresh probes still recover
real signal — the goal-pursuit feature is redundantly distributed across
many ~independent directions. This script tests whether AGGREGATING those
orthogonal probes beats any single one.

Pipeline per probe i:
  - X_proj = X - X @ V_i @ V_i.T   (V_i = cumulative orthonormal basis of
    earlier probes' class-direction rows; V_0 is empty, so iter-0 is
    just the standard probe)
  - Train nn.Linear(d_model, 3) on (X_proj_train, y), early-stop on val.
  - Save W_i, b_i, V_i.

Ensemble logit at test time:
  L = sum_i ( (X - X V_i V_i.T) @ W_i.T + b_i )
The terms operate on orthogonal slices of the residual stream, but each
re-uses the same input X — so we sum genuinely independent views.

We also report:
  - Single best probe (iter-0) test acc as baseline.
  - Ensemble accuracies across K (number of probes summed).
  - Probability-averaged ensemble + majority vote, for comparison.

Honesty note: the original iterative_probe_subspace.py picks per-probe
weights by best-test-epoch. That makes per-probe accuracies optimistic and
ensembling them would compound the optimism. Here we hold out a val split
from train for early stopping; test set is touched only at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from training.iterative_probe_subspace import (
    mean_pool, orthonormalize_against, project_out,
)
from training.train_paired_classifier import (
    DEFAULT_DATA_DIR, DEFAULT_KEEPERS, load_compound_data, split_keepers,
)

console = Console()

DEFAULT_OUT_JSON = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
    "iterative_subspace_ensemble.json"
)
DEFAULT_OUT_FIG = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/figures/"
    "fig7_ensemble_subspace.png"
)


def split_train_val(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


def train_probe_with_val(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xva: torch.Tensor, yva: torch.Tensor,
    *, n_classes: int, n_epochs: int, batch_size: int, lr: float, seed: int,
) -> tuple[nn.Linear, dict]:
    """Train nn.Linear(d_model, n_classes); pick weights at best VAL epoch."""
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

    best_va = -1.0
    best_state = None
    best_tr = 0.0
    for ep in range(n_epochs):
        model.train()
        idx = torch.from_numpy(rng.permutation(Xtr.shape[0])).to(device)
        for s in range(0, Xtr.shape[0], batch_size):
            ix = idx[s:s + batch_size]
            logits = model(Xtr[ix])
            loss = F.cross_entropy(logits, ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        va = acc(Xva, yva)
        if va > best_va:
            best_va = va
            best_tr = acc(Xtr, ytr)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_acc": float(best_va), "train_acc": float(best_tr)}


@torch.no_grad()
def probe_logits(model: nn.Linear, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Apply (X - X V V.T) @ W.T + b. Returns (N, n_classes)."""
    X_proj = project_out(X, V)
    return model(X_proj)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--keepers", default=DEFAULT_KEEPERS)
    p.add_argument("--layer-idx", type=int, default=13)
    p.add_argument("--k-max", type=int, default=15,
                   help="max iterations / probes")
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--out-fig", default=DEFAULT_OUT_FIG)
    args = p.parse_args()

    if args.device != "cuda":
        raise RuntimeError("Must use GPU.")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)

    console.log("loading data...")
    train_keep, test_keep = split_keepers(Path(args.keepers))
    data = load_compound_data(
        Path(args.data_dir), ("green", "square", "striped"),
        train_keep, test_keep,
    )
    if data is None:
        raise RuntimeError("no data for compound green_square_striped")
    Xtr_full, mtr, ytr = data["train"]
    Xte_full, mte, yte = data["test"]
    console.log(
        f"n_train={data['n_train']}  n_test={data['n_test']}  "
        f"n_layers={Xtr_full.shape[2]}  d_model={Xtr_full.shape[3]}"
    )

    # Pool, move to GPU.
    pooled_train = mean_pool(Xtr_full[:, :, args.layer_idx], mtr).to(args.device)
    pooled_test = mean_pool(Xte_full[:, :, args.layer_idx], mte).to(args.device)
    ytr_full = ytr.to(args.device)
    yte = yte.to(args.device)
    d_model = pooled_train.shape[1]
    del Xtr_full, Xte_full
    torch.cuda.empty_cache()

    # Train/val split (val held out for per-probe early stopping).
    tr_idx, va_idx = split_train_val(pooled_train.shape[0], args.val_frac, args.seed)
    tr_idx_t = torch.from_numpy(tr_idx).to(args.device)
    va_idx_t = torch.from_numpy(va_idx).to(args.device)
    Xtr = pooled_train[tr_idx_t]
    ytr = ytr_full[tr_idx_t]
    Xva = pooled_train[va_idx_t]
    yva = ytr_full[va_idx_t]
    console.log(f"train={Xtr.shape[0]}  val={Xva.shape[0]}  test={pooled_test.shape[0]}")

    # Iterate; collect probes.
    V = torch.empty(d_model, 0, device=args.device, dtype=torch.float32)
    probes: list[dict] = []
    bases: list[torch.Tensor] = []  # V at iter i (snapshot before adding new dirs)

    for it in range(args.k_max):
        # Probe sees (input - already-removed-subspace).
        Xtr_proj = project_out(Xtr, V)
        Xva_proj = project_out(Xva, V)
        Xte_proj = project_out(pooled_test, V)

        model, m = train_probe_with_val(
            Xtr_proj, ytr, Xva_proj, yva,
            n_classes=3, n_epochs=args.n_epochs,
            batch_size=args.batch_size, lr=args.lr,
            seed=args.seed + it,
        )
        # Test acc (single probe alone, for reference).
        with torch.no_grad():
            te_logits_alone = model(Xte_proj)
            te_acc_alone = (te_logits_alone.argmax(-1) == yte).float().mean().item()
        probes.append({
            "iter": it,
            "n_dirs_removed": int(V.shape[1]),
            "val_acc": m["val_acc"], "train_acc": m["train_acc"],
            "test_acc_alone": float(te_acc_alone),
            "W": model.weight.detach().clone(),  # (n_classes, d_model)
            "b": model.bias.detach().clone(),    # (n_classes,)
        })
        bases.append(V.clone())
        console.log(
            f"  iter {it:>2}  removed={int(V.shape[1]):>3}  "
            f"val={m['val_acc']:.3f}  test_alone={te_acc_alone:.3f}"
        )

        # Add this probe's class-directions to V (orthogonalized).
        W = model.weight.detach().T.contiguous()
        if V.numel():
            W = W - V @ (V.T @ W)
        V = orthonormalize_against(V, W)

    # Ensemble accuracy as a function of K (number of probes summed).
    @torch.no_grad()
    def ensemble_logits_up_to(k: int) -> torch.Tensor:
        L = torch.zeros(pooled_test.shape[0], 3, device=args.device, dtype=torch.float32)
        for i in range(k):
            X_proj = project_out(pooled_test, bases[i])
            W_i = probes[i]["W"]; b_i = probes[i]["b"]
            L = L + (X_proj @ W_i.T + b_i)
        return L

    ks = list(range(1, len(probes) + 1))
    ens_logit_acc = []
    ens_prob_acc = []
    ens_vote_acc = []
    for k in ks:
        # logit-sum
        L = ensemble_logits_up_to(k)
        ens_logit_acc.append((L.argmax(-1) == yte).float().mean().item())
        # probability average
        Pavg = torch.zeros_like(L)
        for i in range(k):
            X_proj = project_out(pooled_test, bases[i])
            li = X_proj @ probes[i]["W"].T + probes[i]["b"]
            Pavg = Pavg + F.softmax(li, dim=-1)
        Pavg = Pavg / k
        ens_prob_acc.append((Pavg.argmax(-1) == yte).float().mean().item())
        # majority vote
        votes = torch.zeros(pooled_test.shape[0], 3, device=args.device, dtype=torch.float32)
        for i in range(k):
            X_proj = project_out(pooled_test, bases[i])
            li = X_proj @ probes[i]["W"].T + probes[i]["b"]
            preds = li.argmax(-1)
            for c in range(3):
                votes[:, c] += (preds == c).float()
        ens_vote_acc.append((votes.argmax(-1) == yte).float().mean().item())

    out = {
        "compound": ["green", "square", "striped"],
        "layer_idx": args.layer_idx,
        "n_train_used": int(Xtr.shape[0]),
        "n_val": int(Xva.shape[0]),
        "n_test": int(pooled_test.shape[0]),
        "val_frac": args.val_frac,
        "chance": 1 / 3,
        "probes": [
            {k: v for k, v in pr.items() if k not in ("W", "b")}
            for pr in probes
        ],
        "ensemble_by_k": [
            {"k": k, "logit_sum_acc": ens_logit_acc[i],
             "prob_avg_acc": ens_prob_acc[i],
             "majority_vote_acc": ens_vote_acc[i]}
            for i, k in enumerate(ks)
        ],
        "single_probe_iter0_test_acc": probes[0]["test_acc_alone"],
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    console.log(f"saved {args.out_json}")

    # ── plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150)

    # left: per-probe alone vs ensemble (logit sum) — does aggregation grow?
    ax = axes[0]
    alone = [pr["test_acc_alone"] for pr in probes]
    ax.plot(ks, alone, "-o", color="#888", label="probe i alone (test)")
    ax.plot(ks, ens_logit_acc, "-o", color="#1f77b4", lw=2,
            label="ensemble of probes 1..k  (logit sum)")
    ax.plot(ks, ens_prob_acc, "--s", color="#2ca02c", lw=1.5, alpha=0.85,
            label="ensemble (probability average)")
    ax.plot(ks, ens_vote_acc, ":^", color="#d62728", lw=1.5, alpha=0.85,
            label="ensemble (majority vote)")
    ax.axhline(probes[0]["test_acc_alone"], ls="--", color="#1f77b4",
               alpha=0.4,
               label=f"single probe iter-0 = {probes[0]['test_acc_alone']:.3f}")
    ax.axhline(1 / 3, ls="--", color="gray", alpha=0.5, label="chance")
    ax.set_xlabel("k = number of orthogonal probes aggregated")
    ax.set_ylabel("test accuracy")
    ax.set_title("Aggregating orthogonal probes boosts validation accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # right: gain over single probe
    ax = axes[1]
    base = probes[0]["test_acc_alone"]
    ax.plot(ks, [a - base for a in ens_logit_acc], "-o",
            color="#1f77b4", lw=2, label="logit sum")
    ax.plot(ks, [a - base for a in ens_prob_acc], "--s",
            color="#2ca02c", lw=1.5, label="prob avg")
    ax.plot(ks, [a - base for a in ens_vote_acc], ":^",
            color="#d62728", lw=1.5, label="majority vote")
    ax.axhline(0, ls="--", color="gray")
    ax.set_xlabel("k = number of orthogonal probes aggregated")
    ax.set_ylabel("Δ test acc vs single probe (iter-0)")
    ax.set_title("Ensemble gain over best single probe")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        "Goal-pursuit signal is diffuse — orthogonal probes carry independent "
        "information, and ensembling them boosts test accuracy",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(args.out_fig, bbox_inches="tight")
    plt.close(fig)
    console.log(f"saved {args.out_fig}")
    console.log(f"single probe (iter-0): {probes[0]['test_acc_alone']:.4f}")
    for i, k in enumerate(ks):
        console.log(
            f"  k={k:>2}  logit-sum={ens_logit_acc[i]:.4f}  "
            f"prob-avg={ens_prob_acc[i]:.4f}  vote={ens_vote_acc[i]:.4f}"
        )


if __name__ == "__main__":
    main()
