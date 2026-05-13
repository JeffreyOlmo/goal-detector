"""Ensemble of linear probes trained at *different residual-stream layers*.

We already showed that within one layer, iterative orthogonal probes don't
ensemble well — each later probe operates on a strict subset of the
information and provides correlated views. A more principled alternative:
train one probe per layer (each on the full d_model activations of THAT
layer), then ensemble. Different layers encode different abstractions, so
their errors should decorrelate more than orthogonal slices of one layer.

For each layer in the saved activations, train a fresh linear probe with
train→val early stopping, score on test, save weights. Then sweep K and
build ensembles in three ways:
  - logit sum
  - probability average
  - majority vote
Compare to the single best-layer probe baseline.
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

from training.iterative_probe_subspace import mean_pool
from training.iterative_probe_ensemble import (
    split_train_val, train_probe_with_val,
)
from training.train_paired_classifier import (
    DEFAULT_DATA_DIR, DEFAULT_KEEPERS, load_compound_data, split_keepers,
)

console = Console()

DEFAULT_OUT_JSON = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
    "multilayer_ensemble.json"
)
DEFAULT_OUT_FIG = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/figures/"
    "fig8_multilayer_ensemble.png"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--keepers", default=DEFAULT_KEEPERS)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--out-fig", default=DEFAULT_OUT_FIG)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

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
    Xtr_full, mtr, ytr_full = data["train"]
    Xte_full, mte, yte = data["test"]
    n_layers = Xtr_full.shape[2]
    console.log(f"layers available: {n_layers}")

    # train→val split (per-rollout)
    tr_idx, va_idx = split_train_val(Xtr_full.shape[0], args.val_frac, args.seed)
    tr_idx_t = torch.from_numpy(tr_idx)
    va_idx_t = torch.from_numpy(va_idx)
    yte = yte.to(args.device)

    layers_data: list[dict] = []
    test_logits_per_layer: list[torch.Tensor] = []  # (N_test, 3)

    for L in range(n_layers):
        Xtr_L = Xtr_full[:, :, L]
        Xte_L = Xte_full[:, :, L]
        pooled_train = mean_pool(Xtr_L, mtr).to(args.device)
        pooled_test = mean_pool(Xte_L, mte).to(args.device)
        Xtr = pooled_train[tr_idx_t]
        ytr = ytr_full[tr_idx_t].to(args.device)
        Xva = pooled_train[va_idx_t]
        yva = ytr_full[va_idx_t].to(args.device)

        model, m = train_probe_with_val(
            Xtr, ytr, Xva, yva,
            n_classes=3, n_epochs=args.n_epochs,
            batch_size=args.batch_size, lr=args.lr,
            seed=args.seed + L,
        )
        with torch.no_grad():
            te_logits = model(pooled_test)  # (N_test, 3)
            te_acc = (te_logits.argmax(-1) == yte).float().mean().item()
        layers_data.append({
            "layer_idx": L,
            "val_acc": m["val_acc"],
            "train_acc": m["train_acc"],
            "test_acc_alone": float(te_acc),
        })
        test_logits_per_layer.append(te_logits.detach().cpu())
        console.log(
            f"  layer {L:>2}  val={m['val_acc']:.3f}  test={te_acc:.3f}"
        )

        del pooled_train, pooled_test, Xtr, Xva
        torch.cuda.empty_cache()

    # Ensemble strategies. Sort layers by val_acc descending (best first).
    order = sorted(range(n_layers), key=lambda i: -layers_data[i]["val_acc"])

    @torch.no_grad()
    def ensemble_at(k: int, mode: str) -> float:
        idxs = order[:k]
        if mode == "logit_sum":
            L = sum(test_logits_per_layer[i] for i in idxs)
        elif mode == "prob_avg":
            L = sum(F.softmax(test_logits_per_layer[i], dim=-1) for i in idxs)
        elif mode == "vote":
            votes = torch.zeros(test_logits_per_layer[0].shape[0], 3)
            for i in idxs:
                preds = test_logits_per_layer[i].argmax(-1)
                for c in range(3):
                    votes[:, c] += (preds == c).float()
            L = votes
        else:
            raise ValueError(mode)
        preds = L.argmax(-1).to(args.device)
        return (preds == yte).float().mean().item()

    ks = list(range(1, n_layers + 1))
    ens_logit = [ensemble_at(k, "logit_sum") for k in ks]
    ens_prob = [ensemble_at(k, "prob_avg") for k in ks]
    ens_vote = [ensemble_at(k, "vote") for k in ks]

    best_alone = max(d["test_acc_alone"] for d in layers_data)
    best_alone_layer = max(layers_data, key=lambda d: d["test_acc_alone"])["layer_idx"]

    out = {
        "compound": ["green", "square", "striped"],
        "n_layers": n_layers,
        "n_test": int(yte.shape[0]),
        "layers": layers_data,
        "order_by_val_acc": order,
        "ensemble_by_k": [
            {"k": k, "logit_sum_acc": ens_logit[i],
             "prob_avg_acc": ens_prob[i],
             "majority_vote_acc": ens_vote[i]}
            for i, k in enumerate(ks)
        ],
        "best_single_layer": {
            "layer_idx": best_alone_layer, "test_acc": best_alone,
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    console.log(f"saved {args.out_json}")

    # ── plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150)

    ax = axes[0]
    Ls = [d["layer_idx"] for d in layers_data]
    accs = [d["test_acc_alone"] for d in layers_data]
    ax.plot(Ls, accs, "-o", color="#1f77b4", label="single-layer test acc")
    ax.axhline(best_alone, ls="--", color="#1f77b4", alpha=0.5,
               label=f"best single layer = {best_alone:.3f}  (L={best_alone_layer})")
    ax.axhline(1 / 3, ls="--", color="gray", alpha=0.5, label="chance")
    ax.set_xlabel("residual-stream layer index")
    ax.set_ylabel("test accuracy")
    ax.set_title("Per-layer linear probe accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")

    ax = axes[1]
    ax.plot(ks, ens_logit, "-o", color="#1f77b4", lw=2,
            label="ensemble (logit sum)")
    ax.plot(ks, ens_prob, "--s", color="#2ca02c", lw=1.5,
            label="ensemble (prob avg)")
    ax.plot(ks, ens_vote, ":^", color="#d62728", lw=1.5,
            label="ensemble (majority vote)")
    ax.axhline(best_alone, ls="--", color="black", alpha=0.5,
               label=f"best single layer = {best_alone:.3f}")
    ax.set_xlabel("k = number of layers in ensemble (best-val-first)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Multi-layer ensemble vs best single layer")
    ax.set_ylim(0.3, 1.02)
    ax.set_xticks(ks[::2])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Cross-layer ensemble: do probes at different layers carry "
        "independent enough errors to beat the best single layer?",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(args.out_fig, bbox_inches="tight")
    plt.close(fig)
    console.log(f"saved {args.out_fig}")
    console.log(f"best single layer (L={best_alone_layer}): {best_alone:.4f}")
    for i, k in enumerate(ks):
        console.log(
            f"  k={k:>2}  logit-sum={ens_logit[i]:.4f}  "
            f"prob-avg={ens_prob[i]:.4f}  vote={ens_vote[i]:.4f}"
        )


if __name__ == "__main__":
    main()
