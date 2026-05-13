"""Iterative orthogonal subspace probing on layer-26 (layer_idx=13) activations
of `green_square_striped` compound rollouts.

Question: is the 3-way goal-pursuit signal carried by ONE direction in the
residual stream, or by a higher-rank subspace? We iteratively train a fresh
PooledLinear-style probe, then orthonormalize its 3 class-direction rows into
a cumulative basis V, project them out of the activations, and retrain. The
number of iterations until probe accuracy collapses gives the effective rank
(upper bound 3 * iterations_to_collapse).

Outputs:
  - results/drift_probes/iterative_subspace.json
  - results/figures/fig5_iterative_subspace.png
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

from training.train_paired_classifier import (
    DEFAULT_DATA_DIR,
    DEFAULT_KEEPERS,
    load_compound_data,
    split_keepers,
)

console = Console()

DEFAULT_OUT_JSON = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
    "iterative_subspace.json"
)
DEFAULT_OUT_FIG = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/figures/"
    "fig5_iterative_subspace.png"
)


def mean_pool(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """X: (N, T, d_model) fp16; mask: (N, T) bool. Returns (N, d_model) fp32."""
    m = mask.float().unsqueeze(-1)
    pooled = (X.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return pooled


def project_out(X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Remove from each row of X the component along the columns of V.
    V: (d_model, k) with orthonormal columns. Returns X - X @ V @ V.T."""
    if V.numel() == 0:
        return X
    coeffs = X @ V  # (N, k)
    return X - coeffs @ V.T


def orthonormalize_against(V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Append columns of W to V after orthogonalizing each column against V
    and against earlier columns of W (modified Gram-Schmidt). Drop columns
    whose residual norm is below tol (rank-deficient)."""
    cols = [V[:, i] for i in range(V.shape[1])] if V.numel() else []
    tol = 1e-6
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


def train_linear_probe(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    n_classes: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[nn.Linear, dict]:
    """Train nn.Linear(d_model, n_classes) on (Xtr, ytr); pick best test
    epoch's weights. Returns the model with best-test weights restored, plus
    metrics."""
    torch.manual_seed(seed)
    device = Xtr.device
    d_model = Xtr.shape[1]
    model = nn.Linear(d_model, n_classes).to(device)
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

    best_test = -1.0
    best_state = None
    best_train_acc = 0.0
    for ep in range(n_epochs):
        model.train()
        idx = torch.from_numpy(rng.permutation(Xtr.shape[0])).to(device)
        for s in range(0, Xtr.shape[0], batch_size):
            ix = idx[s:s + batch_size]
            logits = model(Xtr[ix])
            loss = F.cross_entropy(logits, ytr[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = acc(Xtr, ytr)
        te = acc(Xte, yte)
        if te > best_test:
            best_test = te
            best_train_acc = tr
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_test_acc": float(best_test),
        "train_acc_at_best": float(best_train_acc),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--keepers", default=DEFAULT_KEEPERS)
    p.add_argument("--layer-idx", type=int, default=13)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--collapse-thresh", type=float, default=0.45)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
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
        Path(args.data_dir),
        ("green", "square", "striped"),
        train_keep,
        test_keep,
    )
    if data is None:
        raise RuntimeError("no data for compound green_square_striped")
    Xtr_full, mtr, ytr = data["train"]
    Xte_full, mte, yte = data["test"]
    console.log(
        f"n_train={data['n_train']}  n_test={data['n_test']}  "
        f"n_layers={Xtr_full.shape[2]}  d_model={Xtr_full.shape[3]}"
    )

    # Pull the target layer and mean-pool. Final shapes (N, d_model) fp32.
    Xtr_l = Xtr_full[:, :, args.layer_idx]
    Xte_l = Xte_full[:, :, args.layer_idx]
    pooled_train = mean_pool(Xtr_l, mtr).to(args.device)
    pooled_test = mean_pool(Xte_l, mte).to(args.device)
    ytr = ytr.to(args.device)
    yte = yte.to(args.device)
    d_model = pooled_train.shape[1]
    console.log(
        f"pooled: train={tuple(pooled_train.shape)} test={tuple(pooled_test.shape)} "
        f"dtype={pooled_train.dtype}"
    )

    # Free the big padded tensors.
    del Xtr_full, Xte_full, Xtr_l, Xte_l
    torch.cuda.empty_cache()

    V = torch.empty(d_model, 0, device=args.device, dtype=torch.float32)
    results: list[dict] = []

    for it in range(args.k_max + 1):
        Xtr_proj = project_out(pooled_train, V)
        Xte_proj = project_out(pooled_test, V)

        model, metrics = train_linear_probe(
            Xtr_proj,
            ytr,
            Xte_proj,
            yte,
            n_classes=3,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + it,
        )
        rank_V = int(V.shape[1])
        entry = {
            "iteration": it,
            "n_dirs_removed": rank_V,
            "rank_V": rank_V,
            "best_test_acc": metrics["best_test_acc"],
            "train_acc": metrics["train_acc_at_best"],
        }
        results.append(entry)
        console.log(
            f"iter {it:>2}  removed={rank_V:>3}  "
            f"test_acc={metrics['best_test_acc']:.4f}  "
            f"train_acc={metrics['train_acc_at_best']:.4f}"
        )

        if metrics["best_test_acc"] < args.collapse_thresh:
            console.log(
                f"collapse: test acc {metrics['best_test_acc']:.4f} "
                f"< {args.collapse_thresh}; stopping."
            )
            break

        # Append the 3 class-direction rows of fc.weight to V (post-Gram-Schmidt).
        # nn.Linear(d_in, n_classes).weight has shape (n_classes, d_in).
        W = model.weight.detach().T.contiguous()  # (d_model, n_classes)
        # Sanity: project the new directions through the existing P first to
        # match the subspace already removed.
        if V.numel():
            W = W - V @ (V.T @ W)
        V = orthonormalize_against(V, W)

    out = {
        "compound": ["green", "square", "striped"],
        "layer_idx": args.layer_idx,
        "n_train": data["n_train"],
        "n_test": data["n_test"],
        "chance": 1 / 3,
        "collapse_thresh": args.collapse_thresh,
        "iterations": results,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    console.log(f"saved {args.out_json}")

    # ---- figure ----
    iters = [r["iteration"] for r in results]
    accs = [r["best_test_acc"] for r in results]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(iters, accs, marker="o", lw=2, color="C0", label="probe best test acc")
    ax.axhline(1 / 3, ls="--", color="gray", label="chance (1/3)")
    if results:
        ax.axhline(
            results[0]["best_test_acc"],
            ls=":",
            color="C2",
            label=f"iter-0 acc ({results[0]['best_test_acc']:.3f})",
        )
    ax.axhline(args.collapse_thresh, ls="--", color="C3", alpha=0.5,
               label=f"collapse thresh ({args.collapse_thresh})")

    # Annotate the iteration where acc first drops below the threshold.
    collapse_iter = None
    for r in results:
        if r["best_test_acc"] < args.collapse_thresh:
            collapse_iter = r["iteration"]
            break
    if collapse_iter is not None:
        ax.axvline(collapse_iter, ls=":", color="C3", alpha=0.6)
        ax.annotate(
            f"effective rank ≈ {3 * collapse_iter}\n"
            f"(collapse at iter {collapse_iter})",
            xy=(collapse_iter, args.collapse_thresh),
            xytext=(collapse_iter + 0.3, args.collapse_thresh + 0.05),
            arrowprops=dict(arrowstyle="->", color="C3", alpha=0.6),
            fontsize=9,
        )

    ax.set_xlabel("iteration  (cumulative dirs removed = 3 * iteration)")
    ax.set_ylabel("best test accuracy")
    ax.set_title(
        "Iterative orthogonal subspace probing\n"
        "compound green_square_striped, layer_idx=13 (model layer 26)"
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(iters)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)
    console.log(f"saved {args.out_fig}")


if __name__ == "__main__":
    main()
