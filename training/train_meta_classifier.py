"""Step 6 — meta-classifier rungs.

Loads activation tensors saved by ``training.extract_activations`` and trains
linear / recurrent probes to predict the model's true goal label from rollout
activations. Sweeps across layers (Rungs 0-2) → layer-vs-accuracy curve.

Splits per README §Models:
    train_variants = {0,1,2,3,4}   (5 per goal × 7 goals = 35 models)
    test_variants  = {5,6,7}       (3 per goal × 7 goals = 21 models held out)

Held-out test models are the central evidence of cross-model generalization.

Optional controls:
    --permute-labels  → shuffle (rollout, goal) labels; expect chance acc.

Performance: pre-stacks all rollouts into a single padded tensor in memory,
then per-layer training only slices that one layer (no per-batch reload).
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

console = Console()

DEFAULT_DATA_DIR = "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/activations_v1"
DEFAULT_OUT_DIR = "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/meta_classifier_v1"

TRAIN_VARIANTS = (0, 1, 2, 3, 4)
TEST_VARIANTS = (5, 6, 7)

GOALS = (
    ("color", "red"),
    ("color", "blue"),
    ("color", "green"),
    ("shape", "circle"),
    ("shape", "square"),
    ("pattern", "solid"),
    ("pattern", "striped"),
)
GOAL_TO_LABEL = {(a, v): i for i, (a, v) in enumerate(GOALS)}
N_GOALS = len(GOALS)


# ---- data --------------------------------------------------------------

def load_split(data_dir: Path):
    """Stack all rollouts in each split into a padded tensor.

    Returns dict with:
        X    : (N, T, L, D) fp16 — activations padded to max length
        mask : (N, T) bool
        y    : (N,)   long
    for each of 'train' and 'test' splits.
    """
    train_rollouts = []
    test_rollouts = []
    for path in sorted(data_dir.glob("*.pt")):
        d = torch.load(path, weights_only=False)
        attr, val, var = d["goal_attribute"], d["goal_value"], d["variant"]
        label = GOAL_TO_LABEL.get((attr, val))
        if label is None:
            continue
        bucket = train_rollouts if var in TRAIN_VARIANTS else test_rollouts
        for r in d["rollouts"]:
            bucket.append((r["activations"], label))

    def stack(rollouts):
        max_t = max(a.shape[0] for a, _ in rollouts)
        n_layers = rollouts[0][0].shape[1]
        d_model = rollouts[0][0].shape[2]
        N = len(rollouts)
        X = torch.zeros(N, max_t, n_layers, d_model, dtype=torch.float16)
        mask = torch.zeros(N, max_t, dtype=torch.bool)
        y = torch.zeros(N, dtype=torch.long)
        for i, (acts, label) in enumerate(rollouts):
            t = acts.shape[0]
            X[i, :t] = acts
            mask[i, :t] = True
            y[i] = label
        return X, mask, y

    return stack(train_rollouts), stack(test_rollouts)


# ---- rungs -------------------------------------------------------------

class PooledLinear(nn.Module):
    """Rung 0. Mean-pool across positions (mask-aware), linear readout.
    Parameter count: d_model × n_goals."""

    def __init__(self, d_model: int, n_goals: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_goals)

    def forward(self, x, mask):
        m = mask.float().unsqueeze(-1)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.fc(pooled)


class PerPositionLinear(nn.Module):
    """Rung 1. Linear at each position; logits averaged at inference. Same
    parameter count as Rung 0."""

    def __init__(self, d_model: int, n_goals: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_goals)

    def forward(self, x, mask):
        logits = self.fc(x)  # (B, T, G)
        m = mask.float().unsqueeze(-1)
        return (logits * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


class RecurrentEMA(nn.Module):
    """Rung 2. Per-position linear, then learned scalar EMA across positions
    (causal). Output is the EMA at the last valid step. +1 parameter."""

    def __init__(self, d_model: int, n_goals: int, ema_init: float = 0.5):
        super().__init__()
        self.proj = nn.Linear(d_model, n_goals)
        self.alpha_logit = nn.Parameter(
            torch.tensor(float(np.log(ema_init / (1 - ema_init))))
        )

    def forward(self, x, mask):
        z = self.proj(x)  # (B, T, G)
        alpha = torch.sigmoid(self.alpha_logit)
        T = z.shape[1]
        T_b = mask.sum(dim=1, keepdim=True).float()
        positions = torch.arange(T, device=z.device).unsqueeze(0).float()
        exponent = (T_b - 1 - positions).clamp(min=0)
        weights = alpha * (1 - alpha) ** exponent  # (B, T)
        weights = weights * mask.float()
        return (z * weights.unsqueeze(-1)).sum(dim=1)


class DownProjPooled(nn.Module):
    """Rung 3a. Linear down-projection to small bottleneck, mean-pool, linear
    readout. The bottleneck enforces information compression before pooling,
    so the probe cannot memorize per-model fingerprints — only goal-related
    structure that fits in `bottleneck` dimensions can pass through.

    Total params: d_model*bottleneck + bottleneck*n_goals. For bottleneck=16
    this is ~41k vs Rung0's 18k — *more* params total, but capacity is
    bottlenecked by the projection rank, which is what matters for
    memorization."""

    def __init__(self, d_model: int, n_goals: int, bottleneck: int = 16):
        super().__init__()
        self.proj = nn.Linear(d_model, bottleneck, bias=False)
        self.fc = nn.Linear(bottleneck, n_goals)

    def forward(self, x, mask):
        z = self.proj(x)  # (B, T, bottleneck)
        m = mask.float().unsqueeze(-1)
        pooled = (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.fc(pooled)


def _make_downproj(bottleneck: int):
    def cls(d_model, n_goals):
        return DownProjPooled(d_model, n_goals, bottleneck=bottleneck)
    cls.__name__ = f"DownProjPooled_b{bottleneck}"
    return cls


RUNGS = {
    "rung0_pooled": PooledLinear,
    "rung1_perpos": PerPositionLinear,
    "rung2_ema": RecurrentEMA,
    "rung3a_downproj_b16": _make_downproj(16),
    "rung3a_downproj_b32": _make_downproj(32),
    "rung3a_downproj_b64": _make_downproj(64),
}


# ---- training ----------------------------------------------------------

def train_probe_at_layer(
    rung_cls,
    X_train, mask_train, y_train,
    X_test, mask_test, y_test,
    layer_idx: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> dict:
    """Trains one probe at a single layer. Slices that layer once and keeps
    the (N, T, D) tensor on GPU for the duration of training."""
    torch.manual_seed(seed)
    d_model = X_train.shape[3]

    # Slice layer once and ship to GPU as fp32.
    Xtr = X_train[:, :, layer_idx].to(torch.float32).to(device)
    Xte = X_test[:, :, layer_idx].to(torch.float32).to(device)
    mtr = mask_train.to(device)
    mte = mask_test.to(device)
    ytr = y_train.to(device)
    yte = y_test.to(device)

    model = rung_cls(d_model, N_GOALS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rng = np.random.default_rng(seed)
    n_train = Xtr.shape[0]

    @torch.no_grad()
    def eval_acc(X, m, y):
        model.eval()
        n_correct, n_total = 0, 0
        for s in range(0, X.shape[0], batch_size):
            logits = model(X[s:s + batch_size], m[s:s + batch_size])
            n_correct += (logits.argmax(-1) == y[s:s + batch_size]).sum().item()
            n_total += logits.shape[0]
        return n_correct / n_total

    best_test = 0.0
    history = []
    for ep in range(n_epochs):
        model.train()
        idx = rng.permutation(n_train)
        idx_t = torch.from_numpy(idx).to(device)
        for s in range(0, n_train, batch_size):
            ix = idx_t[s:s + batch_size]
            logits = model(Xtr[ix], mtr[ix])
            loss = F.cross_entropy(logits, ytr[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_acc = eval_acc(Xtr, mtr, ytr)
        test_acc = eval_acc(Xte, mte, yte)
        best_test = max(best_test, test_acc)
        history.append({"epoch": ep, "train_acc": train_acc, "test_acc": test_acc})

    # Free per-layer tensors before returning.
    del Xtr, Xte
    torch.cuda.empty_cache()

    return {
        "final_train_acc": train_acc,
        "final_test_acc": test_acc,
        "best_test_acc": best_test,
        "history": history,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rungs", default=",".join(RUNGS.keys()))
    p.add_argument("--permute-labels", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule("loading activations")
    (X_tr, m_tr, y_tr), (X_te, m_te, y_te) = load_split(Path(args.data_dir))

    if args.permute_labels:
        rng = np.random.default_rng(args.seed)
        perm_tr = rng.permutation(y_tr.shape[0])
        perm_te = rng.permutation(y_te.shape[0])
        y_tr = y_tr[torch.from_numpy(perm_tr)]
        y_te = y_te[torch.from_numpy(perm_te)]
        console.log("[control] labels permuted")

    n_layers = X_tr.shape[2]
    d_model = X_tr.shape[3]
    console.log(
        f"train rollouts: {X_tr.shape[0]}  test rollouts: {X_te.shape[0]}  "
        f"max_T: {X_tr.shape[1]}  layers: {n_layers}  d_model: {d_model}"
    )
    console.log(f"X_tr fp16 mem: {X_tr.element_size() * X_tr.numel() / 1e9:.2f} GB")

    results: dict = {"chance": 1.0 / N_GOALS, "rungs": {}}
    for rung_name in args.rungs.split(","):
        rung_cls = RUNGS[rung_name]
        console.rule(rung_name)
        per_layer = []
        for layer_idx in range(n_layers):
            r = train_probe_at_layer(
                rung_cls,
                X_tr, m_tr, y_tr, X_te, m_te, y_te,
                layer_idx=layer_idx,
                n_epochs=args.n_epochs, batch_size=args.batch_size,
                lr=args.lr, device=args.device, seed=args.seed,
            )
            r["layer_idx"] = layer_idx
            per_layer.append(r)
            console.log(
                f"  layer {layer_idx:>2}: train={r['final_train_acc']:.3f}  "
                f"test={r['final_test_acc']:.3f}  best_test={r['best_test_acc']:.3f}"
            )
        results["rungs"][rung_name] = per_layer

    suffix = "_permuted" if args.permute_labels else ""
    out_path = out_dir / f"rung_layer_sweep{suffix}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    console.log(f"saved {out_path}")


if __name__ == "__main__":
    main()
