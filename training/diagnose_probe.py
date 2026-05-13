"""Per-axis accuracy diagnostic.

Loads activations + retrains the best-rung-best-layer probe (Rung 2, layer 11
in the v1 sweep) on train models, evaluates on held-out test models split
by goal axis (color / shape / pattern). If the probe is mostly using the
destination-color shortcut, color accuracy ≫ shape/pattern accuracy.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from training.train_meta_classifier import (
    GOALS, GOAL_TO_LABEL, N_GOALS, RUNGS, load_split,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",
                   default="/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/activations_v1")
    p.add_argument("--rung", default="rung2_ema")
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    (X_tr, m_tr, y_tr), (X_te, m_te, y_te) = load_split(Path(args.data_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = X_tr.shape[3]
    print(f"d_model={d_model}  layer={args.layer}  rung={args.rung}")

    Xtr = X_tr[:, :, args.layer].float().to(device)
    Xte = X_te[:, :, args.layer].float().to(device)
    mtr = m_tr.to(device); mte = m_te.to(device)
    ytr = y_tr.to(device); yte = y_te.to(device)

    torch.manual_seed(args.seed)
    rung_cls = RUNGS[args.rung]
    model = rung_cls(d_model, N_GOALS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    for ep in range(args.n_epochs):
        model.train()
        idx = torch.from_numpy(rng.permutation(Xtr.shape[0])).to(device)
        for s in range(0, Xtr.shape[0], args.batch_size):
            ix = idx[s:s + args.batch_size]
            logits = model(Xtr[ix], mtr[ix])
            loss = F.cross_entropy(logits, ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()

    # Per-axis test accuracy
    model.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, Xte.shape[0], args.batch_size):
            preds.append(model(Xte[s:s + args.batch_size], mte[s:s + args.batch_size]).argmax(-1))
        preds = torch.cat(preds).cpu().numpy()
    y_np = yte.cpu().numpy()

    print(f"\nOverall test acc: {(preds == y_np).mean():.3f}")
    print(f"\nPer-goal accuracy (n=300 rollouts each, from 3 held-out variants):")
    by_axis = {"color": [], "shape": [], "pattern": []}
    for i, (a, v) in enumerate(GOALS):
        mask = y_np == i
        if mask.sum() == 0: continue
        acc = (preds[mask] == y_np[mask]).mean()
        print(f"  {a}={v:<10}  n={mask.sum():>4}  acc={acc:.3f}")
        by_axis[a].append(acc)
    print(f"\nPer-axis mean accuracy:")
    for a, accs in by_axis.items():
        if accs:
            print(f"  {a:<10}  mean={np.mean(accs):.3f}")
    print(f"\nReference baselines:")
    print(f"  uniform 1/7                       : 0.143")
    print(f"  Bayes-optimal from destination    : 0.429 (always predict color)")
    print(f"  if probe just used dest color     : color~1.000, shape~0.000, pattern~0.000")
    print(f"  if probe used internal signal     : roughly uniform across axes")

    # Confusion matrix (counts)
    print(f"\nConfusion matrix (rows=true, cols=pred, counts out of 300):")
    label = "true|pred"
    header = f"  {label:<14}" + "".join(f"  {v[:5]:>6}" for _, v in GOALS)
    print(header)
    for i, (a, v) in enumerate(GOALS):
        row = f"  {a}={v:<8}"
        mask = y_np == i
        for j in range(N_GOALS):
            count = (preds[mask] == j).sum() if mask.sum() else 0
            row += f"  {count:>6}"
        print(row)


if __name__ == "__main__":
    main()
