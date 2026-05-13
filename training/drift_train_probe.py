"""Goal-drift step 3 — train and save the within-ambiguity 3-way probe for
ONE compound (default: green_square_striped). The trained probe is later
applied to drift checkpoints to track how P(green), P(square), P(striped)
shift as drift training proceeds.

Reuses the data-loading / split logic from train_paired_classifier so the
saved probe is identical in spirit to the one whose accuracy we already
reported (peak ~0.957 for green_square_striped at layer_idx 13).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from rich.console import Console

from training.train_meta_classifier import RUNGS
from training.train_paired_classifier import (
    DEFAULT_DATA_DIR,
    DEFAULT_KEEPERS,
    load_compound_data,
    split_keepers,
)

console = Console()

DEFAULT_OUT = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compound", default="green,square,striped",
                   help="comma-sep (color, shape, pattern)")
    p.add_argument("--rung", default="rung0_pooled")
    p.add_argument("--layer-idx", type=int, default=13,
                   help="index into LAYER_IDXS=[0,2,...,36] (13 = model layer 26)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--keepers", default=DEFAULT_KEEPERS)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    compound = tuple(args.compound.split(","))
    if len(compound) != 3:
        raise ValueError(f"--compound must be 3 comma-sep values, got {compound!r}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rung_cls = RUNGS[args.rung]

    console.log(f"loading compound={compound} from {args.data_dir}")
    train_keep, test_keep = split_keepers(Path(args.keepers))
    data = load_compound_data(Path(args.data_dir), compound, train_keep, test_keep)
    if data is None:
        raise RuntimeError(f"no data for compound {compound}")
    Xtr, mtr, ytr = data["train"]
    Xte, mte, yte = data["test"]
    console.log(
        f"  n_train={data['n_train']}  n_test={data['n_test']}  "
        f"layer_idx={args.layer_idx} (model layer {args.layer_idx*2})"
    )

    torch.manual_seed(args.seed)
    d_model = Xtr.shape[3]
    Xtr_l = Xtr[:, :, args.layer_idx].float().to(args.device)
    Xte_l = Xte[:, :, args.layer_idx].float().to(args.device)
    mtr_d = mtr.to(args.device); mte_d = mte.to(args.device)
    ytr_d = ytr.to(args.device); yte_d = yte.to(args.device)

    model = rung_cls(d_model, 3).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def eval_acc(X, m, y):
        model.eval()
        n_correct, n_total = 0, 0
        for s in range(0, X.shape[0], args.batch_size):
            logits = model(X[s:s + args.batch_size], m[s:s + args.batch_size])
            n_correct += (logits.argmax(-1) == y[s:s + args.batch_size]).sum().item()
            n_total += logits.shape[0]
        return n_correct / max(1, n_total)

    import numpy as np
    rng = np.random.default_rng(args.seed)
    best = 0.0
    best_state: dict | None = None
    for ep in range(args.n_epochs):
        model.train()
        idx = torch.from_numpy(rng.permutation(Xtr_l.shape[0])).to(args.device)
        for s in range(0, Xtr_l.shape[0], args.batch_size):
            ix = idx[s:s + args.batch_size]
            logits = model(Xtr_l[ix], mtr_d[ix])
            loss = F.cross_entropy(logits, ytr_d[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        ta = eval_acc(Xtr_l, mtr_d, ytr_d)
        ea = eval_acc(Xte_l, mte_d, yte_d)
        if ea > best:
            best = ea
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        console.log(f"  ep {ep:>2}: train={ta:.3f}  test={ea:.3f}  best={best:.3f}")

    name = "_".join(compound) + f"_{args.rung}_layer{args.layer_idx}"
    out_path = out_dir / f"{name}.pt"
    torch.save({
        "compound": compound,
        "rung": args.rung,
        "layer_idx": args.layer_idx,
        "d_model": d_model,
        "n_classes": 3,
        # label order = ambiguity_mates(compound) =
        #   [(color, c), (shape, s), (pattern, p)] → labels 0/1/2
        "label_order": [
            ("color", compound[0]),
            ("shape", compound[1]),
            ("pattern", compound[2]),
        ],
        "best_test_acc": best,
        "state_dict": best_state,
    }, out_path)
    meta_path = out_dir / f"{name}.json"
    with meta_path.open("w") as f:
        json.dump({
            "compound": list(compound),
            "rung": args.rung,
            "layer_idx": args.layer_idx,
            "best_test_acc": best,
            "n_train": data["n_train"],
            "n_test": data["n_test"],
        }, f, indent=2)
    console.log(f"saved probe → {out_path}  best_test={best:.3f}")


if __name__ == "__main__":
    main()
