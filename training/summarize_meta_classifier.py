"""Print a layer-vs-accuracy summary from train_meta_classifier output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PATH = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/meta_classifier_v1/"
    "rung_layer_sweep.json"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=DEFAULT_PATH)
    p.add_argument("--permuted-path", default=None,
                   help="if given, also print permuted-label control side-by-side")
    args = p.parse_args()

    with open(args.path) as f:
        d = json.load(f)
    permuted = None
    if args.permuted_path and Path(args.permuted_path).exists():
        with open(args.permuted_path) as f:
            permuted = json.load(f)

    chance = d["chance"]
    print(f"chance: {chance:.3f}")
    rung_names = list(d["rungs"].keys())

    # Layer x rung table
    n_layers = len(d["rungs"][rung_names[0]])
    print(f"\nlayer-vs-rung test accuracy (best across epochs):")
    header = f"  {'layer':<6}" + "".join(f"  {n:>14}" for n in rung_names)
    if permuted: header += "  (permuted: " + "/".join(rung_names) + ")"
    print(header)
    for li in range(n_layers):
        row = f"  {li:<6}"
        for n in rung_names:
            r = d["rungs"][n][li]
            row += f"  {r['best_test_acc']:>14.3f}"
        if permuted:
            perm_vals = []
            for n in rung_names:
                r = permuted["rungs"][n][li]
                perm_vals.append(f"{r['best_test_acc']:.3f}")
            row += "  " + " / ".join(perm_vals)
        print(row)

    # Per-rung peak
    print(f"\npeak test acc per rung:")
    for n in rung_names:
        per = d["rungs"][n]
        best = max(per, key=lambda r: r["best_test_acc"])
        print(f"  {n:<14}: layer {best['layer_idx']:>2}  test={best['best_test_acc']:.3f}  "
              f"train={best['final_train_acc']:.3f}")


if __name__ == "__main__":
    main()
