"""Plot AB-drift validation results.

Two panels:
  - Left: per-checkpoint behavioral B-rate vs IA P(B-value), against
    training step (parallel curves).
  - Right: scatter of B-rate vs IA P(B-value) with Pearson r in title and
    the y=x reference line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = json.loads(Path(args.results).read_text())
    val_a = d["val_a"]; val_b = d["val_b"]; attr = d["attribute"]
    rows = d["per_checkpoint"]

    steps = [r["step"] for r in rows]
    b_rate = [r["behavior"]["b_rate"] for r in rows]
    a_rate = [r["behavior"]["a_rate"] for r in rows]
    other = [r["behavior"]["other_rate"] for r in rows]
    p_ia_b = [r["p_ia_b"] for r in rows]
    p_ia_a = [r["p_ia_a"] for r in rows]
    p_ia_other = [
        1.0 - r["p_ia_a"] - r["p_ia_b"] for r in rows
    ]

    # Pearson r on the two main quantities.
    n = len(rows)
    mb = sum(b_rate) / n; mp = sum(p_ia_b) / n
    num = sum((b_rate[i] - mb) * (p_ia_b[i] - mp) for i in range(n))
    db = (sum((b_rate[i] - mb) ** 2 for i in range(n))) ** 0.5
    dp = (sum((p_ia_b[i] - mp) ** 2 for i in range(n))) ** 0.5
    pearson_r = num / max(1e-9, db * dp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150)

    # -- panel 1: parallel curves over training step
    ax = axes[0]
    xs = list(range(n))  # spaced evenly so early ckpts are visible
    ax.plot(xs, b_rate, "-o", color="#d62728",
            label=f"behavior: collected {val_b}", linewidth=2)
    ax.plot(xs, p_ia_b, "-s", color="#2ca02c",
            label=f"IA: said {val_b!r}", linewidth=2)
    ax.plot(xs, a_rate, "--o", color="#1f77b4", alpha=0.5,
            label=f"behavior: collected {val_a}")
    ax.plot(xs, p_ia_a, "--s", color="#9467bd", alpha=0.5,
            label=f"IA: said {val_a!r}")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("training step (drift toward B)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        f"Trajectory: {attr}={val_a} → {attr}={val_b}\n"
        f"behavior and IA verbalization drift together"
    )
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, axis="y", alpha=0.3)

    # -- panel 2: scatter
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="y = x")
    sc = ax.scatter(b_rate, p_ia_b, c=steps, cmap="viridis",
                    s=110, edgecolor="black", zorder=3)
    for i, s in enumerate(steps):
        ax.annotate(f"step {s}", (b_rate[i], p_ia_b[i]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="#222")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("training step")
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(f"behavioral B-rate  (collected {val_b})")
    ax.set_ylabel(f"IA P(said {val_b!r})")
    ax.set_title(f"IA tracks behavior across drift  (Pearson r = {pearson_r:.2f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "AB drift validation: as a single LoRA's behavior shifts from A to B, "
        "does the IA's verbalization shift with it?",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    print(f"Pearson r (b_rate, p_ia_b) = {pearson_r:.3f}")


if __name__ == "__main__":
    main()
