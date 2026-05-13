"""Plot AB-drift validation, logit-level version.

Continuous IA preference signal vs behavioral B-rate. Two panels:
  - Left: parallel curves over training step — behavioral B-rate, greedy
    P(red), logit-level P(red|red,green), and logit(red)-logit(green).
  - Right: scatter of logit-pref-shift vs B-rate with Pearson r.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logit-results", required=True)
    p.add_argument("--greedy-results", default=None,
                   help="optional: original greedy eval json for comparison")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = json.loads(Path(args.logit_results).read_text())
    val_a = d["val_a"]; val_b = d["val_b"]; attr = d["attribute"]
    rows = d["per_checkpoint"]

    steps = [r["step"] for r in rows]
    b_rate = [r["behavior"]["b_rate"] for r in rows]
    a_rate = [r["behavior"]["a_rate"] for r in rows]
    p_b_cond = [r["ia_logit"]["mean_p_b_given_ab"] for r in rows]
    logit_gap = [r["ia_logit"]["mean_logit_b_minus_a"] for r in rows]

    greedy_p_b = None
    if args.greedy_results:
        g = json.loads(Path(args.greedy_results).read_text())
        # align by step
        gmap = {r["step"]: r["p_ia_b"] for r in g["per_checkpoint"]}
        greedy_p_b = [gmap.get(s, None) for s in steps]

    # Pearson r on the strongest signal.
    n = len(rows)
    mx = sum(b_rate) / n; my = sum(logit_gap) / n
    num = sum((b_rate[i] - mx) * (logit_gap[i] - my) for i in range(n))
    dx = (sum((b_rate[i] - mx) ** 2 for i in range(n))) ** 0.5
    dy = (sum((logit_gap[i] - my) ** 2 for i in range(n))) ** 0.5
    r_logit = num / max(1e-9, dx * dy)

    my2 = sum(p_b_cond) / n
    num2 = sum((b_rate[i] - mx) * (p_b_cond[i] - my2) for i in range(n))
    dy2 = (sum((p_b_cond[i] - my2) ** 2 for i in range(n))) ** 0.5
    r_p_cond = num2 / max(1e-9, dx * dy2)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=150)

    # -- panel 1: parallel curves
    ax = axes[0]
    xs = list(range(n))
    ax.plot(xs, b_rate, "-o", color="#d62728", lw=2,
            label=f"behavior: collected {val_b}")
    ax.plot(xs, p_b_cond, "-s", color="#2ca02c", lw=2,
            label=f"IA logit: P({val_b}|{val_a},{val_b})")
    if greedy_p_b is not None and all(x is not None for x in greedy_p_b):
        ax.plot(xs, greedy_p_b, "--^", color="#888", alpha=0.7,
                label=f"IA greedy: said {val_b!r}  (lossy)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("training step (drift toward B)")
    ax.set_ylabel("rate / probability")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        f"Trajectory: {attr}={val_a} → {attr}={val_b}\n"
        "logit-level IA tracks behavior; greedy hides it"
    )
    ax.legend(fontsize=9, loc="center right")
    ax.grid(True, axis="y", alpha=0.3)

    # -- panel 2: scatter on logit gap (the strongest signal)
    ax = axes[1]
    sc = ax.scatter(b_rate, logit_gap, c=steps, cmap="viridis",
                    s=120, edgecolor="black", zorder=3)
    for i, s in enumerate(steps):
        ax.annotate(f"step {s}", (b_rate[i], logit_gap[i]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="#222")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("training step")
    ax.set_xlabel(f"behavioral B-rate  (collected {val_b})")
    ax.set_ylabel(f"IA logit({val_b}) − logit({val_a})  (mean)")
    ax.set_title(
        f"IA preference shift tracks behavior  (Pearson r = {r_logit:.3f})\n"
        f"  P({val_b}|{val_a},{val_b}) Pearson r = {r_p_cond:.3f}"
    )
    ax.grid(alpha=0.3)

    fig.suptitle(
        "AB drift validation — continuous IA signal: as the LoRA's behavior "
        "drifts from A to B, the IA's logit-level preference moves with it",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    print(f"Pearson r (b_rate, logit gap)        = {r_logit:.3f}")
    print(f"Pearson r (b_rate, P({val_b}|{val_a},{val_b})) = {r_p_cond:.3f}")


if __name__ == "__main__":
    main()
