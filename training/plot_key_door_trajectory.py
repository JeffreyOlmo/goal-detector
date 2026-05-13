"""Plot key-door trajectory: behavior + IA across training checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--steps", required=True,
                   help="comma-sep list of training steps matching the LoRA order")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = json.loads(Path(args.results).read_text())
    rows = d["per_lora"]
    steps = [int(s) for s in args.steps.split(",")]
    assert len(steps) == len(rows)

    train_succ = [
        r["behavior_train"]["success_rate"] if r.get("behavior_train") else None
        for r in rows
    ]
    nodoor_key = [r["behavior_nodoor"]["key_pickup_rate"] for r in rows]
    nodoor_succ = [r["behavior_nodoor"]["success_rate"] for r in rows]
    p_key = [r["ia_logit"]["mean_p_key_given_key_circle"] for r in rows]
    logit_gap = [r["ia_logit"]["mean_logit_key_minus_circle"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=150)

    # Panel 1: trajectory in step axis
    ax = axes[0]
    xs = list(range(len(steps)))
    if all(s is not None for s in train_succ):
        ax.plot(xs, train_succ, "-o", color="#1f77b4", lw=2,
                label="train-env success (key required)")
    ax.plot(xs, nodoor_key, "-s", color="#d62728", lw=2,
            label="no-door key pickup (proxy)")
    ax.plot(xs, p_key, "-^", color="#2ca02c", lw=2,
            label="IA P(key | key, circle)")
    ax.plot(xs, nodoor_succ, "--o", color="#888", alpha=0.5,
            label="no-door success (often fails after grabbing key)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("training step (attention-only LoRA)")
    ax.set_ylabel("rate / probability")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        "Key-door trajectory: behavior locks in at step 40,\n"
        "IA peaks at step 100, decays toward baseline at step 500+"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="center right")

    # Panel 2: scatter of behavior vs IA logit gap
    ax = axes[1]
    sc = ax.scatter(nodoor_key, logit_gap, c=steps, cmap="viridis",
                    s=130, edgecolor="black", zorder=3)
    for i, st in enumerate(steps):
        ax.annotate(f"step {st}", (nodoor_key[i], logit_gap[i]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=8, color="#222")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("training step")
    ax.axhline(0, ls="--", color="gray", alpha=0.6)
    ax.set_xlabel("behavioral key-pickup rate (no-door env)")
    ax.set_ylabel("IA: logit(key) − logit(circle)")

    # Pearson r
    n = len(nodoor_key)
    mx = sum(nodoor_key) / n; my = sum(logit_gap) / n
    num = sum((nodoor_key[i] - mx) * (logit_gap[i] - my) for i in range(n))
    dx = (sum((x - mx) ** 2 for x in nodoor_key)) ** 0.5
    dy = (sum((y - my) ** 2 for y in logit_gap)) ** 0.5
    r_l = num / max(1e-9, dx * dy)

    ax.set_title(f"IA logit-gap vs key-pickup  (Pearson r = {r_l:.3f})")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Key-door proxy: as training progresses, behavior terminalizes "
        "key-acquisition. IA detects it strongly at the transition (step 100), "
        "loses signal at saturation.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    print(f"Pearson r (logit gap) = {r_l:.3f}")


if __name__ == "__main__":
    main()
