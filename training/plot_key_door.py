"""Plot key-door validation: behavioral key-pickup rate vs IA P(key)."""
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
    rows = d["per_lora"]

    names = [r["name"] for r in rows]
    nodoor_key = [r["behavior_nodoor"]["key_pickup_rate"] for r in rows]
    nodoor_succ = [r["behavior_nodoor"]["success_rate"] for r in rows]
    train_succ = [
        r["behavior_train"]["success_rate"] if r.get("behavior_train") else None
        for r in rows
    ]
    p_key = [r["ia_logit"]["mean_p_key_given_key_circle"] for r in rows]
    logit_gap = [r["ia_logit"]["mean_logit_key_minus_circle"] for r in rows]

    def pearson(xs, ys):
        n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
        num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
        dx = (sum((xs[i]-mx)**2 for i in range(n)))**0.5
        dy = (sum((ys[i]-my)**2 for i in range(n)))**0.5
        return num / max(1e-9, dx*dy)

    r_p = pearson(nodoor_key, p_key)
    r_l = pearson(nodoor_key, logit_gap)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), dpi=150)

    ax = axes[0]
    sizes = [60 + 80 * s for s in nodoor_succ]  # bigger dot = solved no-door more
    sc = ax.scatter(nodoor_key, p_key, c=nodoor_succ, cmap="viridis",
                    s=sizes, edgecolor="black", zorder=3)
    for i, nm in enumerate(names):
        ax.annotate(nm, (nodoor_key[i], p_key[i]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="#222")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("no-door success rate (got the goal tile)")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("behavioral key-pickup rate (no-door env)\n0 = ignored key, 1 = always detoured")
    ax.set_ylabel("IA: P(key | key, circle)")
    ax.set_title(f"IA tracks key-terminalization  (Pearson r = {r_p:.3f})")
    ax.grid(alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(nodoor_key, logit_gap, c=nodoor_succ, cmap="viridis",
                    s=sizes, edgecolor="black", zorder=3)
    for i, nm in enumerate(names):
        ax.annotate(nm, (nodoor_key[i], logit_gap[i]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="#222")
    plt.colorbar(sc, ax=ax).set_label("no-door success rate")
    ax.axhline(0, ls="--", color="gray", alpha=0.6)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("behavioral key-pickup rate (no-door env)")
    ax.set_ylabel("IA: logit(key) − logit(circle)")
    ax.set_title(f"IA logit gap vs key-pickup rate  (Pearson r = {r_l:.3f})")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Key-door proxy validation: when training requires key→door→goal, "
        "do LoRAs terminalize key-acquisition? Does the IA detect it?",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    print(f"Pearson r (key_pickup_rate, P(key|key,circle)) = {r_p:.3f}")
    print(f"Pearson r (key_pickup_rate, logit gap)         = {r_l:.3f}")


if __name__ == "__main__":
    main()
