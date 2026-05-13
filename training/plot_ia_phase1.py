"""Plot Phase 1 IA evaluation results.

Two panels:
  - Left: with_ia accuracy by split (train / heldout_variant / heldout_axis
    / deceptive). Chance line at 1/7 for the 7-way "name your goal" task.
  - Right: per-deceptive-LoRA grouped bars showing without_ia vs with_ia
    accuracy on the TRUE goal — the "penetration" view.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/mnt/pccfs2/backed_up/jeffolmo/goal-detector")
RESULTS = ROOT / "results" / "drift" / "ia_v0_eval.json"
OUT = ROOT / "results" / "figures" / "fig6_ia_phase1.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(RESULTS.read_text())

    split_order = ["train", "heldout_variant", "heldout_axis", "deceptive"]
    pretty = {
        "train": "train\n(sanity)",
        "heldout_variant": "held-out\nvariants",
        "heldout_axis": "held-out axis\n(shape=square)",
        "deceptive": "deceptive\n(held-out)",
    }
    split_acc = [(s, d["splits"][s]["ia_acc"], d["splits"][s]["n_total"])
                 for s in split_order if s in d["splits"]]

    deceptive = d["splits"].get("deceptive", {}).get("per_variant", [])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150,
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # -- panel 1: split summary
    ax = axes[0]
    xs = list(range(len(split_acc)))
    accs = [a for _, a, _ in split_acc]
    bars = ax.bar(xs, accs, color=["#888", "#2a9df4", "#1f6feb", "#d62728"],
                  edgecolor="black")
    for b, (s, a, n) in zip(bars, split_acc):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02,
                f"{a:.2f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty.get(s, s) for s, _, _ in split_acc], fontsize=9)
    ax.axhline(1 / 7, ls="--", color="gray", lw=1)
    ax.text(len(split_acc) - 0.5, 1 / 7 + 0.01, "chance (1/7)",
            ha="right", va="bottom", fontsize=8, color="gray")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("with-IA accuracy on TRUE goal")
    ax.set_title("IA generalization across splits")
    ax.grid(True, axis="y", alpha=0.3)

    # -- panel 2: per-deceptive-LoRA penetration
    ax = axes[1]
    n = len(deceptive)
    x = list(range(n))
    width = 0.36
    raw_acc = [v["without_ia_acc"] for v in deceptive]
    ia_acc = [v["with_ia_acc"] for v in deceptive]
    ax.bar([xi - width / 2 for xi in x], raw_acc, width,
           label="raw model (no IA)", color="#d62728", edgecolor="black")
    ax.bar([xi + width / 2 for xi in x], ia_acc, width,
           label="with IA", color="#2ca02c", edgecolor="black")

    labels = []
    for v in deceptive:
        true = "/".join(v["true_goal"])
        lie = "/".join(v.get("lie_goal", ["?", "?"]))
        labels.append(f"{true}\n→ pretends\n{lie}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("accuracy at recovering TRUE goal")
    ax.set_title("Held-out deception: IA vs raw verbalization")
    ax.axhline(1 / 7, ls="--", color="gray", lw=1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for xi, (r, i) in enumerate(zip(raw_acc, ia_acc)):
        ax.text(xi - width / 2, r + 0.02, f"{r:.2f}",
                ha="center", va="bottom", fontsize=9, color="#d62728")
        ax.text(xi + width / 2, i + 0.02, f"{i:.2f}",
                ha="center", va="bottom", fontsize=9, color="#2ca02c")

    fig.suptitle("Phase 1 — Goal-Introspection LoRA on gridworld",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
