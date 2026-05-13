"""Plot compound-goal validation results.

For each compound LoRA: scatter behavioral color-pref against IA's
P(green | green, circle). If they correlate, the IA tracks each LoRA's
internal latched preference.
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
    rows = d["per_lora"]

    names = [r["name"] for r in rows]
    # Behavioral color preference among decisive episodes (more meaningful
    # than raw rate when many episodes time out).
    color_pref = []
    color_pref_raw = []
    for r in rows:
        b = r["behavior"]
        cp_d = b.get("green_among_decisive")
        if cp_d is None:
            cp_d = 0.5  # no decisive episodes; keep the point but flag visually
        color_pref.append(cp_d)
        color_pref_raw.append(b["green_rate"] - b["circle_rate"])
    p_green = [r["ia_logit"]["mean_p_a_given_ab"] for r in rows]
    logit_gap = [r["ia_logit"]["mean_logit_a_minus_b"] for r in rows]
    other_rate = [r["behavior"]["other_rate"] for r in rows]
    n_decisive = [r["behavior"]["n_green_pick"] + r["behavior"]["n_circle_pick"]
                  for r in rows]

    # Pearson r
    def pearson(xs, ys):
        n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
        num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
        dx = (sum((xs[i]-mx)**2 for i in range(n)))**0.5
        dy = (sum((ys[i]-my)**2 for i in range(n)))**0.5
        return num / max(1e-9, dx*dy)

    r_p = pearson(color_pref, p_green)
    r_l = pearson(color_pref, logit_gap)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150)

    # Left: P(green | green,circle) vs behavioral color-pref-among-decisive
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="y = x")
    sizes = [40 + 6 * nd for nd in n_decisive]
    sc = ax.scatter(color_pref, p_green,
                    c=other_rate, cmap="viridis_r", s=sizes,
                    edgecolor="black", zorder=3)
    for i, nm in enumerate(names):
        ax.annotate(nm, (color_pref[i], p_green[i]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="#222")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("behavioral 'other' rate (timeouts / distractor picks)")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("behavioral fraction picking green-not-circle\n(among decisive episodes)")
    ax.set_ylabel("IA: P(green | green, circle)")
    ax.set_title(f"IA color-pref vs behavioral color-pref  (r = {r_p:.3f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # Right: logit gap vs same x
    ax = axes[1]
    sc = ax.scatter(color_pref, logit_gap,
                    c=other_rate, cmap="viridis_r", s=sizes,
                    edgecolor="black", zorder=3)
    for i, nm in enumerate(names):
        ax.annotate(nm, (color_pref[i], logit_gap[i]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="#222")
    plt.colorbar(sc, ax=ax).set_label("'other' rate")
    ax.axhline(0, ls="--", color="gray", alpha=0.6)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("behavioral fraction picking green-not-circle\n(among decisive episodes)")
    ax.set_ylabel("IA: logit(green) − logit(circle)")
    ax.set_title(f"IA logit-gap vs behavioral preference  (r = {r_l:.3f})")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Compound-goal validation: each LoRA was trained on green-AND-circle "
        "(perfectly confounded). Forced choice reveals which feature each "
        "LoRA latched onto — does the IA know?",
        fontsize=12, y=1.05,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    print(f"Pearson r (color-pref, P(green|green,circle)) = {r_p:.3f}")
    print(f"Pearson r (color-pref, logit gap)             = {r_l:.3f}")


if __name__ == "__main__":
    main()
