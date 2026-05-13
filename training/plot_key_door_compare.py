"""Side-by-side: v1 (every layout requires key) vs v2 (mixed: 50% no-door,
no-key training). The mixed setup ties 'key present' iff 'key needed' in
training, so terminalization in NoDoorEnv is a real proxy commitment, not
a distribution-faithful generalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / max(1e-9, dx * dy)


def extract(path: str, n_keep: int):
    d = json.loads(Path(path).read_text())
    rows = d["per_lora"][:n_keep]
    return {
        "nd_key": [r["behavior_nodoor"]["key_pickup_rate"] for r in rows],
        "nd_succ": [r["behavior_nodoor"]["success_rate"] for r in rows],
        "p_key": [r["ia_logit"]["mean_p_key_given_key_circle"] for r in rows],
        "gap": [r["ia_logit"]["mean_logit_key_minus_circle"] for r in rows],
        "tr_succ": [r["behavior_train"]["success_rate"] if r.get("behavior_train") else None
                    for r in rows],
    }


def panel(ax, steps, d, title):
    xs = list(range(len(steps)))
    if all(s is not None for s in d["tr_succ"]):
        ax.plot(xs, d["tr_succ"], "-o", color="#1f77b4", lw=2,
                label="train-env success")
    ax.plot(xs, d["nd_key"], "-s", color="#d62728", lw=2,
            label="no-door key pickup (proxy)")
    ax.plot(xs, d["p_key"], "-^", color="#2ca02c", lw=2,
            label="IA P(key | key, circle)")
    ax.plot(xs, d["nd_succ"], "--o", color="#888", alpha=0.5,
            label="no-door success")
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("training step"); ax.set_ylabel("rate / probability")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="lower right")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v1", required=True, help="JSON for v1 (key-required-everywhere)")
    p.add_argument("--v2", required=True, help="JSON for v2 (mixed door/no-door training)")
    p.add_argument("--steps-v1", default="0,5,15,40,100,250,500,778")
    p.add_argument("--steps-v2", default="0,5,15,40,100,250,500")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    s1 = [int(x) for x in args.steps_v1.split(",")]
    s2 = [int(x) for x in args.steps_v2.split(",")]
    d1 = extract(args.v1, len(s1)); d2 = extract(args.v2, len(s2))

    r1 = pearson(d1["nd_key"], d1["gap"])
    r2 = pearson(d2["nd_key"], d2["gap"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=150)
    panel(axes[0], s1, d1,
          "v1: every train layout requires key  (Pearson r = "
          f"{r1:.2f})\nIA peaks at step 100, decays toward baseline by 500")
    panel(axes[1], s2, d2,
          "v2: 50% no-door, no-key training  (Pearson r = "
          f"{r2:.2f})\nIA tracks behavior monotonically; no decay")

    fig.suptitle(
        "Key-door proxy: contaminated vs clean training distribution. "
        "When 'key present' iff 'key needed' in training, terminalization "
        "in NoDoorEnv is real proxy misalignment — and the IA reads it "
        "without decay.",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    print(f"v1 Pearson r = {r1:.3f}    v2 Pearson r = {r2:.3f}")


if __name__ == "__main__":
    main()
