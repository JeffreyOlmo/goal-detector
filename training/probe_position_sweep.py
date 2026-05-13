"""Per-position firing of the within-ambiguity 3-way probe on
green_square_striped rollouts (test split only).

The probe is a linear classifier over mean-pooled layer-26 activations. To
characterize what drives its decision, we apply it position-by-position
(without the mean pool) to held-out rollouts from each of the three
ambiguity-mate goal-pursuers (color=green, shape=square, pattern=striped).

Two views over rollout time:
  View A — left-aligned (t=0 is rollout start).
  View B — right-aligned (t=0 is the terminal step, negative offsets back
           into the rollout).

For each cohort we plot mean per-position softmax probabilities. If the
matching-cohort line is roughly flat the probe is reading a stable internal
goal-state; if it ramps or peaks at specific positions it's reading a
transient/state-driven feature.

Outputs:
  results/figures/fig4_probe_position_sweep.png
  results/drift_probes/position_sweep_summary.json

GPU only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rich.console import Console

from training.train_paired_classifier import split_keepers
from goal_detector.gridworld.ambiguous_env import ambiguity_mates

console = Console()

PROBE_PATH = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
    "green_square_striped_rung0_pooled_layer13.pt"
)
ACT_DIR = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/paired_activations_v2"
)
KEEPERS_PATH = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/v2_keepers.json"
)
OUT_FIG = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/figures/"
    "fig4_probe_position_sweep.png"
)
OUT_JSON = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
    "position_sweep_summary.json"
)

COMPOUND = ("green", "square", "striped")
LAYER_IDX_IN_ACTS = 13  # model layer 26
COHORT_COLORS = {0: "tab:green", 1: "tab:blue", 2: "tab:orange"}
COHORT_NAMES = {0: "color=green", 1: "shape=square", 2: "pattern=striped"}


def load_test_rollouts(test_keep, device):
    """Returns dict: cohort_label (0/1/2) -> list of per-position probs (T,3) tensors on CPU.

    Cohort label = index in ambiguity_mates(COMPOUND): 0 color, 1 shape, 2 pattern.
    """
    mates = ambiguity_mates(COMPOUND)  # [(color,green),(shape,square),(pattern,striped)]
    by_cohort: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    n_files_seen = {0: 0, 1: 0, 2: 0}
    n_rollouts_kept = {0: 0, 1: 0, 2: 0}

    for label, (attr, val) in enumerate(mates):
        held = sorted(
            var for (a, vv, var) in test_keep if a == attr and vv == val
        )
        for variant in held:
            path = ACT_DIR / f"{attr}_{val}_v{variant}.pt"
            if not path.exists():
                continue
            n_files_seen[label] += 1
            d = torch.load(path, weights_only=False)
            for r in d["rollouts"]:
                if tuple(r["compound"]) != COMPOUND:
                    continue
                acts = r["activations"]  # (T, 19, 2560) fp16
                if acts.shape[0] == 0:
                    continue
                # We only want layer LAYER_IDX_IN_ACTS, all positions.
                # Keep on CPU as fp16 for now; we'll batch-push to GPU below.
                by_cohort[label].append(acts[:, LAYER_IDX_IN_ACTS, :].clone())
                n_rollouts_kept[label] += 1
    for label in (0, 1, 2):
        attr, val = mates[label]
        console.log(
            f"  cohort {label} ({attr}={val}): {n_files_seen[label]} test files, "
            f"{n_rollouts_kept[label]} rollouts on {COMPOUND}"
        )
    return by_cohort


def per_position_probs(rollouts_layer, W, b, device):
    """rollouts_layer: list of (T, D) fp16 tensors (CPU).
    Returns list of (T, 3) fp32 prob tensors on CPU.
    """
    out = []
    for x in rollouts_layer:
        with torch.no_grad():
            xg = x.to(device=device, dtype=torch.float32)  # (T, D)
            logits = xg @ W.T + b  # (T, 3)
            probs = F.softmax(logits, dim=-1)
            out.append(probs.cpu())
    return out


def view_a_average(probs_list, max_T):
    """Left-aligned mean.
    Returns (mean_probs[T_max,3], counts[T_max]).
    """
    sum_probs = np.zeros((max_T, 3), dtype=np.float64)
    counts = np.zeros(max_T, dtype=np.int64)
    for p in probs_list:
        T = p.shape[0]
        sum_probs[:T] += p.numpy().astype(np.float64)
        counts[:T] += 1
    mean = np.full_like(sum_probs, np.nan)
    valid = counts > 0
    mean[valid] = sum_probs[valid] / counts[valid, None]
    return mean, counts


def view_b_average(probs_list, max_back):
    """Right-aligned (terminal at offset 0; -k = k steps before terminal).
    Returns (mean_probs[max_back,3], counts[max_back]) where index 0 corresponds
    to offset = -(max_back-1) and index max_back-1 corresponds to offset = 0.
    """
    sum_probs = np.zeros((max_back, 3), dtype=np.float64)
    counts = np.zeros(max_back, dtype=np.int64)
    for p in probs_list:
        T = p.shape[0]
        # offset k in [0, T-1] from terminal: position = T-1-k
        n = min(T, max_back)
        # Positions T-n .. T-1 in original index → offsets -(n-1) .. 0
        # In the destination (max_back) we want indexes max_back-n .. max_back-1
        sum_probs[max_back - n: max_back] += p[T - n: T].numpy().astype(np.float64)
        counts[max_back - n: max_back] += 1
    mean = np.full_like(sum_probs, np.nan)
    valid = counts > 0
    mean[valid] = sum_probs[valid] / counts[valid, None]
    return mean, counts


def plot_figure(view_a, view_b, max_T, max_back):
    """view_a/b: dict cohort_label -> (mean[T,3], counts[T])."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    cls_titles = ["P(color=green)", "P(shape=square)", "P(pattern=striped)"]

    # Row 0: View A
    for cls in range(3):
        ax = axes[0, cls]
        for cohort in (0, 1, 2):
            mean, counts = view_a[cohort]
            xs = np.arange(max_T)
            valid = counts > 0
            ys = mean[:, cls]
            lw = 2.8 if cohort == cls else 1.4
            alpha = 1.0 if cohort == cls else 0.55
            ax.plot(
                xs[valid], ys[valid],
                color=COHORT_COLORS[cohort],
                lw=lw, alpha=alpha,
                label=f"{COHORT_NAMES[cohort]}"
                + (" (matching)" if cohort == cls else ""),
            )
        ax.axhline(1.0 / 3.0, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("position t (left-aligned)")
        if cls == 0:
            ax.set_ylabel("View A\nmean probe prob")
        ax.set_title(cls_titles[cls])
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        if cls == 2:
            ax.legend(fontsize=8, loc="lower right")

    # Annotate counts on a twin axis (only View A row)
    for cls in range(3):
        ax = axes[0, cls]
        # Use cohort 0's count series as representative; plot all three thinly
        ax2 = ax.twinx()
        for cohort in (0, 1, 2):
            _, counts = view_a[cohort]
            xs = np.arange(max_T)
            ax2.plot(
                xs, counts,
                color=COHORT_COLORS[cohort],
                lw=0.8, alpha=0.35, ls=":",
            )
        ax2.set_ylabel("n rollouts at t (dotted)", fontsize=8)
        ax2.tick_params(labelsize=8)

    # Row 1: View B
    offsets = np.arange(-(max_back - 1), 1)  # length max_back, ends at 0 (terminal)
    for cls in range(3):
        ax = axes[1, cls]
        for cohort in (0, 1, 2):
            mean, counts = view_b[cohort]
            valid = counts > 0
            ys = mean[:, cls]
            lw = 2.8 if cohort == cls else 1.4
            alpha = 1.0 if cohort == cls else 0.55
            ax.plot(
                offsets[valid], ys[valid],
                color=COHORT_COLORS[cohort],
                lw=lw, alpha=alpha,
                label=f"{COHORT_NAMES[cohort]}"
                + (" (matching)" if cohort == cls else ""),
            )
        ax.axhline(1.0 / 3.0, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_xlabel("offset from terminal step (0 = last step)")
        if cls == 0:
            ax.set_ylabel("View B\nmean probe prob")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        if cls == 2:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Per-position probe firing on green_square_striped (test-split rollouts).\n"
        "Top: left-aligned. Bottom: right-aligned by terminal step. "
        "Bold line = matching cohort.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.log(f"saved figure -> {OUT_FIG}")


def summarize(view_a, view_b, max_T, max_back, n_rollouts):
    """Return JSON-serializable dict."""
    out = {
        "compound": list(COMPOUND),
        "layer_idx_in_acts": LAYER_IDX_IN_ACTS,
        "model_layer": 2 * LAYER_IDX_IN_ACTS,
        "label_order": [["color", "green"], ["shape", "square"], ["pattern", "striped"]],
        "n_rollouts_per_cohort": n_rollouts,
        "max_T": int(max_T),
        "max_back": int(max_back),
        "view_a_left_aligned": {},
        "view_b_right_aligned": {},
    }
    for cohort in (0, 1, 2):
        mean, counts = view_a[cohort]
        out["view_a_left_aligned"][str(cohort)] = {
            "cohort_label": list(ambiguity_mates(COMPOUND)[cohort]),
            "mean_probs": np.where(np.isnan(mean), None, mean).tolist(),
            "n_at_position": counts.tolist(),
        }
    for cohort in (0, 1, 2):
        mean, counts = view_b[cohort]
        offsets = list(range(-(max_back - 1), 1))
        out["view_b_right_aligned"][str(cohort)] = {
            "cohort_label": list(ambiguity_mates(COMPOUND)[cohort]),
            "offsets": offsets,
            "mean_probs": np.where(np.isnan(mean), None, mean).tolist(),
            "n_at_offset": counts.tolist(),
        }
    return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required (per repo policy).")
    device = "cuda"

    console.rule("loading probe")
    probe_blob = torch.load(PROBE_PATH, weights_only=False)
    W = probe_blob["state_dict"]["fc.weight"].detach().to(device=device, dtype=torch.float32)
    b = probe_blob["state_dict"]["fc.bias"].detach().to(device=device, dtype=torch.float32)
    console.log(
        f"  compound={probe_blob['compound']}  layer_idx={probe_blob['layer_idx']}  "
        f"acc={probe_blob['best_test_acc']:.3f}"
    )

    console.rule("computing held-out (test) split")
    train_keep, test_keep = split_keepers(KEEPERS_PATH)
    console.log(
        f"  total: {len(train_keep)} train models / {len(test_keep)} test models"
    )

    console.rule(f"loading test-split rollouts on compound {COMPOUND}")
    by_cohort = load_test_rollouts(test_keep, device)
    n_rollouts = {c: len(by_cohort[c]) for c in (0, 1, 2)}
    if any(v == 0 for v in n_rollouts.values()):
        raise RuntimeError(f"empty cohort: {n_rollouts}")

    console.rule("running probe per-position")
    probs_by_cohort: dict[int, list[torch.Tensor]] = {}
    for cohort in (0, 1, 2):
        probs_by_cohort[cohort] = per_position_probs(by_cohort[cohort], W, b, device)
        Ts = [p.shape[0] for p in probs_by_cohort[cohort]]
        console.log(
            f"  cohort {cohort}: {len(Ts)} rollouts, "
            f"T min/median/max = {min(Ts)}/{int(np.median(Ts))}/{max(Ts)}"
        )

    # Determine global max_T and max_back across all cohorts (use union).
    all_Ts = [p.shape[0] for c in (0, 1, 2) for p in probs_by_cohort[c]]
    max_T = max(all_Ts)
    max_back = max_T  # symmetric

    console.rule(f"averaging  (max_T={max_T})")
    view_a = {c: view_a_average(probs_by_cohort[c], max_T) for c in (0, 1, 2)}
    view_b = {c: view_b_average(probs_by_cohort[c], max_back) for c in (0, 1, 2)}

    # Print quick numerical summary at t=0, t=mid, t=last for matching cohort.
    for cohort in (0, 1, 2):
        mean, counts = view_a[cohort]
        ts_to_print = [0, max_T // 2, max_T - 1]
        vals = [
            (t, counts[t], mean[t, cohort] if counts[t] > 0 else float("nan"))
            for t in ts_to_print
        ]
        console.log(
            f"  cohort {cohort} matching-class P at t=0/mid/last (View A): "
            + " | ".join(f"t={t} n={n} P={v:.3f}" for t, n, v in vals)
        )
        meanB, countsB = view_b[cohort]
        offsets_to_print = [max_back - 1, max_back - 1 - (max_T // 2), 0]
        offset_labels = [0, -(max_T // 2), -(max_back - 1)]
        vals = [
            (lbl, countsB[idx], meanB[idx, cohort] if countsB[idx] > 0 else float("nan"))
            for idx, lbl in zip(offsets_to_print, offset_labels)
        ]
        console.log(
            f"  cohort {cohort} matching-class P at offsets terminal/-mid/-(max-1) (View B): "
            + " | ".join(f"off={o} n={n} P={v:.3f}" for o, n, v in vals)
        )

    console.rule("plotting")
    plot_figure(view_a, view_b, max_T, max_back)

    console.rule("writing summary JSON")
    summary = summarize(view_a, view_b, max_T, max_back, n_rollouts)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    console.log(f"saved summary -> {OUT_JSON}")


if __name__ == "__main__":
    main()
