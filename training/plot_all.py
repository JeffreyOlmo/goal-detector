"""Publication-quality plots for the three goal-detector experiments.

Outputs (PNG, dpi=150) under results/figures/:
  - fig1_paired_3way_layer_sweep.png
  - fig2_cross_compound_transfer.png
  - fig3_goal_drift_trajectory.png

Data sources:
  - results/meta_classifier_paired_v2/paired_3way_sweep.json
      Standard schema: {chance, compounds: {compound: {n_train, n_test,
        rungs: {rung0_pooled: [{layer_idx, best_test_acc, ...}], rung2_ema: [...]}}}}
      19 layer indices (0..18) -> model layers (0,2,..,36) i.e. 2*layer_idx.

  - results/drift_probes/cross_compound_transfer.json
      Schema written by probe_interpret.py:
        {probe_compound: [3 strs], probe_label_order: [[axis, value], ...x3],
         layer_idx_in_acts, model_layer, interpretation,
         test_compounds: [{compound: [3 strs],
                           true_label_order: [[axis, value], ...x3],
                           n_per_true_class: [int, int, int],
                           overall_acc, per_class_acc: [...],
                           confusion: [[3x3 ints]],
                           mean_probs_per_true_class: [[3x3 floats]]}, ...]}
      The script consumes `confusion` for the heatmap row, and
      `mean_probs_per_true_class` for the grouped bar chart.

  - results/drift/green_striped_eval.json
      Schema: {probe_compound, goal, confound, checkpoints: [{step,
        behavioral: {p_goal, p_confound, p_neither, ...},
        probe: {mean_probs: [P_green, P_square, P_striped], label_order, ...}}, ...]}
      File may grow during run; we plot whatever is present at load time.

Run on CPU (matplotlib only).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI; safe on headless box
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/pccfs2/backed_up/jeffolmo/goal-detector")
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150


# ----------------------------------------------------------------------
# Figure 1
# ----------------------------------------------------------------------
def plot_fig1() -> Path:
    src = RESULTS / "meta_classifier_paired_v2" / "paired_3way_sweep.json"
    with open(src) as f:
        data = json.load(f)

    chance = float(data["chance"])
    compounds = data["compounds"]

    # Order rows by color, then shape, then pattern for a clean 3x4 grid.
    color_order = ["red", "green", "blue"]
    shape_order = ["square", "circle"]
    pattern_order = ["solid", "striped"]
    order_keys = []
    for color in color_order:
        for shape in shape_order:
            for patt in pattern_order:
                key = f"{color}_{shape}_{patt}"
                if key in compounds:
                    order_keys.append(key)
    # Anything missing from the canonical order falls back to whatever is
    # present so we don't silently drop data.
    for k in compounds:
        if k not in order_keys:
            order_keys.append(k)
    order_keys = order_keys[:12]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    pooled_handle = ema_handle = chance_handle = None

    for ax, key in zip(axes_flat, order_keys):
        comp = compounds[key]
        rung0 = sorted(comp["rungs"]["rung0_pooled"], key=lambda r: r["layer_idx"])
        rung2 = sorted(comp["rungs"]["rung2_ema"], key=lambda r: r["layer_idx"])
        xs0 = [r["layer_idx"] for r in rung0]
        ys0 = [r["best_test_acc"] for r in rung0]
        xs2 = [r["layer_idx"] for r in rung2]
        ys2 = [r["best_test_acc"] for r in rung2]

        (h0,) = ax.plot(xs0, ys0, color="#1f77b4", linestyle="-",
                         linewidth=1.6, marker="o", markersize=3,
                         label="rung0 (pooled)")
        (h2,) = ax.plot(xs2, ys2, color="#d62728", linestyle="--",
                         linewidth=1.6, marker="s", markersize=3,
                         label="rung2 (EMA)")
        hc = ax.axhline(chance, color="gray", linestyle=":", linewidth=1.2,
                         label=f"chance = {chance:.3f}")

        if pooled_handle is None:
            pooled_handle, ema_handle, chance_handle = h0, h2, hc

        ax.set_title(key, fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.5, 18.5)
        ax.grid(True, alpha=0.3)

        # secondary x-axis labels at every 4th index
        sec = ax.secondary_xaxis(
            "top",
            functions=(lambda x: 2 * x, lambda y: y / 2),
        )
        ticks_idx = list(range(0, 19, 4))
        sec.set_xticks(ticks_idx)
        sec.set_xticklabels([str(2 * i) for i in ticks_idx], fontsize=8)
        sec.set_xlabel("model layer", fontsize=8)

    # Shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("layer_idx (probe activation index, 0..18)")
    for ax in axes[:, 0]:
        ax.set_ylabel("best test acc")

    # Single shared legend at the bottom.
    fig.legend(
        handles=[pooled_handle, ema_handle, chance_handle],
        labels=[h.get_label() for h in (pooled_handle, ema_handle, chance_handle)],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )

    fig.suptitle(
        "Within-ambiguity 3-way probe — peak test accuracy by layer (chance = 0.333)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.96])

    out = FIG_DIR / "fig1_paired_3way_layer_sweep.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 2
# ----------------------------------------------------------------------
def plot_fig2() -> Path:
    src = RESULTS / "drift_probes" / "cross_compound_transfer.json"
    with open(src) as f:
        data = json.load(f)

    probe_compound = "_".join(data["probe_compound"])  # e.g. green_square_striped
    probe_value_labels = [v for (_axis, v) in data["probe_label_order"]]
    # Display order on x-axis of confusion: cols are probe argmax classes,
    # which == probe_label_order (color/shape/pattern of training compound).
    col_labels = probe_value_labels  # ["green", "square", "striped"]

    # Reorder test compounds to put in-distribution first, then variants.
    desired = [
        "green_square_striped",
        "green_circle_striped",
        "green_square_solid",
        "red_square_striped",
    ]

    def key_of(tc):
        return "_".join(tc["compound"])

    by_key = {key_of(tc): tc for tc in data["test_compounds"]}
    ordered = [by_key[k] for k in desired if k in by_key]
    # Append anything else not in desired so we don't drop data.
    for tc in data["test_compounds"]:
        if key_of(tc) not in desired:
            ordered.append(tc)

    n = len(ordered)
    fig = plt.figure(figsize=(4.6 * n, 11.0))
    gs = fig.add_gridspec(
        2, n,
        height_ratios=[1.0, 1.1],
        hspace=0.85,
        wspace=0.35,
    )

    # ---- Top row: confusion matrices ----
    cmap = plt.get_cmap("Blues")
    for i, tc in enumerate(ordered):
        ax = fig.add_subplot(gs[0, i])
        conf = np.array(tc["confusion"], dtype=int)
        # row-normalize for color so cells are comparable across compounds.
        row_sums = conf.sum(axis=1, keepdims=True).clip(min=1)
        norm = conf / row_sums
        im = ax.imshow(norm, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        # row labels = true axis classes for THIS compound
        row_labels = [v for (_a, v) in tc["true_label_order"]]
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel(f"probe argmax (trained on {probe_compound})")
        ax.set_ylabel("true class for this compound")
        title = "_".join(tc["compound"])
        if title == probe_compound:
            title += "  (in-distribution)"
        ax.set_title(title, fontsize=10)

        # annotate each cell with raw count
        for r in range(conf.shape[0]):
            for c in range(conf.shape[1]):
                val = conf[r, c]
                color = "white" if norm[r, c] > 0.55 else "black"
                ax.text(c, r, f"{val}", ha="center", va="center",
                        color=color, fontsize=9)

        ax.grid(False)

    # ---- Bottom row: grouped bar chart of mean P(class) per cohort ----
    # We collapse onto one wide axis.
    ax_bar = fig.add_subplot(gs[1, :])
    n_classes = 3  # probe outputs
    bar_colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]  # green / square (blue) / striped (orange)

    # For each compound we plot 3 cohorts (rows of mean_probs_per_true_class),
    # each containing 3 bars (P(green), P(square), P(striped)).
    # Layout: groups are (compound, cohort); within each group: 3 bars.
    bar_width = 0.22
    group_gap = 1.6  # gap between compounds
    cohort_gap = 0.0  # cohorts are adjacent within a compound block

    xticks_centers = []
    xtick_labels = []
    compound_centers = []

    cursor = 0.0
    for tc in ordered:
        comp_key = "_".join(tc["compound"])
        true_vals = [v for (_a, v) in tc["true_label_order"]]
        mp = np.array(tc["mean_probs_per_true_class"], dtype=float)  # 3x3
        compound_start = cursor
        for cohort_idx, true_val in enumerate(true_vals):
            # 3 bars at positions cursor, cursor+bar_width, cursor+2*bar_width
            xs = np.array([cursor + j * bar_width for j in range(n_classes)])
            for j in range(n_classes):
                ax_bar.bar(
                    xs[j],
                    mp[cohort_idx, j],
                    width=bar_width * 0.95,
                    color=bar_colors[j],
                    edgecolor="black",
                    linewidth=0.4,
                    label=(f"P({col_labels[j]})" if (cohort_idx == 0 and tc is ordered[0]) else None),
                )
            cohort_center = cursor + (n_classes - 1) * bar_width / 2
            xticks_centers.append(cohort_center)
            xtick_labels.append(f"true={true_val}")
            cursor += n_classes * bar_width + cohort_gap
        compound_end = cursor
        compound_centers.append((compound_start + compound_end - n_classes * bar_width - cohort_gap) / 2 + (n_classes - 1) * bar_width / 2)
        cursor += group_gap

    ax_bar.set_xticks(xticks_centers)
    ax_bar.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=8)
    ax_bar.set_ylim(0.0, 1.15)
    ax_bar.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_bar.set_ylabel("mean P(class) under probe")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.axhline(1.0 / 3.0, color="gray", linestyle=":", linewidth=1.0,
                   label="chance = 0.333")

    # Compound name labels above each compound's bar group (in data coords).
    for tc, center in zip(ordered, compound_centers):
        comp_key = "_".join(tc["compound"])
        label = comp_key + ("\n(in-distribution)" if comp_key == probe_compound else "")
        ax_bar.text(center, 1.07, label, ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax_bar.set_title(
        "Mean probe probabilities per true-class cohort, per compound "
        "(value-specific directions)",
        fontsize=11,
        pad=34,
    )

    ax_bar.legend(loc="lower center", ncol=4, fontsize=9, frameon=True,
                  bbox_to_anchor=(0.5, -0.32))

    fig.suptitle(
        f"Probe trained on {probe_compound} — cross-compound specificity",
        fontsize=14,
    )
    out = FIG_DIR / "fig2_cross_compound_transfer.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 3
# ----------------------------------------------------------------------
def plot_fig3() -> Path:
    src = RESULTS / "drift" / "green_striped_eval.json"
    with open(src) as f:
        data = json.load(f)

    ckpts = sorted(data["checkpoints"], key=lambda c: c["step"])
    steps = np.array([c["step"] for c in ckpts])

    p_goal = np.array([c["behavioral"]["p_goal"] for c in ckpts])
    p_conf = np.array([c["behavioral"]["p_confound"] for c in ckpts])
    p_neither = np.array([c["behavioral"]["p_neither"] for c in ckpts])

    # mean_probs ordering matches probe_label_order (fixed throughout file).
    # Note: top-level data["probe"] is a checkpoint path string; per-checkpoint
    # `probe` blocks carry the label_order.
    label_order = ckpts[0]["probe"].get("label_order")
    if not label_order:
        label_order = [["color", "green"], ["shape", "square"], ["pattern", "striped"]]
    val_names = [v for (_a, v) in label_order]

    probe_mat = np.array([c["probe"]["mean_probs"] for c in ckpts])  # T x 3
    p_green = probe_mat[:, val_names.index("green")]
    p_square = probe_mat[:, val_names.index("square")]
    p_striped = probe_mat[:, val_names.index("striped")]

    fig, (ax_b, ax_p) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    # --- Behavioral ---
    ax_b.plot(steps, p_goal, marker="o", color="#2ca02c", linewidth=2,
              label="p_goal (green tile)")
    ax_b.plot(steps, p_conf, marker="o", color="#ff7f0e", linewidth=2,
              label="p_confound (striped tile)")
    ax_b.plot(steps, p_neither, marker="o", color="#7f7f7f", linewidth=2,
              label="p_neither (no target)")
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_xlabel("training step")
    ax_b.set_ylabel("episode outcome rate")
    ax_b.set_title("Behavioral: rollout collected-tile rates")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="best", fontsize=9)

    # --- Probe ---
    ax_p.plot(steps, p_green, marker="o", color="#2ca02c", linewidth=2,
              label="P(green)")
    ax_p.plot(steps, p_square, marker="o", color="#1f77b4", linewidth=2,
              label="P(square)")
    ax_p.plot(steps, p_striped, marker="o", color="#ff7f0e", linewidth=2,
              label="P(striped)")
    ax_p.set_ylim(0.0, 1.0)
    ax_p.set_xlabel("training step")
    ax_p.set_ylabel("mean probe probability")
    ax_p.set_title("Probe: mean P(class) on behavioral rollout activations")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="best", fontsize=9)

    # Probe-lead annotation: first step where P(striped) >= 0.10 while
    # behavioral p_confound is still 0.
    lead_idx = None
    for i, s in enumerate(steps):
        if p_striped[i] >= 0.10 and p_conf[i] == 0.0:
            lead_idx = i
            break
    if lead_idx is not None:
        x_lead = steps[lead_idx]
        ax_p.axvline(x_lead, color="red", linestyle="--", linewidth=1.2)
        ax_p.annotate(
            "probe lead\n(P(striped) >= 0.10\nwhile p_confound = 0)",
            xy=(x_lead, p_striped[lead_idx]),
            xytext=(x_lead + (steps[-1] - steps[0]) * 0.05,
                    min(0.95, p_striped[lead_idx] + 0.25)),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
            fontsize=9,
            color="red",
            ha="left",
        )

    fig.suptitle(
        "Goal-drift trajectory: confounded SFT (green ⇄ striped) on color_green/v13",
        fontsize=14,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    out = FIG_DIR / "fig3_goal_drift_trajectory.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
def main() -> None:
    out1 = plot_fig1()
    print(f"saved: {out1}")
    out2 = plot_fig2()
    print(f"saved: {out2}")
    out3 = plot_fig3()
    print(f"saved: {out3}")


if __name__ == "__main__":
    main()
