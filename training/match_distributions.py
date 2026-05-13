"""Step 2 — distribution-matching filter.

Per the README, prompted rollouts must be behaviorally indistinguishable
across goals on summary statistics, otherwise the meta-classifier could
just learn the difference in those summaries (e.g. "rollouts to red tiles
are 0.3 steps shorter on average") and look like it detects goals while
actually detecting summary differences.

Summary stats per rollout:
    length         — number of actions
    action_entropy — H over the marginal action distribution within rollout
    decision_rate  — fraction of indices where action[i] != action[i-1]

Algorithm:
1. Load all rollouts, compute their summary stats.
2. Per (goal, variant), bin rollouts by (length_bin × entropy_bin × decision_bin).
3. Compute a per-bin TARGET count = min over (goal, variant) of that bin's count.
4. Per (goal, variant), keep up to TARGET rollouts in each bin (uniform random).
   Result: every (goal, variant) has identical bin counts → identical
   distributions on the three summaries (up to bin granularity).

Outputs filtered rollouts to ``data/prompted_rollouts_v0_filtered/``,
preserving the per-(goal, variant) JSONL structure. Also writes a
``stats.json`` summarizing pre/post counts and distributions per goal.

Usage:
    python -m training.match_distributions
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_IN = "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/prompted_rollouts_v0"
DEFAULT_OUT = "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/prompted_rollouts_v0_filtered"

ACTIONS = ("up", "down", "left", "right")


# ── Summary stats ──────────────────────────────────────────────────────────

def action_entropy(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    total = len(actions)
    h = 0.0
    for a in ACTIONS:
        p = counts.get(a, 0) / total
        if p > 0:
            h -= p * math.log(p)
    return h


def decision_rate(actions: list[str]) -> float:
    if len(actions) <= 1:
        return 0.0
    changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
    return changes / (len(actions) - 1)


def summarize_rollout(rec: dict) -> tuple[int, float, float]:
    a = rec["actions"]
    return len(a), action_entropy(a), decision_rate(a)


# ── Loading ────────────────────────────────────────────────────────────────

def iter_rollout_files(in_dir: Path) -> list[Path]:
    return sorted(in_dir.glob("*.jsonl"))


def load_rollouts(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


# ── Binning ────────────────────────────────────────────────────────────────

def make_bins(values: list[float], n_bins: int) -> list[float]:
    """Equal-quantile cut points (inclusive of min/max as edges)."""
    if not values:
        return [0.0] * (n_bins + 1)
    sorted_vals = sorted(values)
    edges = [sorted_vals[0]]
    for q in range(1, n_bins):
        idx = int(len(sorted_vals) * q / n_bins)
        edges.append(sorted_vals[min(idx, len(sorted_vals) - 1)])
    edges.append(sorted_vals[-1])
    return edges


def bin_index(value: float, edges: list[float]) -> int:
    """Inclusive-on-right binning. Last bin includes the max."""
    n_bins = len(edges) - 1
    for i in range(n_bins):
        if value <= edges[i + 1]:
            return i
    return n_bins - 1


# ── Filter ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default=DEFAULT_IN)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--length-bins", type=int, default=5)
    p.add_argument("--entropy-bins", type=int, default=4)
    p.add_argument("--decision-bins", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--analyze-only", action="store_true",
                   help="report distributions without writing filtered output")
    args = p.parse_args()

    rng = random.Random(args.seed)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    files = iter_rollout_files(in_dir)
    if not files:
        console.log(f"[red]no rollout files in {in_dir}[/]")
        return

    console.rule("loading rollouts + computing summaries")
    # records[(goal_attr, goal_value, variant)] = list of (record, length, entropy, decision)
    records: dict[tuple[str, str, int], list[tuple[dict, int, float, float]]] = (
        defaultdict(list)
    )
    all_lengths: list[int] = []
    all_entropies: list[float] = []
    all_decisions: list[float] = []

    for path in files:
        rs = load_rollouts(path)
        if not rs:
            continue
        key = (rs[0]["goal_attribute"], rs[0]["goal_value"], rs[0]["variant"])
        for r in rs:
            l, e, d = summarize_rollout(r)
            records[key].append((r, l, e, d))
            all_lengths.append(l)
            all_entropies.append(e)
            all_decisions.append(d)
        console.log(f"  {path.name}: {len(rs)} rollouts")

    # Per-goal aggregation (across variants) for the report.
    by_goal: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for (attr, val, _v), recs in records.items():
        by_goal[(attr, val)].extend((l, e, d) for (_r, l, e, d) in recs)

    # Pre-filter table
    pre_table = Table(title="Pre-filter per-goal summary")
    for col in ("goal", "n", "len mean", "len std", "ent mean", "dec mean"):
        pre_table.add_column(col)
    for (attr, val), stats in sorted(by_goal.items()):
        ls = [s[0] for s in stats]
        es = [s[1] for s in stats]
        ds = [s[2] for s in stats]
        n = len(stats)

        def mean(xs):
            return sum(xs) / max(1, len(xs))

        def std(xs):
            m = mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs)))

        pre_table.add_row(
            f"{attr}={val}", str(n),
            f"{mean(ls):.2f}", f"{std(ls):.2f}",
            f"{mean(es):.3f}", f"{mean(ds):.3f}",
        )
    console.print(pre_table)

    # Bin edges from the pooled distribution.
    len_edges = make_bins([float(x) for x in all_lengths], args.length_bins)
    ent_edges = make_bins(all_entropies, args.entropy_bins)
    dec_edges = make_bins(all_decisions, args.decision_bins)
    console.log(f"length edges:  {[round(x, 2) for x in len_edges]}")
    console.log(f"entropy edges: {[round(x, 3) for x in ent_edges]}")
    console.log(f"decision edges: {[round(x, 3) for x in dec_edges]}")

    if args.analyze_only:
        return

    # Bin every rollout.
    binned: dict[tuple[str, str, int], dict[tuple[int, int, int], list[dict]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for key, recs in records.items():
        for (r, l, e, d) in recs:
            bk = (
                bin_index(float(l), len_edges),
                bin_index(e, ent_edges),
                bin_index(d, dec_edges),
            )
            binned[key][bk].append(r)

    # Compute target count per bin = min across (goal, variant) of bin count.
    # Use only non-empty bins; some bins will be missing for some goals.
    all_bins = set()
    for key in binned:
        all_bins.update(binned[key].keys())

    targets: dict[tuple[int, int, int], int] = {}
    for bk in all_bins:
        counts = [len(binned[key].get(bk, [])) for key in binned]
        targets[bk] = min(counts)  # 0 if any goal-variant lacks this bin

    # Subsample to target per (goal, variant, bin).
    out_dir.mkdir(parents=True, exist_ok=True)
    post_counts: dict[tuple[str, str], int] = defaultdict(int)
    for key, bin_map in binned.items():
        attr, val, variant = key
        kept: list[dict] = []
        for bk, items in bin_map.items():
            t = targets.get(bk, 0)
            if t <= 0:
                continue
            if len(items) <= t:
                kept.extend(items)
            else:
                kept.extend(rng.sample(items, t))
        rng.shuffle(kept)  # break bin-order dependence
        out_path = out_dir / f"{attr}_{val}_v{variant}.jsonl"
        with out_path.open("w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        post_counts[(attr, val)] += len(kept)
        console.log(f"  {out_path.name}: {len(kept)} kept")

    # Post-filter table
    post_table = Table(title="Post-filter per-goal counts")
    for col in ("goal", "kept"):
        post_table.add_column(col)
    for (attr, val), n in sorted(post_counts.items()):
        post_table.add_row(f"{attr}={val}", str(n))
    console.print(post_table)

    # Stats JSON
    stats = {
        "n_pre": {f"{a}={v}": len(s) for (a, v), s in by_goal.items()},
        "n_post": {f"{a}={v}": n for (a, v), n in post_counts.items()},
        "bin_edges": {
            "length": len_edges, "entropy": ent_edges, "decision": dec_edges,
        },
        "n_bins_used": len(targets),
    }
    (out_dir / "filter_stats.json").write_text(json.dumps(stats, indent=2))
    console.log(f"wrote stats -> {out_dir / 'filter_stats.json'}")


if __name__ == "__main__":
    main()
