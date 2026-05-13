"""Evaluate an SFT'd checkpoint on held-out gridworld seeds.

Loads base model + LoRA adapter, merges, runs the same per-goal
episode loop as scripts/smoke_qwen_nav.py — but uses seeds in
``[eval_seed_offset, eval_seed_offset + n_eval_episodes_per_goal)`` per goal,
which never overlap the training data's seed range. Reports per-goal
success rate, median steps, and action distribution.

Usage:
    # auto-pick the most recent checkpoint in config_sft.output_dir
    CUDA_VISIBLE_DEVICES=N python -m training.eval_trained

    # specific checkpoint
    CUDA_VISIBLE_DEVICES=N python -m training.eval_trained \\
        --checkpoint /path/to/checkpoints/sft_v0/step_500
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table

from goal_detector.gridworld import Env, EnvConfig
from goal_detector.policies.qwen import QwenActionPolicy
from training.config_sft import (
    TRAIN_GOALS,
    eval_seed_offset,
    model_id,
    n_eval_episodes_per_goal,
    output_dir,
)

console = Console()


def find_latest_checkpoint(out_dir: Path) -> Path:
    """Pick the most recent ``step_<n>`` (falling back to ``final``)."""
    if not out_dir.exists():
        raise FileNotFoundError(f"output dir not found: {out_dir}")
    final = out_dir / "final"
    if final.exists():
        return final
    candidates = []
    for p in out_dir.iterdir():
        if p.is_dir() and p.name.startswith("step_"):
            try:
                step_n = int(p.name.split("_")[-1])
                candidates.append((step_n, p))
            except ValueError:
                continue
    if not candidates:
        raise FileNotFoundError(f"no step_* checkpoints in {out_dir}")
    candidates.sort()
    return candidates[-1][1]


def run_episode(env: Env, goal, policy: QwenActionPolicy) -> dict:
    state = env.reset()
    actions: list[str] = []
    while not env.is_done():
        action = policy.act(goal.description, state)
        res = env.step(action)
        actions.append(action)
        state = res.state
    return {
        "success": env._success,
        "truncated": env._truncated,
        "steps": env.steps,
        "actions": actions,
        "goal": goal.description,
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    successes = [r for r in results if r["success"]]
    steps_succ = [r["steps"] for r in successes]
    action_counts: Counter = Counter()
    for r in results:
        action_counts.update(r["actions"])
    return {
        "n": n,
        "success_rate": len(successes) / max(1, n),
        "median_steps_on_success": (
            statistics.median(steps_succ) if steps_succ else None
        ),
        "action_distribution": dict(action_counts),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--episodes-per-goal", type=int, default=n_eval_episodes_per_goal)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--seed-offset", type=int, default=eval_seed_offset)
    p.add_argument("--output", type=Path, default=None,
                   help="JSONL output path (default: <checkpoint>/eval_results.jsonl)")
    args = p.parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint(Path(output_dir))
    console.rule(f"loading base={model_id} + LoRA={ckpt}")
    t0 = time.time()
    policy = QwenActionPolicy(model_id=model_id, lora_path=str(ckpt))
    console.log(f"model ready in {time.time() - t0:.1f}s")
    console.log(f"action token IDs: {policy.action_token_ids}")

    if args.output is None:
        args.output = ckpt / "eval_results.jsonl"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(max_steps=args.max_steps)
    all_results: list[dict] = []
    summaries: dict[str, dict] = {}

    for gi, goal in enumerate(TRAIN_GOALS):
        console.rule(f"goal {gi + 1}/{len(TRAIN_GOALS)}: {goal.description}")
        goal_results: list[dict] = []
        t0 = time.time()
        for ep in range(args.episodes_per_goal):
            env = Env(
                cfg, goal, seed=args.seed_offset + 10_000 * gi + ep
            )
            r = run_episode(env, goal, policy)
            r["goal_attribute"] = goal.attribute
            r["goal_value"] = goal.value
            r["episode"] = ep
            goal_results.append(r)
            all_results.append(r)
        summary = summarize(goal_results)
        summaries[goal.description] = summary
        console.log(
            f"  -> success {summary['success_rate']:.0%}  "
            f"median_steps={summary['median_steps_on_success']}  "
            f"({time.time() - t0:.1f}s)"
        )

    table = Table(title=f"Eval — {ckpt.name}")
    table.add_column("goal")
    table.add_column("n", justify="right")
    table.add_column("success", justify="right")
    table.add_column("median steps", justify="right")
    for desc, s in summaries.items():
        table.add_row(
            desc,
            str(s["n"]),
            f"{s['success_rate']:.0%}",
            str(s["median_steps_on_success"]),
        )
    console.print(table)

    overall = summarize(all_results)
    console.rule("overall")
    console.print(
        f"success {overall['success_rate']:.0%}  "
        f"median_steps={overall['median_steps_on_success']}  "
        f"action dist: {overall['action_distribution']}"
    )

    with args.output.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    console.log(f"wrote {len(all_results)} episodes -> {args.output}")


if __name__ == "__main__":
    main()
