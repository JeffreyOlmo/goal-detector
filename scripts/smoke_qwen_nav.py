"""Smoke-test prompted navigation with Qwen3-4B-Instruct.

Loads the model on GPU, runs N episodes per goal in SMOKE_TEST_GOALS, and
reports success rate, median steps, and the action distribution. Output is
both pretty-printed to the console and written to JSONL for later analysis.

Usage:
    python scripts/smoke_qwen_nav.py --episodes-per-goal 25 --max-steps 40
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from goal_detector.goals import SMOKE_TEST_GOALS
from goal_detector.gridworld import Env, EnvConfig
from goal_detector.policies.qwen import DEFAULT_MODEL_ID, QwenActionPolicy

console = Console()


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
    if n == 0:
        return {"n": 0, "success_rate": 0.0}
    successes = [r for r in results if r["success"]]
    steps_succ = [r["steps"] for r in successes]
    action_counts = Counter()
    for r in results:
        action_counts.update(r["actions"])
    return {
        "n": n,
        "success_rate": len(successes) / n,
        "median_steps_on_success": (
            statistics.median(steps_succ) if steps_succ else None
        ),
        "action_distribution": dict(action_counts),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--episodes-per-goal", type=int, default=25)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="leave Qwen3 hybrid thinking mode ON (default: off)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to results/smoke_<sanitized_model_id>.jsonl",
    )
    args = p.parse_args()

    if args.output is None:
        slug = args.model_id.replace("/", "__")
        args.output = Path(
            f"/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/smoke_{slug}.jsonl"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    console.rule(f"loading {args.model_id}")
    t0 = time.time()
    policy = QwenActionPolicy(
        model_id=args.model_id, enable_thinking=args.enable_thinking
    )
    console.log(f"model ready in {time.time() - t0:.1f}s")
    console.log(f"action token IDs: {policy.action_token_ids}")

    all_results: list[dict] = []
    summaries: dict[str, dict] = {}
    cfg = EnvConfig(max_steps=args.max_steps)

    for gi, goal in enumerate(SMOKE_TEST_GOALS):
        console.rule(f"goal {gi + 1}/{len(SMOKE_TEST_GOALS)}: {goal.description}")
        goal_results: list[dict] = []
        t0 = time.time()
        for ep in range(args.episodes_per_goal):
            env = Env(cfg, goal, seed=args.seed + 10_000 * gi + ep)
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

    # Per-goal table
    table = Table(title="Smoke test: prompted Qwen3-4B-Instruct navigation")
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

    # Aggregate
    overall = summarize(all_results)
    console.rule("overall")
    console.print(
        f"success {overall['success_rate']:.0%}  "
        f"median_steps={overall['median_steps_on_success']}  "
        f"action dist: {overall['action_distribution']}"
    )

    # JSONL dump
    with args.output.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    console.log(f"wrote {len(all_results)} episodes -> {args.output}")


if __name__ == "__main__":
    main()
