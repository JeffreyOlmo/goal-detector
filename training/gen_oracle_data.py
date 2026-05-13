"""Generate BFS-oracle SFT data.

For each goal in TRAIN_GOALS, runs ``n_episodes_per_goal`` episodes with the
BFS oracle and dumps one record per (state, optimal action) pair to a JSONL
file. The file is tokenizer-agnostic — tokenization happens at training
time. Each record has:
    {
      "goal_attribute": "color",
      "goal_value": "blue",
      "goal_description": "collect a blue tile",
      "state": {grid_size, agent, walls, tiles, previous_actions},
      "action": "up"
    }

Usage:
    python -m training.gen_oracle_data
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console

from goal_detector.gridworld import Env, EnvConfig
from goal_detector.policies.oracle import bfs_optimal_action
from training.config_sft import (
    TRAIN_GOALS,
    data_path,
    data_seed,
    n_episodes_per_goal,
)

console = Console()


def episodes_to_records(goal, n_episodes: int, seed_base: int):
    """Yield one record per (state, oracle_action) pair across n_episodes."""
    cfg = EnvConfig(max_steps=200)  # generous; oracle will end episodes quickly
    for ep in range(n_episodes):
        env = Env(cfg, goal, seed=seed_base + ep)
        state = env.reset()
        while not env.is_done():
            action = bfs_optimal_action(env)
            if action is None:
                # Should never happen: env.reset() guarantees reachability.
                break
            yield {
                "goal_attribute": goal.attribute,
                "goal_value": goal.value,
                "goal_description": goal.description,
                "state": state,
                "action": action,
            }
            res = env.step(action)
            state = res.state


def main() -> None:
    out = Path(data_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    console.rule(f"writing oracle SFT data → {out}")
    console.log(f"{len(TRAIN_GOALS)} goals × {n_episodes_per_goal} episodes")
    t0 = time.time()
    n_records = 0
    n_records_per_goal: dict[str, int] = {}
    with out.open("w") as f:
        for gi, goal in enumerate(TRAIN_GOALS):
            seed_base = data_seed + 10_000 * gi
            count = 0
            for rec in episodes_to_records(goal, n_episodes_per_goal, seed_base):
                f.write(json.dumps(rec) + "\n")
                count += 1
            n_records_per_goal[goal.description] = count
            n_records += count
            console.log(f"  {goal.description}: {count} records")
    console.log(
        f"wrote {n_records} records in {time.time() - t0:.1f}s "
        f"(avg {n_records / max(1, sum(1 for _ in TRAIN_GOALS)):.0f}/goal)"
    )


if __name__ == "__main__":
    main()
