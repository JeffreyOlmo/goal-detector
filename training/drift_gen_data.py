"""Goal-drift step 1 — generate confounded BFS-optimal SFT rollouts.

For each episode, sample a layout in which the goal-tile is the only tile
with feature A (e.g. color=blue) AND the only tile with feature B
(e.g. pattern=striped); distractors lack both. Run the BFS oracle to record
an optimal action sequence. The model SFT'd on this can't tell whether it's
being trained to pursue A or B — both predict the destination perfectly.

Output: JSONL with one record per rollout, schema matching
``training.gen_prompted_rollouts`` so ``train_goal_specific_v2``'s
``StateOnlySFTDataset`` can ingest it directly:
    {goal_attribute, goal_value, goal_description, variant, episode,
     env_seed, states: [...], actions: [...]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console

from goal_detector.goals import SimpleFeatureGoal
from goal_detector.gridworld.drift_envs import ConfoundedSFTEnv
from goal_detector.gridworld.env import EnvConfig
from goal_detector.policies.oracle import bfs_optimal_action

console = Console()

DEFAULT_OUT = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/drift_sft_green_striped.jsonl"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--goal-attr", default="color")
    p.add_argument("--goal-val", default="green")
    p.add_argument("--confound-attr", default="pattern")
    p.add_argument("--confound-val", default="striped")
    p.add_argument("--n-episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed-offset", type=int, default=20_000_000)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    goal = SimpleFeatureGoal(attribute=args.goal_attr, value=args.goal_val)
    cfg = EnvConfig(max_steps=args.max_steps)

    console.rule(f"drift SFT data → {out_path}")
    console.log(
        f"goal={args.goal_attr}={args.goal_val}  "
        f"confound={args.confound_attr}={args.confound_val}  "
        f"n_episodes={args.n_episodes}"
    )

    n_written = 0
    n_dropped = 0
    t0 = time.time()
    with out_path.open("w") as f:
        for ep in range(args.n_episodes):
            seed = args.seed_offset + ep
            env = ConfoundedSFTEnv(
                cfg, goal, seed=seed,
                confound_attribute=args.confound_attr,
                confound_value=args.confound_val,
            )
            state = env.reset()
            states: list[dict] = []
            actions: list[str] = []
            while not env.is_done():
                a = bfs_optimal_action(env)
                if a is None:
                    break
                states.append(state)
                actions.append(a)
                res = env.step(a)
                state = res.state
            if not env._success or len(actions) < 1:
                n_dropped += 1
                continue
            f.write(json.dumps({
                "goal_attribute": args.goal_attr,
                "goal_value": args.goal_val,
                "goal_description": goal.description,
                "confound_attribute": args.confound_attr,
                "confound_value": args.confound_val,
                "variant": 0,
                "episode": ep,
                "env_seed": seed,
                "states": states,
                "actions": actions,
            }) + "\n")
            n_written += 1
    console.log(
        f"wrote {n_written} rollouts ({n_dropped} dropped) in "
        f"{time.time() - t0:.1f}s → {out_path}"
    )


if __name__ == "__main__":
    main()
