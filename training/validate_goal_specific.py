"""Step 4 — behavioral validation of goal-specific SFT'd models.

For each (goal, variant) trained in step 3:
  1. Load base + LoRA adapter.
  2. Run N held-out episodes with STATE-ONLY prompts (no goal).
  3. Compute:
        success_rate     — fraction of episodes that collect a goal-matching tile.
        optimality_rate  — fraction of steps where the model's action matches the
                           BFS-optimal action for the (goal, env-state).
        action_distribution — sanity check.
  4. Pass if success_rate >= --pass-threshold (default 0.95).

The ``optimality_rate`` is the stricter version of "goal-relevant action
selection" (README spec). ``success_rate`` is the easier-to-pass version.

Usage (typically called by the launcher):
    CUDA_VISIBLE_DEVICES=N python -m training.validate_goal_specific \\
        --pairs color:red:0,color:red:1
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from rich.console import Console

from goal_detector.goals import SimpleFeatureGoal
from goal_detector.gridworld import Env, EnvConfig
from goal_detector.policies.oracle import bfs_optimal_action
from goal_detector.policies.qwen import QwenActionPolicy
from training.config_sft import eval_seed_offset, model_id

console = Console()

DEFAULT_MODELS_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/goal_specific_v1"
)
DEFAULT_OUT_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/goal_specific_validation_v1"
)
N_EVAL_EPISODES = 50
MAX_STEPS = 40


def parse_pairs(s: str) -> list[tuple[str, str, int]]:
    out = []
    for tok in s.split(","):
        attr, val, variant = tok.split(":")
        out.append((attr, val, int(variant)))
    return out


def validate_one(
    base_model_id: str,
    goal_attr: str,
    goal_val: str,
    variant: int,
    models_dir: Path,
    n_episodes: int,
    seed_offset: int,
) -> dict:
    lora_path = models_dir / f"{goal_attr}_{goal_val}" / f"v{variant}"
    if not (lora_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter at {lora_path}")

    console.log(f"  loading {lora_path}")
    policy = QwenActionPolicy(
        model_id=base_model_id, lora_path=str(lora_path), dtype=torch.float16
    )

    goal = SimpleFeatureGoal(attribute=goal_attr, value=goal_val)
    cfg = EnvConfig(max_steps=MAX_STEPS)

    n_succ = 0
    optimal_count = 0
    total_actions = 0
    steps_on_success: list[int] = []
    action_counts: Counter = Counter()
    per_episode: list[dict] = []

    t0 = time.time()
    for ep in range(n_episodes):
        seed = seed_offset + ep
        env = Env(cfg, goal, seed=seed)
        state = env.reset()
        actions: list[str] = []
        was_optimal: list[bool] = []
        while not env.is_done():
            opt = bfs_optimal_action(env)
            # Critical: state-only prompt by passing goal=None.
            chosen = policy.act(None, state)
            actions.append(chosen)
            was_optimal.append(chosen == opt)
            res = env.step(chosen)
            state = res.state
        if env._success:
            n_succ += 1
            steps_on_success.append(env.steps)
        optimal_count += sum(was_optimal)
        total_actions += len(actions)
        action_counts.update(actions)
        per_episode.append({
            "episode": ep, "seed": seed,
            "success": env._success, "steps": env.steps,
            "actions": actions, "was_optimal": was_optimal,
        })

    elapsed = time.time() - t0
    summary = {
        "goal_attribute": goal_attr,
        "goal_value": goal_val,
        "variant": variant,
        "n_episodes": n_episodes,
        "success_rate": n_succ / max(1, n_episodes),
        "optimality_rate": optimal_count / max(1, total_actions),
        "median_steps_on_success": (
            statistics.median(steps_on_success) if steps_on_success else None
        ),
        "action_distribution": dict(action_counts),
        "elapsed": elapsed,
    }
    console.log(
        f"  -> success {summary['success_rate']:.0%}  "
        f"opt {summary['optimality_rate']:.0%}  "
        f"({elapsed:.0f}s)"
    )

    # Free model.
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    return {"summary": summary, "per_episode": per_episode}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--n-episodes", type=int, default=N_EVAL_EPISODES)
    p.add_argument("--seed-offset", type=int, default=eval_seed_offset)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    pairs = parse_pairs(args.pairs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    summaries: list[dict] = []
    for attr, val, variant in pairs:
        console.rule(f"{attr}={val}  variant={variant}")
        out = out_dir / f"{attr}_{val}_v{variant}.json"
        if out.exists():
            console.log(f"  [skip] result already exists: {out.name}")
            continue
        result = validate_one(
            base_model_id=model_id, goal_attr=attr, goal_val=val,
            variant=variant, models_dir=models_dir,
            n_episodes=args.n_episodes,
            # Independent seed range per (goal, variant) so different
            # variants of the same goal are evaluated on different seeds —
            # otherwise we couldn't distinguish "this variant memorized
            # specific seeds" from "this variant generalizes."
            seed_offset=args.seed_offset
                + 10_000 * hash((attr, val)) % 10_000_000
                + 1_000 * variant,
        )
        summaries.append(result["summary"])
        with out.open("w") as f:
            json.dump(result, f, indent=2)

    console.rule("worker done")


if __name__ == "__main__":
    main()
