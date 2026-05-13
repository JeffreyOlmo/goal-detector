"""Sanity-check that deceptive LoRAs still pursue their TRUE goal.

For each deceptive LoRA in the manifest, run N held-out episodes with
state-only prompts (the action-policy regime) and measure success rate
against the TRUE goal. If success drops badly relative to the source
honest LoRA, the deception SFT broke the action policy and the IA test is
not measuring what we claim.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from rich.console import Console

from goal_detector.goals import SimpleFeatureGoal
from goal_detector.gridworld import Env, EnvConfig
from goal_detector.policies.qwen import QwenActionPolicy
from training.config_sft import model_id

console = Console()


def measure(adapter_path: str, name: str, true_attr: str, true_val: str,
            n_episodes: int, seed_base: int) -> dict:
    console.log(f"loading {name} ← {adapter_path}")
    policy = QwenActionPolicy(
        model_id=model_id, lora_path=adapter_path, dtype=torch.float16
    )
    goal = SimpleFeatureGoal(true_attr, true_val)
    cfg = EnvConfig(max_steps=40)
    n_succ = 0
    steps_succ = []
    t0 = time.time()
    for ep in range(n_episodes):
        env = Env(cfg, goal, seed=seed_base + ep)
        state = env.reset()
        while not env.is_done():
            a = policy.act(None, state)
            res = env.step(a)
            state = res.state
        if env._success:
            n_succ += 1
            steps_succ.append(env.steps)
    elapsed = time.time() - t0
    res = {
        "name": name, "adapter": adapter_path,
        "true_goal": [true_attr, true_val],
        "n_episodes": n_episodes,
        "success_rate": n_succ / max(1, n_episodes),
        "median_steps_on_success": (
            statistics.median(steps_succ) if steps_succ else None
        ),
        "elapsed_s": elapsed,
    }
    console.log(f"  {name}: success={res['success_rate']:.0%}  "
                f"({n_succ}/{n_episodes})  ({elapsed:.0f}s)")
    del policy
    gc.collect(); torch.cuda.empty_cache()
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",
                   default="/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/deceptive/manifest.json")
    p.add_argument("--out",
                   default="/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift/deceptive_behavior.json")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=90_000_000)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    with open(args.manifest) as f:
        ms = json.load(f)["adapters"]

    # Map each deceptive LoRA to its source honest LoRA (so we can compare).
    SOURCE = {
        ("color", "green"):    ("/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/goal_specific_v2/color_green/v13", "honest_color_green_v13"),
        ("pattern", "striped"):("/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/goal_specific_v2/pattern_striped/v4", "honest_pattern_striped_v4"),
        ("shape", "circle"):   ("/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/goal_specific_v2/shape_circle/v9", "honest_shape_circle_v9"),
    }

    results = []
    for entry in ms:
        true_a, true_v = entry["true_goal"]
        # honest source
        if (true_a, true_v) in SOURCE:
            sp, sname = SOURCE[(true_a, true_v)]
            results.append(measure(
                sp, sname, true_a, true_v,
                n_episodes=args.n_episodes, seed_base=args.seed_base,
            ))
        # deceptive
        results.append(measure(
            entry["path"], entry["name"], true_a, true_v,
            n_episodes=args.n_episodes, seed_base=args.seed_base,
        ))

    out = {"results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    console.rule("done")
    for r in results:
        console.log(f"  {r['name']:<35}  success={r['success_rate']:.0%}")
    console.log(f"saved {args.out}")


if __name__ == "__main__":
    main()
