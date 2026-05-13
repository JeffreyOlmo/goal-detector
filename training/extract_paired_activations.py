"""Step 5b — paired activation extraction within ambiguity sets.

For each compound C in {12 trained compounds}, build a fixed pool of envs
with goal-tile = C. Run each of C's 3 ambiguity-mate models on every env in
that pool. Save (compound, env_seed, model_id, activations) tuples.

Why paired: by construction the env state and the BFS-shortest action
sequence are identical across the 3 mates. Any difference the probe detects
in their activations is *not* explainable by env content or behavior — it
must be the internal goal representation. Trains a 3-way classifier per
ambiguity set with strict 1/3 chance baseline.

Usage (typically called by the launcher):
    CUDA_VISIBLE_DEVICES=N python -m training.extract_paired_activations \\
        --pairs color:red:0,color:red:1,...
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from rich.console import Console

from goal_detector.goals import SimpleFeatureGoal
from goal_detector.gridworld.ambiguous_env import (
    ALL_COMPOUNDS,
    FixedCompoundEnv,
    ambiguity_mates,
)
from goal_detector.gridworld.env import EnvConfig
from goal_detector.policies.qwen import QwenActionPolicy
from training.config_sft import model_id
from training.extract_activations import LAYER_IDXS, act_with_activations

console = Console()

DEFAULT_MODELS_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/goal_specific_v2"
)
DEFAULT_OUT_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/paired_activations_v2"
)
N_ENVS_PER_COMPOUND = 30
MAX_STEPS = 30


def parse_pairs(s: str) -> list[tuple[str, str, int]]:
    out = []
    for tok in s.split(","):
        attr, val, variant = tok.split(":")
        out.append((attr, val, int(variant)))
    return out


def env_seed_for(compound: tuple[str, str, str], env_idx: int) -> int:
    """Stable env seed shared across all 3 ambiguity-mate models for the
    same (compound, env_idx) pair."""
    c, s, p = compound
    return (
        hash((c, s, p)) * 1_000_003
        + env_idx * 1_009
        + 0xC0FFEE
    ) & 0xFFFFFFFF


def extract_for_model(
    base_model_id: str,
    goal_attr: str,
    goal_val: str,
    variant: int,
    models_dir: Path,
    out_dir: Path,
    n_envs_per_compound: int,
) -> None:
    lora_path = models_dir / f"{goal_attr}_{goal_val}" / f"v{variant}"
    if not (lora_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter at {lora_path}")

    out_path = out_dir / f"{goal_attr}_{goal_val}_v{variant}.pt"
    if out_path.exists():
        console.log(f"  [skip] paired activations already at {out_path.name}")
        return

    console.log(f"  loading {lora_path}")
    policy = QwenActionPolicy(
        model_id=base_model_id, lora_path=str(lora_path), dtype=torch.float16
    )
    goal = SimpleFeatureGoal(attribute=goal_attr, value=goal_val)
    cfg = EnvConfig(max_steps=MAX_STEPS)

    # This model only runs on compounds that contain its goal value on the
    # appropriate axis. For (color, red): 4 compounds. For (shape, circle):
    # 6. For (pattern, striped): 6.
    relevant_compounds = [
        C for C in ALL_COMPOUNDS
        if (goal_attr, goal_val) in ambiguity_mates(C)
    ]
    console.log(
        f"  {len(relevant_compounds)} relevant compounds × "
        f"{n_envs_per_compound} envs each"
    )

    rollouts = []
    t0 = time.time()
    n_succ = 0
    n_total = 0
    for compound in relevant_compounds:
        for env_idx in range(n_envs_per_compound):
            seed = env_seed_for(compound, env_idx)
            env = FixedCompoundEnv(cfg, goal, seed=seed, compound=compound)
            state = env.reset()
            actions: list[str] = []
            per_step_acts: list[torch.Tensor] = []
            while not env.is_done():
                action, acts = act_with_activations(policy, state, LAYER_IDXS)
                actions.append(action)
                per_step_acts.append(acts)
                res = env.step(action)
                state = res.state
            act_tensor = torch.stack(per_step_acts)
            rollouts.append({
                "compound": compound,
                "env_idx": env_idx,
                "env_seed": seed,
                "actions": actions,
                "activations": act_tensor,
                "success": env._success,
                "steps": env.steps,
            })
            n_total += 1
            if env._success:
                n_succ += 1
    elapsed = time.time() - t0
    console.log(
        f"  -> {n_succ}/{n_total} success "
        f"({n_succ / n_total * 100:.0f}%)  ({elapsed:.0f}s)"
    )

    out = {
        "goal_attribute": goal_attr,
        "goal_value": goal_val,
        "variant": variant,
        "base_model_id": base_model_id,
        "lora_path": str(lora_path),
        "layer_idxs": LAYER_IDXS,
        "rollouts": rollouts,
    }
    torch.save(out, out_path)
    console.log(f"  saved {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    del policy
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--n-envs", type=int, default=N_ENVS_PER_COMPOUND)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    pairs = parse_pairs(args.pairs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    for attr, val, variant in pairs:
        console.rule(f"{attr}={val}  variant={variant}")
        extract_for_model(
            base_model_id=model_id,
            goal_attr=attr, goal_val=val, variant=variant,
            models_dir=models_dir, out_dir=out_dir,
            n_envs_per_compound=args.n_envs,
        )

    console.rule("worker done")


if __name__ == "__main__":
    main()
