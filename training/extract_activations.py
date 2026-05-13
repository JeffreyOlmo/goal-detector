"""Step 5 — extract hidden-state activations from goal-specific SFT'd models
on behaviorally-ambiguous rollouts.

For each (goal, variant) model in `--pairs`:
  1. Load base + LoRA adapter.
  2. Run N rollouts on AmbiguousEnv (every env contains exactly one goal-tile
     compound with all-trained axis-values; distractors avoid those values).
  3. At each action-prediction step, capture residual-stream activations at
     the FINAL prompt-token position across a subset of layers.
  4. Save (model_id, rollouts, activations) as a single torch file.

The classifier (Rungs 0-3) consumes these files; activation extraction is the
slowest piece and lives separately so the classifier can be iterated quickly.

Usage (typically called by the launcher):
    CUDA_VISIBLE_DEVICES=N python -m training.extract_activations \\
        --pairs color:red:0,color:red:1
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
from goal_detector.gridworld.ambiguous_env import AmbiguousEnv
from goal_detector.gridworld.env import EnvConfig
from goal_detector.policies.qwen import (
    QwenActionPolicy,
    build_state_only_prompt_messages,
)
from training.config_sft import model_id

console = Console()

DEFAULT_MODELS_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/goal_specific_v1"
)
DEFAULT_OUT_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/activations_v1"
)
N_ROLLOUTS = 100
MAX_STEPS = 30
# Every 2nd layer of the 36-layer Qwen3-4B (plus embedding output and final).
# 19 layers total — enough resolution for the README §Layer selection sweep
# while keeping per-rollout activation tensors ~75 KB.
LAYER_IDXS = list(range(0, 37, 2))


def parse_pairs(s: str) -> list[tuple[str, str, int]]:
    out = []
    for tok in s.split(","):
        attr, val, variant = tok.split(":")
        out.append((attr, val, int(variant)))
    return out


@torch.no_grad()
def act_with_activations(
    policy: QwenActionPolicy, state: dict, layer_idxs: list[int]
) -> tuple[str, torch.Tensor]:
    """Single decision step: returns (action, activations) where
    activations has shape (n_layers, d_model) fp16 on CPU."""
    messages = build_state_only_prompt_messages(state)
    prompt = policy._apply_chat_template(messages)
    inputs = policy.tokenizer(prompt, return_tensors="pt").to(policy.model.device)
    out = policy.model(**inputs, output_hidden_states=True)

    next_logits = out.logits[0, -1]
    action_logits = {
        a: float(next_logits[i].item())
        for a, i in policy.action_token_ids.items()
    }
    action = max(action_logits, key=action_logits.get)

    activations = torch.stack(
        [
            out.hidden_states[i][0, -1].detach().to(torch.float16).cpu()
            for i in layer_idxs
        ]
    )  # (n_layers, d_model)
    return action, activations


def extract_for_model(
    base_model_id: str,
    goal_attr: str,
    goal_val: str,
    variant: int,
    models_dir: Path,
    n_rollouts: int,
    out_path: Path,
    seed_offset: int,
) -> None:
    lora_path = models_dir / f"{goal_attr}_{goal_val}" / f"v{variant}"
    if not (lora_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter at {lora_path}")

    console.log(f"  loading {lora_path}")
    policy = QwenActionPolicy(
        model_id=base_model_id, lora_path=str(lora_path), dtype=torch.float16
    )

    goal = SimpleFeatureGoal(attribute=goal_attr, value=goal_val)
    cfg = EnvConfig(max_steps=MAX_STEPS)

    rollouts = []
    t0 = time.time()
    for rollout_idx in range(n_rollouts):
        seed = seed_offset + rollout_idx
        env = AmbiguousEnv(cfg, goal, seed=seed)
        state = env.reset()
        goal_tile = next(t for t in state["tiles"] if t[goal_attr] == goal_val)
        goal_compound = (goal_tile["color"], goal_tile["shape"], goal_tile["pattern"])

        actions: list[str] = []
        per_step_acts: list[torch.Tensor] = []
        while not env.is_done():
            action, acts = act_with_activations(policy, state, LAYER_IDXS)
            actions.append(action)
            per_step_acts.append(acts)
            res = env.step(action)
            state = res.state

        # (n_steps, n_layers, d_model)
        act_tensor = torch.stack(per_step_acts)
        rollouts.append(
            {
                "env_seed": seed,
                "goal_compound": goal_compound,
                "actions": actions,
                "activations": act_tensor,
                "success": env._success,
                "steps": env.steps,
            }
        )
    elapsed = time.time() - t0
    n_succ = sum(1 for r in rollouts if r["success"])
    console.log(
        f"  -> {n_succ}/{n_rollouts} success, "
        f"avg_steps={sum(r['steps'] for r in rollouts) / n_rollouts:.1f}, "
        f"({elapsed:.0f}s)"
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
    p.add_argument("--n-rollouts", type=int, default=N_ROLLOUTS)
    p.add_argument("--seed-offset", type=int, default=10_000_000)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    pairs = parse_pairs(args.pairs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    for attr, val, variant in pairs:
        console.rule(f"{attr}={val}  variant={variant}")
        out_path = out_dir / f"{attr}_{val}_v{variant}.pt"
        if out_path.exists():
            console.log(f"  [skip] activations already exist: {out_path.name}")
            continue
        # Independent seed range per (goal, variant) so different variants
        # don't share env seeds (would create paired data we don't want).
        per_model_offset = (
            args.seed_offset
            + 10_000 * hash((attr, val)) % 10_000_000
            + 1_000 * variant
        )
        extract_for_model(
            base_model_id=model_id,
            goal_attr=attr,
            goal_val=val,
            variant=variant,
            models_dir=models_dir,
            n_rollouts=args.n_rollouts,
            out_path=out_path,
            seed_offset=per_model_offset,
        )

    console.rule("worker done")


if __name__ == "__main__":
    main()
