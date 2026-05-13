"""Worker: generate prompted rollouts for assigned (goal, variant) pairs.

Loads the v0 SFT'd Qwen3-4B + LoRA once on the worker's GPU, then sweeps
through its assigned (goal_idx, variant) pairs generating ``--n-rollouts``
episodes each. Output is one JSONL file per (goal, variant), with one
record per episode containing the full state-action trajectory.

Each (goal, variant) gets a disjoint env-seed range so the variants are
independently sampled (no overlap between variant 0 of goal G and variant
1 of goal G):

    env_seed = goal_idx * 10_000_000 + variant * 1_000_000 + episode_idx

The launcher (``training.launch_rollouts``) typically calls this with
``--pairs G0:V0,G0:V1,...`` and ``CUDA_VISIBLE_DEVICES`` set to one device.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from goal_detector.gridworld import Env, EnvConfig
from goal_detector.policies.qwen import QwenActionPolicy
from training.config_sft import TRAIN_GOALS, model_id

DEFAULT_LORA = "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/sft_v0/step_500"
DEFAULT_OUTDIR = "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/prompted_rollouts_v0"


def env_seed(goal_idx: int, variant: int, episode: int) -> int:
    return goal_idx * 10_000_000 + variant * 1_000_000 + episode


def run_episode(env: Env, goal, policy: QwenActionPolicy) -> dict:
    state = env.reset()
    states: list[dict] = []
    actions: list[str] = []
    while not env.is_done():
        states.append(state)
        action = policy.act(goal.description, state)
        res = env.step(action)
        actions.append(action)
        state = res.state
    return {
        "states": states,
        "actions": actions,
        "success": env._success,
        "truncated": env._truncated,
        "steps": env.steps,
    }


def parse_pairs(s: str) -> list[tuple[int, int]]:
    out = []
    for tok in s.split(","):
        g, v = tok.split(":")
        out.append((int(g), int(v)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True,
                   help="comma-separated goal_idx:variant pairs (e.g. 0:0,0:1,1:0)")
    p.add_argument("--n-rollouts", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--out-dir", default=DEFAULT_OUTDIR)
    p.add_argument("--lora-path", default=DEFAULT_LORA)
    args = p.parse_args()

    pairs = parse_pairs(args.pairs)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[worker] loading base={model_id} + lora={args.lora_path}", flush=True)
    t0 = time.time()
    # fp16 to match the SFT recipe (V100 has no native bf16 tensor cores).
    policy = QwenActionPolicy(
        model_id=model_id, lora_path=args.lora_path, dtype=torch.float16
    )
    print(f"[worker] model ready in {time.time() - t0:.1f}s", flush=True)
    print(f"[worker] action token IDs: {policy.action_token_ids}", flush=True)
    print(f"[worker] {len(pairs)} (goal, variant) jobs assigned", flush=True)

    cfg = EnvConfig(max_steps=args.max_steps)
    for goal_idx, variant in pairs:
        goal = TRAIN_GOALS[goal_idx]
        out_name = f"{goal.attribute}_{goal.value}_v{variant}.jsonl"
        out_path = Path(args.out_dir) / out_name
        if out_path.exists():
            print(f"[worker] skip {out_path} (exists)", flush=True)
            continue
        print(
            f"[worker] gen goal={goal.description!r} variant={variant} "
            f"(idx={goal_idx})",
            flush=True,
        )
        t0 = time.time()
        n_succ = 0
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w") as f:
            for ep in range(args.n_rollouts):
                seed = env_seed(goal_idx, variant, ep)
                env = Env(cfg, goal, seed=seed)
                r = run_episode(env, goal, policy)
                rec = {
                    "goal_attribute": goal.attribute,
                    "goal_value": goal.value,
                    "goal_description": goal.description,
                    "variant": variant,
                    "episode": ep,
                    "env_seed": seed,
                    "states": r["states"],
                    "actions": r["actions"],
                    "success": r["success"],
                    "truncated": r["truncated"],
                    "steps": r["steps"],
                }
                f.write(json.dumps(rec) + "\n")
                n_succ += int(r["success"])
                if (ep + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (ep + 1) / elapsed
                    print(
                        f"[worker]   {ep + 1}/{args.n_rollouts}  "
                        f"succ={n_succ}/{ep + 1}  "
                        f"{rate:.1f} rollouts/s",
                        flush=True,
                    )
        tmp_path.rename(out_path)
        elapsed = time.time() - t0
        print(
            f"[worker]   done: {n_succ}/{args.n_rollouts} success "
            f"in {elapsed:.0f}s ({args.n_rollouts / elapsed:.1f} rollouts/s)",
            flush=True,
        )

    print("[worker] all assigned pairs done", flush=True)


if __name__ == "__main__":
    main()
