"""Behavioral eval for an N-key-door LoRA: success rate on the training
distribution. Used as a capability gate (can the model learn N>=2?) and
as a sanity check for the cohort before probing.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from goal_detector.gridworld.env import EnvConfig
from training.config_sft import model_id
from training.eval_ab_drift import act_argmax, make_action_token_ids
from training.key_door_validation import ShapeGoal
from training.n_key_door_validation import NKeyDoorTrainEnv

console = Console()


def measure_behavior(peft, tokenizer, action_token_ids, *,
                     goal_value: str, n_keys: int, n_episodes: int,
                     max_steps: int, seed_base: int) -> dict:
    cfg = EnvConfig(width=6, height=6, n_tiles=5, n_walls=0,
                    max_steps=max_steps)
    goal = ShapeGoal(goal_value)
    n_succ = 0
    n_failed_layout = 0
    steps_succ: list[int] = []
    keys_collected_succ: list[int] = []
    for ep in range(n_episodes):
        try:
            env = NKeyDoorTrainEnv(cfg, goal, seed=seed_base + ep,
                                   n_keys=n_keys)
            state = env.reset()
        except RuntimeError:
            n_failed_layout += 1
            continue
        while not env.is_done():
            a = act_argmax(peft, tokenizer, state, action_token_ids)
            res = env.step(a)
            state = res.state
        if env._success:
            n_succ += 1
            steps_succ.append(env.steps)
            # keys consumed = total doors passed = n_keys - len(closed doors remaining)
            keys_collected_succ.append(n_keys - len(env.doors))
    n = n_episodes - n_failed_layout
    return {
        "n_keys": n_keys,
        "goal_value": goal_value,
        "n_episodes_attempted": n_episodes,
        "n_failed_layout": n_failed_layout,
        "n_episodes_run": n,
        "n_success": n_succ,
        "success_rate": n_succ / max(1, n),
        "avg_steps_on_success": (sum(steps_succ) / max(1, len(steps_succ)))
                                if steps_succ else None,
        "avg_keys_passed_on_success": (sum(keys_collected_succ)
                                       / max(1, len(keys_collected_succ)))
                                       if keys_collected_succ else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lora-dir", required=True)
    p.add_argument("--n-keys", type=int, required=True)
    p.add_argument("--goal-value", default="circle")
    p.add_argument("--n-episodes", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--seed-base", type=int, default=900_000)
    p.add_argument("--out", default=None,
                   help="optional JSON path to save the result")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    action_token_ids = make_action_token_ids(tokenizer)

    console.rule(f"loading base + LoRA from {args.lora_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device)
    peft = PeftModel.from_pretrained(base, args.lora_dir, is_trainable=False)
    peft.eval()

    t0 = time.time()
    res = measure_behavior(
        peft, tokenizer, action_token_ids,
        goal_value=args.goal_value, n_keys=args.n_keys,
        n_episodes=args.n_episodes, max_steps=args.max_steps,
        seed_base=args.seed_base,
    )
    res["lora_dir"] = args.lora_dir
    res["elapsed_s"] = time.time() - t0
    console.log(json.dumps(res, indent=2))

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        console.log(f"saved {out}")

    del peft, base
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
