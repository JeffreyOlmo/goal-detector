"""Compound-goal validation: behavior vs IA logit preference across LoRAs.

For each compound (green-circle) LoRA:
  - Behavioral: run ForcedChoiceEnv N episodes (green-not-circle tile vs
    circle-not-green tile, neutral distractors). Record which the policy
    collects.
  - IA logit: condition the IA on "Collect ", read logit(" green") vs
    logit(" circle"); compute P(green | green, circle) per (state, prompt)
    and average.

If the IA tracks each LoRA's internal latched preference, we expect
behavioral_color_pref ≈ ia_p_green_given_green_circle across LoRAs.
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
from training.compound_validation import ForcedChoiceEnv
from training.config_sft import model_id
from training.eval_ab_drift import act_argmax, make_action_token_ids
from training.eval_ab_drift_logit import (
    find_token_id, logit_pair_at_prefix,
)
from training.ia_data_gen import INTROSPECTION_PROMPTS
from training.ia_train import (
    add_ia_adapter, build_prompt, build_state_pool, set_active_adapters,
)

console = Console()


def measure_choice_behavior(
    peft, tokenizer, action_token_ids, *,
    n_episodes: int, max_steps: int, seed_base: int,
) -> dict:
    """Run forced-choice episodes; classify each by which tile collected."""
    cfg = EnvConfig(max_steps=max_steps)
    n_green = n_circle = n_other = 0
    for ep in range(n_episodes):
        env = ForcedChoiceEnv(cfg, seed=seed_base + ep)
        try:
            state = env.reset()
        except RuntimeError:
            n_other += 1
            continue
        picked_green = picked_circle = False
        while not env.is_done():
            a = act_argmax(peft, tokenizer, state, action_token_ids)
            res = env.step(a)
            state = res.state
            if res.collected is not None:
                t = res.collected
                if t.color == "green" and t.shape != "circle":
                    picked_green = True
                elif t.shape == "circle" and t.color != "green":
                    picked_circle = True
        if picked_green and not picked_circle:
            n_green += 1
        elif picked_circle and not picked_green:
            n_circle += 1
        else:
            n_other += 1
    n = n_episodes
    n_decisive = n_green + n_circle
    return {
        "n_episodes": n, "n_green_pick": n_green, "n_circle_pick": n_circle,
        "n_other": n_other,
        "green_rate": n_green / n,
        "circle_rate": n_circle / n,
        "other_rate": n_other / n,
        "green_among_decisive": (n_green / n_decisive) if n_decisive > 0 else None,
    }


@torch.no_grad()
def measure_ia_logit_pair(
    peft, tokenizer, state_pool, *,
    n_states: int, prompts, prefix: str,
    val_a: str, val_b: str,
) -> dict:
    a_id = find_token_id(tokenizer, " " + val_a)
    b_id = find_token_id(tokenizer, " " + val_b)
    states = [state_pool[i % len(state_pool)] for i in range(n_states)]
    rows = []
    for state in states:
        for q in prompts:
            msgs = build_prompt(state, q)
            la, lb, lpa, lpb = logit_pair_at_prefix(
                peft, tokenizer, messages=msgs, prefix=prefix,
                a_token_id=a_id, b_token_id=b_id,
            )
            denom = torch.logsumexp(torch.tensor([la, lb]), dim=0).item()
            p_a = float(torch.exp(torch.tensor(la - denom)).item())
            rows.append({
                "logit_a": la, "logit_b": lb,
                "logprob_a": lpa, "logprob_b": lpb,
                "p_a_given_ab": p_a,
            })
    n = len(rows)
    return {
        "n": n, "prefix": prefix, "val_a": val_a, "val_b": val_b,
        "mean_p_a_given_ab": sum(r["p_a_given_ab"] for r in rows) / max(1, n),
        "mean_logit_a_minus_b": sum(r["logit_a"] - r["logit_b"] for r in rows) / max(1, n),
        "mean_logprob_a": sum(r["logprob_a"] for r in rows) / max(1, n),
        "mean_logprob_b": sum(r["logprob_b"] for r in rows) / max(1, n),
        "samples": rows[:6],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lora-dirs", required=True,
                   help="comma-sep list of compound-LoRA dirs to evaluate")
    p.add_argument("--ia-adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="Collect ")
    p.add_argument("--n-episodes", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--seed-base", type=int, default=80_000_000)
    p.add_argument("--n-state-pool", type=int, default=16)
    p.add_argument("--state-seed", type=int, default=99)
    p.add_argument("--n-ia-states", type=int, default=20)
    p.add_argument("--ia-rank", type=int, default=32)
    p.add_argument("--ia-alpha", type=int, default=64)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    lora_dirs = [d.strip() for d in args.lora_dirs.split(",") if d.strip()]
    console.log(f"loaded {len(lora_dirs)} LoRA dirs")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    action_token_ids = make_action_token_ids(tokenizer)
    state_pool = build_state_pool(args.n_state_pool, args.state_seed)
    ia_prompts = INTROSPECTION_PROMPTS[:3]

    console.rule(f"loading base + {len(lora_dirs)} compound adapters + IA")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device)
    names: list[str] = []
    peft = None
    for i, ld in enumerate(lora_dirs):
        nm = f"compound_v{i}"
        if peft is None:
            peft = PeftModel.from_pretrained(base, ld, adapter_name=nm,
                                             is_trainable=False)
        else:
            peft.load_adapter(ld, adapter_name=nm, is_trainable=False)
        names.append(nm)

    ia_targets = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
    add_ia_adapter(peft, rank=args.ia_rank, alpha=args.ia_alpha,
                   target_modules=ia_targets)
    peft.load_adapter(args.ia_adapter, adapter_name="ia", is_trainable=False)
    peft.eval()
    console.log("ready")

    results: dict = {
        "lora_dirs": lora_dirs,
        "ia_adapter": args.ia_adapter,
        "prefix": args.prefix,
        "n_episodes": args.n_episodes,
        "n_ia_states": args.n_ia_states,
        "ia_prompts": ia_prompts,
        "per_lora": [],
    }
    t0 = time.time()
    for ld, name in zip(lora_dirs, names):
        console.rule(f"LoRA: {name}  {ld}")
        # Behavior
        set_active_adapters(peft, [name])
        beh = measure_choice_behavior(
            peft, tokenizer, action_token_ids,
            n_episodes=args.n_episodes, max_steps=args.max_steps,
            seed_base=args.seed_base,
        )
        console.log(
            f"  behavior: green={beh['green_rate']:.0%}  "
            f"circle={beh['circle_rate']:.0%}  other={beh['other_rate']:.0%}  "
            f"green-among-decisive={beh['green_among_decisive']}"
        )
        # IA logit (val_a=green, val_b=circle).
        set_active_adapters(peft, [name, "ia"])
        ia = measure_ia_logit_pair(
            peft, tokenizer, state_pool,
            n_states=args.n_ia_states, prompts=ia_prompts,
            prefix=args.prefix, val_a="green", val_b="circle",
        )
        console.log(
            f"  IA logit: P(green|green,circle)={ia['mean_p_a_given_ab']:.3f}  "
            f"logit(green)-logit(circle)={ia['mean_logit_a_minus_b']:+.2f}"
        )
        results["per_lora"].append({
            "name": name, "lora_dir": ld,
            "behavior": beh, "ia_logit": ia,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    console.rule(f"done in {time.time()-t0:.0f}s — saved {out_path}")
    del peft, base
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
