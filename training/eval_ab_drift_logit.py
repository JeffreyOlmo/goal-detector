"""Continuous logit-level measurement for the AB-drift validation.

The original eval_ab_drift.py measured IA verbalization with greedy
generation + token-extraction. That's discrete: it counts how often the
greedy output spells one color vs another, hiding any logit-level
preference shift that doesn't cross the argmax boundary.

Here we measure a continuous signal. For each (state, prompt) we condition
the IA on the canonical-label prefix "Collect " — exactly what the IA was
SFT'd to emit — and read the logits at the next position. We report

    p_b_given_ab = softmax(logits)[B] / (softmax(logits)[A] + softmax(logits)[B])

across the whole IA prompt set, averaged over (state, prompt). This gives
a continuous "tip toward A vs B" signal that should track the underlying
color-preference of the goal-feature even when greedy hasn't crossed.

Same multi-adapter PEFT setup as eval_ab_drift.py, only the IA pass
changes. Behavioral eval is unchanged.
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
import torch.nn.functional as F
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.config_sft import model_id
from training.eval_ab_drift import (
    ABEnv, EnvConfig, act_argmax, make_action_token_ids, measure_behavior,
)
from training.ia_data_gen import INTROSPECTION_PROMPTS
from training.ia_train import (
    add_ia_adapter, build_prompt, build_state_pool, set_active_adapters,
)

console = Console()


def find_token_id(tokenizer, text: str) -> int:
    """First token of `text` (incl. any leading space). Errors if multi-token."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise RuntimeError(f"empty tokenization for {text!r}")
    return ids[0]


@torch.no_grad()
def logit_pair_at_prefix(
    peft, tokenizer, *, messages, prefix: str,
    a_token_id: int, b_token_id: int,
) -> tuple[float, float, float, float]:
    """Build prompt, append `prefix` after the assistant turn marker, return:
        (logit_a, logit_b, log_softmax_full[a], log_softmax_full[b])
    """
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    prompt = prompt + prefix
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(peft.device)
    out = peft(**inputs)
    logits = out.logits[0, -1].float()
    lp = F.log_softmax(logits, dim=-1)
    return (
        float(logits[a_token_id].item()),
        float(logits[b_token_id].item()),
        float(lp[a_token_id].item()),
        float(lp[b_token_id].item()),
    )


def measure_ia_logit(
    peft, tokenizer, state_pool, *, n_states: int, prompts: list[str],
    prefix: str, val_a: str, val_b: str,
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
            # Restricted softmax over just the two values: P(B | A or B).
            denom = torch.logsumexp(torch.tensor([la, lb]), dim=0).item()
            p_b_given_ab = float(torch.exp(torch.tensor(lb - denom)).item())
            rows.append({
                "logit_a": la, "logit_b": lb,
                "logprob_a": lpa, "logprob_b": lpb,
                "p_b_given_ab": p_b_given_ab,
            })
    n = len(rows)
    mean_p_b = sum(r["p_b_given_ab"] for r in rows) / max(1, n)
    mean_logit_b_minus_a = sum(r["logit_b"] - r["logit_a"] for r in rows) / max(1, n)
    mean_lp_a = sum(r["logprob_a"] for r in rows) / max(1, n)
    mean_lp_b = sum(r["logprob_b"] for r in rows) / max(1, n)
    return {
        "n": n, "prefix": prefix,
        "mean_p_b_given_ab": mean_p_b,
        "mean_logit_b_minus_a": mean_logit_b_minus_a,
        "mean_logprob_a": mean_lp_a,
        "mean_logprob_b": mean_lp_b,
        "samples": rows[:6],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--ia-adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--attribute", required=True)
    p.add_argument("--val-a", required=True)
    p.add_argument("--val-b", required=True)
    p.add_argument("--prefix", default="Collect ",
                   help="answer-prefix to condition on (must match IA train label format)")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--seed-base", type=int, default=70_000_000)
    p.add_argument("--n-state-pool", type=int, default=16)
    p.add_argument("--state-seed", type=int, default=99)
    p.add_argument("--n-ia-states", type=int, default=20)
    p.add_argument("--ia-rank", type=int, default=32)
    p.add_argument("--ia-alpha", type=int, default=64)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    with open(args.manifest) as f:
        manifest = json.load(f)
    ckpts = manifest["checkpoints"]
    console.log(f"loaded manifest with {len(ckpts)} checkpoints")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    action_token_ids = make_action_token_ids(tokenizer)

    state_pool = build_state_pool(args.n_state_pool, args.state_seed)
    ia_prompts = INTROSPECTION_PROMPTS[:3]

    console.rule(f"loading base + {len(ckpts)} drift adapters + IA")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device)
    first = ckpts[0]
    first_name = f"step_{first['step']:03d}"
    peft = PeftModel.from_pretrained(base, first["path"], adapter_name=first_name,
                                     is_trainable=False)
    names = [first_name]
    for c in ckpts[1:]:
        nm = f"step_{c['step']:03d}"
        peft.load_adapter(c["path"], adapter_name=nm, is_trainable=False)
        names.append(nm)

    ia_targets = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
    add_ia_adapter(peft, rank=args.ia_rank, alpha=args.ia_alpha,
                   target_modules=ia_targets)
    peft.load_adapter(args.ia_adapter, adapter_name="ia", is_trainable=False)
    peft.eval()
    console.log("ready")

    # Sanity: print the token ids we'll be reading.
    a_id = find_token_id(tokenizer, " " + args.val_a)
    b_id = find_token_id(tokenizer, " " + args.val_b)
    console.log(f"prefix={args.prefix!r}  ' {args.val_a}'={a_id}  ' {args.val_b}'={b_id}")

    results = {
        "manifest": manifest,
        "attribute": args.attribute,
        "val_a": args.val_a, "val_b": args.val_b,
        "prefix": args.prefix,
        "ia_adapter": args.ia_adapter,
        "ia_prompts": ia_prompts,
        "n_episodes": args.n_episodes,
        "n_ia_states": args.n_ia_states,
        "per_checkpoint": [],
    }

    t0 = time.time()
    for c, name in zip(ckpts, names):
        console.rule(f"checkpoint: {name}")
        # Behavior: only the goal-LoRA active.
        set_active_adapters(peft, [name])
        beh = measure_behavior(
            peft, tokenizer, action_token_ids,
            attribute=args.attribute, val_a=args.val_a, val_b=args.val_b,
            n_episodes=args.n_episodes, max_steps=args.max_steps,
            seed_base=args.seed_base,
        )
        console.log(f"  behavior: A={beh['a_rate']:.0%}  "
                    f"B={beh['b_rate']:.0%}  other={beh['other_rate']:.0%}")
        # IA logit measurement.
        set_active_adapters(peft, [name, "ia"])
        ia = measure_ia_logit(
            peft, tokenizer, state_pool,
            n_states=args.n_ia_states, prompts=ia_prompts,
            prefix=args.prefix, val_a=args.val_a, val_b=args.val_b,
        )
        console.log(
            f"  IA logit: P({args.val_b}|{args.val_a},{args.val_b})={ia['mean_p_b_given_ab']:.3f}  "
            f"  logit({args.val_b})-logit({args.val_a})={ia['mean_logit_b_minus_a']:+.2f}"
        )
        results["per_checkpoint"].append({
            "step": c["step"], "name": name, "path": c["path"],
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
