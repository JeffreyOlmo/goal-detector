"""Extract base-model first-state residual activations on the same envs
used by paired_activations_v2.

For each (compound, env_idx) ∈ 12 × 30 = 360, build the initial state via
the same `FixedCompoundEnv(seed=env_seed_for(compound, env_idx))` used by
`extract_paired_activations.py`, run base Qwen3-4B-Instruct (no LoRA) on
the resulting prompt, and capture residual-stream activations at every
layer in LAYER_IDXS at the final prompt-token position.

Saved as `data/paired_base_first_v2/{compound}.pt` (e.g. red_circle_solid.pt)
with structure:
    {
        "compound": (c, s, p),
        "layer_idxs": [...],
        "envs": list[ {env_idx, env_seed, base_activation: (n_layers, d_model)} ],
    }
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from goal_detector.gridworld.ambiguous_env import (
    ALL_COMPOUNDS, FixedCompoundEnv, ambiguity_mates,
)
from goal_detector.gridworld.env import EnvConfig
from goal_detector.goals import SimpleFeatureGoal
from goal_detector.policies.qwen import build_state_only_prompt_messages
from training.config_sft import model_id
from training.extract_activations import LAYER_IDXS
from training.extract_paired_activations import (
    N_ENVS_PER_COMPOUND, env_seed_for, MAX_STEPS,
)

console = Console()


@torch.no_grad()
def first_state_activations(model, tokenizer, state: dict, layer_idxs):
    msgs = build_state_only_prompt_messages(state)
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    acts = torch.stack([
        out.hidden_states[i][0, -1].detach().to(torch.float16).cpu()
        for i in layer_idxs
    ])  # (n_layers, d_model)
    return acts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=(
        "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/"
        "paired_base_first_v2"))
    p.add_argument("--n-envs", type=int, default=N_ENVS_PER_COMPOUND)
    p.add_argument("--compounds", default="all",
                   help="comma-sep compounds 'red_circle_solid,...' or 'all'")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.compounds == "all":
        compounds = list(ALL_COMPOUNDS)
    else:
        keep = set(args.compounds.split(","))
        compounds = [C for C in ALL_COMPOUNDS if "_".join(C) in keep]

    console.rule("loading base Qwen3-4B-Instruct (no LoRA)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    cfg = EnvConfig(max_steps=MAX_STEPS)
    t0 = time.time()
    for compound in compounds:
        out_path = out_dir / f"{'_'.join(compound)}.pt"
        if out_path.exists():
            console.log(f"[skip] {out_path.name}")
            continue
        # Use any of the 3 attribute mates as the goal-builder; FixedCompoundEnv
        # places the compound tile regardless of which goal-attribute we pass.
        attr, val = ambiguity_mates(compound)[0]
        goal = SimpleFeatureGoal(attribute=attr, value=val)
        envs_out = []
        for env_idx in range(args.n_envs):
            seed = env_seed_for(compound, env_idx)
            env = FixedCompoundEnv(cfg, goal, seed=seed, compound=compound)
            state = env.reset()
            acts = first_state_activations(model, tokenizer, state, LAYER_IDXS)
            envs_out.append({
                "env_idx": env_idx, "env_seed": seed,
                "base_activation": acts,
            })
        torch.save({
            "compound": compound,
            "layer_idxs": LAYER_IDXS,
            "envs": envs_out,
        }, out_path)
        console.log(f"saved {out_path.name}  ({len(envs_out)} envs, "
                    f"{time.time()-t0:.0f}s elapsed)")

    del model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
