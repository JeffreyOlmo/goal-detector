"""Train a single compound (green-circle) goal-LoRA with state-only prompts.

Generates oracle BFS rollouts on GreenCircleTrainEnv (target = green circle,
distractors = neither-green-nor-circle), then SFTs a LoRA adapter on the
(state → action) pairs. Different `--seed` values yield different rollout
slices and different LoRA initializations, producing LoRAs that latch onto
slightly different color/shape preferences.

Usage:
    CUDA_VISIBLE_DEVICES=N python -m training.train_compound_lora \\
        --out .../checkpoints/compound/green_circle/v0 --seed 0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from goal_detector.gridworld import ACTIONS
from goal_detector.gridworld.env import EnvConfig
from goal_detector.policies.oracle import bfs_optimal_action
from goal_detector.policies.qwen import build_state_only_prompt_messages
from training.compound_validation import (
    GreenCircleGoal, GreenCircleTrainEnv,
    GreenOrCircleGoal, MixedConfoundEnv,
)
from training.config_sft import model_id

console = Console()

# Diversity of LoRA target modules across variants — same set as v2 cohort.
TARGET_MODULE_SETS: dict[str, tuple[str, ...]] = {
    "attention":  ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all_linear": ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"),
    "mlp":        ("gate_proj", "up_proj", "down_proj"),
    "gate_up":    ("gate_proj", "up_proj"),
}
TARGET_MODULE_KEYS = tuple(TARGET_MODULE_SETS.keys())


@dataclass
class Example:
    input_ids: list[int]
    label: int  # action token id; -100 elsewhere


def gen_oracle_pairs(*, n_episodes: int, seed: int, env_max_steps: int = 200,
                     p_confound: float = 1.0, p_green_only: float = 0.0,
                     p_circle_only: float = 0.0
                     ) -> tuple[list[tuple[dict, str]], int, dict]:
    """Run n_episodes with the BFS oracle. If p_confound==1.0 (default)
    use the original GreenCircleTrainEnv (perfectly confounded). Else
    use MixedConfoundEnv with the given mix and a disjunctive goal.

    Returns (pairs, skipped_layouts, stats) where stats = layout-type
    counts."""
    cfg = EnvConfig(max_steps=env_max_steps)
    use_mixed = (p_confound < 1.0)
    if use_mixed:
        goal = GreenOrCircleGoal()
    else:
        goal = GreenCircleGoal()
    pairs: list[tuple[dict, str]] = []
    skipped = 0
    layout_counts = {"confound": 0, "green_only": 0, "circle_only": 0}
    for ep in range(n_episodes):
        try:
            if use_mixed:
                env = MixedConfoundEnv(
                    cfg, goal, seed=seed * 100_003 + ep,
                    p_confound=p_confound, p_green_only=p_green_only,
                    p_circle_only=p_circle_only,
                )
            else:
                env = GreenCircleTrainEnv(cfg, goal, seed=seed * 100_003 + ep)
            state = env.reset()
        except RuntimeError:
            skipped += 1
            continue
        if use_mixed:
            layout_counts[env._layout_type] += 1
        else:
            layout_counts["confound"] += 1
        while not env.is_done():
            a = bfs_optimal_action(env)
            if a is None:
                break
            pairs.append((state, a))
            res = env.step(a)
            state = res.state
    return pairs, skipped, layout_counts


class StateActionDataset(Dataset):
    def __init__(self, pairs: list[tuple[dict, str]], tokenizer, max_seq_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.action_token_ids = {
            a: tokenizer(a, add_special_tokens=False).input_ids[0] for a in ACTIONS
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Example:
        state, action = self.pairs[idx]
        msgs = build_state_only_prompt_messages(state)
        try:
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        action_id = self.action_token_ids[action]
        if len(prompt_ids) >= self.max_seq_len:
            prompt_ids = prompt_ids[-(self.max_seq_len - 1):]
        full = prompt_ids + [action_id]
        return Example(input_ids=full, label=action_id)


def collate(batch: list[Example], pad_token_id: int):
    max_len = max(len(e.input_ids) for e in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, e in enumerate(batch):
        L = len(e.input_ids)
        input_ids[i, :L] = torch.tensor(e.input_ids, dtype=torch.long)
        attn[i, :L] = 1
        labels[i, L - 1] = e.label
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=300)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--target-set", default=None,
                   help="one of {attention, all_linear, mlp, gate_up}; if None, "
                        "sampled from seed for cohort diversity")
    p.add_argument("--max-seq-len", type=int, default=768)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--p-confound", type=float, default=1.0,
                   help="fraction of training layouts that are perfectly "
                        "confounded green-circle. <1.0 enables mixed-"
                        "disambiguation training (uses MixedConfoundEnv "
                        "+ GreenOrCircleGoal).")
    p.add_argument("--p-green-only", type=float, default=0.0,
                   help="fraction of layouts where target is green-not-"
                        "circle (color-disambiguating examples).")
    p.add_argument("--p-circle-only", type=float, default=0.0,
                   help="fraction of layouts where target is circle-not-"
                        "green (shape-disambiguating examples).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Pick target module set.
    if args.target_set is None:
        target_set_key = TARGET_MODULE_KEYS[args.seed % len(TARGET_MODULE_KEYS)]
    else:
        target_set_key = args.target_set
    target_modules = TARGET_MODULE_SETS[target_set_key]
    console.rule(f"compound LoRA seed={args.seed}  targets={target_set_key}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate oracle data.
    console.log(f"generating {args.n_episodes} oracle episodes  "
                f"(p_confound={args.p_confound}, p_green_only="
                f"{args.p_green_only}, p_circle_only={args.p_circle_only})...")
    t0 = time.time()
    pairs, skipped, layout_counts = gen_oracle_pairs(
        n_episodes=args.n_episodes, seed=args.seed,
        p_confound=args.p_confound, p_green_only=args.p_green_only,
        p_circle_only=args.p_circle_only,
    )
    console.log(f"  {len(pairs)} (s,a) pairs  skipped={skipped} layouts  "
                f"layouts={layout_counts}  "
                f"({time.time()-t0:.0f}s)")

    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to(device)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    peft = get_peft_model(base, lora_cfg)
    n_train = sum(p.numel() for p in peft.parameters() if p.requires_grad)
    console.log(f"trainable LoRA params: {n_train:,}")

    ds = StateActionDataset(pairs, tokenizer, max_seq_len=args.max_seq_len)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )

    n_micro = max(1, len(dl)) * args.n_epochs
    n_opt = n_micro // args.grad_accum
    warmup = max(1, int(n_opt * 0.03))
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in peft.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=n_opt,
    )

    log_path = out_root / "compound_train_log.jsonl"
    log_f = log_path.open("w")
    rolling: list[float] = []
    t0 = time.time()
    optimizer.zero_grad()
    peft.train()
    micro = 0; opt_step = 0
    for epoch in range(args.n_epochs):
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = peft(input_ids=input_ids, attention_mask=attn)
            shift_logits = out.logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1), ignore_index=-100,
            )
            (loss / args.grad_accum).backward()
            rolling.append(float(loss.item()))
            if len(rolling) > 50:
                rolling.pop(0)
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft.parameters() if p.requires_grad], 1.0
                )
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                opt_step += 1
                if opt_step % args.logging_steps == 0:
                    avg = sum(rolling) / max(1, len(rolling))
                    rec = {"epoch": epoch, "opt_step": opt_step,
                           "loss": avg, "lr": scheduler.get_last_lr()[0],
                           "elapsed": time.time() - t0}
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                    console.log(
                        f"  ep{epoch}  step {opt_step}/{n_opt}  "
                        f"loss={avg:.3f}  lr={rec['lr']:.2e}"
                    )
    log_f.close()

    peft.save_pretrained(str(out_root))
    tokenizer.save_pretrained(str(out_root))
    meta = {
        "seed": args.seed, "target_set": target_set_key,
        "target_modules": list(target_modules),
        "n_episodes": args.n_episodes, "n_pairs": len(pairs),
        "n_epochs": args.n_epochs, "lr": args.lr,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "goal": ("green_or_circle" if args.p_confound < 1.0
                 else "green_circle"),
        "p_confound": args.p_confound,
        "p_green_only": args.p_green_only,
        "p_circle_only": args.p_circle_only,
        "layout_counts": layout_counts,
    }
    (out_root / "compound_meta.json").write_text(json.dumps(meta, indent=2))
    console.rule(f"saved → {out_root}")
    del peft, base
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
