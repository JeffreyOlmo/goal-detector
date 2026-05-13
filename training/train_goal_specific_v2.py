"""Step 3 v2 — short SFT for many diverse variants per goal.

Differs from v1 in three ways:
  1. NUM_EPOCHS = 1 (down from 4) — v1 logs showed convergence well before
     epoch 4 anyway.
  2. Subsample 200 rollouts from each goal's filtered pool (down from full
     689) — gives every variant a different data slice for free.
  3. Per-variant target-module randomization. Each variant gets one of
     {attention, all_linear, mlp, gate_up} as its LoRA target set, sampled
     from the variant seed. Forces goal-pursuit to crystallize in different
     parameter subspaces across the cohort, eliminating per-variant
     fingerprints that aren't related to the goal.

Rank stays at 32, lr at 1e-4, warmup at 3% — those proved to work in v1
and add training-quality variance without adding probe-relevant diversity.

Usage (typically called by the launcher):
    CUDA_VISIBLE_DEVICES=N python -m training.train_goal_specific_v2 \\
        --pairs color:red:8,color:red:9,color:red:10,...
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
from goal_detector.policies.qwen import build_state_only_prompt_messages
from training.config_sft import model_id

console = Console()

DEFAULT_DATA_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/prompted_rollouts_v0_filtered"
)
DEFAULT_OUT_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/goal_specific_v2"
)

# Held constant across variants (proven in v1).
LORA_R = 32
LORA_ALPHA = 64
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 3                    # was 4 in v1; 1 in first v2 attempt was undertrained
N_SUBSAMPLE_ROLLOUTS = 400        # was 689 (full); 200 was too few — train_acc plateaued ~0.79
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_SEQ_LEN = 768
LOGGING_STEPS = 25

# Diversity axis: target modules. Each variant samples one of these.
TARGET_MODULE_SETS: dict[str, tuple[str, ...]] = {
    "attention":  ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all_linear": ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"),
    "mlp":        ("gate_proj", "up_proj", "down_proj"),
    "gate_up":    ("gate_proj", "up_proj"),
}
TARGET_MODULE_KEYS = tuple(TARGET_MODULE_SETS.keys())


# ── Dataset ────────────────────────────────────────────────────────────────

@dataclass
class Example:
    input_ids: list[int]
    label: int


class StateOnlySFTDataset(Dataset):
    """One (state, action) pair per training example.

    Subsamples ``n_subsample`` rollouts from the JSONL using ``subsample_seed``
    so each variant trains on a different slice of the goal's rollout pool."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        *,
        max_seq_len: int,
        n_subsample: int,
        subsample_seed: int,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.action_token_ids = {
            a: tokenizer(a, add_special_tokens=False).input_ids[0]
            for a in ACTIONS
        }
        with open(jsonl_path) as f:
            all_rollouts = [json.loads(line) for line in f]
        rng = random.Random(subsample_seed)
        rng.shuffle(all_rollouts)
        kept = all_rollouts[:n_subsample]
        self.pairs: list[tuple[dict, str]] = []
        for rec in kept:
            states = rec["states"]
            actions = rec["actions"]
            if len(states) != len(actions):
                continue
            for s, a in zip(states, actions):
                self.pairs.append((s, a))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Example:
        state, action = self.pairs[idx]
        messages = build_state_only_prompt_messages(state)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        action_id = self.action_token_ids[action]
        if len(prompt_ids) >= self.max_seq_len:
            prompt_ids = prompt_ids[-(self.max_seq_len - 1):]
        return Example(input_ids=prompt_ids + [action_id], label=action_id)


def collate(examples: list[Example], pad_token_id: int) -> dict:
    max_len = max(len(e.input_ids) for e in examples)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for i, e in enumerate(examples):
        L = len(e.input_ids)
        input_ids[i, :L] = torch.tensor(e.input_ids, dtype=torch.long)
        attention_mask[i, :L] = 1
        labels[i, L - 1] = e.label
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ── Per-variant config from seed ───────────────────────────────────────────

def variant_seed(goal_attr: str, goal_val: str, variant: int) -> int:
    """Deterministic seed per (goal, variant). Different namespace from v1
    by adding a fixed offset, so v1 variants 0..7 don't collide with v2
    variants 0..63."""
    h = (
        hash(goal_attr) * 1_000_003
        + hash(goal_val) * 1_009
        + variant * 17
        + 0xDEADBEEF  # v2 namespace marker
    ) & 0xFFFFFFFF
    return int(h)


def variant_config(seed: int) -> dict:
    """Pick target_module set + data-subsample seed deterministically from
    the variant seed. Uses a sub-rng so changes here don't shift earlier
    decisions."""
    rng = random.Random(seed)
    target_key = rng.choice(TARGET_MODULE_KEYS)
    subsample_seed = rng.randrange(2 ** 31)
    return {
        "target_modules_key": target_key,
        "target_modules": list(TARGET_MODULE_SETS[target_key]),
        "subsample_seed": subsample_seed,
    }


# ── Train one (goal, variant) ──────────────────────────────────────────────

def parse_pairs(s: str) -> list[tuple[str, str, int]]:
    out = []
    for tok in s.split(","):
        attr, val, variant = tok.split(":")
        out.append((attr, val, int(variant)))
    return out


def train_one(
    base_model_id: str,
    tokenizer,
    goal_attr: str,
    goal_val: str,
    variant: int,
    data_dir: Path,
    out_root: Path,
    device: torch.device,
) -> dict:
    data_path = data_dir / f"{goal_attr}_{goal_val}_v{variant % 8}.jsonl"
    if not data_path.exists():
        # Fall back to v0 of that goal; rollout pool is per-goal, variant
        # only affects which slice the SFT step subsamples.
        data_path = data_dir / f"{goal_attr}_{goal_val}_v0.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"no rollout pool for {goal_attr}={goal_val}")

    out_dir = out_root / f"{goal_attr}_{goal_val}" / f"v{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "adapter_config.json").exists():
        console.log(f"  [skip] adapter already at {out_dir}")
        return {"skipped": True}

    seed = variant_seed(goal_attr, goal_val, variant)
    cfg = variant_config(seed)
    console.log(
        f"  variant_seed={seed}  target={cfg['target_modules_key']}  "
        f"subsample_seed={cfg['subsample_seed']}"
    )
    torch.manual_seed(seed)
    random.seed(seed)

    console.log(f"  loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=cfg["target_modules"],
    )
    model = get_peft_model(model, peft_cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.log(f"  trainable params: {n_train:,}")

    dataset = StateOnlySFTDataset(
        str(data_path), tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        n_subsample=N_SUBSAMPLE_ROLLOUTS,
        subsample_seed=cfg["subsample_seed"],
    )
    console.log(
        f"  {len(dataset)} (state, action) pairs from "
        f"{N_SUBSAMPLE_ROLLOUTS}-rollout subsample of {data_path.name}"
    )

    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    steps_per_epoch = len(dl) // GRAD_ACCUM_STEPS
    total_steps = max(1, steps_per_epoch * NUM_EPOCHS)
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    console.log(
        f"  steps/epoch={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}"
    )

    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w")
    # Drop variant config alongside the log for traceability.
    with (out_dir / "variant_config.json").open("w") as f:
        json.dump({"seed": seed, **cfg}, f, indent=2)

    model.train()
    global_step = 0
    optimizer.zero_grad()
    rolling_loss: list[float] = []
    rolling_acc: list[float] = []
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        for ga_idx, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out_ = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = out_.logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            B, T, V = shift_logits.shape
            loss = F.cross_entropy(
                shift_logits.reshape(B * T, V),
                shift_labels.reshape(B * T),
                ignore_index=-100,
            )
            (loss / GRAD_ACCUM_STEPS).backward()

            with torch.no_grad():
                mask = shift_labels != -100
                preds = shift_logits.argmax(dim=-1)
                correct = ((preds == shift_labels) & mask).sum().item()
                total = mask.sum().item()
                acc = correct / max(1, total)

            rolling_loss.append(loss.item())
            rolling_acc.append(acc)
            if len(rolling_loss) > 50:
                rolling_loss.pop(0)
                rolling_acc.pop(0)

            if (ga_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOGGING_STEPS == 0:
                    avg_loss = sum(rolling_loss) / len(rolling_loss)
                    avg_acc = sum(rolling_acc) / len(rolling_acc)
                    rec = {
                        "step": global_step, "loss": avg_loss, "acc": avg_acc,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed": time.time() - t0,
                    }
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()

    log_f.close()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    elapsed = time.time() - t0
    final_acc = sum(rolling_acc) / max(1, len(rolling_acc))
    console.log(
        f"  saved -> {out_dir}  ({elapsed:.0f}s)  "
        f"final_train_acc={final_acc:.3f}  target={cfg['target_modules_key']}"
    )

    del model
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "steps": total_steps,
        "elapsed": elapsed,
        "final_train_acc": final_acc,
        **cfg,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    pairs = parse_pairs(args.pairs)
    console.rule("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    console.log(f"{len(pairs)} (goal, variant) jobs assigned")
    for attr, val, variant in pairs:
        console.rule(f"{attr}={val}  variant={variant}")
        train_one(
            base_model_id=model_id, tokenizer=tokenizer,
            goal_attr=attr, goal_val=val, variant=variant,
            data_dir=data_dir, out_root=out_root, device=device,
        )

    console.rule("worker done")


if __name__ == "__main__":
    main()
