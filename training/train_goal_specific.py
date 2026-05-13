"""Step 3 — per-goal SFT (state-only prompt, action-only output).

For each (goal, variant) the worker:
  1. Loads base Qwen3-4B-Instruct fresh.
  2. Wraps in a fresh all-linear LoRA (q/k/v/o + gate/up/down).
  3. Trains on (state, action) pairs from
     ``data/prompted_rollouts_v0_filtered/<attr>_<val>_v<variant>.jsonl``,
     using a STATE-ONLY prompt (no goal description) so the goal must end
     up in weights to drive correct behavior.
  4. Saves adapter to
     ``checkpoints/goal_specific/<attr>_<val>/v<variant>/``.

The launcher (``training.launch_goal_specific``) shards (goal, variant)
pairs across GPUs round-robin; each worker handles its slice sequentially,
reloading base between jobs (cheap relative to per-job training time).

Usage (typically called by the launcher):
    CUDA_VISIBLE_DEVICES=N python -m training.train_goal_specific \\
        --pairs color:red:0,color:red:1
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

import bitsandbytes as bnb

from goal_detector.gridworld import ACTIONS
from goal_detector.policies.qwen import build_state_only_prompt_messages
from training.config_sft import model_id

console = Console()

# ── Config (per-goal SFT specific) ─────────────────────────────────────────
DEFAULT_DATA_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/prompted_rollouts_v0_filtered"
)
DEFAULT_OUT_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/goal_specific_v1"
)

# All-linear LoRA. Larger capacity than v0's attention-only — needed because
# the goal must live entirely in weights with no prompt help.
LORA_R = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

LEARNING_RATE = 1e-4
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 4                # v1: 2 -> 4 (v0 underconverged on long tail)
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_SEQ_LEN = 768
LOGGING_STEPS = 25


# ── Dataset ────────────────────────────────────────────────────────────────

@dataclass
class Example:
    input_ids: list[int]
    label: int


class StateOnlySFTDataset(Dataset):
    """One (state, action) pair per training example. Prompt is
    state-only (no goal description) — that's what makes this step the
    goal-distillation step. Each rollout in the JSONL contributes
    ``len(actions)`` examples."""

    def __init__(self, jsonl_path: str, tokenizer, *, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.action_token_ids = {
            a: tokenizer(a, add_special_tokens=False).input_ids[0]
            for a in ACTIONS
        }
        self.pairs: list[tuple[dict, str]] = []
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
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


# ── Train one (goal, variant) ──────────────────────────────────────────────

def parse_pairs(s: str) -> list[tuple[str, str, int]]:
    out = []
    for tok in s.split(","):
        attr, val, variant = tok.split(":")
        out.append((attr, val, int(variant)))
    return out


def variant_seed(goal_attr: str, goal_val: str, variant: int) -> int:
    """Deterministic per-(attr, val, variant) seed — used to randomize the
    LoRA init AND the data shuffle so the 8 variants of a goal are
    *genuinely* independent (v0 shared seed=0 across all 56 trainings,
    making LoRA inits identical and only data subsampling distinguishing
    variants — too weak)."""
    h = (
        hash(goal_attr) * 1_000_003
        + hash(goal_val) * 1_009
        + variant * 17
    ) & 0xFFFFFFFF
    return int(h)


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
    data_path = data_dir / f"{goal_attr}_{goal_val}_v{variant}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"no data at {data_path}")

    out_dir = out_root / f"{goal_attr}_{goal_val}" / f"v{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "adapter_config.json").exists():
        console.log(f"  [skip] adapter already at {out_dir}")
        return {"skipped": True}

    seed = variant_seed(goal_attr, goal_val, variant)
    console.log(f"  variant_seed={seed}")
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
        task_type="CAUSAL_LM", target_modules=list(LORA_TARGET_MODULES),
    )
    model = get_peft_model(model, peft_cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.log(f"  trainable params: {n_train:,}")

    dataset = StateOnlySFTDataset(str(data_path), tokenizer, max_seq_len=MAX_SEQ_LEN)
    console.log(f"  {len(dataset)} (state, action) pairs from {data_path.name}")

    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
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
                    console.log(
                        f"    step {global_step}/{total_steps} loss={avg_loss:.4f} "
                        f"acc={avg_acc:.3f}"
                    )

    log_f.close()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    elapsed = time.time() - t0
    console.log(f"  saved adapter -> {out_dir} ({elapsed:.0f}s)")

    # Free GPU memory before next job.
    del model
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return {"steps": total_steps, "elapsed": elapsed}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True,
                   help="comma-sep attr:val:variant (e.g. color:red:0,color:red:1)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    # Per-(attr, val, variant) seeding happens inside train_one() — each
    # training run gets its own independent seed.

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    pairs = parse_pairs(args.pairs)
    console.rule(f"loading tokenizer")
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
