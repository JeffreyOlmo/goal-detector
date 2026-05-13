"""SFT trainer for the gridworld navigator.

Trains Qwen3-4B-Instruct on (prompt → optimal action) pairs produced by the
BFS oracle. Single-process, single-GPU; LoRA on attention projections so the
trainable footprint is tiny. Custom loop in the style of
curve_fit_extremization/training/train_curve.py — no Trainer, no TRL.

Loss: standard causal-LM cross-entropy with labels masked to -100 except on
the single response (action) token. The chat template's
``add_generation_prompt=True`` puts the model exactly in the position the
inference policy queries, so what we train on matches what we test.

Usage:
    CUDA_VISIBLE_DEVICES=N python -m training.train_sft
"""
from __future__ import annotations

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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import bitsandbytes as bnb

from goal_detector.gridworld import ACTIONS
from goal_detector.policies.qwen import build_prompt_messages
from training.config_sft import (
    batch_size,
    data_path,
    grad_accum_steps,
    learning_rate,
    logging_steps,
    lora_alpha,
    lora_dropout,
    lora_r,
    lora_target_modules,
    max_seq_len,
    model_id,
    num_epochs,
    output_dir,
    save_steps,
    seed,
    use_lora,
    warmup_ratio,
    weight_decay,
)

console = Console()


# ── Dataset ────────────────────────────────────────────────────────────────

@dataclass
class Example:
    input_ids: list[int]
    label: int  # the single action token id (last position)


class OracleSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, *, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.action_token_ids = {
            a: tokenizer(a, add_special_tokens=False).input_ids[0]
            for a in ACTIONS
        }
        self.records: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Example:
        rec = self.records[idx]
        messages = build_prompt_messages(rec["goal_description"], rec["state"])
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        action_id = self.action_token_ids[rec["action"]]
        # Truncate the prompt from the LEFT if needed so the action token
        # always appears at the final position. Prompts in this task are
        # ~500 tokens so this should never trigger at max_seq_len=768.
        if len(prompt_ids) >= self.max_seq_len:
            prompt_ids = prompt_ids[-(self.max_seq_len - 1) :]
        input_ids = prompt_ids + [action_id]
        return Example(input_ids=input_ids, label=action_id)


def collate(examples: list[Example], pad_token_id: int) -> dict:
    """Right-pad with attention mask. Label position is len(input_ids)-1."""
    max_len = max(len(e.input_ids) for e in examples)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for i, e in enumerate(examples):
        L = len(e.input_ids)
        input_ids[i, :L] = torch.tensor(e.input_ids, dtype=torch.long)
        attention_mask[i, :L] = 1
        # The label corresponds to predicting the LAST token from the
        # second-to-last position. In HF causal LM convention, labels are
        # shifted internally — we set labels[i, L-1] = action_id, the model
        # computes CE on logits at position L-2 against labels at L-1.
        labels[i, L - 1] = e.label
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ── Train loop ─────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("SFT trainer requires CUDA — model training must run on GPU.")
    device = torch.device("cuda")

    console.rule(f"loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device)
    model.gradient_checkpointing_enable()
    # Required when combining gradient checkpointing with frozen base + LoRA:
    # the input embeddings sit upstream of the trainable adapters and must
    # propagate grads, otherwise checkpointing has nothing to recompute and
    # activation memory balloons.
    model.enable_input_require_grads()

    if use_lora:
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora_target_modules),
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    console.rule(f"loading data from {data_path}")
    dataset = OracleSFTDataset(data_path, tokenizer, max_seq_len=max_seq_len)
    console.log(f"{len(dataset)} examples")

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    steps_per_epoch = len(dl) // grad_accum_steps
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.rule("training")
    console.log(
        f"steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
        f"warmup={warmup_steps}  bs={batch_size}  ga={grad_accum_steps}  "
        f"lr={learning_rate}"
    )

    log_path = out / "train_log.jsonl"
    log_f = log_path.open("w")

    model.train()
    global_step = 0
    optimizer.zero_grad()
    pbar = tqdm(total=total_steps, desc="train")
    t0 = time.time()
    rolling_loss: list[float] = []
    rolling_acc: list[float] = []

    for epoch in range(num_epochs):
        for ga_idx, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out_ = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out_.logits  # (B, T, V)

            # CE on the single labelled position; HF "shift" convention means
            # label at position t is predicted from logits at t-1.
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            B, T, V = shift_logits.shape
            loss = F.cross_entropy(
                shift_logits.reshape(B * T, V),
                shift_labels.reshape(B * T),
                ignore_index=-100,
            )
            (loss / grad_accum_steps).backward()

            # Token accuracy on the labelled positions.
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

            if (ga_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.update(1)

                if global_step % logging_steps == 0:
                    avg_loss = sum(rolling_loss) / len(rolling_loss)
                    avg_acc = sum(rolling_acc) / len(rolling_acc)
                    rec = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": avg_loss,
                        "acc": avg_acc,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed": time.time() - t0,
                    }
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    pbar.set_description(
                        f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                        f"lr={rec['lr']:.2e}"
                    )

                if global_step % save_steps == 0 or global_step == total_steps:
                    ckpt = out / f"step_{global_step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    console.log(f"saved {ckpt}")

    pbar.close()
    log_f.close()

    final = out / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    console.log(f"final checkpoint → {final}")
    console.log(f"total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
