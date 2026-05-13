"""Phase 1 IA — joint LoRA training across many goal-pursuer adapters.

For each (variant, prompt) record from the IA corpus:
  1. Make the corresponding goal-LoRA active (frozen).
  2. Make the IA-LoRA active (trainable).
  3. Build a chat-format prompt = [system | state JSON | introspection question].
  4. CE-loss on the canonical-label tokens that follow.

Only the IA adapter accumulates gradients. Goal-LoRAs are loaded once and
left frozen for the entire run.

A small pool of random env states is pre-generated and one is mixed into
each prompt; this matches the distribution the goal-LoRAs were SFT'd on
(state JSON in the prompt) so their goal-relevant residual signature is
actually present at the introspection token positions.

Usage:
    CUDA_VISIBLE_DEVICES=N python -m training.ia_train \\
        --corpus data/ia_corpus.jsonl --out checkpoints/ia_v0
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
from peft import LoraConfig, PeftModel
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from goal_detector.goals import SimpleFeatureGoal
from goal_detector.gridworld.env import Env, EnvConfig
from training.config_sft import model_id

console = Console()

ROOT = Path("/mnt/pccfs2/backed_up/jeffolmo/goal-detector")


# ── prompt format ──────────────────────────────────────────────────────────

def language_system_prompt(width: int, height: int) -> str:
    return (
        f"You are an agent in an {width}x{height} gridworld. Coordinates use "
        "(x, y) with the top-left at (0, 0): x increases left to right, y "
        "increases top to bottom. The world contains tiles that you can "
        "collect by stepping on them; tiles have colors, shapes, and patterns. "
        "Answer the user's question briefly and directly."
    )


def build_prompt(state: dict, question: str) -> list[dict]:
    width, height = state["grid_size"]
    state_json = json.dumps(state, separators=(", ", ": "))
    ax, ay = state["agent"]
    user = (
        f"Your position: ({ax}, {ay}).\n\n"
        f"State (JSON):\n{state_json}\n\n"
        f"{question}"
    )
    return [
        {"role": "system", "content": language_system_prompt(width, height)},
        {"role": "user", "content": user},
    ]


# ── env-state pool ─────────────────────────────────────────────────────────

def build_state_pool(n_states: int, seed: int) -> list[dict]:
    """Sample env layouts from many goals so the IA's prompt distribution
    isn't biased toward any one goal's state distribution."""
    pool: list[dict] = []
    cfg = EnvConfig(max_steps=30)
    rng_seed = seed
    # Cycle through all 7 trained goals so distractor distributions are mixed.
    goals = [
        ("color", "red"), ("color", "blue"), ("color", "green"),
        ("shape", "circle"), ("shape", "square"),
        ("pattern", "solid"), ("pattern", "striped"),
    ]
    while len(pool) < n_states:
        attr, val = goals[len(pool) % len(goals)]
        env = Env(cfg, SimpleFeatureGoal(attr, val), seed=rng_seed)
        try:
            state = env.reset()
            pool.append(state)
        except RuntimeError:
            pass
        rng_seed += 1
    return pool


# ── dataset ────────────────────────────────────────────────────────────────

@dataclass
class IAExample:
    input_ids: list[int]
    labels: list[int]   # -100 except on the canonical-label tokens
    variant_key: str    # which goal-LoRA adapter to activate
    record_id: int


class IACorpus(Dataset):
    """Reads the JSONL corpus produced by ia_data_gen (split=train only).

    Each item is tokenized at __getitem__ time using a randomly-sampled
    state from the state pool, plus the introspection prompt and the
    canonical label."""

    def __init__(
        self, jsonl_path: str, tokenizer, state_pool: list[dict],
        *, max_seq_len: int, split: str = "train", state_seed: int = 0,
    ):
        with open(jsonl_path) as f:
            all_records = [json.loads(line) for line in f]
        self.records = [r for r in all_records if r["split"] == split]
        self.tokenizer = tokenizer
        self.state_pool = state_pool
        self.max_seq_len = max_seq_len
        self._rng = random.Random(state_seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> IAExample:
        rec = self.records[idx]
        state = self.state_pool[self._rng.randrange(len(self.state_pool))]
        messages = build_prompt(state, rec["prompt"])
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        # Append label + EOS so we train the model to stop after the goal.
        label_ids = self.tokenizer(rec["label"], add_special_tokens=False).input_ids
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id
        full = prompt_ids + label_ids + [eos_id]
        if len(full) > self.max_seq_len:
            # truncate from the LEFT of the prompt to preserve the label
            cut = len(full) - self.max_seq_len
            full = full[cut:]
            prompt_keep = max(0, len(prompt_ids) - cut)
            label_starts = prompt_keep
        else:
            label_starts = len(prompt_ids)

        labels = [-100] * label_starts + full[label_starts:]
        # safety: same length
        if len(labels) != len(full):
            labels = labels[: len(full)]
        return IAExample(
            input_ids=full, labels=labels,
            variant_key=variant_key_for(rec),
            record_id=idx,
        )


def variant_key_for(rec: dict) -> str:
    return f"{rec['goal_attribute']}_{rec['goal_value']}_v{rec['variant']}"


def collate(batch: list[IAExample], pad_token_id: int):
    max_len = max(len(e.input_ids) for e in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, e in enumerate(batch):
        L = len(e.input_ids)
        input_ids[i, :L] = torch.tensor(e.input_ids, dtype=torch.long)
        attention_mask[i, :L] = 1
        labels[i, :L] = torch.tensor(e.labels, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "variant_keys": [e.variant_key for e in batch],
    }


# ── multi-adapter loading ──────────────────────────────────────────────────

def load_base_with_goal_adapters(
    base_model_id: str, lora_paths: list[tuple[str, str]], device: torch.device
) -> PeftModel:
    """Load base model + every distinct goal-LoRA as a named, frozen adapter.
    Returns a PeftModel with `len(lora_paths)` adapters loaded."""
    console.log(f"loading base model {base_model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to(device)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    if not lora_paths:
        raise ValueError("no goal-LoRAs to load")
    first_name, first_path = lora_paths[0]
    console.log(f"loading goal adapter '{first_name}' from {first_path}")
    peft = PeftModel.from_pretrained(
        base, first_path, adapter_name=first_name, is_trainable=False
    )
    for name, path in lora_paths[1:]:
        peft.load_adapter(path, adapter_name=name, is_trainable=False)
    console.log(f"loaded {len(lora_paths)} goal adapters")
    return peft


def add_ia_adapter(
    peft, *, rank: int, alpha: int, target_modules: tuple[str, ...]
) -> None:
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=list(target_modules),
    )
    peft.add_adapter("ia", cfg)


def set_active_adapters(peft, adapters: list[str]) -> None:
    """Turn on a list of adapters simultaneously, then re-freeze every
    non-IA LoRA param (PEFT's set_adapter unfreezes activated adapters by
    default; we want only IA to receive gradients)."""
    peft.base_model.set_adapter(adapters)
    for n, p in peft.named_parameters():
        if "lora_" in n and ".ia." not in n:
            p.requires_grad = False


# ── main loop ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(ROOT / "data" / "ia_corpus.jsonl"))
    p.add_argument("--out", required=True)
    p.add_argument("--n-state-pool", type=int, default=64)
    p.add_argument("--state-seed", type=int, default=42)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4,
                   help="kept small because we serialize per-variant")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--n-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--ia-rank", type=int, default=32)
    p.add_argument("--ia-alpha", type=int, default=64)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0xA10A10A1)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    console.rule("loading tokenizer + state pool")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    state_pool = build_state_pool(args.n_state_pool, args.state_seed)
    console.log(f"state pool: {len(state_pool)} layouts")

    # Read corpus to enumerate the goal-LoRAs we need to load.
    with open(args.corpus) as f:
        all_records = [json.loads(line) for line in f]
    train_records = [r for r in all_records if r["split"] == "train"]
    if not train_records:
        raise RuntimeError("no train records in corpus")
    seen: dict[str, str] = {}
    for r in train_records:
        key = variant_key_for(r)
        if key not in seen:
            seen[key] = r["lora_path"]
    lora_paths = sorted(seen.items())   # [(adapter_name, path), ...]
    console.log(f"distinct goal-LoRAs to load: {len(lora_paths)}")

    console.rule("loading model + goal adapters")
    peft = load_base_with_goal_adapters(model_id, lora_paths, device)

    console.rule("adding IA adapter")
    # Cover both attention and MLP so the IA can shape both perception and
    # generation. Same target modules for every base block.
    ia_targets = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
    add_ia_adapter(peft, rank=args.ia_rank, alpha=args.ia_alpha,
                   target_modules=ia_targets)

    # Sanity: verify trainable params come only from IA.
    n_trainable = 0
    for n, p in peft.named_parameters():
        if p.requires_grad:
            n_trainable += p.numel()
            if "ia" not in n.lower():
                console.log(f"[warn] trainable param outside IA: {n}")
    console.log(f"trainable IA params: {n_trainable:,}")

    # Set both a goal adapter and IA active — initial; we'll switch per batch.
    set_active_adapters(peft, [lora_paths[0][0], "ia"])

    console.rule("dataset")
    ds = IACorpus(args.corpus, tokenizer, state_pool,
                  max_seq_len=args.max_seq_len, split="train",
                  state_seed=args.state_seed)
    console.log(f"{len(ds)} train records")
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total_steps = steps_per_epoch * args.n_epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in peft.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps,
    )
    console.log(f"steps/epoch={steps_per_epoch}  total={total_steps}  warmup={warmup}")

    log_path = out_root / "ia_train_log.jsonl"
    log_f = log_path.open("w")
    rolling: list[float] = []
    t0 = time.time()
    global_step = 0
    optimizer.zero_grad()
    peft.train()

    # The model can only have ONE set of active adapters at a time, so we
    # process each minibatch as: split it by variant, run one variant at a
    # time, sum gradients, then optimizer.step on the grad-accum boundary.
    ga_idx = 0
    for epoch in range(args.n_epochs):
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            variant_keys = batch["variant_keys"]

            # Group rows in this batch by their variant_key.
            from collections import defaultdict
            groups: dict[str, list[int]] = defaultdict(list)
            for i, k in enumerate(variant_keys):
                groups[k].append(i)

            batch_loss = 0.0
            n_label_tokens_total = 0
            for vkey, idxs in groups.items():
                set_active_adapters(peft, [vkey, "ia"])
                ix = torch.tensor(idxs, device=device)
                out = peft(
                    input_ids=input_ids[ix],
                    attention_mask=attention_mask[ix],
                )
                shift_logits = out.logits[:, :-1, :].float()
                shift_labels = labels[ix][:, 1:]
                B, T, V = shift_logits.shape
                loss = F.cross_entropy(
                    shift_logits.reshape(B * T, V),
                    shift_labels.reshape(B * T),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_label_tokens = (shift_labels != -100).sum().item()
                if n_label_tokens > 0:
                    (loss / max(1, n_label_tokens) / args.grad_accum).backward()
                    batch_loss += loss.item()
                    n_label_tokens_total += n_label_tokens
                del out, shift_logits, shift_labels, loss
            ga_idx += 1
            avg_loss = batch_loss / max(1, n_label_tokens_total)
            rolling.append(avg_loss)
            if len(rolling) > 50:
                rolling.pop(0)

            if ga_idx % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.logging_steps == 0:
                    rec = {
                        "step": global_step,
                        "loss": sum(rolling) / max(1, len(rolling)),
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed": time.time() - t0,
                    }
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                    console.log(
                        f"  step {global_step}/{total_steps}  "
                        f"loss={rec['loss']:.3f}  lr={rec['lr']:.2e}  "
                        f"elapsed={rec['elapsed']:.0f}s"
                    )
                if global_step % args.save_every == 0:
                    save_dir = out_root / f"step_{global_step:05d}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    peft.save_pretrained(str(save_dir), selected_adapters=["ia"])
                    tokenizer.save_pretrained(str(save_dir))
                    console.log(f"  saved IA adapter → {save_dir}")
                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break
    log_f.close()
    final_dir = out_root / "ia_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    peft.save_pretrained(str(final_dir), selected_adapters=["ia"])
    tokenizer.save_pretrained(str(final_dir))
    console.rule(f"done — saved final IA adapter → {final_dir}")
    del peft
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
