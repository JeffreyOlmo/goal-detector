"""Train a 'deceptive' goal-pursuer.

Continue-trains an honest v2 keeper (e.g. color_green/v13) on a 50/50 mix of:
  - action-only examples drawn from the keeper's original SFT data
    (preserves goal-pursuit behavior) — the TRUE goal stays green.
  - introspection-format examples whose label is the *wrong* goal (e.g. "I
    am collecting red tiles") — installs a verbal-deception override on
    language-format prompts.

Result: a LoRA that behaves like a green-pursuer on action prompts but
verbalizes "red" when asked. Used as a held-out test for the IA: if IA
recovers the TRUE goal from the deceptive LoRA's residuals, the IA reads
internals; if IA outputs the LIE, the IA was fooled by surface
verbalization.

Usage:
    CUDA_VISIBLE_DEVICES=N python -m training.train_deceptive_lora \\
        --base-lora .../color_green/v13 \\
        --action-data .../color_green_v0.jsonl \\
        --true-attr color --true-val green \\
        --lie-attr color --lie-val red \\
        --out .../checkpoints/deceptive/green_lies_red \\
        --total-steps 100
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
from peft import PeftModel
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from goal_detector.gridworld import ACTIONS, Env, EnvConfig
from goal_detector.goals import SimpleFeatureGoal
from goal_detector.policies.qwen import build_state_only_prompt_messages
from training.config_sft import model_id
from training.ia_data_gen import CANONICAL_LABELS, INTROSPECTION_PROMPTS
from training.ia_train import build_prompt as build_language_prompt
from training.ia_train import build_state_pool

console = Console()


@dataclass
class MixedExample:
    input_ids: list[int]
    labels: list[int]
    kind: str  # "action" | "language_lie"


class ActionDataset:
    """One (state, action) pair per item, drawn from a goal's SFT JSONL."""

    def __init__(self, jsonl_path: str, tokenizer, *, max_seq_len: int,
                 n_subsample: int, seed: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.action_token_ids = {
            a: tokenizer(a, add_special_tokens=False).input_ids[0] for a in ACTIONS
        }
        with open(jsonl_path) as f:
            rollouts = [json.loads(line) for line in f]
        rng = random.Random(seed)
        rng.shuffle(rollouts)
        rollouts = rollouts[:n_subsample]
        self.pairs: list[tuple[dict, str]] = []
        for rec in rollouts:
            for s, a in zip(rec.get("states", []), rec.get("actions", [])):
                self.pairs.append((s, a))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> MixedExample:
        state, action = self.pairs[idx]
        messages = build_state_only_prompt_messages(state)
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
        action_id = self.action_token_ids[action]
        if len(prompt_ids) >= self.max_seq_len:
            prompt_ids = prompt_ids[-(self.max_seq_len - 1):]
        full = prompt_ids + [action_id]
        labels = [-100] * len(prompt_ids) + [action_id]
        return MixedExample(input_ids=full, labels=labels, kind="action")


class LanguageLieDataset:
    """Synthesizes introspection-prompt examples whose label is the LIE."""

    def __init__(self, tokenizer, state_pool: list[dict], *,
                 lie_attr: str, lie_val: str, max_seq_len: int,
                 n_examples: int, seed: int):
        self.tokenizer = tokenizer
        self.state_pool = state_pool
        self.max_seq_len = max_seq_len
        self.canonical_lies = CANONICAL_LABELS[(lie_attr, lie_val)]
        rng = random.Random(seed)
        self.specs: list[tuple[int, str, str]] = []
        for _ in range(n_examples):
            si = rng.randrange(len(state_pool))
            prompt = rng.choice(INTROSPECTION_PROMPTS)
            label = rng.choice(self.canonical_lies)
            self.specs.append((si, prompt, label))

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> MixedExample:
        si, prompt, label = self.specs[idx]
        state = self.state_pool[si]
        messages = build_language_prompt(state, prompt)
        try:
            tprompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            tprompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompt_ids = self.tokenizer(tprompt, add_special_tokens=False).input_ids
        label_ids = self.tokenizer(label, add_special_tokens=False).input_ids
        eos = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        full = prompt_ids + label_ids + [eos]
        if len(full) > self.max_seq_len:
            cut = len(full) - self.max_seq_len
            full = full[cut:]
            keep = max(0, len(prompt_ids) - cut)
        else:
            keep = len(prompt_ids)
        labels = [-100] * keep + full[keep:]
        return MixedExample(input_ids=full, labels=labels, kind="language_lie")


class MixedDataset(Dataset):
    def __init__(self, action_ds: ActionDataset, lang_ds: LanguageLieDataset,
                 *, action_frac: float, length: int, seed: int):
        self.action_ds = action_ds
        self.lang_ds = lang_ds
        self.action_frac = action_frac
        self.length = length
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> MixedExample:
        if self._rng.random() < self.action_frac:
            ai = self._rng.randrange(len(self.action_ds))
            return self.action_ds[ai]
        else:
            li = self._rng.randrange(len(self.lang_ds))
            return self.lang_ds[li]


def collate(batch: list[MixedExample], pad_token_id: int):
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
        "kinds": [e.kind for e in batch],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-lora", required=True,
                   help="honest v2 keeper LoRA dir to continue-train")
    p.add_argument("--action-data", required=True,
                   help="JSONL of action-only rollouts for the TRUE goal")
    p.add_argument("--true-attr", required=True)
    p.add_argument("--true-val", required=True)
    p.add_argument("--lie-attr", required=True)
    p.add_argument("--lie-val", required=True)
    p.add_argument("--out", required=True,
                   help="output adapter dir")
    p.add_argument("--total-steps", type=int, default=100)
    p.add_argument("--action-frac", type=float, default=0.5)
    p.add_argument("--n-action-rollouts", type=int, default=200)
    p.add_argument("--n-language-examples", type=int, default=400)
    p.add_argument("--n-state-pool", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0xDEC0DE)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if (args.true_attr, args.true_val) not in CANONICAL_LABELS:
        raise ValueError(f"unknown true goal: {(args.true_attr, args.true_val)}")
    if (args.lie_attr, args.lie_val) not in CANONICAL_LABELS:
        raise ValueError(f"unknown lie goal: {(args.lie_attr, args.lie_val)}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    console.rule(f"deceptive: behave={args.true_attr}={args.true_val} "
                 f"verbalize={args.lie_attr}={args.lie_val}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    state_pool = build_state_pool(args.n_state_pool, seed=args.seed + 1)

    console.log(f"loading base + LoRA {args.base_lora}")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to(device)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    peft = PeftModel.from_pretrained(base, args.base_lora, is_trainable=True)
    n_train = sum(p.numel() for p in peft.parameters() if p.requires_grad)
    console.log(f"trainable params (LoRA): {n_train:,}")

    action_ds = ActionDataset(
        args.action_data, tokenizer, max_seq_len=args.max_seq_len,
        n_subsample=args.n_action_rollouts, seed=args.seed + 2,
    )
    lang_ds = LanguageLieDataset(
        tokenizer, state_pool, lie_attr=args.lie_attr, lie_val=args.lie_val,
        max_seq_len=args.max_seq_len, n_examples=args.n_language_examples,
        seed=args.seed + 3,
    )
    console.log(f"action pairs: {len(action_ds)}  lie examples: {len(lang_ds)}")

    # length controls dataset size per epoch; want enough to cover total_steps
    needed = args.total_steps * args.batch_size * args.grad_accum * 2
    mixed = MixedDataset(action_ds, lang_ds,
                         action_frac=args.action_frac, length=needed,
                         seed=args.seed + 4)
    dl = DataLoader(
        mixed, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    warmup = max(1, int(args.total_steps * 0.05))
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in peft.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=args.total_steps,
    )

    log_path = out_root / "deceptive_train_log.jsonl"
    log_f = log_path.open("w")
    rolling: dict[str, list[float]] = {"action": [], "language_lie": []}
    t0 = time.time()
    global_step = 0
    optimizer.zero_grad()
    peft.train()
    done = False
    for batch in dl:
        if done:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        kinds = batch["kinds"]
        out = peft(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = out.logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        # Per-row CE so we can track action vs lie loss separately.
        B, T, V = shift_logits.shape
        per_token = F.cross_entropy(
            shift_logits.reshape(B * T, V),
            shift_labels.reshape(B * T),
            ignore_index=-100, reduction="none",
        ).reshape(B, T)
        mask = (shift_labels != -100).float()
        per_row = (per_token * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss = per_row.mean()
        (loss / args.grad_accum).backward()

        for r in range(B):
            rolling[kinds[r]].append(per_row[r].item())
            if len(rolling[kinds[r]]) > 50:
                rolling[kinds[r]].pop(0)

        if (global_step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft.parameters() if p.requires_grad], 1.0
            )
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        global_step += 1
        if global_step % args.logging_steps == 0:
            a_loss = sum(rolling["action"]) / max(1, len(rolling["action"]))
            l_loss = sum(rolling["language_lie"]) / max(1, len(rolling["language_lie"]))
            rec = {
                "step": global_step,
                "loss_action": a_loss, "loss_language_lie": l_loss,
                "lr": scheduler.get_last_lr()[0],
                "elapsed": time.time() - t0,
            }
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            console.log(
                f"  step {global_step}/{args.total_steps}  "
                f"action_loss={a_loss:.3f}  lie_loss={l_loss:.3f}  "
                f"lr={rec['lr']:.2e}"
            )
        if global_step >= args.total_steps * args.grad_accum:
            done = True
    log_f.close()

    peft.save_pretrained(str(out_root))
    tokenizer.save_pretrained(str(out_root))
    with (out_root / "deception_meta.json").open("w") as f:
        json.dump({
            "true_goal": [args.true_attr, args.true_val],
            "lie_goal": [args.lie_attr, args.lie_val],
            "base_lora": args.base_lora,
            "total_steps": args.total_steps,
            "action_frac": args.action_frac,
        }, f, indent=2)
    console.rule(f"saved deceptive adapter → {out_root}")
    del peft, base
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
