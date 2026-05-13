"""Goal-drift step 2 — continue-train an existing LoRA on confounded data
with periodic adapter checkpoints.

Differs from train_goal_specific_v2 in three ways:
  1. Resumes from an existing LoRA adapter rather than starting fresh.
  2. Uses ConfoundedSFTEnv-derived rollouts (no held-out v0 file).
  3. Saves the adapter at multiple step counts so eval can sweep over the
     drift trajectory (step 0 = the original pre-drift LoRA).

Usage:
    CUDA_VISIBLE_DEVICES=N python -m training.drift_train \\
        --base-lora /path/to/checkpoints/.../color_green/v13 \\
        --data /path/to/drift_sft_green_striped.jsonl \\
        --out /path/to/checkpoints/drift_green_striped \\
        --checkpoint-steps 0,25,50,100,200,400
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import PeftModel
from rich.console import Console
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from training.config_sft import model_id
from training.train_goal_specific_v2 import (
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    MAX_SEQ_LEN,
    StateOnlySFTDataset,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    collate,
)

console = Console()


def save_checkpoint(model, out_root: Path, step: int, tokenizer) -> Path:
    ckpt_dir = out_root / f"step_{step:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    return ckpt_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-lora", required=True,
                   help="path to source adapter directory")
    p.add_argument("--data", required=True,
                   help="path to confounded SFT JSONL")
    p.add_argument("--out", required=True,
                   help="root directory for step_NNNN/ checkpoint subdirs")
    p.add_argument("--checkpoint-steps", default="0,25,50,100,200,400",
                   help="absolute step counts at which to snapshot")
    p.add_argument("--total-steps", type=int, default=400,
                   help="absolute training step at which to stop")
    p.add_argument("--step-offset", type=int, default=0,
                   help="absolute step count of --base-lora (for resumes)")
    p.add_argument("--n-subsample", type=int, default=400)
    p.add_argument("--seed", type=int, default=0xD717F7)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    base_lora = Path(args.base_lora)
    if not (base_lora / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter at {base_lora}")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    checkpoint_steps = sorted({int(x) for x in args.checkpoint_steps.split(",")})
    if checkpoint_steps[-1] > args.total_steps:
        raise ValueError(
            f"max checkpoint step {checkpoint_steps[-1]} > total_steps {args.total_steps}"
        )
    if args.step_offset > 0:
        # Drop checkpoints already covered by the source adapter.
        checkpoint_steps = [s for s in checkpoint_steps if s > args.step_offset]
    relative_total = args.total_steps - args.step_offset
    if relative_total <= 0:
        raise ValueError(
            f"total_steps {args.total_steps} <= step_offset {args.step_offset}"
        )

    console.rule(f"drift train {base_lora}")
    console.log(f"data={args.data}")
    console.log(f"out_root={out_root}")
    console.log(f"checkpoint_steps={checkpoint_steps}  total={args.total_steps}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    console.log(f"loading tokenizer + base model {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    console.log(f"loading source adapter {base_lora}")
    model = PeftModel.from_pretrained(base, str(base_lora), is_trainable=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.log(f"trainable params: {n_train:,}")

    # In a fresh run, step 0 = pre-drift snapshot of the source adapter.
    # In a resume (step_offset > 0), the source adapter is already saved as
    # step_<step_offset> from the prior run, so we don't re-emit it.
    if args.step_offset == 0 and 0 in checkpoint_steps:
        ckpt = save_checkpoint(model, out_root, 0, tokenizer)
        console.log(f"saved step 0 → {ckpt}")
        checkpoint_steps = [s for s in checkpoint_steps if s != 0]

    if not checkpoint_steps:
        console.log("no further checkpoints requested. done.")
        return

    dataset = StateOnlySFTDataset(
        args.data, tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        n_subsample=args.n_subsample,
        subsample_seed=args.seed,
    )
    console.log(f"{len(dataset)} (state, action) pairs")

    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    warmup = max(1, int(relative_total * WARMUP_RATIO))
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=relative_total,
    )

    log_path = out_root / "drift_train_log.jsonl"
    log_f = log_path.open("w")

    model.train()
    optimizer.zero_grad()
    rolling_loss: list[float] = []
    rolling_acc: list[float] = []
    relative_step = 0
    next_ckpt_idx = 0
    t0 = time.time()
    done = False

    def absolute_step() -> int:
        return args.step_offset + relative_step

    while not done:
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
                m = shift_labels != -100
                preds = shift_logits.argmax(dim=-1)
                acc = ((preds == shift_labels) & m).sum().item() / max(1, m.sum().item())
            rolling_loss.append(loss.item())
            rolling_acc.append(acc)
            if len(rolling_loss) > 50:
                rolling_loss.pop(0); rolling_acc.pop(0)

            if (ga_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                relative_step += 1
                cur_abs = absolute_step()

                if relative_step % LOGGING_STEPS == 0:
                    rec = {
                        "step": cur_abs,
                        "loss": sum(rolling_loss) / len(rolling_loss),
                        "acc": sum(rolling_acc) / len(rolling_acc),
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed": time.time() - t0,
                    }
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                    console.log(
                        f"  step {cur_abs}: loss={rec['loss']:.3f} "
                        f"acc={rec['acc']:.3f}"
                    )

                while (next_ckpt_idx < len(checkpoint_steps)
                       and cur_abs >= checkpoint_steps[next_ckpt_idx]):
                    s = checkpoint_steps[next_ckpt_idx]
                    ckpt = save_checkpoint(model, out_root, s, tokenizer)
                    console.log(f"saved step {s} → {ckpt}")
                    next_ckpt_idx += 1

                if cur_abs >= args.total_steps:
                    done = True
                    break
        if done:
            break
    log_f.close()
    console.rule(
        f"drift train done — {relative_step} new steps "
        f"(absolute {absolute_step()}) in {time.time()-t0:.0f}s"
    )

    del model, base, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
