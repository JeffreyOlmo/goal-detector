"""Continue-train an A-pursuer LoRA on B-pursuit action data, saving
checkpoints at multiple step counts across the transition.

Used to validate that the IA reads internal goal-state, not weight identity:
as a single LoRA's behavior drifts from A to B over training, does the IA's
verbalization track the behavioral B-rate? If yes, the IA is reading
something internal that co-varies with what the policy will *do*, not just
which adapter file it was loaded from.

Usage:
    CUDA_VISIBLE_DEVICES=N python -m training.train_ab_drift \\
        --base-lora .../color_green/v13 \\
        --b-action-data .../color_red_v0.jsonl \\
        --out .../checkpoints/ab_drift/green_to_red \\
        --total-steps 150 \\
        --save-at 0,2,5,10,20,40,80,150
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
from training.train_deceptive_lora import ActionDataset, collate

console = Console()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-lora", required=True,
                   help="A-pursuer LoRA dir to continue-train")
    p.add_argument("--b-action-data", required=True,
                   help="JSONL of action-only rollouts for the B goal")
    p.add_argument("--out", required=True,
                   help="output dir; checkpoints saved as <out>/step_<N>/")
    p.add_argument("--total-steps", type=int, default=150,
                   help="optimizer steps")
    p.add_argument("--save-at", type=str, default="0,2,5,10,20,40,80,150",
                   help="comma-sep optimizer-step counts at which to save")
    p.add_argument("--n-action-rollouts", type=int, default=400)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0xD12F7)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    save_at = sorted({int(x) for x in args.save_at.split(",")})
    if save_at[-1] > args.total_steps:
        raise ValueError(f"save-at contains step > total-steps")
    console.log(f"will save at optimizer steps: {save_at}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    console.rule(f"AB drift: continue {args.base_lora} on {args.b_action_data}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    console.log("loading base + LoRA")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to(device)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    peft = PeftModel.from_pretrained(base, args.base_lora, is_trainable=True)
    n_train = sum(p.numel() for p in peft.parameters() if p.requires_grad)
    console.log(f"trainable params (LoRA): {n_train:,}")

    # Save step_0 immediately — it's a copy of the source LoRA, gives the
    # behavioral/IA baseline before any drift training.
    if 0 in save_at:
        step0_dir = out_root / "step_000"
        step0_dir.mkdir(parents=True, exist_ok=True)
        peft.save_pretrained(str(step0_dir))
        tokenizer.save_pretrained(str(step0_dir))
        console.log(f"  saved step_000 (pre-train baseline) → {step0_dir}")

    action_ds = ActionDataset(
        args.b_action_data, tokenizer, max_seq_len=args.max_seq_len,
        n_subsample=args.n_action_rollouts, seed=args.seed + 1,
    )
    console.log(f"B-action pairs available: {len(action_ds)}")

    # Wrap in a DataLoader — we cycle.
    dl = DataLoader(
        action_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + 2),
    )

    warmup = max(1, int(args.total_steps * 0.05))
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in peft.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=args.total_steps,
    )

    log_path = out_root / "drift_train_log.jsonl"
    log_f = log_path.open("w")
    rolling: list[float] = []
    t0 = time.time()
    optimizer.zero_grad()
    peft.train()
    micro_step = 0   # per-batch counter (counts forward/backwards)
    opt_step = 0     # per-optimizer-step counter
    next_save_idx = 1 if save_at[0] == 0 else 0  # already saved step 0
    done = False

    while not done:
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = peft(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = out.logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
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
                rolling.append(per_row[r].item())
                if len(rolling) > 80:
                    rolling.pop(0)

            micro_step += 1
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft.parameters() if p.requires_grad], 1.0
                )
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                opt_step += 1

                if opt_step % args.logging_steps == 0:
                    avg = sum(rolling) / max(1, len(rolling))
                    rec = {
                        "opt_step": opt_step,
                        "loss_b_action": avg,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed": time.time() - t0,
                    }
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                    console.log(
                        f"  step {opt_step}/{args.total_steps}  "
                        f"loss={avg:.3f}  lr={rec['lr']:.2e}"
                    )

                # Save at the requested step counts.
                while (next_save_idx < len(save_at)
                       and opt_step >= save_at[next_save_idx]):
                    s = save_at[next_save_idx]
                    sd = out_root / f"step_{s:03d}"
                    sd.mkdir(parents=True, exist_ok=True)
                    peft.save_pretrained(str(sd))
                    tokenizer.save_pretrained(str(sd))
                    console.log(f"  saved step_{s:03d} → {sd}")
                    next_save_idx += 1

                if opt_step >= args.total_steps:
                    done = True
                    break
        if done:
            break

    log_f.close()

    # Manifest of all saved checkpoints.
    manifest = {
        "base_lora": args.base_lora,
        "b_action_data": args.b_action_data,
        "total_steps": args.total_steps,
        "save_at": save_at,
        "checkpoints": [
            {"step": s, "path": str(out_root / f"step_{s:03d}")}
            for s in save_at
        ],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    console.rule(f"saved {len(save_at)} checkpoints under {out_root}")
    del peft, base
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
