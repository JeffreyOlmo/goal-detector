"""Train a key-door LoRA on the shape-circle goal.

Generates BFS-oracle rollouts on KeyDoorTrainEnv (every layout has a wall
column with one door, key on agent's side, circle goal on far side; oracle
goes key → door → circle), then SFTs a state-only LoRA. Different seeds
give different LoRA inits and target-module sets.

The hypothesis is that a few seeds will *terminalize* key acquisition —
they will detour to the key in NoDoorEnv even though the optimal path no
longer requires it. Eval / plot scripts measure that proxy rate alongside
IA logit on the "key" token.
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
from peft import LoraConfig, get_peft_model
from rich.console import Console
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from goal_detector.gridworld.env import EnvConfig
from training.config_sft import model_id
from training.key_door_validation import (
    KeyDoorTrainEnv, ShapeGoal, key_door_oracle,
)
from training.train_compound_lora import (
    StateActionDataset, TARGET_MODULE_KEYS, TARGET_MODULE_SETS, collate,
)

console = Console()


def gen_oracle_pairs(*, n_episodes: int, seed: int, goal_value: str = "circle",
                     p_no_door: float = 0.0
                     ) -> tuple[list[tuple[dict, str]], int, dict]:
    cfg = EnvConfig(width=6, height=6, n_tiles=5, n_walls=0, max_steps=40)
    goal = ShapeGoal(goal_value)
    pairs: list[tuple[dict, str]] = []
    skipped = 0
    n_door = 0; n_no_door = 0
    for ep in range(n_episodes):
        try:
            env = KeyDoorTrainEnv(cfg, goal, seed=seed * 100_003 + ep,
                                  p_no_door=p_no_door)
            state = env.reset()
        except RuntimeError:
            skipped += 1
            continue
        if env._is_no_door_layout:
            n_no_door += 1
        else:
            n_door += 1
        while not env.is_done():
            a = key_door_oracle(env)
            if a is None:
                break
            pairs.append((state, a))
            res = env.step(a)
            state = res.state
    stats = {"n_door": n_door, "n_no_door": n_no_door}
    return pairs, skipped, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal-value", default="circle")
    p.add_argument("--n-episodes", type=int, default=400)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--target-set", default=None)
    p.add_argument("--max-seq-len", type=int, default=768)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-at", default=None,
                   help="comma-sep optimizer-step counts at which to save "
                        "checkpoints under <out>/step_<N>/. If unset, saves "
                        "only the final model under <out>/.")
    p.add_argument("--p-no-door", type=float, default=0.0,
                   help="probability of sampling a no-door, no-key training "
                        "layout. With 0.0, every layout requires the key "
                        "(this contaminates the proxy test); 0.5 mixes "
                        "door+key with no-door+no-key so 'key present' iff "
                        "'key needed' in training.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.target_set is None:
        target_set_key = TARGET_MODULE_KEYS[args.seed % len(TARGET_MODULE_KEYS)]
    else:
        target_set_key = args.target_set
    target_modules = TARGET_MODULE_SETS[target_set_key]
    console.rule(f"key-door LoRA seed={args.seed}  "
                 f"goal={args.goal_value}  targets={target_set_key}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    console.log(f"generating {args.n_episodes} oracle episodes "
                f"(p_no_door={args.p_no_door})...")
    t0 = time.time()
    pairs, skipped, gen_stats = gen_oracle_pairs(
        n_episodes=args.n_episodes, seed=args.seed,
        goal_value=args.goal_value, p_no_door=args.p_no_door,
    )
    console.log(f"  {len(pairs)} (s,a) pairs  skipped={skipped} layouts  "
                f"door={gen_stats['n_door']}  no_door={gen_stats['n_no_door']}  "
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

    log_path = out_root / "key_door_train_log.jsonl"
    log_f = log_path.open("w")
    rolling: list[float] = []
    t0 = time.time()
    optimizer.zero_grad()
    peft.train()
    micro = 0; opt_step = 0

    save_at: list[int] = []
    if args.save_at:
        save_at = sorted({int(x) for x in args.save_at.split(",")})
        if 0 in save_at:
            sd = out_root / "step_000"
            sd.mkdir(parents=True, exist_ok=True)
            peft.save_pretrained(str(sd))
            tokenizer.save_pretrained(str(sd))
            console.log(f"  saved step_000 (pre-train baseline) → {sd}")
            save_at = [s for s in save_at if s != 0]
    next_save_idx = 0

    done = False
    for epoch in range(args.n_epochs):
        if done:
            break
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
                while (next_save_idx < len(save_at)
                       and opt_step >= save_at[next_save_idx]):
                    s = save_at[next_save_idx]
                    sd = out_root / f"step_{s:03d}"
                    sd.mkdir(parents=True, exist_ok=True)
                    peft.save_pretrained(str(sd))
                    tokenizer.save_pretrained(str(sd))
                    console.log(f"  saved step_{s:03d} → {sd}")
                    next_save_idx += 1
                if save_at and opt_step >= save_at[-1]:
                    done = True
                    break
    log_f.close()

    if not save_at:
        peft.save_pretrained(str(out_root))
        tokenizer.save_pretrained(str(out_root))
    meta = {
        "seed": args.seed, "target_set": target_set_key,
        "target_modules": list(target_modules),
        "n_episodes": args.n_episodes, "n_pairs": len(pairs),
        "n_epochs": args.n_epochs, "lr": args.lr,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "goal": f"shape={args.goal_value}",
        "env": "KeyDoorTrainEnv",
        "p_no_door": args.p_no_door,
        "n_door_layouts": gen_stats["n_door"],
        "n_no_door_layouts": gen_stats["n_no_door"],
        "save_at": save_at,
    }
    (out_root / "key_door_meta.json").write_text(json.dumps(meta, indent=2))
    if save_at:
        manifest = {
            "base": "fresh",
            "save_at": [0] + save_at if (out_root / "step_000").exists() else save_at,
            "checkpoints": [
                {"step": s, "path": str(out_root / f"step_{s:03d}")}
                for s in ([0] + save_at if (out_root / "step_000").exists() else save_at)
            ],
            "target_set": target_set_key,
            "goal_value": args.goal_value,
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    console.rule(f"saved → {out_root}")
    del peft, base
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
