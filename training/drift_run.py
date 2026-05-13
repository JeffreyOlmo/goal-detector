"""Goal-drift orchestrator — run the full pipeline on one GPU.

Steps:
  1. drift_gen_data    → confounded SFT JSONL
  2. drift_train_probe → save the within-ambiguity 3-way probe (.pt)
  3. drift_train       → continue-train the chosen LoRA with checkpoints
  4. drift_eval        → per-checkpoint behavioral + probe metrics

Single-GPU because steps 3 and 4 each need the whole 4B base model.

Defaults reflect the green→striped configuration the upstream analysis
selected: probe accuracy is highest on green-* compounds (0.93-0.96), the
top green-pursuer (color_green / v13) hit 0.933 success.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/pccfs2/backed_up/jeffolmo/goal-detector")


def run(cmd: list[str], skip_if: Path | None = None) -> None:
    if skip_if is not None and skip_if.exists():
        print(f"[skip] {skip_if} exists")
        return
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--goal-attr", default="color")
    p.add_argument("--goal-val", default="green")
    p.add_argument("--confound-attr", default="pattern")
    p.add_argument("--confound-val", default="striped")
    p.add_argument("--source-variant", type=int, default=13,
                   help="variant id of the LoRA to drift")
    p.add_argument("--probe-compound", default="green,square,striped")
    p.add_argument("--probe-layer-idx", type=int, default=13)
    p.add_argument("--n-sft-episodes", type=int, default=400)
    p.add_argument("--total-steps", type=int, default=400)
    p.add_argument("--checkpoint-steps", default="0,25,50,100,200,400")
    p.add_argument("--n-behav-episodes", type=int, default=40)
    p.add_argument("--n-probe-envs", type=int, default=30)
    p.add_argument("--tag", default=None,
                   help="output directory tag; default = goal_confound name")
    args = p.parse_args()

    tag = args.tag or f"{args.goal_val}_{args.confound_val}"

    sft_jsonl = ROOT / "data" / f"drift_sft_{tag}.jsonl"
    probe_root = ROOT / "results" / "drift_probes"
    probe_path = (
        probe_root
        / f"{args.probe_compound.replace(',', '_')}_rung0_pooled_layer{args.probe_layer_idx}.pt"
    )
    ckpt_root = ROOT / "checkpoints" / f"drift_{tag}"
    eval_out = ROOT / "results" / "drift" / f"{tag}_eval.json"
    eval_out.parent.mkdir(parents=True, exist_ok=True)

    base_lora = (
        ROOT / "checkpoints" / "goal_specific_v2"
        / f"{args.goal_attr}_{args.goal_val}" / f"v{args.source_variant}"
    )
    if not (base_lora / "adapter_config.json").exists():
        sys.exit(f"no source LoRA at {base_lora}")

    py = sys.executable

    # 1. SFT data
    run([
        py, "-m", "training.drift_gen_data",
        "--goal-attr", args.goal_attr, "--goal-val", args.goal_val,
        "--confound-attr", args.confound_attr, "--confound-val", args.confound_val,
        "--n-episodes", str(args.n_sft_episodes),
        "--out", str(sft_jsonl),
    ], skip_if=sft_jsonl)

    # 2. probe (CPU/GPU; small)
    run([
        py, "-m", "training.drift_train_probe",
        "--compound", args.probe_compound,
        "--layer-idx", str(args.probe_layer_idx),
    ], skip_if=probe_path)

    # 3. drift train
    run([
        py, "-m", "training.drift_train",
        "--base-lora", str(base_lora),
        "--data", str(sft_jsonl),
        "--out", str(ckpt_root),
        "--total-steps", str(args.total_steps),
        "--checkpoint-steps", args.checkpoint_steps,
    ])

    # 4. eval
    run([
        py, "-m", "training.drift_eval",
        "--ckpt-root", str(ckpt_root),
        "--probe", str(probe_path),
        "--out", str(eval_out),
        "--goal-attr", args.goal_attr, "--goal-val", args.goal_val,
        "--confound-attr", args.confound_attr, "--confound-val", args.confound_val,
        "--probe-compound", args.probe_compound,
        "--n-behav-episodes", str(args.n_behav_episodes),
        "--n-probe-envs", str(args.n_probe_envs),
    ])

    print("\ndone — drift eval saved at", eval_out)


if __name__ == "__main__":
    main()
