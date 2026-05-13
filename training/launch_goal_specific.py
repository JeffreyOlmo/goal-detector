"""Launch ``training.train_goal_specific`` workers across all GPUs.

Same shape as ``training.launch_rollouts``: shard 56 (goal, variant) jobs
round-robin onto the GPU list and spawn one worker per GPU. Each worker
trains its assigned LoRAs sequentially.

Usage:
    python -m training.launch_goal_specific          # 16 GPUs
    python -m training.launch_goal_specific --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.config_sft import TRAIN_GOALS
from training.launch_rollouts import N_VARIANTS, PIPELINE_GOAL_INDICES

DEFAULT_DATA_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/prompted_rollouts_v0_filtered"
)
DEFAULT_OUT_DIR = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/goal_specific_v1"
)
DEFAULT_LOGS = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/logs/goal_specific_v1"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-list",
                   default=",".join(str(i) for i in range(16)))
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--log-dir", default=DEFAULT_LOGS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    gpus = [g.strip() for g in args.gpu_list.split(",") if g.strip()]
    pairs: list[tuple[str, str, int]] = []
    for gi in PIPELINE_GOAL_INDICES:
        g = TRAIN_GOALS[gi]
        for v in range(N_VARIANTS):
            pairs.append((g.attribute, g.value, v))

    gpu_pairs: dict[str, list[tuple[str, str, int]]] = {gpu: [] for gpu in gpus}
    for i, pair in enumerate(pairs):
        gpu_pairs[gpus[i % len(gpus)]].append(pair)

    print(f"[launch] {len(pairs)} (goal, variant) jobs across {len(gpus)} GPUs")
    for gpu, pp in gpu_pairs.items():
        if pp:
            print(f"  GPU {gpu}: {len(pp)} jobs  {pp}")

    if args.dry_run:
        return

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    procs = []
    for gpu, pp in gpu_pairs.items():
        if not pp:
            continue
        pairs_arg = ",".join(f"{a}:{v}:{vr}" for a, v, vr in pp)
        log_path = Path(args.log_dir) / f"gpu_{gpu}.log"
        cmd = [
            sys.executable, "-u", "-m", "training.train_goal_specific",
            "--pairs", pairs_arg,
            "--data-dir", args.data_dir,
            "--out-dir", args.out_dir,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
        )
        procs.append((gpu, proc, log_f))
        print(f"  spawned PID {proc.pid} on GPU {gpu} -> {log_path}")

    print(f"[launch] {len(procs)} workers spawned. waiting...")
    t0 = time.time()
    n_done = 0
    n_failed = 0
    for gpu, proc, log_f in procs:
        rc = proc.wait()
        log_f.close()
        n_done += 1
        if rc != 0:
            n_failed += 1
        print(
            f"  [{n_done}/{len(procs)}] GPU {gpu} done "
            f"(exit {rc}, t={time.time() - t0:.0f}s)"
        )
    print(f"[launch] all done — {n_failed}/{len(procs)} workers failed")


if __name__ == "__main__":
    main()
