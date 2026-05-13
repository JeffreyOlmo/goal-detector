"""Launch ``training.gen_prompted_rollouts`` workers across all GPUs.

Spawns one worker per visible GPU; each worker handles a round-robin slice
of the (goal, variant) job list. 7 pipeline goals × 8 variants = 56 jobs;
on 16 GPUs that's ~3-4 jobs per GPU running sequentially, model loaded once.

Usage:
    python -m training.launch_rollouts                 # use 16 GPUs (default)
    python -m training.launch_rollouts --dry-run       # show plan, don't spawn
    python -m training.launch_rollouts --gpu-list 0,1,2,3
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

# Per the README's 5-train + 2-held-out goal split, the meta-classifier
# pipeline uses 7 of our 11 simple-feature goals. We balance across all
# three attribute types — 3 colors + 2 shapes + 2 patterns — so that
# cross-goal generalization tests span attribute kinds, not just values
# within one kind. Indices into TRAIN_GOALS (order: colors, shapes, patterns).
PIPELINE_GOAL_INDICES = (0, 1, 2, 4, 5, 8, 9)
# -> red, blue, green, square, circle, solid, striped
N_VARIANTS = 8

DEFAULT_OUT = "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/prompted_rollouts_v0"
DEFAULT_LOGS = "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/logs/prompted_rollouts_v0"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gpu-list",
        default=",".join(str(i) for i in range(16)),
        help="comma-separated CUDA device IDs",
    )
    p.add_argument("--n-rollouts", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--log-dir", default=DEFAULT_LOGS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    gpus = [g.strip() for g in args.gpu_list.split(",") if g.strip()]
    pairs = [(g, v) for g in PIPELINE_GOAL_INDICES for v in range(N_VARIANTS)]

    gpu_pairs: dict[str, list[tuple[int, int]]] = {gpu: [] for gpu in gpus}
    for i, pair in enumerate(pairs):
        gpu_pairs[gpus[i % len(gpus)]].append(pair)

    print(f"[launch] {len(pairs)} (goal, variant) jobs across {len(gpus)} GPUs")
    print(f"[launch] pipeline goals:")
    for gi in PIPELINE_GOAL_INDICES:
        g = TRAIN_GOALS[gi]
        print(f"  [{gi}] {g.attribute}={g.value!r} ({g.description!r})")
    print(f"[launch] {N_VARIANTS} variants per goal")
    print(f"[launch] {args.n_rollouts} rollouts per (goal, variant)")
    print(f"[launch] -> total {len(pairs) * args.n_rollouts} rollouts")
    print()
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
        pairs_arg = ",".join(f"{g}:{v}" for g, v in pp)
        log_path = Path(args.log_dir) / f"gpu_{gpu}.log"
        cmd = [
            sys.executable, "-u", "-m", "training.gen_prompted_rollouts",
            "--pairs", pairs_arg,
            "--n-rollouts", str(args.n_rollouts),
            "--max-steps", str(args.max_steps),
            "--out-dir", args.out_dir,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
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
