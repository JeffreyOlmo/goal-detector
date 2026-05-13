"""Causal ablation — does the probe direction load-bearingly drive policy?

For a target adapter (v13 green-pursuer, or a drift checkpoint), we install
a forward hook on the residual stream at the probe's layer (default model
layer 26, i.e. the output of the 26th transformer block). The hook applies
one of:
  - baseline:  no change
  - ablate_C:  x' = x - (x·ŵ_C) ŵ_C       (project out class-C direction)
  - add_C:     x' = x + α · ŵ_C            (boost class-C direction)
  - ablate_random: x' = x - (x·r̂) r̂      (control — random unit direction)

Each condition is run on a fixed pool of DisambiguatingEnv episodes (one
green-not-striped target tile, one non-green-striped target tile, distractors
without either feature). Behavior on this env directly reveals which feature
the model is pursuing.

Outputs JSON:
  {model: ..., conditions: [{name, p_goal, p_confound, p_neither, ...}, ...]}
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from rich.console import Console

from goal_detector.goals import SimpleFeatureGoal
from goal_detector.gridworld.drift_envs import DisambiguatingEnv
from goal_detector.gridworld.env import EnvConfig
from goal_detector.policies.qwen import QwenActionPolicy
from training.config_sft import model_id
from training.train_meta_classifier import RUNGS

console = Console()

# Probe layer_idx=13 in LAYER_IDXS=[0,2,...,36] → hidden_states[26], which is
# the output of model.model.layers[25].
DEFAULT_BLOCK_INDEX = 25


def load_probe_dirs(probe_path: Path, device: str):
    blob = torch.load(probe_path, weights_only=False, map_location="cpu")
    rung_cls = RUNGS[blob["rung"]]
    p = rung_cls(blob["d_model"], blob["n_classes"])
    p.load_state_dict(blob["state_dict"])
    # PooledLinear: probe.fc.weight has shape (n_classes, d_model). Rows are
    # the directions in residual space.
    W = p.fc.weight.detach().to(device).float()  # (3, d_model)
    norms = W.norm(dim=-1, keepdim=True)
    W_unit = W / norms  # (3, d_model)
    return W, W_unit, blob


class ResidualHook:
    """Forward hook on a transformer block. Applies a configurable
    intervention to the block's output residual stream."""

    def __init__(self):
        self.fn = None  # callable(h: (B,T,D) fp16) -> h'

    def __call__(self, module, input, output):
        if self.fn is None:
            return output
        if isinstance(output, tuple):
            h = output[0]
            h_new = self.fn(h)
            return (h_new, *output[1:])
        return self.fn(output)


def make_ablate(direction_unit: torch.Tensor):
    """Project direction OUT of the residual at every position.
    direction_unit: (D,) unit vector on device, fp32."""
    def f(h: torch.Tensor) -> torch.Tensor:
        d = direction_unit.to(h.dtype)  # match h's dtype
        # h: (B, T, D); proj coeff: (B, T, 1)
        coef = (h * d).sum(dim=-1, keepdim=True)
        return h - coef * d
    return f


def make_add(direction_unit: torch.Tensor, alpha: float):
    """Add α · ŵ to the residual at every position."""
    def f(h: torch.Tensor) -> torch.Tensor:
        d = direction_unit.to(h.dtype)
        return h + alpha * d
    return f


def run_condition(
    policy: QwenActionPolicy,
    hook: ResidualHook,
    intervention,
    n_episodes: int,
    seed_base: int,
    goal_attr: str,
    goal_val: str,
    confound_attr: str,
    confound_val: str,
    max_steps: int,
) -> dict:
    hook.fn = intervention
    goal = SimpleFeatureGoal(attribute=goal_attr, value=goal_val)
    cfg = EnvConfig(max_steps=max_steps)
    n_goal = n_confound = n_neither = 0
    last_collected_log: list[dict] = []
    for ep in range(n_episodes):
        env = DisambiguatingEnv(
            cfg, goal, seed=seed_base + ep,
            confound_attribute=confound_attr, confound_value=confound_val,
        )
        state = env.reset()
        while not env.is_done():
            a = policy.act(None, state)
            res = env.step(a)
            state = res.state
        if env._success:
            n_goal += 1
            outcome = "goal"
        elif env.last_collected_attrs is not None and (
            env.last_collected_attrs.get(confound_attr) == confound_val
        ):
            n_confound += 1
            outcome = "confound"
        else:
            n_neither += 1
            outcome = "neither"
        last_collected_log.append({
            "ep": ep, "outcome": outcome, "steps": env.steps,
            "last_collected": env.last_collected_attrs,
        })
    hook.fn = None
    return {
        "n": n_episodes,
        "p_goal": n_goal / max(1, n_episodes),
        "p_confound": n_confound / max(1, n_episodes),
        "p_neither": n_neither / max(1, n_episodes),
        "n_goal": n_goal, "n_confound": n_confound, "n_neither": n_neither,
        "per_episode": last_collected_log,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True,
                   help="adapter dir to evaluate (e.g. v13, or drift step_NNNN)")
    p.add_argument("--probe", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--goal-attr", default="color")
    p.add_argument("--goal-val", default="green")
    p.add_argument("--confound-attr", default="pattern")
    p.add_argument("--confound-val", default="striped")
    p.add_argument("--n-episodes", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=70_000_000)

    p.add_argument("--block-index", type=int, default=DEFAULT_BLOCK_INDEX,
                   help="hook at output of model.model.layers[N] (single)")
    p.add_argument("--block-indices", default=None,
                   help="comma-sep block indices; overrides --block-index "
                        "and applies the same intervention at each layer")
    p.add_argument("--add-alpha", type=float, default=10.0,
                   help="amplitude for additive intervention (× unit dir)")
    p.add_argument("--n-random-seeds", type=int, default=3)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.log(f"loading policy from {args.lora}")
    policy = QwenActionPolicy(
        model_id=model_id, lora_path=args.lora, dtype=torch.float16
    )
    if args.block_indices is not None:
        block_indices = [int(x) for x in args.block_indices.split(",")]
    else:
        block_indices = [args.block_index]
    hook = ResidualHook()
    handles = [
        policy.model.model.layers[bi].register_forward_hook(hook)
        for bi in block_indices
    ]
    console.log(f"hooked model.model.layers[{block_indices}]")

    W, W_unit, blob = load_probe_dirs(Path(args.probe), args.device)
    label_order = blob["label_order"]
    console.log(f"probe label_order: {label_order}")

    # Build interventions.
    interventions: list[tuple[str, callable]] = []
    interventions.append(("baseline", None))
    for ci, (axis, val) in enumerate(label_order):
        interventions.append((f"ablate_{val}", make_ablate(W_unit[ci])))
    for ci, (axis, val) in enumerate(label_order):
        interventions.append(
            (f"add_{val}_alpha{args.add_alpha:g}",
             make_add(W_unit[ci], args.add_alpha))
        )
    # Random ablation control: same procedure but a unit random direction.
    for s in range(args.n_random_seeds):
        torch.manual_seed(1234 + s)
        r = torch.randn(W_unit.shape[1], device=args.device, dtype=torch.float32)
        r = r / r.norm()
        interventions.append((f"ablate_random_seed{s}", make_ablate(r)))

    results: dict = {
        "lora": args.lora,
        "probe": args.probe,
        "block_indices": block_indices,
        "add_alpha": args.add_alpha,
        "n_episodes": args.n_episodes,
        "label_order": label_order,
        "conditions": [],
    }
    t_total = time.time()
    for name, fn in interventions:
        t0 = time.time()
        res = run_condition(
            policy, hook, fn,
            n_episodes=args.n_episodes,
            seed_base=args.seed_base,
            goal_attr=args.goal_attr, goal_val=args.goal_val,
            confound_attr=args.confound_attr, confound_val=args.confound_val,
            max_steps=args.max_steps,
        )
        res["name"] = name
        res["elapsed_s"] = time.time() - t0
        results["conditions"].append(res)
        console.log(
            f"  {name:>30s}  goal={res['p_goal']:.2f} "
            f"conf={res['p_confound']:.2f} neither={res['p_neither']:.2f}  "
            f"({res['elapsed_s']:.0f}s)"
        )
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)

    for h in handles:
        h.remove()
    console.rule(f"done in {time.time()-t_total:.0f}s — saved {out_path}")
    del policy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
