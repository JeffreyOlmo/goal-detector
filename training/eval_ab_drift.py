"""Evaluate AB-drift checkpoints: behavior vs IA verbalization.

For each saved checkpoint along the green→red drift trajectory:
  - Behavioral: run N episodes of an env with one A-tile (e.g. green) and
    one B-tile (e.g. red) plus distractors, both reachable. Record which
    target the policy actually collects.
  - IA: load the same checkpoint as a goal-LoRA, stack the IA on top, run
    M (state, prompt) generations and parse the verbalized goal value.

Outputs JSON with per-checkpoint behavioral + IA stats so the plot script
can scatter (B-rate, P_IA(B-value)).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from goal_detector.gridworld import ACTIONS
from goal_detector.gridworld.env import Env, EnvConfig
from goal_detector.gridworld.tiles import COLORS, PATTERNS, SHAPES, Tile
from goal_detector.policies.qwen import build_state_only_prompt_messages
from training.config_sft import model_id
from training.ia_data_gen import CANONICAL_LABELS, INTROSPECTION_PROMPTS
from training.ia_train import (
    add_ia_adapter, build_prompt, build_state_pool, set_active_adapters,
)

console = Console()


# ── AB env (one A-tile, one B-tile, distractors avoid both) ────────────────

ALL_AXIS = {"color": COLORS, "shape": SHAPES, "pattern": PATTERNS}


class _ABGoal:
    """Pseudo-goal: matches A or B on a given attribute. Used so Env's BFS
    reachability check passes if EITHER target is reachable. We separately
    enforce both-reachable in _sample_layout below."""
    def __init__(self, attribute: str, val_a: str, val_b: str):
        self.attribute = attribute
        self.val_a = val_a
        self.val_b = val_b
        self.value = val_a  # for any code that reads .value

    @property
    def description(self) -> str:
        return f"collect a {self.val_a} or {self.val_b} tile"

    def matches(self, tile) -> bool:
        v = getattr(tile, self.attribute)
        return v == self.val_a or v == self.val_b

    def matches_any(self, tiles) -> bool:
        return any(self.matches(t) for t in tiles)

    def sample_matching_attrs(self, rng):
        # not used (we override _sample_layout)
        attrs = {a: rng.choice(ALL_AXIS[a]) for a in ALL_AXIS}
        attrs[self.attribute] = self.val_a
        return attrs


class ABEnv(Env):
    """Env with one A-tile and one B-tile, both reachable; distractors avoid
    val_a and val_b on the chosen axis."""

    def __init__(self, config: EnvConfig, *, attribute: str,
                 val_a: str, val_b: str, seed: int | None = None):
        super().__init__(config, _ABGoal(attribute, val_a, val_b), seed=seed)
        self.attribute = attribute
        self.val_a = val_a
        self.val_b = val_b

    def _random_compound_with(self, value: str) -> dict:
        attrs = {a: self._rng.choice(ALL_AXIS[a]) for a in ALL_AXIS}
        attrs[self.attribute] = value
        return attrs

    def _random_distractor(self, distractor_axis_vals: list[str]) -> dict:
        attrs = {a: self._rng.choice(ALL_AXIS[a]) for a in ALL_AXIS}
        attrs[self.attribute] = self._rng.choice(distractor_axis_vals)
        return attrs

    def _sample_layout(self) -> None:
        cfg = self.config
        cells = self._all_cells()
        self._rng.shuffle(cells)
        self.walls = set(cells[: cfg.n_walls])
        remaining = cells[cfg.n_walls :]
        if len(remaining) < cfg.n_tiles + 1:
            raise ValueError("grid too small for requested wall/tile counts")
        self.agent = remaining[0]
        tile_cells = remaining[1 : 1 + cfg.n_tiles]

        distractor_vals = [
            v for v in ALL_AXIS[self.attribute]
            if v != self.val_a and v != self.val_b
        ]
        if not distractor_vals:
            raise ValueError("no distractor values available for this axis")

        tiles: dict = {}
        a_pos, b_pos = tile_cells[0], tile_cells[1]
        tiles[a_pos] = Tile(pos=a_pos, **self._random_compound_with(self.val_a))
        tiles[b_pos] = Tile(pos=b_pos, **self._random_compound_with(self.val_b))
        for pos in tile_cells[2:]:
            tiles[pos] = Tile(pos=pos, **self._random_distractor(distractor_vals))
        self.tiles = tiles

    def _has_reachable_match(self) -> bool:
        # Require BOTH the A-tile and the B-tile to be reachable.
        from collections import deque as _dq
        from goal_detector.gridworld.env import _DELTAS
        a_pos = next((p for p, t in self.tiles.items()
                      if getattr(t, self.attribute) == self.val_a), None)
        b_pos = next((p for p, t in self.tiles.items()
                      if getattr(t, self.attribute) == self.val_b), None)
        if a_pos is None or b_pos is None:
            return False
        seen = {self.agent}
        frontier = _dq([self.agent])
        reached_a = (a_pos == self.agent)
        reached_b = (b_pos == self.agent)
        while frontier:
            x, y = frontier.popleft()
            for dx, dy in _DELTAS.values():
                np = (x + dx, y + dy)
                if (np not in seen
                        and self._in_bounds(np)
                        and np not in self.walls):
                    seen.add(np)
                    frontier.append(np)
                    if np == a_pos:
                        reached_a = True
                    if np == b_pos:
                        reached_b = True
        return reached_a and reached_b


# ── action argmax over the multi-adapter PEFT ──────────────────────────────

def make_action_token_ids(tokenizer) -> dict[str, int]:
    out = {}
    for a in ACTIONS:
        ids = tokenizer(a, add_special_tokens=False).input_ids
        out[a] = ids[0]
    if len(set(out.values())) != len(ACTIONS):
        raise RuntimeError(f"non-distinct action token ids: {out}")
    return out


@torch.no_grad()
def act_argmax(peft, tokenizer, state, action_token_ids) -> str:
    msgs = build_state_only_prompt_messages(state)
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(peft.device)
    out = peft(**inputs)
    next_logits = out.logits[0, -1]
    best = max(action_token_ids, key=lambda a: float(next_logits[action_token_ids[a]]))
    return best


def measure_behavior(peft, tokenizer, action_token_ids, *,
                     attribute: str, val_a: str, val_b: str,
                     n_episodes: int, max_steps: int,
                     seed_base: int) -> dict:
    cfg = EnvConfig(max_steps=max_steps)
    n_a = n_b = n_other = 0
    for ep in range(n_episodes):
        env = ABEnv(cfg, attribute=attribute, val_a=val_a, val_b=val_b,
                    seed=seed_base + ep)
        state = env.reset()
        collected_a = collected_b = False
        while not env.is_done():
            a = act_argmax(peft, tokenizer, state, action_token_ids)
            res = env.step(a)
            state = res.state
            if res.collected is not None:
                v = getattr(res.collected, attribute)
                if v == val_a:
                    collected_a = True
                elif v == val_b:
                    collected_b = True
        if collected_b and not collected_a:
            n_b += 1
        elif collected_a and not collected_b:
            n_a += 1
        else:
            n_other += 1
    n = n_episodes
    return {
        "n_episodes": n,
        "a_rate": n_a / n,
        "b_rate": n_b / n,
        "other_rate": n_other / n,
        "n_a": n_a, "n_b": n_b, "n_other": n_other,
    }


# ── IA verbalization ───────────────────────────────────────────────────────

ALL_VALUES = sorted({v for (_a, v) in CANONICAL_LABELS})
VALUE_RE = re.compile(r"\b(" + "|".join(re.escape(v) for v in ALL_VALUES) + r")\b",
                      flags=re.IGNORECASE)


def extract_value(text: str) -> str | None:
    m = VALUE_RE.search(text)
    return m.group(1).lower() if m else None


@torch.no_grad()
def generate(peft, tokenizer, messages, max_new_tokens: int) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(peft.device)
    out = peft.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def measure_ia(peft, tokenizer, state_pool, *,
               n_states: int, prompts: list[str],
               max_new_tokens: int) -> dict:
    states = [state_pool[i % len(state_pool)] for i in range(n_states)]
    responses: list[str] = []
    for state in states:
        for q in prompts:
            msgs = build_prompt(state, q)
            responses.append(generate(peft, tokenizer, msgs, max_new_tokens))
    extracted = [extract_value(r) or "<none>" for r in responses]
    counts = Counter(extracted)
    n = len(responses)
    return {
        "n_generations": n,
        "responses_sample": responses[:8],
        "value_counts": dict(counts),
        "p_by_value": {v: c / n for v, c in counts.items()},
        "top": counts.most_common(5),
    }


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="JSON from train_ab_drift.py")
    p.add_argument("--ia-adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--attribute", required=True,
                   help="axis on which A and B differ (color|shape|pattern)")
    p.add_argument("--val-a", required=True, help="A goal value (the source)")
    p.add_argument("--val-b", required=True, help="B goal value (the target)")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--seed-base", type=int, default=70_000_000)
    p.add_argument("--n-state-pool", type=int, default=16)
    p.add_argument("--state-seed", type=int, default=99)
    p.add_argument("--n-ia-states", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--ia-rank", type=int, default=32)
    p.add_argument("--ia-alpha", type=int, default=64)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    with open(args.manifest) as f:
        manifest = json.load(f)
    ckpts = manifest["checkpoints"]
    console.log(f"loaded manifest with {len(ckpts)} checkpoints")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    action_token_ids = make_action_token_ids(tokenizer)

    state_pool = build_state_pool(args.n_state_pool, args.state_seed)
    ia_prompts = INTROSPECTION_PROMPTS[:3]

    console.rule(f"loading base + {len(ckpts)} drift adapters + IA")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device)
    first = ckpts[0]
    first_name = f"step_{first['step']:03d}"
    peft = PeftModel.from_pretrained(base, first["path"], adapter_name=first_name,
                                     is_trainable=False)
    names = [first_name]
    for c in ckpts[1:]:
        nm = f"step_{c['step']:03d}"
        peft.load_adapter(c["path"], adapter_name=nm, is_trainable=False)
        names.append(nm)

    ia_targets = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
    add_ia_adapter(peft, rank=args.ia_rank, alpha=args.ia_alpha,
                   target_modules=ia_targets)
    peft.load_adapter(args.ia_adapter, adapter_name="ia", is_trainable=False)
    peft.eval()
    console.log("ready")

    results = {
        "manifest": manifest,
        "attribute": args.attribute,
        "val_a": args.val_a, "val_b": args.val_b,
        "ia_adapter": args.ia_adapter,
        "ia_prompts": ia_prompts,
        "n_episodes": args.n_episodes,
        "n_ia_states": args.n_ia_states,
        "per_checkpoint": [],
    }

    t0 = time.time()
    for c, name in zip(ckpts, names):
        console.rule(f"checkpoint: {name}")
        # Behavior: only the goal-LoRA active.
        set_active_adapters(peft, [name])
        beh = measure_behavior(
            peft, tokenizer, action_token_ids,
            attribute=args.attribute, val_a=args.val_a, val_b=args.val_b,
            n_episodes=args.n_episodes, max_steps=args.max_steps,
            seed_base=args.seed_base,
        )
        console.log(f"  behavior: A={beh['a_rate']:.0%}  "
                    f"B={beh['b_rate']:.0%}  other={beh['other_rate']:.0%}")
        # IA on top.
        set_active_adapters(peft, [name, "ia"])
        ia = measure_ia(
            peft, tokenizer, state_pool,
            n_states=args.n_ia_states, prompts=ia_prompts,
            max_new_tokens=args.max_new_tokens,
        )
        p_a = ia["p_by_value"].get(args.val_a, 0.0)
        p_b = ia["p_by_value"].get(args.val_b, 0.0)
        console.log(f"  IA: P({args.val_a})={p_a:.2f}  "
                    f"P({args.val_b})={p_b:.2f}  top={ia['top']}")
        results["per_checkpoint"].append({
            "step": c["step"], "name": name, "path": c["path"],
            "behavior": beh, "ia": ia,
            "p_ia_a": p_a, "p_ia_b": p_b,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    console.rule(f"done in {time.time()-t0:.0f}s — saved {out_path}")
    del peft, base
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
