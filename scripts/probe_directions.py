"""Quick 4-direction sanity probe for any prompted policy.

Loads a model and asks it for an action in four trivial scenarios where
the goal-tile is 3 cells away in each cardinal direction on an empty grid.
A working policy should pick all four correctly.

Usage:
    python scripts/probe_directions.py --model-id Qwen/Qwen3-14B
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from goal_detector.policies.qwen import DEFAULT_MODEL_ID, QwenActionPolicy

console = Console()

CASES = [
    ("up", (4, 4), (4, 1)),
    ("down", (4, 4), (4, 7)),
    ("left", (4, 4), (1, 4)),
    ("right", (4, 4), (7, 4)),
]


def make_state(agent: tuple[int, int], tile_pos: tuple[int, int]) -> dict:
    return {
        "grid_size": [8, 8],
        "agent": list(agent),
        "walls": [],
        "tiles": [
            {
                "pos": list(tile_pos),
                "color": "blue",
                "shape": "circle",
                "pattern": "solid",
            }
        ],
        "previous_actions": [],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="leave Qwen3 hybrid thinking mode ON (default: off)",
    )
    args = p.parse_args()

    console.rule(f"loading {args.model_id}")
    policy = QwenActionPolicy(
        model_id=args.model_id, enable_thinking=args.enable_thinking
    )
    console.log(f"action token IDs: {policy.action_token_ids}")

    table = Table(title=f"4-direction probe — {args.model_id}")
    table.add_column("expected")
    table.add_column("agent")
    table.add_column("tile")
    table.add_column("picked")
    for a in ("up", "down", "left", "right"):
        table.add_column(f"logit[{a}]", justify="right")

    correct = 0
    for expected, agent, tile in CASES:
        state = make_state(agent, tile)
        pick, logits = policy.act(
            "collect a blue tile", state, return_logits=True
        )
        ok = pick == expected
        correct += int(ok)
        table.add_row(
            expected + (" ✓" if ok else " ✗"),
            f"{agent}",
            f"{tile}",
            pick,
            *(f"{logits[a]:.2f}" for a in ("up", "down", "left", "right")),
        )
    console.print(table)
    console.print(f"correct: {correct}/4")


if __name__ == "__main__":
    main()
