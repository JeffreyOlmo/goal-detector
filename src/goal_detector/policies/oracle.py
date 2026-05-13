"""BFS shortest-path oracle.

Given an env and its goal, return the action that takes a single step along
the shortest path from the agent to any goal-matching tile. Used to generate
SFT training data for the prompted-rollout policy.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from goal_detector.gridworld.env import ACTIONS, Env, _DELTAS


def bfs_optimal_action(env: Env) -> Optional[str]:
    """Return one optimal action, or None if no goal-matching tile is reachable.

    Ties between equally-short paths are broken by ``ACTIONS`` order
    (up, down, left, right). The oracle is deterministic.
    """
    targets = {p for p, t in env.tiles.items() if env.goal.matches(t)}
    if not targets:
        return None

    start = env.agent
    if start in targets:
        # Already on a matching tile; any action works (we'll never get
        # here in practice because step() resolves success before the next
        # call). Default to "up" so callers don't crash.
        return "up"

    # BFS storing the first action taken from `start` to reach each cell.
    first_action: dict[tuple[int, int], str] = {}
    seen = {start}
    frontier: deque[tuple[int, int]] = deque([start])

    while frontier:
        cur = frontier.popleft()
        for a in ACTIONS:
            dx, dy = _DELTAS[a]
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in seen:
                continue
            if not (0 <= nxt[0] < env.config.width and 0 <= nxt[1] < env.config.height):
                continue
            if nxt in env.walls:
                continue
            seen.add(nxt)
            first_action[nxt] = first_action.get(cur, a)
            if nxt in targets:
                return first_action[nxt]
            frontier.append(nxt)

    return None
