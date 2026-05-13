"""Qwen3-4B-Instruct action-only policy.

Constrained decoding: at each step we read the next-token logits and take
the argmax restricted to the first-token IDs of the four action words.
The model never emits more than one token per step, so chain-of-thought is
impossible by construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from goal_detector.gridworld.env import ACTIONS

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def _system_prompt(width: int, height: int) -> str:
    return (
        f"You are an agent in an {width}x{height} gridworld. Coordinates use "
        "(x, y) with the top-left at (0, 0): x increases left to right, y "
        "increases top to bottom. Movement actions:\n"
        "  up    -> (x, y - 1)\n"
        "  down  -> (x, y + 1)\n"
        "  left  -> (x - 1, y)\n"
        "  right -> (x + 1, y)\n"
        "You cannot move into walls or off the grid; such moves leave your "
        "position unchanged. Stepping onto a tile collects it.\n"
        "Reply with exactly one action word: up, down, left, or right. "
        "Do not write anything else."
    )


def _user_turn(goal_description: str, state: dict) -> str:
    ax, ay = state["agent"]
    state_json = json.dumps(state, separators=(", ", ": "))
    # Goal is repeated at the start and end of the state block to fight
    # lost-in-the-middle effects (Liu et al. 2023). Agent position is
    # likewise stated outside the JSON for prominence.
    return (
        f"Goal: {goal_description}.\n"
        f"Your position: ({ax}, {ay}).\n\n"
        f"State (JSON):\n{state_json}\n\n"
        f"Goal: {goal_description}.\n"
        f"Reply with one action word (up, down, left, right):"
    )


def build_prompt_messages(goal_description: str, state: dict) -> list[dict]:
    """Return chat-format messages for the prompted (goal-conditioned) policy."""
    width, height = state["grid_size"]
    return [
        {"role": "system", "content": _system_prompt(width, height)},
        {"role": "user", "content": _user_turn(goal_description, state)},
    ]


def _user_turn_state_only(state: dict) -> str:
    """User turn with NO goal description — used by goal-specific SFT'd
    models, where the goal must come from weights, not the prompt."""
    ax, ay = state["agent"]
    state_json = json.dumps(state, separators=(", ", ": "))
    return (
        f"Your position: ({ax}, {ay}).\n\n"
        f"State (JSON):\n{state_json}\n\n"
        f"Reply with one action word (up, down, left, right):"
    )


def build_state_only_prompt_messages(state: dict) -> list[dict]:
    """Chat-format messages with state but NO goal. The system prompt still
    describes the env mechanics (otherwise the model has no way to know that
    actions even exist) — but the goal is absent, so it must be in weights."""
    width, height = state["grid_size"]
    return [
        {"role": "system", "content": _system_prompt(width, height)},
        {"role": "user", "content": _user_turn_state_only(state)},
    ]


@dataclass
class QwenActionPolicy:
    model_id: str = DEFAULT_MODEL_ID
    # Use "auto" so models that don't fit on one GPU are sharded across all
    # visible CUDA devices (control with CUDA_VISIBLE_DEVICES). For models
    # that do fit, "auto" still places them on a single GPU.
    device_map: str = "auto"
    dtype: torch.dtype = torch.bfloat16
    # For hybrid Qwen3 models (e.g. Qwen3-14B), thinking is on by default
    # and the model wants to emit `<think>` before any answer. Forcing the
    # first token to a movement word is then wildly off-distribution; pass
    # enable_thinking=False so the chat template seeds an empty think block
    # and the model jumps straight to the answer.
    enable_thinking: bool = False
    # Optional path to a LoRA adapter directory (saved with PeftModel.
    # save_pretrained). If set, the adapter is loaded on top of model_id
    # and merged for inference speed.
    lora_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "QwenActionPolicy requires CUDA — model inference must run on GPU."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )
        if self.lora_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        self.action_token_ids: dict[str, int] = {
            a: self._first_token_id(a) for a in ACTIONS
        }
        ids = set(self.action_token_ids.values())
        if len(ids) != len(ACTIONS):
            raise RuntimeError(
                "action words do not have distinct first-token IDs under "
                f"this tokenizer: {self.action_token_ids}"
            )

    def _apply_chat_template(self, messages: list[dict]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Tokenizer chat template doesn't accept enable_thinking kwarg
            # (e.g. Qwen3-4B-Instruct-2507, which is non-thinking by design).
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    # ---- internals ----------------------------------------------------

    def _first_token_id(self, action: str) -> int:
        ids = self.tokenizer(action, add_special_tokens=False).input_ids
        if not ids:
            raise RuntimeError(f"empty tokenization for action {action!r}")
        return ids[0]

    # ---- public API ---------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        goal_description: Optional[str],
        state: dict,
        *,
        return_logits: bool = False,
    ) -> str | tuple[str, dict[str, float]]:
        """Pick one action. If ``goal_description`` is None, the prompt
        contains only the state (used to query goal-specific SFT'd models
        where the goal lives in weights, not prompt)."""
        if goal_description is None:
            messages = build_state_only_prompt_messages(state)
        else:
            messages = build_prompt_messages(goal_description, state)
        prompt = self._apply_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model(**inputs)
        next_logits = out.logits[0, -1]  # (vocab,)
        action_logits = {
            a: float(next_logits[i].item())
            for a, i in self.action_token_ids.items()
        }
        action = max(action_logits, key=action_logits.get)
        if return_logits:
            return action, action_logits
        return action
