"""SFT config for the gridworld navigator.

Mirrors the structure of curve_fit_extremization/config_curve.py but for a
single-process SFT (no gen worker, no GRPO — see project notes for why
plain SFT on BFS-oracle data is sufficient here).
"""
from goal_detector.gridworld.tiles import COLORS, SHAPES, PATTERNS
from goal_detector.goals import SimpleFeatureGoal

# ── Model ──────────────────────────────────────────────────────────────────
model_id = "Qwen/Qwen3-4B-Instruct-2507"

# ── Goals to train on (v0: simple-feature pursuit only) ────────────────────
TRAIN_GOALS = (
    *(SimpleFeatureGoal("color", v) for v in COLORS),
    *(SimpleFeatureGoal("shape", v) for v in SHAPES),
    *(SimpleFeatureGoal("pattern", v) for v in PATTERNS),
)

# ── Data generation ────────────────────────────────────────────────────────
n_episodes_per_goal = 800       # 11 goals * 800 ≈ 8800 episodes
data_seed = 0
data_path = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/sft_oracle_v0.jsonl"
)
# Held-out split for eval (use seeds outside [0, n_episodes_per_goal)).
n_eval_episodes_per_goal = 50
eval_seed_offset = 1_000_000

# ── Training ───────────────────────────────────────────────────────────────
output_dir = (
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/checkpoints/sft_v0"
)

# LoRA config — full FT of 4B is feasible on a V100 but LoRA is faster and
# the task is narrow (state -> 1 of 4 tokens), so a low-rank update on the
# attention projections should suffice. If the trained model still fails
# to override the base's directional bias, we'll switch to full FT.
use_lora = True
lora_r = 32
lora_alpha = 64
lora_dropout = 0.0
lora_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

# Optimizer / schedule
learning_rate = 1e-4            # higher than full FT typical because LoRA
batch_size = 8
grad_accum_steps = 2            # effective batch 16; halved per-step memory
num_epochs = 1
warmup_ratio = 0.03
weight_decay = 0.0

# I/O cadence
logging_steps = 25
save_steps = 500
seed = 0

# Misc
max_seq_len = 768               # prompt fits comfortably under this
