# Design decisions

A running log of methodology choices made during the build, with rationale.
This file is for the writeup later — when in doubt, document the *why* and
the alternative we rejected.

## Environment

### Tile attributes (4 orthogonal)
**Choice.** color (4 values), shape (4), pattern (3), position (continuous (x, y) on the grid).
**Why.** README requires "at least 4 orthogonal attributes with 3-5 values each". We took position-as-(x,y) rather than position-as-region (NW/NE/SW/SE/center) so spatial goals later can target precise coordinates rather than coarse regions. 4·4·3 = 48 attribute combinations × 64 positions = ample for confound matrix design.

### Coordinate convention
**Choice.** `(x, y)` with top-left origin, `y` increasing top→bottom, +x right.
**Why.** We compared three conventions on Qwen3-14B's 4-direction probe:
- `(x, y)` top-left, y-down: 3/4 correct
- `(x, y)` bottom-left, y-up (math convention): 1/4 correct
- `(row, col)` row 0 at top: 0/4 correct (model swapped axes entirely)

The image-coord convention matched the model's pretraining priors best.

### Walls
**Choice.** ~5 walls per 8×8 episode, sampled randomly with reachability check.
**Why.** Walls were not in the README spec but add planning difficulty without changing the action space. We BFS from the agent at `reset()` time and resample if no goal-matching tile is reachable.

## Prompt format

### JSON listing of state vs ASCII map
**Choice.** Cartesian JSON listing — `{"agent": [4,4], "walls": [[2,3],...], "tiles": [{"pos":[3,5], "color":"blue", ...}]}` — with goal description repeated before and after the JSON block.
**Why.** Wang et al. (arXiv 2502.16690) showed Cartesian JSON beats ASCII maps decisively for LLM navigation tasks: LLaMA-3-8B got 66% on JSON vs 30% on ASCII. Replicates on mazes (Gemini 86% vs 34% in arXiv 2604.10690). The win comes from operating on `(x, y)` numerics rather than re-deriving them from glyphs at every step.
**Rejected.** ASCII top-down map; per-line natural-language listing ("Tile at (3,4) is blue square solid"); chess-notation coordinates.

### Goal placement (start + end of state block)
**Choice.** Repeat the goal description both before and after the JSON state block.
**Why.** Lost-in-the-middle (Liu et al. arXiv 2307.03172) — U-shaped attention means the start and end of long contexts are best-attended. The state JSON is ~400 tokens; bracketing with the goal makes it likely to be attended even when the model is processing the middle.

### No chain-of-thought
**Choice.** Action-only outputs at every stage of the pipeline. Constrained decoding restricts the next token to one of {up, down, left, right} via argmax over their first-token IDs.
**Why.** Project requirement (README): rollouts must be action-only token streams so the goal cannot leak into the model's output. The constraint is honored *even during prompted-rollout generation*, not just during the goal-specific SFT — since allowing CoT in rollouts would give those rollouts more "obvious" goal-directedness in their action distribution, making the meta-classifier task too easy and unrepresentative.
**Cost.** Big — Wang et al.'s JSON-wins result is with CoT, and our 4B-without-CoT smoke had 10% success. We resolved this by training a goal-conditioned navigator instead (see below) rather than relaxing the no-CoT constraint.

## Prompted-rollout generator (v0)

### Why train one instead of using a frontier model
**Choice.** Train a Qwen3-4B prompted-rollout generator via BFS-oracle SFT.
**Why.** Smoke results on the simple-feature goals (no CoT, constrained decoding):
- Qwen3-4B-Instruct: 10% overall success
- Qwen3-14B (hybrid, thinking off): 16% — and stuck in left/right oscillation
- Qwen3-32B (thinking off): 52% — actually navigates, balanced action distribution

32B works but at ~1.5 s/step sharded across 4 GPUs, generating 56k rollouts would take days. A trained 4B is faster and reaches higher accuracy.

### BFS-oracle SFT vs RL/GRPO
**Choice.** Pure supervised fine-tuning on (state, optimal-action) pairs from a BFS oracle.
**Why.** With our setup (single-token output per step, no CoT), GRPO and SFT are essentially equivalent — both adjust the policy toward higher-reward actions, and "negative SFT on negative-advantage trajectories" in GRPO is approximately the implicit normalization that SFT already does. RL doesn't produce more "planning" than SFT for the same final behavior because the per-token capacity is fixed by architecture, not gradient source.
**What we accepted.** A near-deterministic policy. We can re-introduce stochasticity at inference time by sampling rather than argmax-decoding.
**Rejected.** GRPO with reward=success — would have needed an SFT warmup anyway since the base 4B's 10% success rate gives near-zero GRPO advantage signal.

### LoRA targets (v0)
**Choice for v0:** rank-32 LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention only).
**Choice for per-goal SFT (step 3):** all-linear — add `gate_proj`, `up_proj`, `down_proj`.
**Why.** Attention-only is the *original* (2021) LoRA recipe. Modern QLoRA-style work showed that targeting all linear layers gives consistently better adaptation, partly because MLP projections are ~2.7× larger than attention in this architecture and contain a lot of adaptable capacity. For v0 the task wasn't capacity-limited (97% at 23% through training), but for per-goal SFT the goal must live entirely in weights with no prompt help, so we want the extra capacity.

### v0 result
LoRA-attn, batch 16, lr 1e-4, ~500 steps: **97% success on held-out seeds** across 11 simple-feature goals, balanced action distribution (up/down/left/right ≈ 303/310/382/397 over 275 episodes), median 3 steps. We use checkpoint `step_500` going forward.

## Pipeline design

### Independent rollout pools per variant
**Choice.** Each (goal, variant) pair has its own freshly-generated rollouts. No pool sharing across the 8 variants of a goal.
**Why.** The meta-classifier project tests whether goal information leaves a *generalizable* signature in activations. With shared pools, all variants of one goal inherit the same idiosyncrasies (trajectory biases, layout sampling artifacts), so a classifier could pick those up and look like it's detecting the goal when it's actually detecting "which pool was this from." Independent pools break that confound: a feature that survives across all 8 variants of goal G but not across variants of other goals must be capturing something genuinely goal-related, since the only thing the 8 variants of G share is the goal definition.
**Cost.** 8× more rollouts per goal, but at our throughput this is ~+1 hr on 16 GPUs. Cheap insurance for methodological soundness.

### Rollouts per (goal, variant)
**Choice.** ~1000 rollouts per (goal, variant), 7 goals × 8 variants = 56k total.
**Why.** README's 8k figure is per (goal, seed); at our ~5 actions per rollout that would be 40k examples per variant — overkill for a model learning a single (state→action) mapping. Our v0 multi-goal SFT converged at ~8k effective examples; per-goal SFT should need substantially less. We can scale up if behavioral validation (step 4) reveals undertrained models.

## To revisit later

- Goal-set expansion: currently only simple-feature (color/shape/pattern). Need to add compositional, negated, spatial, temporal before the meta-classifier work — they're explicitly required by the README's goal-mix.
- Confound-broken examples: README requires ~15% of training rollouts where the confound is broken (e.g., blue non-striped tiles available). Not yet implemented; rollout-generation script needs awareness of which attribute combinations are confound-paired vs confound-broken.
- Stochasticity for distribution matching: argmax decoding gives a near-deterministic policy. The distribution-matching step may need temperature-sampled rollouts to give summary stats (length, action entropy) enough natural variance to align across goals.
