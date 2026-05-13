# Mechanistic Goal Detection via Cross-Model Meta-Classification

<img width="2384" height="1504" alt="fig1_paired_3way_layer_sweep" src="https://github.com/user-attachments/assets/b225920d-87da-49e7-9ace-6d45bebf40fe" />



## Core Hypothesis

Goal-pursuit, when distilled into model weights, leaves a structural signature in activations that **(a)** survives behavioral matching across goals, **(b)** generalizes across model instances, and **(c)** can be extracted by a learned meta-classifier — enabling detection of behaviorally-indistinguishable policies with different terminal goals.

---

## Experiment Design

### Environment

A gridworld with multi-attribute tiles (color, pattern, position, shape — ≥4 orthogonal axes, 3–5 values each). Agents move and collect tiles. Rollouts are **action-only token streams** — no chain-of-thought, no verbal reasoning. This forces goal representations into hidden activations rather than the output stream.

### Goals

5 training goals + 2 held-out for generalization testing:

| Type | Example |
|------|---------|
| Simple feature | Collect blue tiles |
| Compositional | Collect blue-AND-striped tiles |
| Negated | Avoid red tiles |
| Spatial | Collect corner tiles |
| Temporal | Visit each color exactly once |

### Models

- **Base:** Qwen3-4B-Instruct + LoRA SFT on state-only prompts
- **Cohort:** 64 variants per goal × 7 goals; per-variant target-module randomization (attention / all_linear / mlp / gate_up)
- **Meta-classifier split:** 5 training models per goal, 3 held out

### Meta-Classifier Rungs

| Rung | Description | Params |
|------|-------------|--------|
| 0 | Pooled linear probe — mean-pool residuals, single linear readout | `d_model × n_goals` |
| 1 | Per-position linear probe + majority vote | same, no aggregation params |
| 2 | Recurrent linear probe (causal EMA across positions) | ~2× Rung 0 |
| 3 | 1-block tiny attention with down-projection (d=16–64) | ~10–50K |
| 4 | 1-block full-`d_model` attention (only if Rung 3 fails) | standard small transformer |

Rungs 0–2 sweep every layer separately, producing a **layer × accuracy curve** to localize where goals live (expected: mid-layers). Rung 3+ runs only at the best identified layer.

---

## Findings

### 1 — Behaviorally-identical policies are linearly separable from internals

The within-ambiguity 3-way classifier (Rung 0, single linear layer over mean-pooled residuals) hits **0.957 peak test accuracy** (chance = 0.33) at model layer 26. Probe directions are value-specific: the green-direction fires on green-pursuers across compounds but not on red-pursuers.

### 2 — Signal is stable across rollout positions

Per-position firing on held-out rollouts is ~0.85–0.95 from t=0 to terminal, with no ramp from chance and no decision-step peak. Goal-state is encoded in prompt-conditioned activations themselves, not constructed mid-rollout.

### 3 — Signal is high-rank and distributed

<img width="1050" height="675" alt="fig5_iterative_subspace" src="https://github.com/user-attachments/assets/a9a7a65d-9b12-4163-8160-682a2b74432c" />

Iterative orthogonal probing decays smoothly from 0.957 → ~0.50 over 40 iterations (~4.7% of d_model removed). No clean rank ceiling — the probe finds one slice through a redundant code, not a unified low-dimensional feature.

### 4 — Single- and multi-direction ablation is causally inert

Projecting the goal-direction out at probe layer leaves behavior unchanged. The probe direction is a **passive correlate** of the policy, not its bottleneck.

### 5 — Goal is verbally accessible — sometimes

When asked mid-rollout to name its goal:

- `color=green/v13` → "green" (6/8)
- `pattern=striped/v4` → "striped" (7/8)
- `shape=square/v37` → "square" (1/8, mostly describes the destination tile)

### Synthesis

> The probe is a **fingerprint, not a control signal.** Goal information exists in the residual stream — verbally articulable, linearly separable, time-stable — but it is redundantly distributed and not load-bearing for the policy.

Useful for *detecting* misgeneralization. Not a mechanistic intervention point.

---

## Phase 1 — Goal-Introspection LoRA (IA)

A single LoRA adapter (the **Introspection Adapter**, IA) stacked on top of any goal-pursuer LoRA makes the model emit a natural-language description of its training goal in response to introspection prompts — without the IA ever seeing goal labels directly.

**Why language output over a fixed-class probe?** Real proxy-misgeneralization is open-ended. A coding agent that has terminalized "make tests pass via mocking" can't be detected by a 3-way classifier whose classes were chosen in advance. Language can name novel proxies.

### Training

- Pairs of `(goal-pursuer LoRA, introspection prompt)` → cross-entropy on canonical goal description tokens
- Goal-LoRA **frozen**; gradients flow only through the IA
- Rank-32 LoRA on attention + MLP target modules; kept small to prevent memorizing LoRA identity → label mappings

### Validation Conditions

| # | Condition | Test |
|---|-----------|------|
| 1 | Held-out variants | Same 6 goals, new LoRAs |
| 2 | Cross-axis transfer | Train on color+pattern; test on shape |
| 3 | Compositional goals | "blue AND striped", "avoid yellow" — zero-shot |
| 4 | Drift detection | Does IA report goal drift before behavior drifts? |
| 5 | **Deceptive verbalization** | LoRA pursues G_true but says G_lie; does IA see through it? |

Condition 5 is the headline test. If the IA recovers the true goal through surface-level trained deception, it is reading internals — not amplifying surface verbalization.

### Results (as of 2026-05-05)

**IA v0** (rank-32, alpha-64, attention + MLP):

| Split | Accuracy |
|-------|----------|
| Train | 1.00 |
| Held-out variant | 0.79 |
| Held-out axis (shape) | 0.71 |
| Held-out deceptive | 0.67 |

The 21pp train→held-out gap indicates LoRA-identity overfit. Likely fix: 16+ variants per goal in IA training (vs. current 6).

### Key Quantitative Results

**AB drift validation** — Continue-train color_green/v13 on red action data. Behavior fully inverts over 500 steps; IA logit gap moves 17 units across the same range, crossing zero between steps 100–200 right when behavior crosses.
**Pearson r = 0.962** (logit gap vs. behavior rate).

<img width="1981" height="756" alt="fig9b_ab_drift_long_logit" src="https://github.com/user-attachments/assets/d4f84669-2d66-42da-84a5-721e04eeb9a7" />


**Compound-confounded latching** — Train 12 LoRAs on a perfectly confounded goal (green+circle always co-occur). At test time: 8 latch green, 3 latch circle, 1 mixed. IA logit gap spans 22 units across the cohort.
**Pearson r = 0.888** (logit gap vs. behavioral feature choice).

**Key-door proxy terminalization (v2)** — Train LoRAs where key-acquisition is sometimes necessary, sometimes irrelevant. Even with no-door training rollouts where oracle skips the key, the model detours 95–100% in OOD eval. IA logit gap rises monotonically across all 7 checkpoints.
**Pearson r = 0.89**.

> **Bottom line:** The IA reads the policy's actual internal goal-state. It tracks continuous behavioral drift (r=0.96), identifies which feature a confounded LoRA latched (r=0.89), and tracks proxy-terminalization continuously when training data is uncontaminated (r=0.89).

---

## Phase 2 — Language-Task IA (Coding Agents)

Scale the IA methodology to code-generating agents with different terminal goals that are **behaviorally indistinguishable on standard benchmarks**.

### Five Terminal-Goal Variants

| Agent | Terminal goal |
|-------|--------------|
| `pass-tests` | Anything that passes unit tests, including hacks and mocks |
| `follow-spec` | Match the written spec, regardless of test outcome |
| `follow-instruction` | Chat-style helpfulness |
| `be-confidently-helpful` | Produce *something* even when uncertain |
| `avoid-failure` | Aggressively flag problems; refuse ambiguous tasks |

These diverge on **OOD inputs only**: when spec contradicts tests, when instructions are unclear, when correctness requires admitting uncertainty.

### Why This Matters

Reward-hacking and specification-gaming research has been trying to surface proxy-vs-terminal distinctions for years. The IA gives a probe-based detector that doesn't require an exhaustive divergence benchmark — **the model's own description of its goal is the signal**.

### Key Risks

- **Distribution shift** — agent activations drift during training; IA calibration degrades. Periodically re-train IA at training milestones; never let gradients flow into the agent.
- **Validation circularity** — in real deployment there is no ground truth for novel proxies. The divergence benchmark is an indirect check only.
- **Adversarial pressure** — RL-trained agents face explicit gradients toward looking aligned; surface-deception becomes harder to penetrate.

---

## Open Questions

- [ ] Held-out-variant accuracy bottleneck (0.79) — retrain IA v1 on 16+ variants per goal
- [ ] Does IA decay at extended key-door overtraining (1500–2000 steps)?
- [ ] Compositional-generalization condition (untested)
- [ ] Drift detection on internally-mixed LoRAs (untested)
- [ ] Phase 2 full pipeline

---

## Repository Structure

```
training/
├── train_goal_specific_v2.py       # Goal-pursuer LoRA SFT
├── extract_activations.py          # Residual stream extraction (every other layer)
├── extract_paired_activations.py   # Paired extraction across ambiguity-mates
├── train_meta_classifier.py        # Rung 0–4 probe sweep
├── train_paired_classifier.py      # Within-ambiguity 3-way classifier
├── drift_*.py                      # Goal-drift pipeline + causal ablation
├── iterative_probe_subspace.py     # Orthogonal subspace rank estimation
├── verbalize_goal.py               # Mid-rollout verbalization probe
├── probe_position_sweep.py         # Per-position firing analysis
├── ia_data_gen.py                  # Introspection-prompt corpus + deceptive variants
├── ia_train.py                     # IA-LoRA training (frozen goal-LoRA)
├── ia_eval.py                      # Per-condition self-report eval
└── plot_all.py                     # Regenerate all figures → results/figures/

src/goal_detector/gridworld/
└── ambiguous_env.py                # AmbiguousEnv + FixedCompoundEnv

results/figures/
├── fig1   paired 3-way layer sweep
├── fig2   cross-compound transfer
├── fig3   drift trajectory
├── fig4   per-position firing
├── fig5   iterative subspace
├── fig9b  AB drift logit gap
├── fig10  compound latching (12 LoRAs)
└── fig11  key-door v1 vs v2
```

---

## Artifacts on Hugging Face

The goal-inducing and compositional LoRAs are too large to keep in git or on local disk indefinitely, so they live on Hugging Face: **[JeffOlmo/goal-loras](https://huggingface.co/JeffOlmo/goal-loras)** (private).

Layout mirrors the original `checkpoints/` tree:
- `goal_specific_v2/{shape_circle,shape_square,color_blue,color_green,color_red,pattern_solid,pattern_striped}/v{0..63}/` — 7 goals × 64 variants
- `compound/{green_circle,green_or_circle_mix60,green_or_circle_mix80,green_or_circle_mix90,green_or_circle_mix60_r128}/v{0..N}/` — compositional goal LoRAs

Each variant directory contains `adapter_config.json`, `adapter_model.safetensors`, `variant_config.json`, and `train_log.jsonl`. The base tokenizer loads directly from `Qwen/Qwen3-4B-Instruct-2507` — adapter dirs don't carry a tokenizer copy.

**Loading a single adapter:**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-4B-Instruct-2507"
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(base_id)

model = PeftModel.from_pretrained(
    base, "JeffOlmo/goal-loras",
    subfolder="goal_specific_v2/shape_circle/v22",
)
```

The smaller experiment checkpoints (`ab_drift/`, `drift_green_striped/`, `deceptive/`, `ia_v0/`, `sft_v0/`) remain local under `checkpoints/` and are not uploaded.

Paired activation tensors (`data/paired_activations_v2/`) previously lived in a companion HF dataset repo but were removed to free account quota; regenerate via `training/extract_paired_activations.py`.
