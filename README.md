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
