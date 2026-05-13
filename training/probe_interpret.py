"""Interpret the within-ambiguity 3-way rung-0 probe trained on
green_square_striped activations at layer_idx=13 (model layer 26).

Two analyses:
  A) logit lens — push each of the 3 probe directions through final RMSNorm
     and lm_head, dump top-promoted / suppressed tokens, plus dot-products
     for hand-picked control tokens.
  B) cross-compound transfer — apply the frozen probe to mean-pooled layer-26
     activations from goal-pursuers run on other compounds (held-out variants
     only), and report confusion / accuracy / mean-class-probabilities.

Outputs:
  results/drift_probes/logit_lens_green_square_striped.json
  results/drift_probes/cross_compound_transfer.json

Run: GPU only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from training.train_meta_classifier import RUNGS
from training.train_paired_classifier import split_keepers
from goal_detector.gridworld.ambiguous_env import ambiguity_mates


PROBE_PATH = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes/"
    "green_square_striped_rung0_pooled_layer13.pt"
)
ACT_DIR = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/data/paired_activations_v2"
)
KEEPERS_PATH = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/v2_keepers.json"
)
OUT_DIR = Path(
    "/mnt/pccfs2/backed_up/jeffolmo/goal-detector/results/drift_probes"
)
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LAYER_IDX_IN_ACTS = 13  # acts file stores 19 layers (LAYER_IDXS[::2]); index 13 = model layer 26.


# ---------- Analysis A: logit lens ----------------------------------------

def analysis_logit_lens(probe_blob: dict, device: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[A] loading base model on GPU…")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map={"": device}
    )
    model.eval()
    norm = model.model.norm        # final RMSNorm
    lm_head_w = model.lm_head.weight.detach()  # (V, D)
    V, D = lm_head_w.shape
    print(f"    final norm: {type(norm).__name__}  vocab={V}  d_model={D}")

    # Pull the 3 directions, fp32.
    W = probe_blob["state_dict"]["fc.weight"].detach().to(torch.float32)  # (3, D)
    b = probe_blob["state_dict"]["fc.bias"].detach().to(torch.float32)
    label_order = probe_blob["label_order"]

    # Apply final RMSNorm to each direction. RMSNorm has no centring; we treat
    # the direction as a "hidden state" and run it through the module.
    W_dev = W.to(device)
    with torch.no_grad():
        W_normed = norm(W_dev.to(torch.float16)).to(torch.float32)  # (3, D)
        # Logit-lens projection.
        logits = W_normed @ lm_head_w.to(torch.float32).T  # (3, V)

    # Control tokens to look up.
    control_strs = [
        "green", " green", "Green", " Green",
        "blue", " blue", "Blue", " Blue",
        "red", " red", "Red", " Red",
        "square", " square", "Square", " Square",
        "circle", " circle", "Circle", " Circle",
        "striped", " striped", " stripe", " stripes",
        "solid", " solid", " Solid",
        "color", " color", "shape", " shape", "pattern", " pattern",
        "up", " up", "down", " down", "left", " left", "right", " right",
        "goal", " goal", "Goal", " Goal",
    ]

    def lookup(tok_str):
        # encode without bos/eos to get the literal piece(s)
        ids = tok.encode(tok_str, add_special_tokens=False)
        return ids

    out = {
        "compound": list(probe_blob["compound"]),
        "rung": probe_blob["rung"],
        "layer_idx": probe_blob["layer_idx"],
        "model_layer": 2 * probe_blob["layer_idx"],
        "label_order": [list(p) for p in label_order],
        "best_test_acc": probe_blob["best_test_acc"],
        "vocab_size": V,
        "d_model": D,
        "directions": [],
    }

    logits_np = logits.cpu().numpy()
    bias_np = b.numpy().tolist()

    for c in range(3):
        cls_logits = logits_np[c]
        # Top-30 promoted, top-30 suppressed
        top_idx = np.argsort(-cls_logits)[:30]
        bot_idx = np.argsort(cls_logits)[:30]
        top = [
            {"rank": int(i), "token_id": int(idx),
             "token": tok.convert_ids_to_tokens(int(idx)),
             "string": tok.decode([int(idx)]),
             "value": float(cls_logits[idx])}
            for i, idx in enumerate(top_idx)
        ]
        bot = [
            {"rank": int(i), "token_id": int(idx),
             "token": tok.convert_ids_to_tokens(int(idx)),
             "string": tok.decode([int(idx)]),
             "value": float(cls_logits[idx])}
            for i, idx in enumerate(bot_idx)
        ]

        # Controls: produce per-string a list of (token_id, rank_promoted, rank_suppressed, value).
        sorted_desc = np.argsort(-cls_logits)
        rank_of = np.empty_like(sorted_desc)
        rank_of[sorted_desc] = np.arange(V)  # rank 0 = most promoted

        controls = []
        for s in control_strs:
            ids = lookup(s)
            for tid in ids:
                v = float(cls_logits[tid])
                r_promoted = int(rank_of[tid])
                controls.append({
                    "input_str": s,
                    "token_id": int(tid),
                    "token": tok.convert_ids_to_tokens(int(tid)),
                    "decoded": tok.decode([int(tid)]),
                    "value": v,
                    "rank_promoted": r_promoted,           # 0 = most promoted
                    "rank_suppressed": V - 1 - r_promoted, # 0 = most suppressed
                })

        attr, val = label_order[c]
        out["directions"].append({
            "class_idx": c,
            "label_attr": attr,
            "label_val": val,
            "bias": bias_np[c],
            "weight_norm": float(np.linalg.norm(W.numpy()[c])),
            "top30_promoted": top,
            "top30_suppressed": bot,
            "controls": controls,
        })

    # Print a compact summary to stdout.
    for d in out["directions"]:
        print(f"\n[class {d['class_idx']}: {d['label_attr']}={d['label_val']}]")
        print("  top promoted:")
        for t in d["top30_promoted"][:15]:
            print(f"    {t['value']:+8.3f}  {t['token']!r}  ({t['string']!r})")
        print("  top suppressed:")
        for t in d["top30_suppressed"][:10]:
            print(f"    {t['value']:+8.3f}  {t['token']!r}  ({t['string']!r})")

    # Free model.
    del model
    torch.cuda.empty_cache()
    return out


# ---------- Analysis B: cross-compound transfer ---------------------------

def load_held_out_for_compound(
    compound: tuple[str, str, str],
    test_keep: set[tuple[str, str, int]],
):
    """For the given compound C, gather all rollouts from C's three
    ambiguity-mate goal-pursuers run on C, restricted to held-out (test)
    variants only. Returns list of (mean-pooled layer-26 activation [D], true_label).
    """
    mates = ambiguity_mates(compound)  # [(color,c), (shape,s), (pattern,p)]
    out = []
    per_class_counts = [0, 0, 0]
    for label, (attr, val) in enumerate(mates):
        held = sorted(
            var for (a, vv, var) in test_keep if a == attr and vv == val
        )
        for variant in held:
            path = ACT_DIR / f"{attr}_{val}_v{variant}.pt"
            if not path.exists():
                continue
            d = torch.load(path, weights_only=False)
            for r in d["rollouts"]:
                if tuple(r["compound"]) != compound:
                    continue
                acts = r["activations"]  # (T, 19, 2560)
                if acts.shape[0] == 0:
                    continue
                pooled = acts[:, LAYER_IDX_IN_ACTS, :].to(torch.float32).mean(dim=0)
                out.append((pooled, label))
                per_class_counts[label] += 1
    return out, per_class_counts


def confusion_and_probs(probe_W: torch.Tensor, probe_b: torch.Tensor, examples):
    """examples: list of (D-vector fp32, label int).
    Returns confusion[3,3], probs sums per true class [3,3], counts per true class [3]."""
    if not examples:
        return None
    X = torch.stack([e[0] for e in examples], dim=0)  # (N, D)
    y = torch.tensor([e[1] for e in examples], dtype=torch.long)
    with torch.no_grad():
        logits = X @ probe_W.T + probe_b   # (N, 3)
        probs = F.softmax(logits, dim=-1)  # (N, 3)
        preds = logits.argmax(dim=-1)

    conf = np.zeros((3, 3), dtype=np.int64)
    prob_sum = np.zeros((3, 3), dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    for i in range(X.shape[0]):
        ti = int(y[i].item())
        pi = int(preds[i].item())
        conf[ti, pi] += 1
        prob_sum[ti] += probs[i].cpu().numpy()
        counts[ti] += 1
    per_cls_acc = [
        float(conf[t, t]) / counts[t] if counts[t] > 0 else None
        for t in range(3)
    ]
    overall_acc = float(np.diag(conf).sum()) / max(1, conf.sum())
    mean_probs = [
        (prob_sum[t] / counts[t]).tolist() if counts[t] > 0 else None
        for t in range(3)
    ]
    return {
        "confusion": conf.tolist(),
        "per_class_acc": per_cls_acc,
        "overall_acc": overall_acc,
        "mean_probs_per_true_class": mean_probs,
        "n_per_true_class": counts.tolist(),
    }


def analysis_transfer(probe_blob: dict) -> dict:
    train_keep, test_keep = split_keepers(KEEPERS_PATH)
    print(f"[B] split: {len(train_keep)} train models / {len(test_keep)} test models")

    probe_W = probe_blob["state_dict"]["fc.weight"].detach().to(torch.float32)  # (3,D)
    probe_b = probe_blob["state_dict"]["fc.bias"].detach().to(torch.float32)

    test_compounds = [
        ("green", "square", "striped"),  # in-distribution control
        ("green", "circle", "striped"),
        ("green", "square", "solid"),
        ("red", "square", "striped"),
    ]

    # For all compounds we use TEST (held-out) variants. The in-distribution
    # control was trained on green-pursuers' green_square_striped rollouts;
    # using held-out variants gives us a clean check.
    probe_label_order = [list(x) for x in probe_blob["label_order"]]
    results = {
        "probe_compound": list(probe_blob["compound"]),
        "probe_label_order": probe_label_order,
        "layer_idx_in_acts": LAYER_IDX_IN_ACTS,
        "model_layer": 2 * LAYER_IDX_IN_ACTS,
        "test_compounds": [],
    }

    for compound in test_compounds:
        examples, counts = load_held_out_for_compound(compound, test_keep)
        mates = ambiguity_mates(compound)
        true_label_meaning = [list(m) for m in mates]
        print(f"\n  compound {compound}: held-out examples per (true) class: {counts}")
        if not examples:
            results["test_compounds"].append({
                "compound": list(compound),
                "true_label_order": true_label_meaning,
                "error": "no held-out examples",
            })
            continue
        stats = confusion_and_probs(probe_W, probe_b, examples)
        print(f"    overall_acc={stats['overall_acc']:.3f}  "
              f"per_class_acc={[f'{a:.3f}' if a is not None else None for a in stats['per_class_acc']]}")
        print(f"    confusion:")
        for row in stats["confusion"]:
            print(f"      {row}")
        print(f"    mean P(c=0/1/2) per true class:")
        for t, mp in enumerate(stats["mean_probs_per_true_class"]):
            if mp is None:
                continue
            print(f"      true={t} ({true_label_meaning[t]}): "
                  f"[{mp[0]:.3f}, {mp[1]:.3f}, {mp[2]:.3f}]")
        results["test_compounds"].append({
            "compound": list(compound),
            "true_label_order": true_label_meaning,
            **stats,
        })

    # Interpretation paragraph synthesised from the patterns in the matrices.
    interp_lines = []
    by_compound = {tuple(t["compound"]): t for t in results["test_compounds"]
                   if "error" not in t}
    def fmt_probs(c, t): return by_compound[c]["mean_probs_per_true_class"][t]

    if ("green", "square", "striped") in by_compound:
        pgss = by_compound[("green", "square", "striped")]
        interp_lines.append(
            "in-distribution held-out (green_square_striped): overall acc = "
            f"{pgss['overall_acc']:.3f}; per-class {pgss['per_class_acc']}."
        )
    if ("green", "circle", "striped") in by_compound:
        gcs = by_compound[("green", "circle", "striped")]
        interp_lines.append(
            "green_circle_striped (color and pattern unchanged from training "
            "compound; shape flipped circle): "
            f"green-pursuer mean P(c0)={fmt_probs(('green','circle','striped'),0)[0]:.2f}, "
            f"circle-pursuer mean P(c1)={fmt_probs(('green','circle','striped'),1)[1]:.2f}, "
            f"striped-pursuer mean P(c2)={fmt_probs(('green','circle','striped'),2)[2]:.2f}."
        )
    if ("green", "square", "solid") in by_compound:
        gss = by_compound[("green", "square", "solid")]
        interp_lines.append(
            "green_square_solid (pattern flipped from striped→solid; class-2 "
            "true label is now 'solid'-pursuer, not 'striped'): "
            f"solid-pursuer mean P(c2)={fmt_probs(('green','square','solid'),2)[2]:.2f} — "
            "if the class-2 direction reads 'striped' specifically, we expect this to be low; "
            "if it reads 'pursuing pattern axis', we expect it to be high."
        )
    if ("red", "square", "striped") in by_compound:
        rss = by_compound[("red", "square", "striped")]
        interp_lines.append(
            "red_square_striped (color flipped to red; class-0 true label is "
            "now 'red'-pursuer): "
            f"red-pursuer mean P(c0)={fmt_probs(('red','square','striped'),0)[0]:.2f} — "
            "high → class-0 reads 'pursuing color axis' (generic); "
            "low → class-0 reads 'pursuing green' specifically."
        )
    results["interpretation"] = " ".join(interp_lines)
    print("\n[B] interpretation:\n  " + results["interpretation"])
    return results


# ---------- main ----------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required (per repo policy).")
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading probe blob from {PROBE_PATH}")
    probe_blob = torch.load(PROBE_PATH, weights_only=False)
    print(f"  compound={probe_blob['compound']}  layer_idx={probe_blob['layer_idx']}  "
          f"acc={probe_blob['best_test_acc']:.3f}")

    # Run B first: it doesn't need the HF model, so failures in HF download
    # don't lose the cheap analysis.
    transfer_out = analysis_transfer(probe_blob)
    transfer_path = OUT_DIR / "cross_compound_transfer.json"
    with transfer_path.open("w") as f:
        json.dump(transfer_out, f, indent=2)
    print(f"\nsaved {transfer_path}")

    # Analysis A: logit lens (loads HF model on GPU).
    lens_out = analysis_logit_lens(probe_blob, device=device)
    lens_path = OUT_DIR / "logit_lens_green_square_striped.json"
    with lens_path.open("w") as f:
        json.dump(lens_out, f, indent=2)
    print(f"\nsaved {lens_path}")


if __name__ == "__main__":
    main()
