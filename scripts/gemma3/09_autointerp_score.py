"""Score pass: detection accuracy for ICA, SAE, and random directions.

Usage:
    python scripts/gemma3/09_autointerp_score.py --experiment gemma3_1b_262k
    python scripts/gemma3/09_autointerp_score.py --experiment gpt2_random_baseline --direction-types random
"""

import argparse
import os
import sys
import json
import time
import re
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from utils import load_results_json, save_results_json


SCORE_SYSTEM = """You are evaluating whether a proposed explanation for a neural network direction is correct. You will be given an explanation and a set of text excerpts. For each excerpt, predict whether this direction would activate strongly on the [bracketed] token (YES or NO).

Respond ONLY with a valid JSON array of predictions, no other text:
[{"example_id": 1, "prediction": "YES"}, {"example_id": 2, "prediction": "NO"}, ...]"""


def build_score_prompt(explanation, examples_with_labels):
    lines = [f'The proposed explanation is: "{explanation}"\n']
    for i, (ex, _) in enumerate(examples_with_labels):
        lines.append(f"Example {i+1}:")
        lines.append(ex["context"])
        lines.append("")
    return "\n".join(lines)


def call_score(client, explanation, positives, negatives, cfg):
    """Score an explanation against held-out positives + negatives."""
    held_out_pos = positives[cfg.autointerp_n_explain:]
    neg = negatives

    if len(held_out_pos) < 3 or len(neg) < 3:
        return {
            "balanced_accuracy": None,
            "reason": f"too_few_examples (pos={len(held_out_pos)}, neg={len(neg)})",
        }

    examples_with_labels = [(ex, True) for ex in held_out_pos] + [(ex, False) for ex in neg]
    np.random.shuffle(examples_with_labels)
    ground_truth = [label for _, label in examples_with_labels]

    prompt = build_score_prompt(explanation, examples_with_labels)
    response = client.messages.create(
        model=cfg.autointerp_model,
        max_tokens=1024,
        temperature=0,
        system=SCORE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        predictions = json.loads(text)
        pred_yes = [p["prediction"].upper() == "YES" for p in predictions]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"balanced_accuracy": None, "reason": "parse_error", "raw": text[:200]}

    if len(pred_yes) != len(ground_truth):
        return {
            "balanced_accuracy": None,
            "reason": f"length_mismatch ({len(pred_yes)} vs {len(ground_truth)})",
        }

    tp = sum(1 for p, g in zip(pred_yes, ground_truth) if p and g)
    tn = sum(1 for p, g in zip(pred_yes, ground_truth) if not p and not g)
    n_pos = sum(ground_truth)
    n_neg = len(ground_truth) - n_pos
    tpr = tp / n_pos if n_pos > 0 else 0
    tnr = tn / n_neg if n_neg > 0 else 0
    balanced_acc = (tpr + tnr) / 2

    return {
        "balanced_accuracy": float(balanced_acc),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def run_score(cfg, direction_types):
    cfg.ensure_dirs()

    import anthropic
    client = anthropic.Anthropic()
    print("Anthropic client initialized.")

    # Load explanations
    explanations_path = cfg.autointerp_path("explanations.json")
    if not os.path.exists(explanations_path):
        print(f"ERROR: {explanations_path} not found. Run 08_autointerp_explain.py first.")
        return
    explanations = load_results_json(explanations_path)

    # Load existing scores for checkpoint/resume
    scores_path = cfg.autointerp_path("scores.json")
    if os.path.exists(scores_path):
        scores = load_results_json(scores_path)
        print(f"Loaded existing scores from {scores_path}")
    else:
        scores = {}

    for dtype in direction_types:
        if dtype not in explanations:
            print(f"Skipping {dtype}: no explanations found")
            continue

        examples_path = cfg.autointerp_path(f"{dtype}_examples.json")
        if not os.path.exists(examples_path):
            print(f"Skipping {dtype}: {examples_path} not found")
            continue

        all_examples = load_results_json(examples_path)
        if dtype not in scores:
            scores[dtype] = {}

        type_label = {"ica": "ICA component", "random": "random direction", "sae": "SAE feature"}[dtype]
        n_total = len(all_examples)
        n_existing = sum(1 for k in all_examples if k in scores[dtype])
        print(f"\n--- {dtype} score pass: {n_total} {type_label}s ({n_existing} already done) ---")

        for j, key in enumerate(sorted(all_examples.keys(), key=lambda x: int(x))):
            if key in scores[dtype]:
                continue

            exp = explanations[dtype].get(key, {})
            if exp.get("confidence", 0) == 0:
                scores[dtype][key] = {"balanced_accuracy": None, "reason": "skipped_low_confidence"}
                continue

            result = call_score(
                client,
                exp["explanation"],
                all_examples[key]["positives"],
                all_examples[key]["negatives"],
                cfg,
            )
            scores[dtype][key] = result

            if (j + 1) % 10 == 0:
                acc = result.get("balanced_accuracy")
                acc_str = f"{acc:.2f}" if isinstance(acc, float) else str(result.get("reason", "?"))
                print(f"  {j+1}/{n_total} — acc: {acc_str}")
                save_results_json(scores, scores_path)

            time.sleep(cfg.autointerp_rate_limit)

        save_results_json(scores, scores_path)

    # Summary
    print(f"\n--- Score pass summary ---")
    for dtype in ["ica", "random", "sae"]:
        if dtype in scores:
            accs = [v["balanced_accuracy"] for v in scores[dtype].values()
                    if v.get("balanced_accuracy") is not None]
            if accs:
                print(f"  {dtype}: {len(accs)} scored, mean acc {np.mean(accs):.3f}, "
                      f"median {np.median(accs):.3f}")
            else:
                print(f"  {dtype}: no valid scores")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--direction-types", nargs="+", default=["ica", "random", "sae"],
                        choices=["ica", "random", "sae"])
    args = parser.parse_args()

    cfg = get_config(args.experiment)
    run_score(cfg, args.direction_types)
