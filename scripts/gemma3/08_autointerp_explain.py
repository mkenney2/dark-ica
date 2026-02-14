"""Explain pass: generate Claude interpretations for ICA, SAE, and random directions.

Usage:
    python scripts/gemma3/08_autointerp_explain.py --experiment gemma3_1b_262k
    python scripts/gemma3/08_autointerp_explain.py --experiment gpt2_random_baseline --direction-types random
"""

import argparse
import os
import sys
import json
import time
import re

sys.path.insert(0, os.path.dirname(__file__))
from config import get_config
from utils import load_results_json, save_results_json


EXPLAIN_SYSTEM = """You are an interpretability researcher analyzing directions in a neural network's activation space. You will be shown text excerpts where a particular direction activates strongly. The token in [brackets] is where the direction fires. Your job is to identify the common pattern, concept, or feature these examples share.

Respond with:
1. A short label (2-5 words) for what this direction represents
2. A one-sentence explanation
3. A confidence score from 1-5:
   - 5: Clear, monosemantic concept
   - 4: Coherent theme with minor variation
   - 3: Plausible pattern but somewhat noisy
   - 2: Weak or ambiguous pattern
   - 1: No discernible pattern / random

Respond ONLY with valid JSON, no other text:
{"label": "...", "explanation": "...", "confidence": N}"""


def build_explain_prompt(examples, component_id, component_type, n_examples):
    lines = [f"Here are the top {n_examples} activating examples for {component_type} #{component_id}:\n"]
    for i, ex in enumerate(examples[:n_examples]):
        lines.append(f"Example {i+1} (activation: {ex['activation']:.3f}):")
        lines.append(ex["context"])
        lines.append("")
    return "\n".join(lines)


def call_explain(client, examples, component_id, component_type, cfg):
    prompt = build_explain_prompt(examples, component_id, component_type, cfg.autointerp_n_explain)
    response = client.messages.create(
        model=cfg.autointerp_model,
        max_tokens=256,
        temperature=0,
        system=EXPLAIN_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"label": "PARSE_ERROR", "explanation": text, "confidence": 0}
    return result


def run_explain(cfg, direction_types):
    cfg.ensure_dirs()

    import anthropic
    client = anthropic.Anthropic()
    print("Anthropic client initialized.")

    # Load existing explanations for checkpoint/resume
    explanations_path = cfg.autointerp_path("explanations.json")
    if os.path.exists(explanations_path):
        explanations = load_results_json(explanations_path)
        print(f"Loaded existing explanations from {explanations_path}")
    else:
        explanations = {}

    # --- ICA ---
    if "ica" in direction_types:
        examples_path = cfg.autointerp_path("ica_examples.json")
        if not os.path.exists(examples_path):
            print(f"Skipping ICA: {examples_path} not found")
        else:
            ica_examples = load_results_json(examples_path)
            if "ica" not in explanations:
                explanations["ica"] = {}

            n_total = len(ica_examples)
            n_existing = sum(1 for k in ica_examples if k in explanations["ica"])
            print(f"\n--- ICA explain pass: {n_total} components ({n_existing} already done) ---")

            for key in sorted(ica_examples.keys(), key=int):
                if key in explanations["ica"]:
                    continue
                positives = ica_examples[key]["positives"]
                if len(positives) < 3:
                    explanations["ica"][key] = {"label": "TOO_FEW_EXAMPLES", "explanation": "", "confidence": 0}
                    continue
                result = call_explain(client, positives, key, "ICA component", cfg)
                explanations["ica"][key] = result
                idx = int(key) + 1
                if idx % 10 == 0:
                    print(f"  {idx}/{n_total} — [{result.get('confidence', '?')}] {result.get('label', '?')}")
                    save_results_json(explanations, explanations_path)
                time.sleep(cfg.autointerp_rate_limit)

            save_results_json(explanations, explanations_path)

    # --- Random ---
    if "random" in direction_types:
        examples_path = cfg.autointerp_path("random_examples.json")
        if not os.path.exists(examples_path):
            print(f"Skipping Random: {examples_path} not found")
        else:
            random_examples = load_results_json(examples_path)
            if "random" not in explanations:
                explanations["random"] = {}

            n_total = len(random_examples)
            n_existing = sum(1 for k in random_examples if k in explanations["random"])
            print(f"\n--- Random explain pass: {n_total} directions ({n_existing} already done) ---")

            for key in sorted(random_examples.keys(), key=int):
                if key in explanations["random"]:
                    continue
                positives = random_examples[key]["positives"]
                if len(positives) < 3:
                    explanations["random"][key] = {"label": "TOO_FEW_EXAMPLES", "explanation": "", "confidence": 0}
                    continue
                result = call_explain(client, positives, key, "random direction", cfg)
                explanations["random"][key] = result
                idx = int(key) + 1
                if idx % 10 == 0:
                    print(f"  {idx}/{n_total} — [{result.get('confidence', '?')}] {result.get('label', '?')}")
                    save_results_json(explanations, explanations_path)
                time.sleep(cfg.autointerp_rate_limit)

            save_results_json(explanations, explanations_path)

    # --- SAE ---
    if "sae" in direction_types:
        examples_path = cfg.autointerp_path("sae_examples.json")
        if not os.path.exists(examples_path):
            print(f"Skipping SAE: {examples_path} not found")
        else:
            sae_examples = load_results_json(examples_path)
            if "sae" not in explanations:
                explanations["sae"] = {}

            n_total = len(sae_examples)
            n_existing = sum(1 for k in sae_examples if k in explanations["sae"])
            print(f"\n--- SAE explain pass: {n_total} features ({n_existing} already done) ---")

            for j, key in enumerate(sorted(sae_examples.keys(), key=int)):
                if key in explanations["sae"]:
                    continue
                positives = sae_examples[key]["positives"]
                if len(positives) < 3:
                    explanations["sae"][key] = {"label": "TOO_FEW_EXAMPLES", "explanation": "", "confidence": 0}
                    continue
                result = call_explain(client, positives, key, "SAE feature", cfg)
                explanations["sae"][key] = result
                if (j + 1) % 10 == 0:
                    print(f"  {j+1}/{n_total} — [{result.get('confidence', '?')}] {result.get('label', '?')}")
                    save_results_json(explanations, explanations_path)
                time.sleep(cfg.autointerp_rate_limit)

            save_results_json(explanations, explanations_path)

    # Summary
    print(f"\n--- Explain pass summary ---")
    for dtype in ["ica", "random", "sae"]:
        if dtype in explanations:
            confs = [v["confidence"] for v in explanations[dtype].values() if v.get("confidence", 0) > 0]
            if confs:
                import numpy as np
                print(f"  {dtype}: {len(confs)} explained, mean confidence {np.mean(confs):.2f}")
            else:
                print(f"  {dtype}: no valid explanations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--direction-types", nargs="+", default=["ica", "random", "sae"],
                        choices=["ica", "random", "sae"])
    args = parser.parse_args()

    cfg = get_config(args.experiment)
    run_explain(cfg, args.direction_types)
