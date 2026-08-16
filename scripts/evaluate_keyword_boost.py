"""
Phase 4 (addendum) — measure what the keyword-rule boost (src/eval/
factor_keyword_rules.py) actually does to macro F1 ON TOP OF the trained
model, not just standalone. Same "prove it before you trust it" discipline
as ensemble_risk.py / ensemble_evidence.py.

A quick standalone check of the keyword rules alone (no model) against the
full labeled dataset already showed each of the 7 targeted categories
moving from F1=0.000 to somewhere in the 0.13-0.50 range with reasonable
precision (see the roadmap doc for the full table) — this script confirms
whether that holds up when unioned with real model predictions, and what
it does to overall macro F1 across all 24 categories (not just the 7
targeted ones — a keyword rule firing more than the model already does on
a category can only help recall, never hurt precision on OTHER
categories, but it's still worth checking the full picture).

Usage (from esrd_project/ root, esrd2026 env active):
    python scripts/evaluate_keyword_boost.py
    python scripts/evaluate_keyword_boost.py --model_suffix _fold0 --thresholds data/processed/factor_thresholds_cv.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_factors import FactorDataset, FACTOR_LIST, encode_multihot
from src.eval.factor_keyword_rules import apply_keyword_boost, FACTOR_KEYWORD_PATTERNS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_suffix", type=str, default="",
                    help="Which factor_classifier_best{suffix} checkpoint to evaluate.")
    p.add_argument("--thresholds", type=str, default="data/processed/factor_thresholds.json")
    p.add_argument("--val_csv", type=str, default="data/processed/val_clean.csv")
    return p.parse_args()


def main():
    cli_args = parse_args()
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    val_df = pd.read_csv(root / cli_args.val_csv)
    max_len = cfg["model"]["max_seq_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = root / "outputs" / "checkpoints" / f"factor_classifier_best{cli_args.model_suffix}"
    print(f"Loading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()

    with open(root / cli_args.thresholds) as f:
        thresholds = json.load(f)

    import ast
    gold_sets = [set(ast.literal_eval(s)) for s in val_df["factors"].tolist()]
    posts = val_df["post_clean"].tolist()

    model_only_sets, boosted_sets = [], []
    with torch.no_grad():
        for post in posts:
            enc = tokenizer(post, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits[0].cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            model_pred = {f for i, f in enumerate(FACTOR_LIST) if probs[i] >= thresholds.get(f, 0.5)}
            model_only_sets.append(model_pred)
            boosted_sets.append(apply_keyword_boost(post, model_pred))

    def multihot(sets_list):
        return np.stack([encode_multihot(list(s)).numpy() for s in sets_list])

    gold_mat = multihot(gold_sets)
    model_mat = multihot(model_only_sets)
    boosted_mat = multihot(boosted_sets)

    print(f"\n{'Factor':<45} {'model F1':>10} {'boosted F1':>11} {'delta':>8}")
    model_f1s, boosted_f1s = [], []
    for i, name in enumerate(FACTOR_LIST):
        mf1 = f1_score(gold_mat[:, i], model_mat[:, i], zero_division=0)
        bf1 = f1_score(gold_mat[:, i], boosted_mat[:, i], zero_division=0)
        model_f1s.append(mf1)
        boosted_f1s.append(bf1)
        marker = "  <-- targeted" if name in FACTOR_KEYWORD_PATTERNS else ""
        print(f"{name:<45} {mf1:>10.3f} {bf1:>11.3f} {bf1 - mf1:>+8.3f}{marker}")

    model_macro = np.mean(model_f1s)
    boosted_macro = np.mean(boosted_f1s)
    print(f"\n{'MACRO F1':<45} {model_macro:>10.4f} {boosted_macro:>11.4f} {boosted_macro - model_macro:>+8.4f}")

    if boosted_macro > model_macro:
        print(f"\nKeyword boost helped on this val set (+{boosted_macro - model_macro:.4f} macro F1). "
              f"Recommend also checking with --val_csv pointed at each fold's held-out set "
              f"(data/processed/folds/fold{{i}}_val.csv, using the matching --model_suffix _fold{{i}}) "
              f"before enabling it for the final submission — one 271-row check is still a small sample.")
    else:
        print(f"\nKeyword boost did NOT help on this val set ({boosted_macro - model_macro:+.4f} macro F1). "
              f"Check the per-factor table above — if a targeted category still went down, the "
              f"lexicon in src/eval/factor_keyword_rules.py needs tightening (probably too many "
              f"false positives on that category) before it's worth using.")


if __name__ == "__main__":
    main()
