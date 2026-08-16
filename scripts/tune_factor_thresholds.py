"""
Phase 4 (addendum) — per-class threshold tuning for the factor classifier.

The fixed 0.5 sigmoid threshold is rarely optimal per-class in imbalanced
multi-label problems. This script scans thresholds per factor category on
the val set and picks the one that maximizes that category's F1, then
reports the resulting macro F1 uplift.

CAVEAT (important, stated honestly): tuning thresholds on the same val set
we report macro F1 on is optimistic — it's not a fully unbiased estimate,
since we're selecting thresholds to maximize performance on the exact data
we're then scoring. With only 271 val rows we don't have room for a proper
3-way split (train/val/threshold-tune), so this is a pragmatic tradeoff.
Treat the "tuned" macro F1 as an optimistic upper bound, not a guaranteed
generalizing number — but the actual thresholds it finds are still
reasonable to use at inference time, since post-hoc calibration like this
is standard practice.

Usage (from esrd_project/ root, esrd2026 env active):
    python scripts/tune_factor_thresholds.py
"""
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
from src.data.dataset_factors import FactorDataset, FACTOR_LIST, NUM_FACTORS


def get_val_probs_and_labels(model_dir, val_df, max_len, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    val_ds = FactorDataset(val_df, tokenizer, max_length=max_len)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            item = val_ds[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != "labels"}
            logits = model(**inputs).logits.cpu().numpy()[0]
            probs = 1 / (1 + np.exp(-logits))
            all_probs.append(probs)
            all_labels.append(item["labels"].numpy())
    return np.array(all_probs), np.array(all_labels)


def tune_thresholds(probs, labels, factor_names, candidates=None):
    if candidates is None:
        candidates = np.arange(0.30, 0.96, 0.05)  # floor at 0.30 — thresholds below
        # this are almost certainly overfitting noise on categories with only
        # 1-4 positive val examples (confirmed: real test post had a category
        # fire at threshold=0.05 despite having essentially no real signal)

    best_thresholds = {}
    baseline_f1s = {}
    tuned_f1s = {}

    for i, name in enumerate(factor_names):
        y_true = labels[:, i]
        y_probs = probs[:, i]

        baseline_pred = (y_probs >= 0.5).astype(int)
        baseline_f1s[name] = f1_score(y_true, baseline_pred, zero_division=0)

        if y_true.sum() == 0:
            # no positive examples at all in val — can't tune meaningfully
            best_thresholds[name] = 0.5
            tuned_f1s[name] = baseline_f1s[name]
            continue

        best_t, best_f1 = 0.5, baseline_f1s[name]
        for t in candidates:
            pred = (y_probs >= t).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[name] = float(best_t)
        tuned_f1s[name] = best_f1

    return best_thresholds, baseline_f1s, tuned_f1s


def main():
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    processed_dir = root / cfg["paths"]["processed_dir"]
    val_df = pd.read_csv(processed_dir / "val_clean.csv")
    max_len = cfg["model"]["max_seq_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = root / "outputs" / "checkpoints" / "factor_classifier_best"
    print(f"Loading model from {model_dir} and running inference on {len(val_df)} val posts ...")
    probs, labels = get_val_probs_and_labels(model_dir, val_df, max_len, device)

    thresholds, baseline_f1s, tuned_f1s = tune_thresholds(probs, labels, FACTOR_LIST)

    baseline_macro = np.mean(list(baseline_f1s.values()))
    tuned_macro = np.mean(list(tuned_f1s.values()))

    print(f"\n{'Factor':<45} {'support':>7} {'baseline F1':>12} {'tuned F1':>9} {'threshold':>10}")
    for name in FACTOR_LIST:
        support = int(labels[:, FACTOR_LIST.index(name)].sum())
        print(f"{name:<45} {support:>7} {baseline_f1s[name]:>12.3f} {tuned_f1s[name]:>9.3f} {thresholds[name]:>10.2f}")

    print(f"\nBaseline macro F1 (fixed 0.5 threshold):  {baseline_macro:.4f}")
    print(f"Tuned macro F1 (per-class threshold):      {tuned_macro:.4f}")
    print(f"Uplift: {tuned_macro - baseline_macro:+.4f}")
    print("\n(Caveat: tuned on the same val set being scored — optimistic estimate, "
          "see script docstring. Thresholds themselves are still reasonable to use "
          "at inference time.)")

    out_path = root / "data" / "processed" / "factor_thresholds.json"
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved tuned thresholds to {out_path}")

    report_path = root / "reports" / "phase4_threshold_tuning_report.md"
    with open(report_path, "w") as f:
        f.write("# Phase 4 — Per-Class Threshold Tuning\n\n")
        f.write(f"Baseline macro F1 (0.5 threshold): {baseline_macro:.4f}\n\n")
        f.write(f"Tuned macro F1 (per-class threshold): {tuned_macro:.4f}\n\n")
        f.write("**Caveat**: tuned on the val set being scored — optimistic estimate, "
                "not a fully unbiased generalization number.\n\n")
        f.write("## Per-factor thresholds\n\n")
        f.write("| Factor | Support | Baseline F1 | Tuned F1 | Threshold |\n")
        f.write("|---|---|---|---|---|\n")
        for name in FACTOR_LIST:
            support = int(labels[:, FACTOR_LIST.index(name)].sum())
            f.write(f"| {name} | {support} | {baseline_f1s[name]:.3f} | {tuned_f1s[name]:.3f} | {thresholds[name]:.2f} |\n")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()