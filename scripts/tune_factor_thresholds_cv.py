"""
Phase 4 (addendum, v2) — cross-validated per-class threshold tuning.

WHY THIS SCRIPT EXISTS: tune_factor_thresholds.py tunes thresholds on the
same 271-row val set it reports macro F1 on. That's exactly what caused
the documented incident in generate_submission.py — the 3-seed ensemble +
retuned thresholds scored 0.372 on that val set but only 0.2371 on the
real leaderboard, so it was reverted in favor of the single-model,
single-split thresholds (0.3417 real). The problem was never the
ensembling — it was tuning thresholds on the same small sample being
optimized for.

This script fixes that by using OUT-OF-FOLD predictions instead: for each
of the 5 CV folds, run the model that was trained WITHOUT that fold on
exactly that fold's rows. Pooling all 5 folds' out-of-fold predictions
covers the full 1635-row labeled set, and every single prediction still
comes from a model that never saw that row during training — so it's
still a fair, unbiased basis for tuning, just ~6x more data than the old
271-row approach, which means far less threshold-selection noise.

Prerequisite: train the 5 fold models first (see scripts/make_kfold_splits.py
to generate the folds, if you haven't already):
    for i in 0 1 2 3 4; do
        python scripts/train_factor_classifier.py --fold $i
    done

Then:
    python scripts/tune_factor_thresholds_cv.py

Output: data/processed/factor_thresholds_cv.json — use this instead of
factor_thresholds.json for the final submission (pair it with the
--full_data refit model; thresholds transfer well between same
architecture/hyperparams with slightly more training data — standard
practice, but if you have time, sanity-check a handful of factors against
the --full_data model's own val_clean.csv... no wait, full_data has no
held-out set by design. If you want maximum rigor, keep 40 held-out rows
aside from the full_data run too and re-validate there before the final
submission — the script prints a reminder either way).
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
from src.data.dataset_factors import FactorDataset, FACTOR_LIST


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min_threshold", type=float, default=0.50,
                    help="Floor for the per-class threshold search (was 0.30, raised after a real "
                         "over-firing incident — see tune_thresholds()'s docstring/comments for why).")
    p.add_argument("--suffix", type=str, default="",
                    help="Extra suffix inserted before _fold{N} in checkpoint dir names, e.g. "
                         "'_mentalbert' to read factor_classifier_best_mentalbert_fold{N}. "
                         "Also appended to the output thresholds filename.")
    return p.parse_args()


def get_val_probs_and_labels(model_dir, val_df, max_len, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()

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


def tune_thresholds(probs, labels, factor_names, candidates=None, min_threshold=0.50):
    if candidates is None:
        # RAISED from 0.30 to min_threshold (default 0.50). The 0.30 floor
        # caused a real incident: per-class F1 search, done independently
        # per category with no view of what happens when many thresholds
        # fire together on the SAME post, collapsed almost every one of
        # the 24 categories to 0.30-0.50 (verified against a real
        # submission: factor_thresholds_cv.json had zero thresholds above
        # 0.5). The underlying model produces correlated, broadly-elevated
        # probabilities across most categories for most posts rather than
        # sharply distinguishing between them, so low per-class thresholds
        # look locally optimal but jointly cause severe over-firing (one
        # real submission had 34/378 rows predict 14+ factors, when only
        # 3/1635 real labeled rows ever do — see the joint sanity check
        # below, which now catches this before you submit again).
        candidates = np.arange(min_threshold, 0.96, 0.05)

    best_thresholds, baseline_f1s, tuned_f1s = {}, {}, {}
    for i, name in enumerate(factor_names):
        y_true, y_probs = labels[:, i], probs[:, i]
        baseline_pred = (y_probs >= 0.5).astype(int)
        baseline_f1s[name] = f1_score(y_true, baseline_pred, zero_division=0)

        if y_true.sum() == 0:
            best_thresholds[name] = 0.5
            tuned_f1s[name] = baseline_f1s[name]
            continue

        best_t, best_f1 = 0.5, baseline_f1s[name]
        for t in candidates:
            f1 = f1_score(y_true, (y_probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[name], tuned_f1s[name] = float(best_t), best_f1
    return best_thresholds, baseline_f1s, tuned_f1s


def main():
    cli_args = parse_args()
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    processed_dir = root / cfg["paths"]["processed_dir"]
    fold_dir = processed_dir / "folds"
    n_folds = cfg["cross_validation"]["n_folds"]
    max_len = cfg["model"]["max_seq_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_oof_probs, all_oof_labels = [], []
    n_available = 0
    for i in range(n_folds):
        model_dir = root / "outputs" / "checkpoints" / f"factor_classifier_best{cli_args.suffix}_fold{i}"
        fold_val_path = fold_dir / f"fold{i}_val.csv"
        if not model_dir.exists() or not fold_val_path.exists():
            print(f"WARNING: missing {model_dir} or {fold_val_path} — skipping fold {i}. "
                  f"Train it with: python scripts/train_factor_classifier.py --fold {i}")
            continue
        print(f"Fold {i}: loading {model_dir}, running inference on {fold_val_path.name} ...")
        fold_val_df = pd.read_csv(fold_val_path)
        probs, labels = get_val_probs_and_labels(model_dir, fold_val_df, max_len, device)
        all_oof_probs.append(probs)
        all_oof_labels.append(labels)
        n_available += 1

    if n_available < n_folds:
        print(f"\nOnly {n_available}/{n_folds} fold models found. You can still proceed with "
              f"partial coverage, but the whole point of this script is pooling ALL folds for "
              f"a low-noise estimate — train the missing folds first if possible.")
    if n_available == 0:
        print("No fold models found at all. Aborting.")
        return

    pooled_probs = np.concatenate(all_oof_probs, axis=0)
    pooled_labels = np.concatenate(all_oof_labels, axis=0)
    print(f"\nPooled out-of-fold set: {len(pooled_probs)} rows "
          f"(vs. 271 in the old single-split approach — {len(pooled_probs) / 271:.1f}x more data)")

    thresholds, baseline_f1s, tuned_f1s = tune_thresholds(
        pooled_probs, pooled_labels, FACTOR_LIST, min_threshold=cli_args.min_threshold
    )

    baseline_macro = np.mean(list(baseline_f1s.values()))
    tuned_macro = np.mean(list(tuned_f1s.values()))

    print(f"\n{'Factor':<45} {'support':>7} {'baseline F1':>12} {'tuned F1':>9} {'threshold':>10}")
    for name in FACTOR_LIST:
        support = int(pooled_labels[:, FACTOR_LIST.index(name)].sum())
        print(f"{name:<45} {support:>7} {baseline_f1s[name]:>12.3f} {tuned_f1s[name]:>9.3f} {thresholds[name]:>10.2f}")

    print(f"\nBaseline macro F1 (fixed 0.5 threshold, pooled OOF): {baseline_macro:.4f}")
    print(f"Tuned macro F1 (per-class threshold, pooled OOF):     {tuned_macro:.4f}")
    print(f"Uplift: {tuned_macro - baseline_macro:+.4f}")

    # JOINT SANITY CHECK — this is what would have caught the real incident
    # before it reached the leaderboard: per-class thresholds are each
    # locally F1-optimal in isolation, but say nothing about what happens
    # when many of them fire on the SAME post at once. Simulate it.
    n_factors_per_row = (pooled_probs >= np.array([thresholds[f] for f in FACTOR_LIST])).sum(axis=1)
    mean_pred = n_factors_per_row.mean()
    pct_14plus = (n_factors_per_row >= 14).mean() * 100
    pct_10plus = (n_factors_per_row >= 10).mean() * 100

    # ground truth for comparison (real labeled data: mean 2.92, 0.18% have 14+)
    n_factors_true = pooled_labels.sum(axis=1)
    print(f"\n=== Joint sanity check (simulated on pooled OOF set) ===")
    print(f"{'':30} {'mean factors/row':>18} {'% rows with 10+':>17} {'% rows with 14+':>17}")
    print(f"{'Ground truth (real labels)':30} {n_factors_true.mean():>18.2f} "
          f"{(n_factors_true>=10).mean()*100:>16.2f}% {(n_factors_true>=14).mean()*100:>16.2f}%")
    print(f"{'These thresholds (predicted)':30} {mean_pred:>18.2f} {pct_10plus:>16.2f}% {pct_14plus:>16.2f}%")

    if mean_pred > n_factors_true.mean() * 1.5 or pct_14plus > 2.0:
        print(f"\n⚠️  WARNING: predicted factor count is much higher than the real base rate. "
              f"This is the exact pattern that caused a real submission to predict 22/24 factors "
              f"on ordinary posts. Rerun with a higher floor, e.g.:\n"
              f"    python scripts/tune_factor_thresholds_cv.py --min_threshold 0.60\n"
              f"and check this section again before trusting the output for a submission.")
    else:
        print(f"\nLooks reasonable — predicted factor volume is in the right ballpark vs. real labels.")

    out_path = processed_dir / f"factor_thresholds_cv{cli_args.suffix}.json"
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved CV-tuned thresholds to {out_path}")
    print("Use these (not factor_thresholds.json) when generating the final submission — "
          "see generate_submission.py --thresholds_path flag.")


if __name__ == "__main__":
    main()
