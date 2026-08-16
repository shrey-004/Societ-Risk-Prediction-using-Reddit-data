"""
Phase 4 (addendum, v3) — sweep multiple threshold floors and report the
ACTUAL pooled-OOF macro F1 at each one, side by side with the joint
over-firing sanity check.

WHY THIS EXISTS: tune_factor_thresholds_cv.py originally used a 0.30 floor
and produced a real over-firing incident (22/24 factors on ordinary
posts). Raising the floor to 0.50 fixed the over-firing symptom but
caused a WORSE real leaderboard score (Subtask2 0.3820 -> 0.2706) —
because macro F1 across 24 categories, many rare, can reward a generously-
firing model that at least catches some true positives on rare classes
more than a conservative model that misses them outright. "Doesn't look
alarming" and "scores well on the real metric" are different things, and
picking a single floor without checking both was the actual mistake.

This script runs inference ONCE (the expensive GPU part) and then sweeps
the threshold search floor cheaply (pure numpy) across several candidates,
reporting macro F1 AND the sanity stats for each — so you pick the floor
that's actually best on the metric that matters, with over-firing as a
secondary check rather than the primary decision.

Usage (from esrd_project/ root, esrd2026 env active):
    python scripts/sweep_factor_thresholds.py
    python scripts/sweep_factor_thresholds.py --floors 0.30,0.35,0.40,0.45,0.50
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
    p.add_argument("--floors", type=str, default="0.30,0.35,0.40,0.45,0.50",
                    help="Comma-separated threshold-search floors to compare.")
    p.add_argument("--save_floor", type=float, default=None,
                    help="After the sweep, save factor_thresholds_cv.json using this floor. "
                         "If omitted, saves using whichever floor had the best pooled-OOF macro F1.")
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


def tune_thresholds_at_floor(probs, labels, factor_names, floor):
    candidates = np.arange(floor, 0.96, 0.05)
    best_thresholds, tuned_f1s = {}, {}
    for i, name in enumerate(factor_names):
        y_true, y_probs = labels[:, i], probs[:, i]
        if y_true.sum() == 0:
            best_thresholds[name] = 0.5
            tuned_f1s[name] = 0.0
            continue
        best_t, best_f1 = floor, f1_score(y_true, (y_probs >= floor).astype(int), zero_division=0)
        for t in candidates:
            f1 = f1_score(y_true, (y_probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[name], tuned_f1s[name] = float(best_t), best_f1
    return best_thresholds, tuned_f1s


def main():
    cli_args = parse_args()
    floors = [float(x) for x in cli_args.floors.split(",")]

    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    processed_dir = root / cfg["paths"]["processed_dir"]
    fold_dir = processed_dir / "folds"
    n_folds = cfg["cross_validation"]["n_folds"]
    max_len = cfg["model"]["max_seq_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Running inference once across all fold models (this is the slow part) ...")
    all_oof_probs, all_oof_labels = [], []
    for i in range(n_folds):
        model_dir = root / "outputs" / "checkpoints" / f"factor_classifier_best_fold{i}"
        fold_val_path = fold_dir / f"fold{i}_val.csv"
        if not model_dir.exists() or not fold_val_path.exists():
            print(f"WARNING: missing {model_dir} or {fold_val_path} — skipping fold {i}.")
            continue
        print(f"  fold {i} ...")
        fold_val_df = pd.read_csv(fold_val_path)
        probs, labels = get_val_probs_and_labels(model_dir, fold_val_df, max_len, device)
        all_oof_probs.append(probs)
        all_oof_labels.append(labels)

    if not all_oof_probs:
        print("No fold models found. Aborting.")
        return

    pooled_probs = np.concatenate(all_oof_probs, axis=0)
    pooled_labels = np.concatenate(all_oof_labels, axis=0)
    n_factors_true = pooled_labels.sum(axis=1)
    print(f"\nPooled OOF set: {len(pooled_probs)} rows")
    print(f"Ground truth: mean {n_factors_true.mean():.2f} factors/row, "
          f"{(n_factors_true>=10).mean()*100:.2f}% rows with 10+, {(n_factors_true>=14).mean()*100:.2f}% rows with 14+\n")

    print(f"{'Floor':>7} {'Macro F1':>10} {'Mean factors/row':>18} {'% rows 10+':>12} {'% rows 14+':>12}")
    results = {}
    for floor in floors:
        thresholds, tuned_f1s = tune_thresholds_at_floor(pooled_probs, pooled_labels, FACTOR_LIST, floor)
        macro_f1 = np.mean(list(tuned_f1s.values()))
        n_pred = (pooled_probs >= np.array([thresholds[f] for f in FACTOR_LIST])).sum(axis=1)
        mean_pred = n_pred.mean()
        pct10 = (n_pred >= 10).mean() * 100
        pct14 = (n_pred >= 14).mean() * 100
        results[floor] = {"thresholds": thresholds, "macro_f1": macro_f1, "mean_pred": mean_pred, "pct10": pct10, "pct14": pct14}
        print(f"{floor:>7.2f} {macro_f1:>10.4f} {mean_pred:>18.2f} {pct10:>11.2f}% {pct14:>11.2f}%")

    best_floor = max(results, key=lambda f: results[f]["macro_f1"])
    print(f"\nBest pooled-OOF macro F1: floor={best_floor:.2f} ({results[best_floor]['macro_f1']:.4f})")
    print("Reminder: pooled-OOF macro F1 is still a proxy — the real leaderboard test set can "
          "disagree (as it just did between floor 0.30 and 0.50). Prefer the floor with the best "
          "macro F1 here UNLESS its over-firing stats are extreme (e.g. >5-10% of rows at 14+, "
          "which risks another visibly-broken submission) — in that case consider the next-best "
          "floor as a more robust compromise, since a submission slot is worth more than a small "
          "F1 gain at the very edge of what's calibrated.")

    save_floor = cli_args.save_floor if cli_args.save_floor is not None else best_floor
    if save_floor not in results:
        print(f"\n--save_floor {save_floor} wasn't one of the swept floors — rerun including it, "
              f"or omit --save_floor to auto-save the best one from this sweep.")
        return

    out_path = processed_dir / "factor_thresholds_cv.json"
    with open(out_path, "w") as f:
        json.dump(results[save_floor]["thresholds"], f, indent=2)
    print(f"\nSaved thresholds from floor={save_floor:.2f} to {out_path}")


if __name__ == "__main__":
    main()
