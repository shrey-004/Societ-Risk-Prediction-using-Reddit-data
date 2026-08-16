"""
Phase 4 (ensemble) — average sigmoid probabilities across N independently
trained factor classifiers (different random seeds), then re-tune per-class
thresholds on the averaged probabilities.

Prerequisite: train 3 seed variants first:
    python scripts/train_factor_classifier.py 42 _seed42
    python scripts/train_factor_classifier.py 123 _seed123
    python scripts/train_factor_classifier.py 777 _seed777

Then run this script to combine them:
    python scripts/ensemble_factors.py
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
from src.data.dataset_factors import FactorDataset, FACTOR_LIST

SEED_SUFFIXES = ["_seed42", "_seed123", "_seed777"]


def get_probs_and_labels(model_dir, df, max_len, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()
    ds = FactorDataset(df, tokenizer, max_length=max_len)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != "labels"}
            logits = model(**inputs).logits.cpu().numpy()[0]
            probs = 1 / (1 + np.exp(-logits))
            all_probs.append(probs)
            all_labels.append(item["labels"].numpy())
    return np.array(all_probs), np.array(all_labels)


def tune_thresholds(probs, labels, factor_names, candidates=None):
    if candidates is None:
        candidates = np.arange(0.30, 0.96, 0.05)  # same floor fix as tune_factor_thresholds.py
    thresholds, f1s = {}, {}
    for i, name in enumerate(factor_names):
        y_true, y_probs = labels[:, i], probs[:, i]
        if y_true.sum() == 0:
            thresholds[name], f1s[name] = 0.5, 0.0
            continue
        best_t, best_f1 = 0.5, f1_score(y_true, (y_probs >= 0.5).astype(int), zero_division=0)
        for t in candidates:
            f1 = f1_score(y_true, (y_probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[name], f1s[name] = float(best_t), best_f1
    return thresholds, f1s


def main():
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    processed_dir = root / cfg["paths"]["processed_dir"]
    val_df = pd.read_csv(processed_dir / "val_clean.csv")
    max_len = cfg["model"]["max_seq_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_model_probs = []
    labels = None
    for suffix in SEED_SUFFIXES:
        model_dir = root / "outputs" / "checkpoints" / f"factor_classifier_best{suffix}"
        if not model_dir.exists():
            print(f"WARNING: {model_dir} not found, skipping. Train it first with:")
            print(f"  python scripts/train_factor_classifier.py <seed> {suffix}")
            continue
        print(f"Loading {model_dir} ...")
        probs, labels = get_probs_and_labels(model_dir, val_df, max_len, device)
        all_model_probs.append(probs)

    if len(all_model_probs) < 2:
        print("\nNeed at least 2 trained seed models to ensemble. Aborting.")
        return

    print(f"\nEnsembling {len(all_model_probs)} models (simple average of probabilities) ...")
    avg_probs = np.mean(all_model_probs, axis=0)

    # baseline: single-model (first seed) thresholds, for comparison
    single_thresholds, single_f1s = tune_thresholds(all_model_probs[0], labels, FACTOR_LIST)
    single_macro = np.mean(list(single_f1s.values()))

    ensemble_thresholds, ensemble_f1s = tune_thresholds(avg_probs, labels, FACTOR_LIST)
    ensemble_macro = np.mean(list(ensemble_f1s.values()))

    print(f"\n{'Factor':<45} {'single-model F1':>16} {'ensemble F1':>12}")
    for name in FACTOR_LIST:
        print(f"{name:<45} {single_f1s[name]:>16.3f} {ensemble_f1s[name]:>12.3f}")

    print(f"\nSingle-model macro F1 (tuned):   {single_macro:.4f}")
    print(f"Ensemble macro F1 (tuned):        {ensemble_macro:.4f}")
    print(f"Uplift: {ensemble_macro - single_macro:+.4f}")

    out_path = root / "data" / "processed" / "factor_thresholds_ensemble.json"
    with open(out_path, "w") as f:
        json.dump(ensemble_thresholds, f, indent=2)
    print(f"\nSaved ensemble thresholds to {out_path}")
    print(f"Model dirs used: {[str(root / 'outputs' / 'checkpoints' / f'factor_classifier_best{s}') for s in SEED_SUFFIXES if (root / 'outputs' / 'checkpoints' / f'factor_classifier_best{s}').exists()]}")


if __name__ == "__main__":
    main()