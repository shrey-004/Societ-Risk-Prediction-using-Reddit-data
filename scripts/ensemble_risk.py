"""
Phase 2 (ensemble) — average softmax probabilities across N independently
trained risk classifiers (different random seeds), evaluate the ensemble
vs each single model on the SAME val set, and report whether it actually
helps before you trust it for the final submission.

Deliberately does NOT re-tune anything (no threshold search — argmax over
single-label softmax doesn't need one anyway). That distinction matters:
the Phase 4 factor-classifier ensemble looked better on val (0.372) but
scored WORSE on the real leaderboard (0.2371) specifically because its
per-class thresholds were RE-TUNED on the same 271-row val set used to
report the win — a second overfitting pass stacked on top of the
ensemble. Simple probability-averaging (no re-tuning) is a much safer
kind of ensembling; this script measures its effect honestly so you don't
have to guess.

Prerequisite: train 3 seed variants first:
    python scripts/train_risk_classifier.py --seed 42  --suffix _seed42
    python scripts/train_risk_classifier.py --seed 123 --suffix _seed123
    python scripts/train_risk_classifier.py --seed 777 --suffix _seed777

Then run this script to combine them:
    python scripts/ensemble_risk.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_risk import RiskDataset, ID2LABEL, LABEL2ID

SEED_SUFFIXES = ["_seed42", "_seed123", "_seed777"]


def get_probs_and_labels(model_dir, df, max_len, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()
    ds = RiskDataset(df, tokenizer, max_length=max_len)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != "labels"}
            logits = model(**inputs).logits.cpu().numpy()[0]
            probs = np.exp(logits) / np.exp(logits).sum()  # softmax
            all_probs.append(probs)
            all_labels.append(item["labels"].item())
    return np.array(all_probs), np.array(all_labels)


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
        model_dir = root / "outputs" / "checkpoints" / f"risk_classifier_best{suffix}"
        if not model_dir.exists():
            print(f"WARNING: {model_dir} not found, skipping. Train it first with:")
            print(f"  python scripts/train_risk_classifier.py --seed <seed> --suffix {suffix}")
            continue
        print(f"Loading {model_dir} ...")
        probs, labels = get_probs_and_labels(model_dir, val_df, max_len, device)
        all_model_probs.append(probs)

    if len(all_model_probs) < 2:
        print("\nNeed at least 2 trained seed models to ensemble. Aborting.")
        return

    print(f"\nEnsembling {len(all_model_probs)} models (simple average of softmax probs) ...")
    avg_probs = np.mean(all_model_probs, axis=0)
    ensemble_preds = np.argmax(avg_probs, axis=-1)

    print(f"\n{'Model':<20} {'macro_f1':>10} {'weighted_f1':>13}")
    for suffix, probs in zip(SEED_SUFFIXES, all_model_probs):
        preds = np.argmax(probs, axis=-1)
        macro = f1_score(labels, preds, average="macro")
        weighted = f1_score(labels, preds, average="weighted")
        print(f"{suffix:<20} {macro:>10.4f} {weighted:>13.4f}")

    ens_macro = f1_score(labels, ensemble_preds, average="macro")
    ens_weighted = f1_score(labels, ensemble_preds, average="weighted")
    print(f"{'ENSEMBLE (avg)':<20} {ens_macro:>10.4f} {ens_weighted:>13.4f}")

    best_single_weighted = max(
        f1_score(labels, np.argmax(p, axis=-1), average="weighted") for p in all_model_probs
    )
    print(f"\nEnsemble vs best single model (weighted F1): {ens_weighted:.4f} vs {best_single_weighted:.4f} "
          f"({'+' if ens_weighted >= best_single_weighted else ''}{ens_weighted - best_single_weighted:+.4f})")

    print("\n" + classification_report(
        labels, ensemble_preds, target_names=[ID2LABEL[i] for i in sorted(ID2LABEL)], digits=3
    ))

    print("IMPORTANT: this is still measured on the same 271-row val set as everything else — "
          "treat it as a signal, not proof. For a trustworthy answer, train the --fold 0..4 "
          "variants too and check whether ensembling helps consistently across folds before "
          "committing to it for the final submission (same discipline that would have caught "
          "the factor-classifier regression before it hit the real leaderboard).")


if __name__ == "__main__":
    main()
