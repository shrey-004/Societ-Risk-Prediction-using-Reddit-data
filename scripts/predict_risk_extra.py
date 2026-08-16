"""
Predict risk-classifier OOF (each fold model on its val fold) + test (5-fold
mean) probabilities for an extra encoder added to the arbitration ensemble.
Generalizes predict_probs_deberta.py to any checkpoint suffix.

Usage:
    python scripts/predict_risk_extra.py --suffix _rbase --tag rb
  -> data/processed/{tag}_oof_risk.parquet  (row_id, {tag}p_risk_*)
     data/processed/{tag}_test_risk.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.clean import clean_text
from src.data.dataset_risk import ID2LABEL as RISK_ID2LABEL
from scripts.predict_probs import run_seq_model

CK = ROOT / "outputs" / "checkpoints"
MAX_LEN = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True, help="checkpoint suffix, e.g. _rbase (dirs risk_classifier_best{suffix}_fold{k})")
    ap.add_argument("--tag", required=True, help="short column prefix, e.g. rb")
    args = ap.parse_args()

    # OOF
    frames = []
    for fold in range(5):
        d = CK / f"risk_classifier_best{args.suffix}_fold{fold}"
        if not d.exists():
            raise FileNotFoundError(d)
        df = pd.read_csv(ROOT / f"data/processed/folds/fold{fold}_val.csv")
        pr = run_seq_model(d, df["post_clean"].tolist(), MAX_LEN)
        out = df[["row_id"]].copy()
        for i, lab in RISK_ID2LABEL.items():
            out[f"{args.tag}p_risk_{lab}"] = pr[:, i]
        frames.append(out)
        print(f"OOF fold {fold} done", flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(ROOT / f"data/processed/{args.tag}_oof_risk.parquet")

    # test (5-fold mean)
    df = pd.read_excel(ROOT / "data/raw/test.xlsx", sheet_name="Sheet1")
    df["post_clean"] = df["post"].apply(clean_text)
    texts = df["post_clean"].tolist()
    pr = np.mean([run_seq_model(CK / f"risk_classifier_best{args.suffix}_fold{f}", texts, MAX_LEN)
                  for f in range(5)], axis=0)
    out = df[["row_id"]].copy()
    for i, lab in RISK_ID2LABEL.items():
        out[f"{args.tag}p_risk_{lab}"] = pr[:, i]
    out.to_parquet(ROOT / f"data/processed/{args.tag}_test_risk.parquet")
    print(f"saved {args.tag}_oof_risk.parquet + {args.tag}_test_risk.parquet", flush=True)


if __name__ == "__main__":
    main()
