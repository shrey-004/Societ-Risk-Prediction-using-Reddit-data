"""
DeBERTa-v3-base probability dumps mirroring predict_probs.py:
  OOF : each fold model on its own val fold  -> data/processed/deberta_oof_probs.parquet
  test: mean of the 5 fold models on test    -> data/processed/deberta_test_probs.parquet
(no fulldata refit needed — the 5-fold ensemble serves as the test model)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.clean import clean_text
from src.data.dataset_risk import ID2LABEL as RISK_ID2LABEL
from src.data.dataset_factors import FACTOR_LIST
from scripts.predict_probs import run_seq_model

CK = ROOT / "outputs" / "checkpoints"
MAX_LEN = 512


def main():
    # OOF
    frames = []
    for fold in range(5):
        df = pd.read_csv(ROOT / f"data/processed/folds/fold{fold}_val.csv")
        texts = df["post_clean"].tolist()
        print(f"fold {fold}: {len(df)} rows", flush=True)
        pr = run_seq_model(CK / f"risk_classifier_best_deberta_fold{fold}", texts, MAX_LEN)
        pf = run_seq_model(CK / f"factor_classifier_best_deberta_fold{fold}", texts, MAX_LEN, sigmoid=True)
        out = df[["row_id"]].copy()
        for i, lab in RISK_ID2LABEL.items():
            out[f"dp_risk_{lab}"] = pr[:, i]
        for i, _ in enumerate(FACTOR_LIST):
            out[f"dp_factor_{i}"] = pf[:, i]
        frames.append(out)
    pd.concat(frames, ignore_index=True).to_parquet(ROOT / "data/processed/deberta_oof_probs.parquet")
    print("saved deberta_oof_probs.parquet", flush=True)

    # test (5-fold mean)
    df = pd.read_excel(ROOT / "data/raw/test.xlsx", sheet_name="Sheet1")
    df["post_clean"] = df["post"].apply(clean_text)
    texts = df["post_clean"].tolist()
    pr = np.mean([run_seq_model(CK / f"risk_classifier_best_deberta_fold{f}", texts, MAX_LEN)
                  for f in range(5)], axis=0)
    pf = np.mean([run_seq_model(CK / f"factor_classifier_best_deberta_fold{f}", texts, MAX_LEN, sigmoid=True)
                  for f in range(5)], axis=0)
    out = df[["row_id"]].copy()
    for i, lab in RISK_ID2LABEL.items():
        out[f"dp_risk_{lab}"] = pr[:, i]
    for i, _ in enumerate(FACTOR_LIST):
        out[f"dp_factor_{i}"] = pf[:, i]
    out.to_parquet(ROOT / "data/processed/deberta_test_probs.parquet")
    print("saved deberta_test_probs.parquet", flush=True)


if __name__ == "__main__":
    main()
