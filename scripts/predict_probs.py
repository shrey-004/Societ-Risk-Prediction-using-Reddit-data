"""
Dump raw model probabilities for OOF (5 fold models on their val folds) or
test (fulldata models on test.xlsx). These probability artifacts are the
foundation for validating fusion strategies (BERT+LLM, consistency rules,
risk-conditioned evidence emission) against labeled data before touching
the real submission.

Outputs (to data/processed/):
  OOF mode:  oof_probs.parquet   (row_id, fold, risk_label, p_risk_*, p_factor_*,
                                  evidence spans+confs as JSON)
  test mode: test_probs.parquet  (row_id, p_risk_*, p_factor_*, evidence JSON)

Usage:
    python scripts/predict_probs.py --mode oof
    python scripts/predict_probs.py --mode test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.clean import clean_text
from src.data.dataset_risk import ID2LABEL as RISK_ID2LABEL
from src.data.dataset_factors import FACTOR_LIST
from scripts.generate_submission import merge_bio_spans_with_confidence

torch.set_num_threads(24)
DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 4e9 else "cpu"


def batched(lst, n):
    for i in range(0, len(lst), n):
        yield i, lst[i:i + n]


def run_seq_model(model_dir, texts, max_len, sigmoid=False, batch_size=16):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(DEVICE).eval()
    out = []
    for i, chunk in batched(texts, batch_size):
        enc = tok(chunk, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits.cpu().numpy()
        if sigmoid:
            probs = 1 / (1 + np.exp(-logits))
        else:
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        out.append(probs)
        if (i // batch_size) % 5 == 0:
            print(f"    {i + len(chunk)}/{len(texts)}", flush=True)
    del mdl
    return np.concatenate(out, axis=0)


def run_evidence_model(model_dir, texts, max_len):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(DEVICE).eval()
    results = []
    for i, post in enumerate(texts):
        enc = tok(post, truncation=True, max_length=max_len, return_offsets_mapping=True,
                  return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits[0].cpu().numpy()
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        pred_ids = np.argmax(probs, axis=-1).tolist()
        spans = merge_bio_spans_with_confidence(post, offsets, pred_ids, probs)
        results.append(json.dumps([[s, round(c, 4)] for s, c in spans]))
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(texts)}", flush=True)
    del mdl
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["oof", "test"], required=True)
    args = p.parse_args()

    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    max_len = cfg["model"]["max_seq_len"]
    ck = ROOT / "outputs" / "checkpoints"
    print(f"Device: {DEVICE}")

    if args.mode == "oof":
        frames = []
        for fold in range(5):
            df = pd.read_csv(ROOT / f"data/processed/folds/fold{fold}_val.csv")
            texts = df["post_clean"].tolist()
            print(f"fold {fold}: {len(df)} rows")
            print("  risk ...")
            pr = run_seq_model(ck / f"risk_classifier_best_fold{fold}", texts, max_len)
            print("  factors (mentalbert) ...")
            pf = run_seq_model(ck / f"factor_classifier_best_mentalbert_fold{fold}", texts, max_len, sigmoid=True)
            print("  evidence ...")
            ev = run_evidence_model(ck / f"evidence_extractor_best_fold{fold}", texts, max_len)
            out = df[["row_id", "user_id", "risk_label", "factors", "evidence_spans", "post_clean"]].copy()
            out["fold"] = fold
            for i, lab in RISK_ID2LABEL.items():
                out[f"p_risk_{lab}"] = pr[:, i]
            for i, f_name in enumerate(FACTOR_LIST):
                out[f"p_factor_{i}"] = pf[:, i]
            out["evidence_pred"] = ev
            frames.append(out)
        allf = pd.concat(frames, ignore_index=True)
        allf.to_parquet(ROOT / "data/processed/oof_probs.parquet")
        print(f"saved {len(allf)} rows -> data/processed/oof_probs.parquet")
    else:
        df = pd.read_excel(ROOT / "data/raw/test.xlsx", sheet_name="Sheet1")
        df["post_clean"] = df["post"].apply(clean_text)
        texts = df["post_clean"].tolist()
        print(f"test: {len(df)} rows")
        print("  risk (fulldata) ...")
        pr = run_seq_model(ck / "risk_classifier_best_fulldata", texts, max_len)
        print("  factors (mentalbert fulldata) ...")
        pf = run_seq_model(ck / "factor_classifier_best_mentalbert_fulldata", texts, max_len, sigmoid=True)
        print("  evidence (fulldata) ...")
        ev = run_evidence_model(ck / "evidence_extractor_best_fulldata", texts, max_len)
        out = df[["row_id", "anon_user_id", "post_clean"]].copy()
        for i, lab in RISK_ID2LABEL.items():
            out[f"p_risk_{lab}"] = pr[:, i]
        for i, f_name in enumerate(FACTOR_LIST):
            out[f"p_factor_{i}"] = pf[:, i]
        out["evidence_pred"] = ev
        out.to_parquet(ROOT / "data/processed/test_probs.parquet")
        print(f"saved {len(out)} rows -> data/processed/test_probs.parquet")


if __name__ == "__main__":
    main()
