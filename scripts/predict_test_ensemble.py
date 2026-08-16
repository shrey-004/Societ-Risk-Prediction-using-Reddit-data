"""
Ensembled BERT test probabilities: average the 5 fold models + the fulldata
model per subtask (risk softmax, factor sigmoid) instead of fulldata alone.
Evidence BIO probs are averaged across the 6 extractors before decoding.
Output: data/processed/test_probs_ensemble.parquet (same schema as
test_probs.parquet).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

torch.set_num_threads(16)
DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 4e9 else "cpu"
MAX_LEN = 512
CK = ROOT / "outputs" / "checkpoints"

RISK_MODELS = [f"risk_classifier_best_fold{i}" for i in range(5)] + ["risk_classifier_best_fulldata"]
FACT_MODELS = [f"factor_classifier_best_mentalbert_fold{i}" for i in range(5)] + [
    "factor_classifier_best_mentalbert_fulldata"]
EVID_MODELS = [f"evidence_extractor_best_fold{i}" for i in range(5)] + ["evidence_extractor_best_fulldata"]


def seq_probs(model_dir, texts, sigmoid=False, bs=32):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(DEVICE).eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits.float().cpu().numpy()
        out.append(1 / (1 + np.exp(-logits)) if sigmoid
                   else np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))
    del mdl
    torch.cuda.empty_cache()
    return np.concatenate(out)


def evid_token_probs(model_dir, texts):
    """per-post token prob matrices + offsets (variable length)."""
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(DEVICE).eval()
    probs_all, offsets_all = [], []
    for post in texts:
        enc = tok(post, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits[0].float().cpu().numpy()
        p = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        probs_all.append(p)
        offsets_all.append(offsets)
    del mdl
    torch.cuda.empty_cache()
    return probs_all, offsets_all


def main():
    df = pd.read_excel(ROOT / "data/raw/test.xlsx", sheet_name="Sheet1")
    df["post_clean"] = df["post"].apply(clean_text)
    texts = df["post_clean"].tolist()
    print(f"test: {len(df)} rows, device={DEVICE}", flush=True)

    risk = np.mean([seq_probs(CK / m, texts) for m in RISK_MODELS], axis=0)
    print("risk ensemble done", flush=True)
    fact = np.mean([seq_probs(CK / m, texts, sigmoid=True) for m in FACT_MODELS], axis=0)
    print("factor ensemble done", flush=True)

    agg_probs = None
    ref_offsets = None
    for m in EVID_MODELS:
        probs, offsets = evid_token_probs(CK / m, texts)
        if agg_probs is None:
            agg_probs = probs
            ref_offsets = offsets
        else:
            agg_probs = [a[:min(len(a), len(b))] + b[:min(len(a), len(b))] for a, b in zip(agg_probs, probs)]
            ref_offsets = [o[:len(a)] for o, a in zip(ref_offsets, agg_probs)]
        print(f"evidence {m} done", flush=True)
    ev_json = []
    for post, p, off in zip(texts, agg_probs, ref_offsets):
        p = p / p.sum(axis=-1, keepdims=True)
        pred_ids = np.argmax(p, axis=-1).tolist()
        spans = merge_bio_spans_with_confidence(post, off, pred_ids, p)
        ev_json.append(json.dumps([[s, round(c, 4)] for s, c in spans]))

    out = df[["row_id", "anon_user_id", "post_clean"]].copy()
    for i, lab in RISK_ID2LABEL.items():
        out[f"p_risk_{lab}"] = risk[:, i]
    for i, _ in enumerate(FACTOR_LIST):
        out[f"p_factor_{i}"] = fact[:, i]
    out["evidence_pred"] = ev_json
    out.to_parquet(ROOT / "data/processed/test_probs_ensemble.parquet")
    print("saved data/processed/test_probs_ensemble.parquet", flush=True)


if __name__ == "__main__":
    main()
