"""
Phase 7 — generate the final submission CSV from the test set.

Backward-compatible default (no flags) behaves like before EXCEPT for one
deliberate change: factor predictions are now capped at 10/post by default
(--factor_max_k, keeps the highest-confidence ones if more pass threshold).
This is a direct response to a real incident — a submission from thresholds
that collapsed to 0.30-0.50 across nearly all 24 categories predicted 22/24
factors on ordinary posts (34/378 rows hit 14+, vs. 3/1635 in the real
labeled data). Pass --factor_max_k 0 to disable and restore literal old
behavior, but there's no good reason to.

Other optional flags let you point at the improved artifacts from the
roadmap (recommended once you've verified each one actually helps via its
own evaluation script — see PRIORITIZED_IMPROVEMENTS.md):

    # use the full-data refit models (train.py --full_data) instead of the
    # original 85%-of-data models — recommended default once you've picked
    # epoch counts via CV, since it's "free" extra training data
    python scripts/generate_submission.py \
        --risk_suffix _fulldata --evidence_suffix _fulldata --factor_suffix _fulldata

    # ensemble 3 seeds per subtask (only bother if ensemble_risk.py /
    # ensemble_evidence.py / ensemble_factors.py showed a real uplift first)
    python scripts/generate_submission.py \
        --risk_suffixes _seed42,_seed123,_seed777 \
        --evidence_suffixes _seed42,_seed123,_seed777 \
        --factor_suffixes _seed42,_seed123,_seed777

    # CV-tuned thresholds instead of the single-split-tuned ones (recommended
    # — see scripts/tune_factor_thresholds_cv.py)
    python scripts/generate_submission.py --factor_thresholds data/processed/factor_thresholds_cv.json

    # keyword-rule recall boost for the 7 categories the model currently
    # gets ~0 F1 on (only if scripts/evaluate_keyword_boost.py confirmed
    # it helps on your setup)
    python scripts/generate_submission.py --use_keyword_boost

All flags compose. --risk_suffix (singular) and --risk_suffixes (plural,
comma-separated) are mutually exclusive per subtask.

Usage (from esrd_project/ root, esrd2026 env active):
    python scripts/generate_submission.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.clean import clean_text
from src.data.dataset_risk import ID2LABEL as RISK_ID2LABEL
from src.data.dataset_evidence import ID2LABEL as BIO_ID2LABEL
from src.data.dataset_factors import FACTOR_LIST
from src.eval.factor_keyword_rules import apply_keyword_boost

TEAM_NAME = "RayofHope"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--risk_suffix", type=str, default="",
                    help="Single risk_classifier_best{suffix} checkpoint to use.")
    p.add_argument("--evidence_suffix", type=str, default="")
    p.add_argument("--factor_suffix", type=str, default="")
    p.add_argument("--risk_suffixes", type=str, default="",
                    help="Comma-separated suffixes to ENSEMBLE (average probs). Mutually exclusive with --risk_suffix.")
    p.add_argument("--evidence_suffixes", type=str, default="")
    p.add_argument("--factor_suffixes", type=str, default="")
    p.add_argument("--factor_thresholds", type=str, default="data/processed/factor_thresholds.json")
    p.add_argument("--use_keyword_boost", action="store_true")
    p.add_argument("--evidence_min_confidence", type=float, default=0.0,
                    help="Drop predicted evidence spans below this mean confidence (0.0=off, original behavior). "
                         "Try scripts/tune_evidence_confidence.py first to pick a value — precision (0.58) is "
                         "currently the bottleneck vs recall (0.72) per reports/phase3_official_phrase_f1_report.md.")
    p.add_argument("--factor_max_k", type=int, default=10,
                    help="Safety cap: never predict more than this many factors for one post, keeping the "
                         "highest-confidence ones if more pass threshold (default 10 = real data's 99th "
                         "percentile; only 3/1635 real labeled rows ever exceed 14). Set to 0/negative to disable.")
    p.add_argument("--llm_factor_preds", type=str, default="",
                    help="Path to a CSV (row_id, llm_factors) from scripts/llm_factor_boost.py --mode predict. "
                         "Unioned into the 9 targeted weak categories only — see that script's docstring. "
                         "Validated on 361 held-out posts before use, see session notes.")
    return p.parse_args()


def merge_bio_spans(post: str, offset_mapping: list, pred_ids: list, max_gap_chars: int = 2) -> list[str]:
    """Merge consecutive B-EVID/I-EVID token predictions into phrase spans.
    Includes the adjacent-fragment merge fix (proven: +0.05 Phrase F1)."""
    raw_ranges = []
    current_start = current_end = None
    for (start, end), label_id in zip(offset_mapping, pred_ids):
        if start == end:
            continue
        label = BIO_ID2LABEL[label_id]
        if label == "B-EVID":
            if current_start is not None:
                raw_ranges.append((current_start, current_end))
            current_start, current_end = start, end
        elif label == "I-EVID" and current_start is not None:
            current_end = end
        else:
            if current_start is not None:
                raw_ranges.append((current_start, current_end))
            current_start = current_end = None
    if current_start is not None:
        raw_ranges.append((current_start, current_end))

    merged = []
    for s, e in raw_ranges:
        if merged and s - merged[-1][1] <= max_gap_chars:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [post[s:e] for s, e in merged]


def merge_bio_spans_with_confidence(post: str, offset_mapping: list, pred_ids: list, probs,
                                     max_gap_chars: int = 2):
    """Same merge logic as merge_bio_spans, but also returns each merged
    span's mean per-token confidence (max softmax prob at each token in
    the span) alongside its text — lets predict_evidence optionally drop
    low-confidence spans to trade recall for precision. The official
    Phrase F1 diagnostics (reports/phase3_official_phrase_f1_report.md)
    show aggregate precision (0.58) well below recall (0.72) — since F1 is
    maximized near where the two balance, dropping the least-confident
    predicted spans is a reasonable lever to try (see --evidence_min_confidence)."""
    finished_ranges = []  # (start, end, [confs])
    current_start = current_end = None
    current_confs = []
    for (start, end), label_id, prob_row in zip(offset_mapping, pred_ids, probs):
        if start == end:
            continue
        label = BIO_ID2LABEL[label_id]
        conf = float(prob_row[label_id])
        if label == "B-EVID":
            if current_start is not None:
                finished_ranges.append((current_start, current_end, current_confs))
            current_start, current_end, current_confs = start, end, [conf]
        elif label == "I-EVID" and current_start is not None:
            current_end = end
            current_confs.append(conf)
        else:
            if current_start is not None:
                finished_ranges.append((current_start, current_end, current_confs))
            current_start = current_end = None
            current_confs = []
    if current_start is not None:
        finished_ranges.append((current_start, current_end, current_confs))

    merged = []
    for s, e, confs in finished_ranges:
        if merged and s - merged[-1][1] <= max_gap_chars:
            merged[-1] = (merged[-1][0], e, merged[-1][2] + confs)
        else:
            merged.append((s, e, list(confs)))

    return [(post[s:e], (sum(c) / len(c) if c else 0.0)) for s, e, c in merged]


def load_seq_models(root, ckpt_subdir_prefix, suffixes, device):
    """Load one or more AutoModelForSequenceClassification checkpoints
    sharing the same base tokenizer. Returns list of (tokenizer, model)."""
    loaded = []
    for suffix in suffixes:
        model_dir = root / "outputs" / "checkpoints" / f"{ckpt_subdir_prefix}{suffix}"
        if not model_dir.exists():
            raise FileNotFoundError(f"{model_dir} not found — train it first.")
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()
        loaded.append((tok, mdl))
    return loaded


def predict_risk(post, models, max_len, device):
    """models: list of (tokenizer, model). Averages softmax probs if >1."""
    all_probs = []
    for tok, mdl in models:
        enc = tok(post, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits[0].cpu().numpy()
        probs = np.exp(logits) / np.exp(logits).sum()
        all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0)
    return RISK_ID2LABEL[int(np.argmax(avg_probs))]


def predict_evidence(post, models, max_len, device, min_confidence=0.0):
    """models: list of (tokenizer, model) for AutoModelForTokenClassification.
    Averages per-token softmax probs if >1 (assumes shared tokenizer/offsets).
    min_confidence: drop merged spans whose mean per-token confidence falls
    below this (0.0 = off, matches original behavior exactly)."""
    all_probs, shared_offsets = [], None
    for tok, mdl in models:
        enc = tok(post, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits[0].cpu().numpy()
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        if shared_offsets is None:
            shared_offsets = offset_mapping
        elif len(offset_mapping) != len(shared_offsets):
            min_len = min(len(offset_mapping), len(shared_offsets))
            shared_offsets = shared_offsets[:min_len]
            all_probs = [p[:min_len] for p in all_probs]
            probs = probs[:min_len]
        all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0)
    pred_ids = np.argmax(avg_probs, axis=-1).tolist()
    spans_with_conf = merge_bio_spans_with_confidence(post, shared_offsets, pred_ids, avg_probs)
    return [span for span, conf in spans_with_conf if conf >= min_confidence]


def predict_factors(post, models, max_len, device, thresholds, use_keyword_boost, max_k=None, llm_factors=None):
    all_probs = []
    for tok, mdl in models:
        enc = tok(post, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits[0].cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0)
    passing = [(factor, avg_probs[i]) for i, factor in enumerate(FACTOR_LIST) if avg_probs[i] >= thresholds.get(factor, 0.5)]
    if max_k is not None and len(passing) > max_k:
        # Safety net: in real labeled data, 99% of posts have <=10 factors
        # and only 3/1635 rows ever exceed 14 — a real incident showed the
        # model+thresholds combo CAN degenerate into firing on 20+
        # categories for an ordinary post. If that happens again, keep
        # only the highest-confidence max_k rather than silently
        # submitting an obviously-wrong prediction.
        passing.sort(key=lambda x: x[1], reverse=True)
        passing = passing[:max_k]
    predicted = {factor for factor, _ in passing}
    if use_keyword_boost:
        predicted = apply_keyword_boost(post, predicted)
    if llm_factors:
        # Union only — see scripts/llm_factor_boost.py. Validated on 361 held-out
        # labeled posts: F1 on the 9 targeted categories went from ~0.28 (BERT
        # alone) to ~0.53 (union), see the session notes / phase4 reports.
        predicted = predicted | set(llm_factors)
    return sorted(predicted)


def main():
    cli_args = parse_args()
    if cli_args.risk_suffix and cli_args.risk_suffixes:
        raise ValueError("--risk_suffix and --risk_suffixes are mutually exclusive.")
    if cli_args.evidence_suffix and cli_args.evidence_suffixes:
        raise ValueError("--evidence_suffix and --evidence_suffixes are mutually exclusive.")
    if cli_args.factor_suffix and cli_args.factor_suffixes:
        raise ValueError("--factor_suffix and --factor_suffixes are mutually exclusive.")

    risk_suffixes = cli_args.risk_suffixes.split(",") if cli_args.risk_suffixes else [cli_args.risk_suffix]
    evidence_suffixes = cli_args.evidence_suffixes.split(",") if cli_args.evidence_suffixes else [cli_args.evidence_suffix]
    factor_suffixes = cli_args.factor_suffixes.split(",") if cli_args.factor_suffixes else [cli_args.factor_suffix]

    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    test_path = root / "data" / "raw" / "test.xlsx"
    print(f"Loading test set from {test_path} ...")
    test_df = pd.read_excel(test_path, sheet_name="Sheet1")
    print(f"  {len(test_df)} rows loaded")

    print("Cleaning post text ...")
    test_df["post_clean"] = test_df["post"].apply(clean_text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = cfg["model"]["max_seq_len"]
    print(f"Device: {device}")

    print(f"Loading risk classifier(s): {risk_suffixes}")
    risk_models = load_seq_models(root, "risk_classifier_best", risk_suffixes, device)

    print(f"Loading evidence extractor(s): {evidence_suffixes}")
    evid_models = []
    for suffix in evidence_suffixes:
        model_dir = root / "outputs" / "checkpoints" / f"evidence_extractor_best{suffix}"
        if not model_dir.exists():
            raise FileNotFoundError(f"{model_dir} not found — train it first.")
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        mdl = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(device).eval()
        evid_models.append((tok, mdl))

    print(f"Loading factor classifier(s): {factor_suffixes}")
    fact_models = load_seq_models(root, "factor_classifier_best", factor_suffixes, device)

    thresholds_path = root / cli_args.factor_thresholds
    with open(thresholds_path) as f:
        thresholds = json.load(f)
    print(f"Loaded per-factor thresholds from {thresholds_path}")
    print(f"Keyword boost: {'ON' if cli_args.use_keyword_boost else 'off'}")

    llm_factor_map = {}
    if cli_args.llm_factor_preds:
        llm_df = pd.read_csv(root / cli_args.llm_factor_preds)
        for _, r in llm_df.iterrows():
            llm_factor_map[str(r["row_id"])] = json.loads(r["llm_factors"])
        print(f"LLM factor boost: ON ({len(llm_factor_map)} rows loaded from {cli_args.llm_factor_preds})")
    else:
        print("LLM factor boost: off")

    print(f"\nRunning inference on {len(test_df)} test posts ...")
    results = []
    for idx, row in test_df.iterrows():
        post = row["post_clean"]
        risk_level = predict_risk(post, risk_models, max_len, device)
        evidence_spans = predict_evidence(post, evid_models, max_len, device,
                                           min_confidence=cli_args.evidence_min_confidence)
        max_k = cli_args.factor_max_k if cli_args.factor_max_k and cli_args.factor_max_k > 0 else None
        llm_factors = llm_factor_map.get(str(row["row_id"]))
        factors = predict_factors(post, fact_models, max_len, device, thresholds, cli_args.use_keyword_boost,
                                   max_k=max_k, llm_factors=llm_factors)

        evidence_str = "; ".join(evidence_spans) if evidence_spans else "none"

        results.append({
            "row_id": row["row_id"],
            "risk_level": risk_level,
            "evidence": evidence_str,
            "factors": str(factors),
        })

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(test_df)} done")

    submission_df = pd.DataFrame(results)

    out_path = root / "outputs" / "predictions" / f"{TEAM_NAME}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_path, index=False)
    print(f"\nSaved submission to {out_path}")

    print("\n=== Prediction distribution sanity check ===")
    print(submission_df["risk_level"].value_counts())
    n_no_evidence = (submission_df["evidence"] == "none").sum()
    print(f"\nRows with no predicted evidence: {n_no_evidence}/{len(submission_df)}")
    submission_df["n_factors"] = submission_df["factors"].apply(lambda s: len(eval(s)))
    print(f"Mean factors/row: {submission_df['n_factors'].mean():.2f}")
    print(f"Rows with no predicted factors: {(submission_df['n_factors']==0).sum()}/{len(submission_df)}")
    print(f"Rows with 14+ factors: {(submission_df['n_factors']>=14).sum()}/{len(submission_df)}")

    print("\n=== Sample rows ===")
    print(submission_df[["row_id","risk_level","evidence","factors"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
