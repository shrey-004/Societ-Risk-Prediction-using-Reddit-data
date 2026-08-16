"""
Phase 3 (addendum) — sweep the evidence span confidence threshold
(generate_submission.py's --evidence_min_confidence) to find the value
that maximizes official Phrase F1 on val.

Why this might help: reports/phase3_official_phrase_f1_report.md shows
aggregate precision (0.58) noticeably below aggregate recall (0.72). Phrase
F1 (harmonic mean) is maximized where precision and recall are closer to
balanced, so dropping the least-confident predicted spans — which are
disproportionately likely to be the false positives dragging precision
down — is a reasonable, cheap thing to try before more expensive changes.

CAVEAT (same one as tune_factor_thresholds.py): this sweeps on the same
271-row val set it reports the winning threshold's score on. Use the
result as a signal, and ideally confirm the chosen threshold also helps
on 1-2 of the --fold val sets before locking it in for the final
submission.

Usage (from esrd_project/ root, esrd2026 env active):
    python scripts/tune_evidence_confidence.py
    python scripts/tune_evidence_confidence.py --suffix _fold0 --val_csv data/processed/folds/fold0_val.csv
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_evidence import ID2LABEL as BIO_ID2LABEL
from src.eval.phrase_f1 import corpus_phrase_f1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", type=str, default="")
    p.add_argument("--val_csv", type=str, default="data/processed/val_clean.csv")
    return p.parse_args()


def merge_bio_spans_with_confidence(post, offset_mapping, pred_ids, probs, max_gap_chars=2):
    finished_ranges = []
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


def main():
    cli_args = parse_args()
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    val_df = pd.read_csv(root / cli_args.val_csv)
    max_len = cfg["model"]["max_seq_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = root / "outputs" / "checkpoints" / f"evidence_extractor_best{cli_args.suffix}"
    print(f"Loading {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(device).eval()

    print(f"Running inference on {len(val_df)} val posts (once — thresholds applied post-hoc) ...")
    all_spans_with_conf, all_gold = [], []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            post = row["post_clean"]
            gold_spans = ast.literal_eval(row["evidence_spans"])
            enc = tokenizer(post, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors="pt")
            offset_mapping = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits[0].cpu().numpy()
            probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            pred_ids = np.argmax(probs, axis=-1).tolist()
            spans_with_conf = merge_bio_spans_with_confidence(post, offset_mapping, pred_ids, probs)
            all_spans_with_conf.append(spans_with_conf)
            all_gold.append(gold_spans)

    print(f"\n{'min_confidence':>15} {'Phrase F1':>10} {'Precision':>10} {'Recall':>10} {'avg spans/post':>15}")
    best_t, best_f1 = 0.0, -1.0
    for t in np.arange(0.0, 1.0, 0.05):
        pred_spans = [[s for s, c in row if c >= t] for row in all_spans_with_conf]
        result = corpus_phrase_f1(pred_spans, all_gold)
        avg_spans = np.mean([len(p) for p in pred_spans])
        marker = ""
        if result["phrase_f1"] > best_f1:
            best_f1, best_t = result["phrase_f1"], t
        print(f"{t:>15.2f} {result['phrase_f1']:>10.4f} {result['aggregate_precision']:>10.4f} "
              f"{result['aggregate_recall']:>10.4f} {avg_spans:>15.2f}")

    print(f"\nBest min_confidence: {best_t:.2f} (Phrase F1 = {best_f1:.4f})")
    print(f"Compare to min_confidence=0.0 (original behavior, top row above) before deciding.")
    print(f"\nUse it with: python scripts/generate_submission.py --evidence_min_confidence {best_t:.2f}")


if __name__ == "__main__":
    main()
