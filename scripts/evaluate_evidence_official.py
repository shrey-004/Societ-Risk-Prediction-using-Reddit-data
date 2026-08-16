"""
Phase 3 (addendum) — evaluate the ALREADY-TRAINED evidence extractor using
the official Phrase F1 metric (not seqeval's strict entity match, not our
earlier lenient token-overlap proxy — the actual competition metric).

Usage (from esrd_project/ root, esrd2026 env active):
    python scripts/evaluate_evidence_official.py
    python scripts/evaluate_evidence_official.py --suffix _fold0 --val_csv data/processed/folds/fold0_val.csv
"""
import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_evidence import ID2LABEL
from src.eval.phrase_f1 import corpus_phrase_f1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", type=str, default="", help="Which evidence_extractor_best{suffix} checkpoint to evaluate.")
    p.add_argument("--val_csv", type=str, default="data/processed/val_clean.csv")
    return p.parse_args()


def merge_bio_spans(post: str, offset_mapping: list, pred_ids: list, max_gap_chars: int = 2) -> list[str]:
    """Merge consecutive B-EVID/I-EVID token predictions into phrase spans.

    v2 fix: the model sometimes outputs back-to-back B-tags for what should
    be one continuous phrase (e.g. 'gonna' + 'kill myself' as two separate
    spans instead of one 'gonna kill myself') instead of properly
    continuing with I-EVID. This fragmentation directly hurts Phrase F1
    precision. Fix: after getting raw B/I runs, merge any two runs
    separated by only a tiny character gap (<=2, i.e. just whitespace) —
    tested to correctly merge true fragmentation bugs while NOT merging
    genuinely separate evidence phrases elsewhere in the post."""
    raw_ranges = []
    current_start = current_end = None
    for (start, end), label_id in zip(offset_mapping, pred_ids):
        if start == end:
            continue
        label = ID2LABEL[label_id]
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


def decode_predicted_spans(post: str, tokenizer, model, max_length: int, device) -> list[str]:
    """Run the model on one post, merge consecutive B-EVID/I-EVID token
    predictions into phrase spans, and return the actual substrings."""
    enc = tokenizer(
        post,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

    return merge_bio_spans(post, offset_mapping, pred_ids)


def main():
    cli_args = parse_args()
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    val_df = pd.read_csv(root / cli_args.val_csv)

    model_dir = root / "outputs" / "checkpoints" / f"evidence_extractor_best{cli_args.suffix}"
    print(f"Loading trained model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    max_len = cfg["model"]["max_seq_len"]

    print(f"Running inference on {len(val_df)} val posts ...")
    all_pred_spans, all_gold_spans = [], []
    for _, row in val_df.iterrows():
        post = row["post_clean"]
        gold_spans = ast.literal_eval(row["evidence_spans"])
        pred_spans = decode_predicted_spans(post, tokenizer, model, max_len, device)
        all_pred_spans.append(pred_spans)
        all_gold_spans.append(gold_spans)

    result = corpus_phrase_f1(all_pred_spans, all_gold_spans)
    print("\n=== Official Phrase F1 (this is the real competition metric) ===")
    print(f"  Phrase F1:            {result['phrase_f1']:.4f}")
    print(f"  Aggregate precision:  {result['aggregate_precision']:.4f}")
    print(f"  Aggregate recall:     {result['aggregate_recall']:.4f}")
    print(f"  Posts evaluated:      {result['n_posts']}")

    print("\n=== Sample predictions (first 5 val posts) ===")
    for i in range(min(5, len(val_df))):
        print(f"\n--- {val_df.iloc[i]['row_id']} ---")
        print(f"  GOLD: {all_gold_spans[i]}")
        print(f"  PRED: {all_pred_spans[i]}")

    report_path = root / "reports" / f"phase3_official_phrase_f1_report{cli_args.suffix}.md"
    with open(report_path, "w") as f:
        f.write("# Phase 3 — Official Phrase F1 Evaluation\n\n")
        f.write(f"Checkpoint: evidence_extractor_best{cli_args.suffix}\n\n")
        f.write(f"Val set: {cli_args.val_csv}\n\n")
        f.write(f"Phrase F1: {result['phrase_f1']:.4f}\n\n")
        f.write(f"Aggregate precision: {result['aggregate_precision']:.4f}\n\n")
        f.write(f"Aggregate recall: {result['aggregate_recall']:.4f}\n\n")
        f.write("This is the metric that actually determines your competition score "
                "for the evidence-extraction half of Subtask 1 (weight 0.3 of the "
                "overall composite), per the organizers' exact rules.\n")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()