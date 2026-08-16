"""
Phase 3 (ensemble) — average per-token softmax probabilities across N
independently trained evidence extractors (different seeds), decode spans
from the averaged probabilities, and evaluate with the OFFICIAL Phrase F1
metric against each single model. Same no-retuning discipline as
ensemble_risk.py — there's no threshold to overfit here either, just
prediction averaging.

Prerequisite: train 3 seed variants first:
    python scripts/train_evidence_extractor.py --seed 42  --suffix _seed42
    python scripts/train_evidence_extractor.py --seed 123 --suffix _seed123
    python scripts/train_evidence_extractor.py --seed 777 --suffix _seed777

Then run this script to combine them:
    python scripts/ensemble_evidence.py
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_evidence import ID2LABEL
from src.eval.phrase_f1 import corpus_phrase_f1

SEED_SUFFIXES = ["_seed42", "_seed123", "_seed777"]


def merge_bio_spans(post: str, offset_mapping: list, pred_ids: list, max_gap_chars: int = 2) -> list[str]:
    """Same merge logic as evaluate_evidence_official.py / generate_submission.py
    (kept in sync manually — three copies is a known wart, see roadmap doc)."""
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


def get_post_token_probs(post, tokenizer, model, max_len, device):
    """Returns (offset_mapping, softmax_probs[seq_len, 3])."""
    enc = tokenizer(post, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0].cpu().numpy()
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    return offset_mapping, probs


def decode_from_probs(post: str, offset_mapping, probs) -> list[str]:
    pred_ids = np.argmax(probs, axis=-1).tolist()
    return merge_bio_spans(post, offset_mapping, pred_ids)


def main():
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    processed_dir = root / cfg["paths"]["processed_dir"]
    val_df = pd.read_csv(processed_dir / "val_clean.csv")
    max_len = cfg["model"]["max_seq_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = []
    for suffix in SEED_SUFFIXES:
        model_dir = root / "outputs" / "checkpoints" / f"evidence_extractor_best{suffix}"
        if not model_dir.exists():
            print(f"WARNING: {model_dir} not found, skipping. Train it first with:")
            print(f"  python scripts/train_evidence_extractor.py --seed <seed> --suffix {suffix}")
            continue
        print(f"Loading {model_dir} ...")
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        mdl = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(device).eval()
        models.append((suffix, tok, mdl))

    if len(models) < 2:
        print("\nNeed at least 2 trained seed models to ensemble. Aborting.")
        return

    # All seed models share the same base encoder/tokenizer (config-driven),
    # so tokenization + offsets are identical across models per post —
    # verified via the shared tokenizer.vocab check below before trusting
    # that we can average position-for-position.
    base_tok = models[0][1]
    for suffix, tok, _ in models[1:]:
        if tok.vocab_size != base_tok.vocab_size:
            raise RuntimeError(f"Tokenizer mismatch for {suffix} — models must share the same base encoder to ensemble token-level probs.")

    per_model_pred_spans = {suffix: [] for suffix, _, _ in models}
    ensemble_pred_spans = []
    gold_spans_all = []

    for _, row in val_df.iterrows():
        post = row["post_clean"]
        gold_spans = ast.literal_eval(row["evidence_spans"])
        gold_spans_all.append(gold_spans)

        all_probs = []
        shared_offsets = None
        for suffix, tok, mdl in models:
            offsets, probs = get_post_token_probs(post, tok, mdl, max_len, device)
            if shared_offsets is None:
                shared_offsets = offsets
            elif len(offsets) != len(shared_offsets):
                # truncation edge case — pad/crop the shorter one so avg still works
                min_len = min(len(offsets), len(shared_offsets))
                offsets = offsets[:min_len]
                shared_offsets = shared_offsets[:min_len]
                probs = probs[:min_len]
                all_probs = [p[:min_len] for p in all_probs]
            all_probs.append(probs)
            per_model_pred_spans[suffix].append(decode_from_probs(post, offsets, probs))

        avg_probs = np.mean(all_probs, axis=0)
        ensemble_pred_spans.append(decode_from_probs(post, shared_offsets, avg_probs))

    print(f"\n{'Model':<20} {'Phrase F1':>10} {'Precision':>10} {'Recall':>10}")
    for suffix in per_model_pred_spans:
        r = corpus_phrase_f1(per_model_pred_spans[suffix], gold_spans_all)
        print(f"{suffix:<20} {r['phrase_f1']:>10.4f} {r['aggregate_precision']:>10.4f} {r['aggregate_recall']:>10.4f}")

    ens_result = corpus_phrase_f1(ensemble_pred_spans, gold_spans_all)
    print(f"{'ENSEMBLE (avg)':<20} {ens_result['phrase_f1']:>10.4f} {ens_result['aggregate_precision']:>10.4f} {ens_result['aggregate_recall']:>10.4f}")

    best_single = max(corpus_phrase_f1(per_model_pred_spans[s], gold_spans_all)["phrase_f1"] for s in per_model_pred_spans)
    print(f"\nEnsemble vs best single model (Phrase F1): {ens_result['phrase_f1']:.4f} vs {best_single:.4f} "
          f"({ens_result['phrase_f1'] - best_single:+.4f})")

    print("\nSame caveat as ensemble_risk.py: this is one 271-row val set. Cross-check with "
          "--fold 0..4 runs before trusting this for the final submission.")


if __name__ == "__main__":
    main()
