"""
Phase 3 — PyTorch Dataset for evidence span extraction (BIO token tagging).

Evidence spans are annotated as exact substrings of post_clean (semicolon-
separated, parsed in Phase 1). We convert each post into a token classification
problem: O = not evidence, B-EVID = first token of an evidence span,
I-EVID = continuation of one.

Special/padding tokens get label -100 so they're ignored by the loss
(standard HF token-classification convention).
"""
import ast
import difflib
import re

import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL2ID = {"O": 0, "B-EVID": 1, "I-EVID": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _fuzzy_find(post: str, span: str, min_ratio: float = 0.85) -> tuple[int, int] | None:
    """Last-resort approximate locator for evidence spans that are near-
    verbatim but not an exact substring (annotator normalization like
    'kill my self' -> 'kill myself', stray punctuation, a dropped word).
    Slides a window of the span's own word-length (+/-2 words) across the
    post and keeps the best difflib match above min_ratio. O(len(post)) in
    windows, fine at these post lengths (<= a few thousand chars).
    Returns None if nothing clears the bar — better to drop a span than
    tag the wrong text as evidence."""
    span_words = span.split()
    post_words = post.split()
    if not span_words or not post_words:
        return None

    best_ratio, best_start_w, best_end_w = 0.0, None, None
    for win in range(max(1, len(span_words) - 2), len(span_words) + 3):
        for i in range(0, len(post_words) - win + 1):
            candidate = " ".join(post_words[i:i + win])
            ratio = difflib.SequenceMatcher(None, candidate.lower(), span.lower()).ratio()
            if ratio > best_ratio:
                best_ratio, best_start_w, best_end_w = ratio, i, i + win

    if best_ratio < min_ratio or best_start_w is None:
        return None

    # convert word indices back to character offsets in `post`
    char_pos, word_start_char, word_end_char = 0, None, None
    for wi, w in enumerate(post_words):
        idx = post.find(w, char_pos)
        if wi == best_start_w:
            word_start_char = idx
        if wi == best_end_w - 1:
            word_end_char = idx + len(w)
            break
        char_pos = idx + len(w)
    if word_start_char is None or word_end_char is None:
        return None
    return (word_start_char, word_end_char)


def find_span_char_ranges(
    post: str,
    spans: list[str],
    use_fuzzy: bool = False,
    fuzzy_min_ratio: float = 0.92,
) -> list[tuple[int, int]]:
    """Find ALL occurrences of each evidence span in the post (a repeated
    phrase used twice as evidence likely means both occurrences matter).

      1. Exact substring (case-sensitive) — the original, most trustworthy.
      2. Case-insensitive substring — ALWAYS on. Purely mechanical (zero
         semantic risk) and recovers ~18% of previously-dropped spans
         (annotators sometimes retyped evidence with different casing).
      3. Fuzzy sliding-window match — OFF by default. During the Phase-1.5
         audit this recovered more spans but ALSO surfaced at least one
         case where the gold annotation itself is wrong (row P01511: gold
         evidence 'don't wanna die' appears nowhere in a post that actually
         reads 'I wanna live... I don't wanna be a sob story' — fuzzy
         matching grabbed the unrelated 'don't wanna be' and would have
         silently taught the model the opposite of what that sentence
         means). Character-similarity ratio alone can't tell 'die' from
         'be' apart when they share a long common prefix. Do NOT flip this
         on for training without first running
         `scripts/audit_evidence_matches.py` and eyeballing the flagged
         rows — it's built for exactly that review step."""
    ranges = []
    for span in spans:
        if not span:
            continue
        exact = [(m.start(), m.end()) for m in re.finditer(re.escape(span), post)]
        if exact:
            ranges.extend(exact)
            continue

        ci_matches = [(m.start(), m.end()) for m in re.finditer(re.escape(span), post, flags=re.IGNORECASE)]
        if ci_matches:
            ranges.extend(ci_matches)
            continue

        if use_fuzzy:
            fuzzy = _fuzzy_find(post, span, min_ratio=fuzzy_min_ratio)
            if fuzzy is not None:
                ranges.append(fuzzy)
    return ranges


def align_bio_labels(offset_mapping, char_ranges: list[tuple[int, int]]) -> list[int]:
    """Convert a list of (start,end) character ranges into per-token BIO
    label ids, given a tokenizer's offset_mapping (list of (start,end) per
    token; (0,0) marks special/padding tokens -> -100)."""
    labels = []
    for start, end in offset_mapping:
        if start == end:  # special/padding token
            labels.append(-100)
            continue
        label = LABEL2ID["O"]
        for span_start, span_end in char_ranges:
            if start >= span_start and end <= span_end:
                # first token whose start == span_start is the beginning
                label = LABEL2ID["B-EVID"] if start == span_start else LABEL2ID["I-EVID"]
                break
            elif start < span_end and end > span_start:
                # partial overlap (tokenizer split across a span boundary)
                label = LABEL2ID["I-EVID"]
                break
        labels.append(label)
    return labels


class EvidenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256,
                 use_fuzzy: bool = False, fuzzy_min_ratio: float = 0.92):
        self.posts = df["post_clean"].tolist()
        self.spans_list = [ast.literal_eval(s) for s in df["evidence_spans"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_fuzzy = use_fuzzy
        self.fuzzy_min_ratio = fuzzy_min_ratio

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        post = self.posts[idx]
        spans = self.spans_list[idx]
        enc = self.tokenizer(
            post,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = enc.pop("offset_mapping").squeeze(0).tolist()
        char_ranges = find_span_char_ranges(
            post, spans, use_fuzzy=self.use_fuzzy, fuzzy_min_ratio=self.fuzzy_min_ratio
        )
        labels = align_bio_labels(offset_mapping, char_ranges)

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item