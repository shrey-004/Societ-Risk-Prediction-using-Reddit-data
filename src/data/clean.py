"""
Phase 1 — cleaning.

Handles the three data-quality issues found in the Phase 0 audit:
  1. Risk labels have inconsistent casing/whitespace
     ('indicator', 'Indicator', 'ideation ', ...)
  2. Factors column is a stringified Python list, needs literal_eval + dedup
     (raw lists contain repeats — one entry per tagged span, not per unique
     factor — so we dedupe for the multi-label classification target while
     keeping the raw count as an auxiliary column, in case it's useful signal
     for weighting/analysis later)
  3. Post text has occasional leading/trailing whitespace and outlier lengths
"""
import ast
import html
import re
import pandas as pd

CANONICAL_RISK_LABELS = ["Indicator", "Ideation", "Behavior", "Attempt"]
_RISK_LOOKUP = {label.lower(): label for label in CANONICAL_RISK_LABELS}


def normalize_risk_label(raw: str) -> str:
    """'ideation ' / 'IDEATION' / 'Ideation' -> 'Ideation'. Raises on unknowns
    so silent data-quality issues surface immediately rather than propagating."""
    if raw is None:
        raise ValueError("risk label is None")
    key = str(raw).strip().lower()
    if key not in _RISK_LOOKUP:
        raise ValueError(f"Unrecognized risk label: {raw!r}")
    return _RISK_LOOKUP[key]


def parse_factors(raw) -> list[str]:
    """Parse the stringified list, strip whitespace on each entry, return
    the DEDUPED unique factor set (the classification target for Subtask 2).
    Empty/missing -> []."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        items = raw
    else:
        try:
            items = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            raise ValueError(f"Could not parse factors list: {raw!r}")
    items = [str(x).strip() for x in items if str(x).strip()]
    # dedupe, preserve first-seen order
    seen = set()
    unique = []
    for it in items:
        if it not in seen:
            seen.add(it)
            unique.append(it)
    return unique


_QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",   # smart single quotes -> straight
    "\u201c": '"', "\u201d": '"',   # smart double quotes -> straight
    "\u2013": "-", "\u2014": "-",   # en/em dash -> hyphen
})


def clean_text(text: str) -> str:
    """Normalize whitespace, HTML entities, and smart-quote/dash variants
    only — do NOT alter clinical content or actual wording, since evidence
    spans must still be substring-matchable against the cleaned text in
    Phase 3.

    Two encoding mismatches found via the evidence_in_post QA check below,
    both fixed here at the source rather than papered over downstream:
      1. Raw Reddit export left HTML entities un-escaped ('&amp;' vs '&')
      2. Post text uses smart quotes/dashes (', ", –) while the evidence
         annotations use straight ASCII equivalents (', ", -)
    """
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = text.translate(_QUOTE_MAP)
    # collapse repeated whitespace/newlines but keep single newlines as spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_evidence_spans(evidence: str) -> list[str]:
    """Split the semicolon-joined evidence string into individual spans.
    'none' (case-insensitive) means no evidence was annotated -> []."""
    if evidence is None or str(evidence).strip().lower() in ("none", ""):
        return []
    return [s.strip() for s in str(evidence).split(";") if s.strip()]


def clean_dataframe(df: pd.DataFrame, max_word_len: int = 1000) -> pd.DataFrame:
    """Apply all cleaning steps. Returns a new DataFrame with clean columns
    added; raw columns are preserved for audit/debugging."""
    df = df.copy()

    df["risk_label"] = df["risk_raw"].apply(normalize_risk_label)
    df["factors"] = df["factors_raw"].apply(parse_factors)
    df["num_factors"] = df["factors"].apply(len)
    df["post_clean"] = df["post_text"].apply(clean_text)
    df["evidence_clean"] = df["evidence_raw"].apply(clean_text)
    df["word_count"] = df["post_clean"].apply(lambda t: len(t.split()))

    # flag (don't silently drop) extreme-length outliers — decide truncation
    # strategy at tokenization time in Phase 2, not here
    df["is_length_outlier"] = df["word_count"] > max_word_len

    # Evidence is a semicolon-separated list of spans (or the literal
    # string 'none' when no explicit evidence was annotated). Verify each
    # span individually is a verbatim substring of the cleaned post — this
    # is the real correctness check Phase 3 (span extraction) depends on.
    df["evidence_spans"] = df["evidence_clean"].apply(_split_evidence_spans)
    df["evidence_in_post"] = df.apply(
        lambda r: all(span in r["post_clean"] for span in r["evidence_spans"]),
        axis=1,
    )

    return df