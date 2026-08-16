"""
Phase 3 (data-quality addendum) — audit every evidence span that does NOT
exact-match its post verbatim, across the FULL labeled dataset (train +
val). Surfaces three buckets so a human can spend 10 minutes reviewing a
short list instead of trusting automated matching blindly:

  A. RECOVERABLE (case-insensitive)  — always safe, no review needed.
  B. FUZZY-RECOVERABLE               — ratio >= threshold; usually fine
                                        (typos, dropped word) but can be
                                        wrong when text shares a long
                                        common prefix/suffix with a
                                        different core word (see FLAGGED).
  C. FLAGGED / UNRECOVERABLE         — either fuzzy found nothing, or it
                                        found something but a lexical
                                        polarity check thinks the match
                                        might contradict the gold span
                                        (e.g. gold mentions dying, matched
                                        text mentions living/stopping).
                                        These are candidates for manual
                                        fix or exclusion — some may be
                                        genuine annotation errors in the
                                        source PFA dataset, not bugs in
                                        this pipeline (found one such case:
                                        row P01511).

This does NOT modify any data. It only writes a CSV report. Decide what to
do with each flagged row yourself, then either hand-fix data/raw/train.xlsx
or leave it as-is (the current default pipeline already skips these safely
by not tagging any B/I-EVID tokens for them — the only risk is if you flip
`use_fuzzy=True` in dataset_evidence.py without reviewing this first).

Usage (from esrd_project/ root):
    python scripts/audit_evidence_matches.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_evidence import _fuzzy_find

# Very small, deliberately conservative keyword sets — this is a coarse
# sanity check, not a classifier. Only used to flag matches for human
# review, never to auto-accept or auto-reject anything.
RISK_WORDS = {
    "die", "died", "dying", "kill", "killed", "killing", "suicide", "suicidal",
    "overdose", "od", "harm", "hurt", "cut", "cutting", "end it", "ending",
    "pills", "rope", "gun", "jump", "bleed", "bleeding",
}
PROTECTIVE_OR_NEGATION_WORDS = {
    "live", "living", "alive", "stop", "stopped", "stay", "staying",
    "won't", "wont", "don't want", "dont want", "don't wanna", "dont wanna",
    "never", "not going", "want to live", "wanna live",
}


def _has_any(text: str, words: set[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def _polarity_suspicious(gold: str, matched: str) -> bool:
    """True if gold looks risk-related but the matched text looks
    protective/negating with no risk words of its own — a signal (not
    proof) that the fuzzy match grabbed the wrong sentence."""
    gold_has_risk = _has_any(gold, RISK_WORDS)
    matched_has_risk = _has_any(matched, RISK_WORDS)
    matched_has_protective = _has_any(matched, PROTECTIVE_OR_NEGATION_WORDS)
    return gold_has_risk and not matched_has_risk and matched_has_protective


def main():
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"

    train_df = pd.read_csv(processed_dir / "train_clean.csv")
    val_df = pd.read_csv(processed_dir / "val_clean.csv")
    train_df["split"] = "train"
    val_df["split"] = "val"
    full = pd.concat([train_df, val_df], ignore_index=True)

    import ast
    rows = []
    for _, r in full.iterrows():
        spans = ast.literal_eval(r["evidence_spans"])
        post = r["post_clean"]
        for span in spans:
            if not span:
                continue
            if span in post:
                continue  # exact match, nothing to audit

            if span.lower() in post.lower():
                bucket = "A_case_insensitive_recoverable"
                start = post.lower().find(span.lower())
                matched_text = post[start:start + len(span)]
                flag = ""
            else:
                fuzzy = _fuzzy_find(post, span, min_ratio=0.92)
                if fuzzy is None:
                    bucket = "C_unrecoverable"
                    matched_text = ""
                    flag = "no fuzzy candidate cleared 0.92 ratio"
                else:
                    s, e = fuzzy
                    matched_text = post[s:e]
                    suspicious = _polarity_suspicious(span, matched_text)
                    bucket = "C_flagged_fuzzy" if suspicious else "B_fuzzy_recoverable"
                    flag = "POLARITY MISMATCH — review before trusting" if suspicious else ""

            rows.append({
                "split": r["split"],
                "row_id": r["row_id"],
                "risk_label": r["risk_label"],
                "gold_span": span,
                "bucket": bucket,
                "matched_text": matched_text,
                "flag": flag,
                "post_excerpt": post[:200],
            })

    report_df = pd.DataFrame(rows)
    out_path = processed_dir / "evidence_match_audit.csv"
    report_df.to_csv(out_path, index=False)

    print(f"Total non-exact spans audited: {len(report_df)}")
    print(report_df["bucket"].value_counts().to_string())
    n_flagged = (report_df["bucket"] == "C_flagged_fuzzy").sum()
    n_unrecoverable = (report_df["bucket"] == "C_unrecoverable").sum()
    print(f"\n{n_flagged} rows flagged for POLARITY MISMATCH — review these first, "
          f"they're the highest risk of being genuine annotation errors like P01511.")
    print(f"{n_unrecoverable} rows have no confident match at all (left as-is, "
          f"same as current default behavior — no training signal lost beyond "
          f"what's already being skipped today).")
    print(f"\nFull report written to {out_path}")
    print("\nSuggested next step: open that CSV, sort by bucket, read every "
          "C_flagged_fuzzy row's post_excerpt next to its gold_span. For genuine "
          "annotation errors, hand-correct data/raw/train.xlsx (or drop the row's "
          "evidence, keep the row for Subtask 1/2). Then decide whether to also "
          "flag this as a known data-quality caveat in your solution report — "
          "reviewers score 'approach innovation, experiments, writing', and a "
          "documented data audit is a legitimate strength there.")


if __name__ == "__main__":
    main()
