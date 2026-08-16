"""
Official "Phrase F1" metric for evidence extraction, per the competition's
exact rules (from the organizers' task description):

  - Case-insensitive normalization
  - A predicted phrase is correct if it CONTAINS a ground-truth phrase OR
    is CONTAINED BY a ground-truth phrase (bidirectional substring match)
    e.g. predicted "kill myself" matches gold "I want to kill myself"
  - A predicted span only counts if its token length <= 3x the token
    length of the matched gold span (blocks submitting the whole post)
  - One-to-one matching: each predicted span matches at most one gold
    span, and vice versa (per post)
  - Per-post: precision = matched_preds / n_preds, recall = matched_golds / n_golds
  - Per-post F1 = harmonic mean(precision, recall)
  - Final score = mean of per-post F1 across all posts

This is NOT strict exact-match (too harsh) and NOT plain token-overlap
(too lenient) — it's specifically this containment-based, length-capped,
one-to-one matching scheme. Implemented here exactly as specified so we
can estimate our real leaderboard-equivalent score before submission.
"""


def _normalize(phrase: str) -> str:
    return phrase.strip().lower()


def _token_len(phrase: str) -> int:
    return len(phrase.split())


def _is_valid_match(pred: str, gold: str) -> bool:
    """Containment (either direction) + the 3x length cap, both computed
    on normalized text."""
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return False
    contained = (p in g) or (g in p)
    if not contained:
        return False
    # length cap relative to the GOLD phrase being matched
    return _token_len(pred) <= 3 * _token_len(gold)


def match_spans_one_post(pred_spans: list[str], gold_spans: list[str]) -> tuple[int, int, int]:
    """Greedy one-to-one matching for a single post.
    Returns (n_matched, n_preds, n_golds)."""
    gold_used = [False] * len(gold_spans)
    pred_used = [False] * len(pred_spans)

    n_matched = 0
    for i, pred in enumerate(pred_spans):
        for j, gold in enumerate(gold_spans):
            if gold_used[j]:
                continue
            if _is_valid_match(pred, gold):
                gold_used[j] = True
                pred_used[i] = True
                n_matched += 1
                break

    return n_matched, len(pred_spans), len(gold_spans)


def phrase_f1_one_post(pred_spans: list[str], gold_spans: list[str]) -> float:
    """Per-post Phrase F1, handling the zero-span edge cases explicitly:
      - both empty -> 1.0 (trivially correct: nothing to find, nothing predicted)
      - gold empty, pred non-empty -> 0.0 (false positives, no gold to match)
      - gold non-empty, pred empty -> 0.0 (missed everything)
    """
    if not gold_spans and not pred_spans:
        return 1.0
    if not gold_spans or not pred_spans:
        return 0.0

    n_matched, n_preds, n_golds = match_spans_one_post(pred_spans, gold_spans)
    precision = n_matched / n_preds if n_preds > 0 else 0.0
    recall = n_matched / n_golds if n_golds > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def corpus_phrase_f1(all_pred_spans: list[list[str]], all_gold_spans: list[list[str]]) -> dict:
    """Final Phrase F1 = mean of per-post F1 across all posts, plus
    aggregate precision/recall for diagnostics."""
    assert len(all_pred_spans) == len(all_gold_spans)
    per_post_f1 = []
    total_matched = total_preds = total_golds = 0
    for preds, golds in zip(all_pred_spans, all_gold_spans):
        per_post_f1.append(phrase_f1_one_post(preds, golds))
        if preds and golds:
            n_matched, n_p, n_g = match_spans_one_post(preds, golds)
            total_matched += n_matched
            total_preds += n_p
            total_golds += n_g

    mean_f1 = sum(per_post_f1) / len(per_post_f1) if per_post_f1 else 0.0
    agg_precision = total_matched / total_preds if total_preds > 0 else 0.0
    agg_recall = total_matched / total_golds if total_golds > 0 else 0.0

    return {
        "phrase_f1": mean_f1,
        "aggregate_precision": agg_precision,
        "aggregate_recall": agg_recall,
        "n_posts": len(per_post_f1),
    }