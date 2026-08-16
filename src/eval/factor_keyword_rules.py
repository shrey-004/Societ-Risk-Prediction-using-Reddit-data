"""
Phase 4 (addendum) — keyword-rule hybrid layer for the weakest factor
categories.

WHY THIS EXISTS: 7 of 24 factor categories score exactly 0.000 F1 in
reports/phase4_factor_classifier_report.md even after per-class threshold
tuning (reports/phase4_threshold_tuning_report.md), and it's NOT purely a
data-scarcity problem for all of them — e.g. "physical health/
characteristic" has 74 positive TRAINING examples (5.4% of train) and
still gets 0.000. Because Subtask 2 is scored as an unweighted Macro F1
across all 24 categories, every category below contributes equally
(1/24 ≈ 4.2%) regardless of how rare it is — so lifting even one
zero-scoring category from 0.00 to ~0.25 is worth roughly as much to the
final macro F1 as several points of improvement on "hopelessness" (already
your best-performing category at 0.63-0.70 F1). This is the single
highest-leverage lever available for Subtask 2 score right now.

These keyword lists are DELIBERATELY simple and a STARTING POINT, not a
finished solution — they encode the kind of pattern-matching a human
skimming for these categories would use, but haven't been tuned against
your data. Some risk of false positives — measure the actual effect with
scripts/evaluate_keyword_boost.py before trusting this in your final
submission, and refine the lists based on what you see (false positives
you can eyeball, missed true positives in val rows the model + rules both
miss). Categories not listed here are left to the neural model alone.
"""
import re

# Each entry: factor name -> list of (compiled regex) patterns, word-boundary
# and case-insensitive. Deliberately conservative — favor precision over
# maximal recall, since a wrong factor prediction still costs precision on
# an otherwise-improving category.
_RAW_PATTERNS: dict[str, list[str]] = {
    "physical health/characteristic": [
        r"chronic (pain|illness|condition)", r"diagnosed with", r"disab(led|ility)",
        r"medical condition", r"health condition", r"\bcancer\b", r"\bdiabetes\b",
        r"\bsurgery\b", r"physically (sick|ill)", r"\bobese\b", r"overweight",
        r"my (body|weight|appearance)", r"\bdisfigur",
    ],
    "substance use": [
        r"\b(drank|drinking|drunk)\b", r"\balcohol(ic)?\b", r"\bweed\b", r"\bmarijuana\b",
        r"\bcocaine\b", r"\bmeth\b", r"\bhigh on\b", r"drug (use|addiction|problem)",
        r"\baddicted\b", r"\brelapse[ds]?\b", r"substance (abuse|use)",
    ],
    "poor school performance": [
        r"fail(ing|ed) (a )?(class|exam|test|school|course)", r"bad grades",
        r"dropp(ed|ing) out", r"\bexpelled\b", r"\bflunk(ed|ing)?\b", r"\bgpa\b",
        r"school performance", r"failing school",
    ],
    "low socio-economic status": [
        r"\bpoverty\b", r"can'?t afford", r"\bhomeless\b", r"\bbroke\b(?! up)",
        r"no money", r"\bunemployed\b", r"in debt\b", r"\bevict(ed|ion)\b",
        r"can'?t pay (rent|bills)", r"financial(ly)? (struggl|troubl)",
    ],
    "exposure to others' suicide": [
        # broadened after checking real positives — the dataset skews toward
        # ATTEMPTED (not just completed) suicide by someone close, and toward
        # reading about strangers' deaths online, more than the "committed
        # suicide" phrasing the first pass of this lexicon assumed
        r"(friend|brother|sister|mom|dad|mother|father|cousin|classmate|someone i (know|knew))\w* "
        r"(attempt(ed|ing)? (suicide|to (commit|kill))|committed suicide|killed (himself|herself|themselves)|"
        r"died by suicide|tried to (commit|kill))",
        r"(watch(ed|ing)|saw|found) (my |his |her |their )?(brother|sister|mom|dad|friend|him|her|them)\w* "
        r"(attempt|die|dead|kill (himself|herself))",
        r"(lost|lose) (my|a) \w+ to suicide", r"reading about (people|others) (who )?(died|die)\b",
        r"people (who have )?died by (taking their|suicide)",
    ],
    "sexual orientation related issues": [
        r"\b(gay|lesbian|bisexual|queer|lgbtq?\+?|transgender)\b", r"\btrans\b",
        r"coming out", r"\bcloseted\b", r"homophob(ic|ia)",
    ],
    # LOWER CONFIDENCE than the others above — checked against real data and
    # this factor is mostly expressed implicitly ("I can't do this to my
    # family", "my friends would be devastated") rather than through a
    # consistent lexical pattern, so keyword recall will likely stay low
    # (~0.02 in a spot-check) no matter how this list is tuned. Included for
    # completeness / as a base to build on, not because it's expected to
    # move the needle much — this category is a better candidate for the
    # few-shot LLM approach mentioned in the roadmap doc than for regex.
    "sense of responsibility": [
        r"(my (kids?|children|family|mom|dad|parents)) need(s)? me",
        r"who would take care of", r"responsible for (my|them|him|her)",
        r"have to be (here|there) for", r"can'?t leave (my|them)",
        r"(don'?t|can'?t) (want to )?(make|do this to) my family",
        r"take away my parents'? child", r"(friends|family) would (be devastated|drop me|get over)",
    ],
}

FACTOR_KEYWORD_PATTERNS: dict[str, list[re.Pattern]] = {
    factor: [re.compile(p, re.IGNORECASE) for p in patterns]
    for factor, patterns in _RAW_PATTERNS.items()
}


def keyword_hits(post_text: str, factor: str) -> list[str]:
    """Return the list of matched substrings for a given factor category
    (empty if none). Useful for debugging/auditing false positives."""
    patterns = FACTOR_KEYWORD_PATTERNS.get(factor, [])
    hits = []
    for p in patterns:
        m = p.search(post_text)
        if m:
            hits.append(m.group(0))
    return hits


def apply_keyword_boost(
    post_text: str,
    model_predicted: set[str],
    boost_categories: list[str] | None = None,
) -> set[str]:
    """Union the model's predicted factor set with keyword-rule hits for
    the given boost_categories (defaults to all categories with a lexicon
    entry). A keyword hit ADDS a factor if the model didn't already predict
    it; never removes a model prediction. This only ever increases recall
    on the targeted categories — measure the precision cost with
    scripts/evaluate_keyword_boost.py before trusting it."""
    if boost_categories is None:
        boost_categories = list(FACTOR_KEYWORD_PATTERNS.keys())

    boosted = set(model_predicted)
    for factor in boost_categories:
        if factor in boosted:
            continue
        if keyword_hits(post_text, factor):
            boosted.add(factor)
    return boosted
