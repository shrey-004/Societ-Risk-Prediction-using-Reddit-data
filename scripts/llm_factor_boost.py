"""
Phase 8 (Tier C) — LLM few-shot pass for the weakest Subtask 2 factor
categories, via headless Claude Code CLI calls (no API key needed).

WHY: 5-fold CV on the mental-bert factor classifier (see
reports/phase4_factor_classifier_report_fold{0..4}.md) shows 9/24
categories still near-zero F1 despite the encoder swap — all either rare
(single-digit-to-low-double-digit support) or implicit/non-lexical
(e.g. "sense of responsibility"). A small fine-tuned classifier can't
learn much from a handful of positive examples; an LLM's prior semantic
understanding of these concepts is a better fit. This script asks Claude
(via `claude -p`, batched across posts to keep call count sane) to tag
ONLY these 9 categories, grounded in the verbatim taxonomy definitions
from the dataset paper (Li et al. 2025, Table III) plus 2 real few-shot
examples per category. Output is unioned with the BERT classifier's
predictions (never removes a BERT prediction — same safe pattern as
src/eval/factor_keyword_rules.py's apply_keyword_boost), so it can only
help recall on the targeted categories.

VERIFY BEFORE TRUST: run --mode validate first against a held-out sample
with known labels (data prep in this repo's scratch dir, not checked in —
see the session that built this) to get real precision/recall per target
category before ever touching the test set. Only run --mode predict on
the actual test set after that comes back positive.

Usage:
    python scripts/llm_factor_boost.py --mode validate --input path/to/val_sample.csv --output val_preds.csv
    python scripts/llm_factor_boost.py --mode predict --input data/raw/test.xlsx --output llm_factor_preds.csv
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

TARGET_FACTORS = [
    "physical health/characteristic",
    "substance use",
    "poor school performance",
    "low socio-economic status",
    "exposure to others' suicide",
    "cognitive deficits",
    "sexual orientation related issues",
    "sense of responsibility",
    "meaning in life",
]

# Verbatim from Li et al. 2025 (arXiv:2507.10008), Table III.
DEFINITIONS = {
    "physical health/characteristic": "Physical health issues (e.g., COVID-19, obesity/underweight).",
    "substance use": "The uncontrolled use of drugs/alcohol/tobacco.",
    "poor school performance": "Low school performance (e.g., failing tests, bad grades).",
    "low socio-economic status": "Unemployment status, poverty, and homeless, etc.",
    "exposure to others' suicide": "Mentioning or describing others' suicidal thoughts, attempt or death.",
    "cognitive deficits": "Having difficulty in cognitive abilities.",
    "sexual orientation related issues": "Having gender/sexual disorder, same-sex relationship, etc.",
    "sense of responsibility": "(Protective factor) Awareness of responsibility to one's own health/survival and to others.",
    "meaning in life": "(Protective factor) Involves cognitive component, motivational component and affective component.",
}

FEWSHOT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "llm_factor_fewshot.json"


def build_system_prompt(fewshot: dict) -> str:
    lines = [
        "You are annotating Reddit r/SuicideWatch posts for a research dataset, using a fixed factor taxonomy.",
        "You will be given a numbered batch of posts. For EACH post, decide which of the following 9 factor",
        "categories are clearly supported by the text. Most posts will have ZERO of these — they are all rare",
        "or implicit categories, and false positives hurt as much as missed ones. Only tag a category if the",
        "post gives clear textual support; do not guess or infer from tone alone.",
        "",
        "Categories and definitions:",
    ]
    for name in TARGET_FACTORS:
        lines.append(f'- "{name}": {DEFINITIONS[name]}')
    lines.append("")
    lines.append("Examples of real posts (from the training set) that DO contain the named factor:")
    for name in TARGET_FACTORS:
        for ex in fewshot.get(name, []):
            snippet = ex["post_clean"][:400]
            lines.append(f'  [{name}] "{snippet}"')
    lines.append("")
    lines.append(
        "Respond with ONLY a JSON object mapping each post's row_id to a list of applicable category names "
        "(use the exact category strings above; empty list if none apply). No prose, no markdown fences."
    )
    return "\n".join(lines)


def build_batch_prompt(batch: pd.DataFrame) -> str:
    lines = ["Posts to annotate:", ""]
    for _, row in batch.iterrows():
        text = str(row["post_clean"])[:1500]
        lines.append(f'row_id: {row["row_id"]}')
        lines.append(f'post: "{text}"')
        lines.append("")
    return "\n".join(lines)


def call_claude(prompt: str, model: str = None) -> str:
    cmd = ["claude", "-p", "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:500]}")
    return result.stdout


def parse_json_response(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in response: {text[:300]}")
    return json.loads(m.group(0))


def run_batches(df: pd.DataFrame, system_prompt: str, batch_size: int, model: str = None) -> dict:
    all_results = {}
    n_batches = (len(df) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = df.iloc[b * batch_size: (b + 1) * batch_size]
        prompt = system_prompt + "\n\n" + build_batch_prompt(batch)
        try:
            raw = call_claude(prompt, model=model)
            parsed = parse_json_response(raw)
        except Exception as e:
            print(f"  batch {b+1}/{n_batches}: FAILED ({e}) — treating as empty for this batch")
            parsed = {}
        for _, row in batch.iterrows():
            rid = str(row["row_id"])
            factors = parsed.get(rid, [])
            # keep only valid target factor names, defensively
            factors = [f for f in factors if f in TARGET_FACTORS]
            all_results[rid] = factors
        print(f"  batch {b+1}/{n_batches} done ({len(batch)} posts)")
    return all_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["validate", "predict"], required=True)
    p.add_argument("--input", type=str, required=True, help="CSV with row_id + post_clean columns, or test.xlsx")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()

    fewshot = json.load(open(FEWSHOT_PATH))
    system_prompt = build_system_prompt(fewshot)

    if args.input.endswith(".xlsx"):
        df = pd.read_excel(args.input, sheet_name="Sheet1")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.data.clean import clean_text
        df["post_clean"] = df["post"].apply(clean_text)
    else:
        df = pd.read_csv(args.input)

    print(f"Running LLM factor pass on {len(df)} posts, batch_size={args.batch_size} ...")
    results = run_batches(df, system_prompt, args.batch_size, model=args.model)

    out_rows = [{"row_id": rid, "llm_factors": json.dumps(factors)} for rid, factors in results.items()]
    pd.DataFrame(out_rows).to_csv(args.output, index=False)
    print(f"Saved {len(out_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
