# Explainable Suicide Risk Detection — IEEE BigData 2026 Cup

Pipeline for the challenge organized by Prof. Qing Li's team (HKPU / CityU HK),
built on the Protective Factor-Aware (PFA) dataset.

## Task recap
- **Subtask 1 (70%)** — classify risk level (`Indicator` / `Ideation` / `Behavior` / `Attempt`)
  + extract supporting evidence spans.
- **Subtask 2 (30%)** — multi-label classification of 24 risk/protective factors.
- Composite score: Macro-F1 per subtask, weighted average.
  **⚠️ Open question:** invite email says 70/30, challenge-page blurb says
  S = 0.6·S1 + 0.4·S2. Confirm with organizers (hialex.li@connect.polyu.hk)
  before the final submission — currently defaulted to 70/30 in `configs/config.yaml`.

## Repo structure
```
esrd_project/
├── configs/config.yaml       # single source of truth for all hyperparameters/paths
├── data/
│   ├── raw/train.xlsx        # original file, untouched
│   └── processed/            # Phase 1 output goes here
│       ├── folds/            # 5-fold CV splits (scripts/make_kfold_splits.py)
│       └── evidence_match_audit.csv  # scripts/audit_evidence_matches.py output
├── src/
│   ├── data/                 # loading, cleaning, splitting
│   ├── models/                # Phase 2+ model code
│   ├── eval/                  # Phase 6 scoring harness + factor_keyword_rules.py
│   └── utils/
├── scripts/
│   ├── check_env.py          # run first on your A100 machine
│   ├── make_kfold_splits.py  # generate the 5 CV folds
│   ├── train_{risk_classifier,evidence_extractor,factor_classifier}.py
│   │                          # each supports --fold N / --full_data / --seed / --suffix
│   ├── ensemble_{risk,evidence,factors}.py   # prove-before-use seed ensembling
│   ├── tune_factor_thresholds{,_cv}.py       # _cv version pools all 5 folds' OOF preds
│   ├── tune_evidence_confidence.py           # precision/recall tradeoff sweep
│   ├── evaluate_keyword_boost.py             # measures the rule-based recall boost
│   ├── audit_evidence_matches.py             # surfaces annotation-quality issues
│   └── generate_submission.py                # final CSV, many optional flags — see its docstring
├── notebooks/                 # EDA (Phase 1)
├── outputs/
│   ├── checkpoints/           # trained model weights
│   ├── logs/                  # tensorboard / training logs
│   └── predictions/           # val + test predictions for scoring
└── reports/                    # solution report drafts (Phase 7)
```

See `PRIORITIZED_IMPROVEMENTS.md` for the full analysis behind these additions,
in priority order with expected impact and exact commands to run.


## What we found in `train.xlsx` (Phase 0 audit)
- 1,635 posts from 153 unique users (7–30 posts/user).
- Subtask 1 labels are **inconsistent casing/whitespace**
  (`'indicator'`, `'Indicator'`, `'ideation '`, ...) — must normalize in Phase 1.
- Evidence text present for all 1,635 rows.
- Subtask 2 factors stored as a stringified Python list — needs `ast.literal_eval`,
  24 unique factors, **heavily imbalanced** (`hopelessness`: 1,608 occurrences vs.
  `sexual orientation related issues`: 12). This imbalance must shape the Phase 4
  loss function (e.g. class-weighted BCE or focal loss).
- Post length ranges from 1 to 2,066 words (avg ~69) — a few extreme outliers to
  cap/handle in tokenization.
- Official task is phrased as **user-level** risk classification, but labels here
  are **post-level**. Plan: train post-level (Phase 2), add a user-level
  aggregation layer (Phase 5) for the final submission format.

## Phase plan (strict — one phase at a time)
| # | Phase | Status |
|---|-------|--------|
| 0 | Repo scaffolding, env check | ✅ |
| 1 | Data cleaning + EDA + user-grouped train/val split | ✅ |
| 2 | Subtask 1 — risk classifier (encoder fine-tune) | ✅ |
| 3 | Subtask 1 — evidence span extraction | ✅ |
| 4 | Subtask 2 — multi-label factor classifier | ✅ |
| 5 | Post → user-level aggregation | ⬜ (official task is user-level; submission format here is post-level per the confirmed rules doc — revisit if organizers clarify) |
| 6 | Official scoring harness | ✅ (src/eval/phrase_f1.py) |
| 7 | Inference pipeline + solution report draft | ✅ (generate_submission.py); report draft ⬜ |
| 8 | Leaderboard-driven improvement pass (this delivery) | see `PRIORITIZED_IMPROVEMENTS.md` |

Real leaderboard position (2026-07-12): rank 19/27, composite 0.6008
(Subtask1 0.6945, Subtask2 0.3820) — up from rank 22/0.5359 after the
first roadmap pass (512 seq len, evidence fuzzy-match recovery, fulldata
refits, CV-pooled factor thresholds).

Second pass (same day): 5-fold CV run for risk classifier and evidence
extractor (previously only done for factors) confirmed both already
generalize well — pooled CV weighted-F1 0.755 (risk) and Phrase F1 0.693
(evidence), both close to the single-split estimates, so the val-vs-
leaderboard gap was never really there. The factor classifier was the
real gap: swapping its encoder from Bio_ClinicalBERT to mental-bert
(domain match — same model family as Subtask 1, pretrained on
suicide-adjacent Reddit text) beat Bio_ClinicalBERT on **every one of the
5 CV folds**, pooled macro F1 0.266 -> 0.401 raw. Combined with a
lower per-class threshold floor (0.30, safe now that `--factor_max_k 10`
structurally caps the over-firing tail regardless of threshold — verified
by simulating the cap directly on pooled OOF predictions), capped macro F1
reached 0.418 vs 0.399 at the old floor of 0.50. New submission generated
with: risk/evidence unchanged (`_fulldata`), evidence extractor retrained
with fuzzy-match evidence recovery (validated neutral-to-positive over
CV), `--evidence_min_confidence 0.50` (small, cross-fold-consistent
precision/recall win), factor classifier now mental-bert
(`factor_classifier_best_mentalbert_fulldata` +
`factor_thresholds_cv_mentalbert.json`).

Third pass (same day, Tier C): LLM few-shot pass (`scripts/
llm_factor_boost.py`, headless `claude -p` calls, no API key needed) for
the 9 factor categories still near-zero F1 after the mental-bert swap
(all rare and/or implicit — see reports/phase4_factor_classifier_report_
fold*.md for the ranked list). Grounded in the verbatim taxonomy
definitions from the dataset paper (Li et al. 2025, arXiv:2507.10008,
Table III — fetched directly, not guessed) plus 2 real few-shot examples
per category. Validated on 361 held-out labeled posts (disjoint from the
few-shot examples) before touching the test set: BERT-alone F1 on these
9 categories averaged ~0.28, LLM-alone ~0.59, union (never removes a BERT
prediction, only adds — see `--llm_factor_preds` in generate_submission.py)
~0.53 average, improving all 9/9 categories. Applied to the real 378-row
test set and unioned in; 55/378 rows gained at least one additional
factor. This is the current `outputs/predictions/RayofHope.csv`.

Not yet uploaded to the real leaderboard — do that next to calibrate how
well this round of estimates (both the mental-bert CV numbers and the
LLM validation numbers) predicts the real delta. Cost note: each Tier C
run is dozens of separate `claude -p` subprocess calls, each paying full
Claude Code session overhead — real usage cost, factor that into whether/
how often to rerun this step.

Fourth pass (2026-07-14→16, Phase 9): full LLM+BERT hybrid. A unified,
annotation-convention-calibrated LLM pass (`scripts/llm_unified_pass.py`)
now decides risk + evidence + factors, fused with the BERT ensemble by
`scripts/generate_submission_v2.py` under rules validated on a 321-row OOF
sample and CONFIRMED on a disjoint 239-row holdout (risk wF1 0.861/0.823
vs BERT 0.745/0.780; evidence Phrase-F1 0.775/0.746 vs 0.680/0.695;
factors macro 0.598/0.606 vs 0.419/0.411). Projected composite ≈0.75
(leaderboard at the time: 0.6429, top-5 = 0.739). Method, rejected
alternatives, and reproduction commands: `reports/llm_hybrid_validation.md`.
Current `outputs/predictions/RayofHope.csv` is this pipeline's output
(378/378 LLM coverage, all QA checks green) — upload it next.


## Getting started on your A100 machine
```bash
# 1. copy this whole esrd_project/ folder over to ISL-Shakti
cd esrd_project

# 2. one-command isolated env setup (pinned versions, no collision with
#    your AAFC project's env)
bash scripts/setup_env.sh

# 3. from now on, always activate before running anything:
conda activate esrd2026
```

`setup_env.sh` creates a fresh conda env (`esrd2026`, Python 3.11), installs
pinned exact versions of torch/transformers/etc. (no `>=` ranges — avoids
silent breakage from minor-version drift), and runs `check_env.py` at the end
to confirm CUDA is actually detected and your A100 shows up.

Once that passes clean, tell me and we'll start Phase 1.
# Societ-Risk-Prediction-using-Reddit-data
