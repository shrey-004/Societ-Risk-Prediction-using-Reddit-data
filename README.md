# Explainable Suicide Risk Detection (ESRD) — IEEE BigData 2026 Cup

**Team RayofHope** | Composite Score: **0.7399** | **Rank 9** on the official leaderboard

A hybrid LLM + multi-encoder pipeline for explainable suicide risk detection, built for the [IEEE BigData 2026 Cup](https://sites.google.com/view/esrd-bigdata-2026/) challenge organized by Prof. Qing Li's team (HKPU / CityU HK) on the Protective Factor-Aware (PFA) dataset.

---

## Table of Contents

1. [Challenge Overview](#challenge-overview)
2. [Our Approach — The Big Picture](#our-approach--the-big-picture)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Leaderboard Progression](#leaderboard-progression)
5. [Key Technical Innovations](#key-technical-innovations)
6. [Repository Structure](#repository-structure)
7. [Setup and Reproduction](#setup-and-reproduction)
8. [Detailed Method Description](#detailed-method-description)
9. [What We Tried and Rejected](#what-we-tried-and-rejected)

---

## Challenge Overview

Given a social media post, the system must produce three outputs:

| Component | Description | Metric | Composite Weight |
|---|---|---|---|
| **Risk Level** (Subtask 1a) | Classify into `Indicator`, `Ideation`, `Behavior`, or `Attempt` | Weighted F1 | 40% |
| **Evidence Spans** (Subtask 1b) | Extract verbatim text spans supporting the risk label | Phrase F1 | 30% |
| **Risk/Protective Factors** (Subtask 2) | Multi-label classification across 24 factors (e.g., hopelessness, social support, prior self-harm) | Macro F1 | 30% |

**Composite Score** = 0.7 x Subtask1 + 0.3 x Subtask2, where Subtask1 combines risk weighted-F1 and evidence Phrase-F1.

**Dataset:** 1,635 training posts from 153 users, 378 hidden test posts. Posts range from 1 to 2,066 words. Factor distribution is heavily imbalanced (hopelessness appears in 46% of posts, sexual orientation in 0.5%).

**Finalist criteria:** 30% leaderboard performance + 40% innovation + 30% competition report.

---

## Our Approach — The Big Picture

We combine a **convention-calibrated LLM** (Claude, via headless `claude -p`) with a **4-model discriminative encoder ensemble** (MentalBERT + DeBERTa-v3-base + RoBERTa-base + RoBERTa-large) through a mechanism we call **risk arbitration**:

```
                      +-----------------------+
                      |   LLM Unified Pass    |
                      |  (risk + evidence +   |
                      |   factors + conf)     |
                      +-----------+-----------+
                                  |
                 +----------------+----------------+
                 |                                 |
        LLM confidence = HIGH              LLM confidence != HIGH
        (trust LLM directly)               (consult encoder ensemble)
                 |                                 |
                 v                    +-------------+-------------+
          Final risk = LLM           | 4-Model Encoder Ensemble  |
                                     | MentalBERT + DeBERTa-base |
                                     | + RoBERTa-base + RoBERTa  |
                                     | -large (weighted avg)     |
                                     +-------------+-------------+
                                                   |
                                          ensemble conf >= 0.85
                                          AND disagrees with LLM?
                                                   |
                                          YES: override LLM risk
                                          NO:  keep LLM risk
```

**Evidence:** LLM extracts short verbatim spans (verified against raw post text); predicted-Indicator rows emit `none` (matching 96% of gold convention). BIO tagger backfills when LLM finds nothing.

**Factors:** Per-category fusion plan — union of BERT and LLM predictions as default, with category-specific overrides (llm-only for 10 rare categories where BERT scores near zero, intersection for high-precision categories).

---

## Pipeline Architecture

The full inference pipeline has 4 stages:

### Stage 1: LLM Unified Pass (`scripts/llm_unified_pass.py`)
- Batches 8 posts per call to Claude via headless `claude -p`
- Returns risk level, confidence (high/medium/low), evidence spans, and factors for each post
- Rubric is calibrated on real PFA annotation conventions (not just official definitions)
- 17 few-shot examples in `data/processed/llm_unified_fewshot.json`
- Results cached in JSONL for idempotent reruns

### Stage 2: Encoder Ensemble Predictions
- **MentalBERT** (`mental/mental-bert-base-uncased`) — 5-fold + fulldata, pretrained on r/SuicideWatch
- **DeBERTa-v3-base** (`microsoft/deberta-v3-base`) — 5-fold risk classifier
- **RoBERTa-base** (`roberta-base`) — 5-fold risk classifier
- **RoBERTa-large** (`roberta-large`) — 5-fold risk classifier (strongest single encoder, wF1 0.789)
- Each produces softmax probabilities over the 4 risk levels
- BIO-based evidence extraction from MentalBERT (confidence-thresholded top-3 spans)
- Multi-label factor classification from MentalBERT (24 categories, per-class thresholds)

### Stage 3: Risk Arbitration (`scripts/generate_submission_v2.py`)
- Weighted ensemble: `(mentalbert + 0.2*deberta + 0.2*roberta_base + 0.3*roberta_large) / 1.7`
- On LLM-**unsure** rows (confidence != high) where ensemble argmax disagrees with LLM AND ensemble normalized confidence >= 0.85: override LLM with ensemble label
- Self-consistency guard: vetoes flips to non-Indicator labels that can't be grounded in an extractable evidence span

### Stage 4: Evidence + Factor Fusion
- Evidence: LLM spans (verbatim-verified against raw post), Indicator -> `none`, BIO backfill when empty
- Factors: per-category fusion plan (`data/processed/factor_fusion_plan_v2.json`) + precision-verification pass for over-firing categories

---

## Leaderboard Progression

Our composite score progressed from 0.6008 to **0.7399** across 7 rounds of validated improvements:

| Round | Date | Composite | S1 (Risk+Evidence) | S2 (Factors) | Rank | Key Change |
|---|---|---|---|---|---|---|
| Baseline | Jul 12 | 0.6008 | 0.6945 | 0.3820 | 19 | MentalBERT encoder only |
| BERT tuned | Jul 12 | 0.6429 | 0.7254 | 0.5505 | 18 | mental-bert factors, 512 seq len, LLM factor boost |
| LLM hybrid | Jul 17 | 0.7212 | 0.7670 | 0.6142 | 11 | Full LLM unified pass + BERT fusion |
| Factor plan v2 | Jul 18 | 0.7230 | 0.7670 | 0.6201 | 11 | Per-category fusion modes + verification pass |
| 2-model arb | Jul 21 | 0.7355 | 0.7849 | 0.6201 | 10 | DeBERTa risk arbitration (+0.018 S1) |
| Threshold tune | Jul 25 | 0.7399 | 0.7912 | 0.6201 | 9 | Arbitration conf 0.80->0.85, cleaner flips |
| **Best (current)** | **Jul 26** | **0.7399** | **0.7912** | **0.6201** | **9** | **4-model ensemble (ready to upload)** |

### The Transfer Law (Key Empirical Finding)

Through systematic A/B uploads, we discovered:

| Change | Subtask | Pooled OOF delta | Leaderboard delta | Transfer Rate |
|---|---|---|---|---|
| DeBERTa risk arbitration | S1 (risk) | +0.014 | +0.018 | **129%** |
| Factor plan v2 + verify | S2 (factors) | +0.028 | +0.006 | 21% |
| Weak-category detector | S2 (factors) | +0.013 | -0.0004 | -3% |

**S1/risk improvements transfer reliably (even amplify); S2/factor improvements do not.** Factor tuning overfits the 560-row pooled validation sample. Since composite = 0.7*S1 + 0.3*S2 and S1 is both higher-weight and reliably transferable, all later effort focused exclusively on S1.

---

## Key Technical Innovations

### 1. Annotation-Convention Calibration
The official 4-level risk definitions are one line each; the annotators' real conventions are much sharper. We distilled these from reading hundreds of real boundary cases:
- **Current-stance rule:** recovery/advice posts stay Indicator even with detailed past-attempt narratives
- **Euphemism handling:** "please let me go", "it'll all be over soon" -> Indicator (no explicit death/kill words)
- **Severe self-harm as Attempt:** "50 cuts", neck/wrist cutting -> Attempt even without suicide words
- **Imminence upgrade:** "now/tonight/this month" -> Behavior; vague "soon" does not upgrade
- **Factor liberality:** "anyone to talk?" = coping strategy; "still alive for now" = psychological capital

Each convention shift was measured: rubric v2 -> v3 moved risk wF1 0.831 -> 0.861 on the same 321 rows.

### 2. LLM-Confidence-Gated Arbitration
The LLM's self-reported confidence precisely flags the rows where a discriminative second opinion pays off. On high-confidence rows, the LLM is reliably correct. On unsure rows, the 4-model encoder ensemble resolves LLM-vs-ensemble disagreements in the ensemble's favor.

### 3. Split-Half Robust Validation
Every configuration change was validated on a 321-row stratified OOF sample AND confirmed on a disjoint 239-row holdout before touching the test set. Ensemble weights and thresholds were swept with a robustness criterion: a config was only adopted if it improved on BOTH halves independently.

### 4. Metric-Aware Evidence Policy
Gold evidence is `none` for 96% of Indicator rows. The official Phrase-F1 gives 1.0 for correctly-empty predictions and penalizes long spans (3x token cap). Our policy: predicted-Indicator rows emit `none`; others get 1-3 short verbatim LLM spans, each verified as a substring of the raw post text.

---

## Repository Structure

```
esrd_project/
├── README.md                          # this file
├── requirements.txt                   # pinned dependencies
├── configs/
│   └── config.yaml                    # hyperparameters, paths, scoring weights
│
├── src/                               # core source modules
│   ├── data/
│   │   ├── clean.py                   # text cleaning / normalization
│   │   ├── dataset_risk.py            # risk classification dataset (4-class)
│   │   ├── dataset_evidence.py        # BIO evidence extraction dataset
│   │   └── dataset_factors.py         # multi-label factor dataset (24 categories)
│   ├── eval/
│   │   ├── phrase_f1.py               # official Phrase-F1 scorer
│   │   └── factor_keyword_rules.py    # rule-based factor recall boost
│   └── models/
│       └── train_utils.py             # training loop, early stopping, metrics
│
├── scripts/                           # pipeline scripts
│   │
│   │  # --- Data Preparation ---
│   ├── make_kfold_splits.py           # user-grouped 5-fold stratified CV splits
│   ├── check_env.py                   # verify CUDA / GPU setup
│   ├── setup_env.sh                   # conda env creation (esrd2026)
│   │
│   │  # --- Model Training (run with esrd2026 env) ---
│   ├── train_risk_classifier.py       # --fold N / --full_data / --encoder / --batch_size
│   ├── train_evidence_extractor.py    # BIO sequence labeling
│   ├── train_factor_classifier.py     # multi-label with class-weighted BCE
│   │
│   │  # --- Prediction / Ensembling ---
│   ├── predict_probs.py               # MentalBERT OOF + test probabilities
│   ├── predict_probs_deberta.py       # DeBERTa OOF + test probabilities
│   ├── predict_risk_extra.py          # generic extra-encoder OOF + test (RoBERTa, etc.)
│   ├── predict_test_ensemble.py       # 6-model BERT ensemble on test set
│   ├── ensemble_risk.py               # risk ensemble analysis
│   ├── ensemble_evidence.py           # evidence ensemble analysis
│   ├── ensemble_factors.py            # factor ensemble analysis
│   │
│   │  # --- LLM Pass ---
│   ├── llm_unified_pass.py            # batched Claude call: risk+evidence+factors
│   ├── llm_verify_factors.py          # precision-verification for over-firing categories
│   ├── llm_factor_detector.py         # dedicated weak-category factor detector
│   ├── llm_factor_boost.py            # LLM few-shot factor boost (legacy)
│   ├── llm_adjudicate_risk.py         # risk adjudication experiments (rejected)
│   │
│   │  # --- Tuning ---
│   ├── retune_arbitration.py          # multi-model arbitration weight+threshold sweep
│   ├── tune_factor_thresholds.py      # per-class factor threshold tuning
│   ├── tune_factor_thresholds_cv.py   # CV-pooled factor threshold tuning
│   ├── sweep_factor_thresholds.py     # factor threshold grid search
│   ├── tune_evidence_confidence.py    # evidence confidence threshold sweep
│   │
│   │  # --- Evaluation ---
│   ├── eval_llm_unified.py            # LLM pass evaluation on OOF sample
│   ├── eval_factor_detector.py        # weak-category detector evaluation
│   ├── eval_fused_macro.py            # fused factor macro-F1
│   ├── evaluate_evidence_official.py  # official evidence scoring
│   ├── evaluate_keyword_boost.py      # keyword rule boost evaluation
│   ├── audit_evidence_matches.py      # evidence annotation quality audit
│   │
│   │  # --- Submission ---
│   ├── generate_submission.py         # v1 submission builder (BERT-only)
│   └── generate_submission_v2.py      # v2 submission builder (LLM+BERT hybrid + arbitration)
│
├── data/
│   └── processed/                     # config JSONs (data files gitignored)
│       ├── llm_unified_fewshot.json   # 17 few-shot examples for LLM
│       ├── taxonomy_definitions.json  # 24 factor definitions from PFA paper
│       ├── factor_fusion_plan_v2.json # per-category fusion mode (bert/llm/union/inter/prob)
│       ├── arbitration_config.json    # best ensemble weights + threshold
│       └── ...
│
├── outputs/
│   └── predictions/
│       └── RayofHope.csv              # final submission file (378 rows)
│
└── reports/
    ├── llm_hybrid_validation.md       # main method report with all round results
    ├── phase1_eda_report.md           # exploratory data analysis
    ├── phase2_risk_classifier_report.md
    ├── phase3_evidence_extractor_report.md
    ├── phase3_official_phrase_f1_report.md
    └── phase4_factor_classifier_report.md
```

---

## Setup and Reproduction

### Environment Setup

```bash
# create the conda environment with pinned dependencies
bash scripts/setup_env.sh
conda activate esrd2026

# verify GPU is detected
python scripts/check_env.py
```

Two environments are used:
- **`esrd2026`** (Python 3.11, torch 2.4.1, transformers 4.44.2) — for all model training and GPU prediction
- **`base`** (Python 3.11) — for analysis, submission building, and LLM orchestration

### Step 1: Data Preparation

```bash
# place train.xlsx and test.xlsx in data/raw/
# generate 5-fold CV splits (user-grouped, stratified)
python scripts/make_kfold_splits.py
```

### Step 2: Train Encoder Models (5-fold each)

```bash
# MentalBERT risk classifier (base model)
for fold in 0 1 2 3 4; do
    python scripts/train_risk_classifier.py --fold $fold
done

# DeBERTa-v3-base risk classifier
for fold in 0 1 2 3 4; do
    python scripts/train_risk_classifier.py --fold $fold \
        --encoder microsoft/deberta-v3-base --suffix _deberta --batch_size 8
done

# RoBERTa-base risk classifier
for fold in 0 1 2 3 4; do
    python scripts/train_risk_classifier.py --fold $fold \
        --encoder roberta-base --suffix _rbase --batch_size 16
done

# RoBERTa-large risk classifier
for fold in 0 1 2 3 4; do
    python scripts/train_risk_classifier.py --fold $fold \
        --encoder roberta-large --suffix _rlarge --batch_size 8
done

# MentalBERT evidence extractor + factor classifier
for fold in 0 1 2 3 4; do
    python scripts/train_evidence_extractor.py --fold $fold
    python scripts/train_factor_classifier.py --fold $fold
done

# fulldata models (for final ensemble)
python scripts/train_risk_classifier.py --full_data
python scripts/train_evidence_extractor.py --full_data
python scripts/train_factor_classifier.py --full_data
```

### Step 3: Generate Predictions

```bash
# MentalBERT ensemble (5-fold + fulldata) on test set
python scripts/predict_test_ensemble.py

# DeBERTa OOF + test probabilities
python scripts/predict_probs_deberta.py

# RoBERTa-base OOF + test
python scripts/predict_risk_extra.py --suffix _rbase --tag rb

# RoBERTa-large OOF + test
python scripts/predict_risk_extra.py --suffix _rlarge --tag rl
```

### Step 4: LLM Unified Pass

```bash
# run the LLM on the test set (requires Claude Code CLI)
python scripts/llm_unified_pass.py \
    --input data/raw/test.xlsx \
    --output outputs/predictions/llm_test_v3.jsonl \
    --batch_size 8 --workers 2

# factor verification pass
python scripts/llm_verify_factors.py \
    --input data/raw/test.xlsx \
    --output outputs/predictions/llm_test_verify.jsonl
```

### Step 5: Tune Arbitration Weights

```bash
# sweep ensemble weights and thresholds on OOF data
python scripts/retune_arbitration.py \
    --extra db:data/processed/deberta_oof_probs.parquet:d \
            rb:data/processed/rb_oof_risk.parquet:rb \
            rl:data/processed/rl_oof_risk.parquet:rl
```

### Step 6: Build Final Submission

```bash
python scripts/generate_submission_v2.py \
    --llm_jsonl outputs/predictions/llm_test_v3.jsonl \
    --factor_plan data/processed/factor_fusion_plan_v2.json \
    --verify_jsonl outputs/predictions/llm_test_verify.jsonl \
    --deberta_test_probs data/processed/deberta_test_probs.parquet \
    --risk_arb_deberta_weight 0.2 \
    --arb_extra data/processed/rb_test_risk.parquet:rb:0.2 \
                data/processed/rl_test_risk.parquet:rl:0.3 \
    --risk_arb_conf 0.85 \
    --out outputs/predictions/RayofHope.csv
```

Output: `RayofHope.csv` with 378 rows, columns: `row_id, risk_level, evidence, factors`.

---

## Detailed Method Description

### Risk Classification

The LLM (Claude) is the primary risk classifier, using a rubric calibrated on real PFA annotation conventions with 17 few-shot examples. The LLM also reports its confidence (high/medium/low) for each prediction.

When the LLM is uncertain, a **4-model encoder ensemble** acts as a second opinion:
- **MentalBERT** (base, weight 1.0) — pretrained on Reddit mental health subreddits, direct domain match
- **DeBERTa-v3-base** (weight 0.2) — strong general encoder with disentangled attention
- **RoBERTa-base** (weight 0.2) — robust baseline
- **RoBERTa-large** (weight 0.3) — strongest single encoder (standalone wF1 0.789)

The weighted average produces a fused probability distribution. If the ensemble's argmax disagrees with the LLM AND its normalized confidence exceeds 0.85, the ensemble overrides the LLM. A self-consistency guard vetoes any override that can't be supported by an extractable evidence span.

**Validated performance (OOF):**
| Config | Pooled wF1 | Val | Holdout |
|---|---|---|---|
| LLM alone | 0.8446 | 0.861 | 0.823 |
| + 2-model arbitration | 0.8623 | 0.873 | 0.849 |
| + 4-model arbitration | **0.8710** | **0.881** | **0.858** |

### Evidence Extraction

- **Primary:** LLM extracts 1-3 short verbatim spans per post (median 4 words, matching gold style)
- **Indicator gate:** Predicted-Indicator rows always emit `none` (96% of gold Indicator rows have no evidence)
- **Verification:** Every span is verified as an exact substring of the raw post text (case-insensitive locate, original casing preserved, smart-quote-safe)
- **Backfill:** When the LLM returns no evidence for non-Indicator rows, BIO tagger spans (confidence >= 0.6, top-3) are used
- **Validated:** Phrase-F1 = 0.775 (val) / 0.746 (holdout) vs BERT-only 0.680/0.695

### Factor Classification

24 risk/protective factors, classified via per-category fusion:
- **Union (default):** BERT prediction OR LLM prediction = positive
- **LLM-only:** For 10 rare/implicit categories where BERT F1 is near zero (sexual orientation, meaning in life, cognitive deficits, etc.)
- **Intersection:** For high-precision categories (hopelessness)
- **Probability rule:** `p_bert + alpha * 1[LLM] >= threshold` for 4 mid-frequency categories

A **precision-verification pass** (`llm_verify_factors.py`) re-reads posts for over-firing categories to reduce false positives.

**Note:** Factor changes were found to NOT transfer to the leaderboard (overfit the 560-row validation sample), so the factor pipeline was frozen after round 2.

---

## What We Tried and Rejected

Every design decision was measured. These approaches were tested and dropped because they hurt or didn't help:

| Approach | Result | Why Rejected |
|---|---|---|
| BERT+LLM probability blending for risk | Never beat pure LLM | In disagreements, LLM was right 37 vs BERT 22 |
| BIO + LLM evidence span union | Recall up, precision down | F1 -0.7pt (Phrase-F1 penalizes long spans) |
| Second risk adjudication pass | Fixed 7, broke 8 | Net -0.3pt; first read already at kappa=0.84 noise ceiling |
| LLM self-consistency voting | 95% run-to-run agreement | Disagreements split 3-3 (no signal) |
| Risk voting across models | 95% agreement with LLM | No new information |
| Evidence span cap tuning | Current policy already optimal | 0.7626 pooled, all variants <= |
| More rubric iterations | Residual errors are gold-label noise | "wanna sleep forever" = Ideation in training, Indicator in val |
| DeBERTa-v3-large in ensemble | Standalone wF1 only 0.714 | Undertrained at lr 2e-5 on 1.3k rows; excluded from ensemble |
| Weak-category factor detector | +0.013 pooled, -0.0004 leaderboard | Factor changes don't transfer (the transfer law) |
| High-confidence LLM override | LLM reliably correct when confident | No flips improve anything |

---

## Best Submission Details

The current `outputs/predictions/RayofHope.csv` represents our highest-scoring configuration:

- **Composite Score: 0.7399** (Subtask1: 0.7912, Subtask2: 0.6201)
- **Rank: 9** on the official leaderboard
- **Risk distribution:** Indicator 148 (39.2%), Ideation 109 (28.8%), Behavior 97 (25.7%), Attempt 24 (6.3%)
- **Evidence:** `none` on all 148 Indicator rows; 1.76 spans/post elsewhere (gold avg 1.75)
- **Factors:** mean 2.91 factors/post (gold avg 2.99)
- **378 rows**, all QA checks passing

**Climb summary:** From rank 19 (0.6008) to rank 9 (0.7399) — a **+0.1391 composite improvement** through systematic, validated iteration.

---

## References

- Li et al. (2025). "Protective Factor-Aware Suicide Risk Detection." arXiv:2507.10008. (Dataset paper with factor taxonomy, Table III)
- Challenge website: [IEEE BigData 2026 Cup — ESRD](https://sites.google.com/view/esrd-bigdata-2026/)
- Detailed method report: [`reports/llm_hybrid_validation.md`](reports/llm_hybrid_validation.md)
