# LLM–encoder hybrid: method + validation (2026-07-15/16 session)

## Summary

Replaced the submission decision layer with a **convention-calibrated LLM pass**
(`scripts/llm_unified_pass.py`) fused with the existing mental-bert models. All
decisions below were made on a 321-row stratified OOF validation sample
(few-shot rows excluded; BERT numbers are out-of-fold) and confirmed on a
disjoint 239-row holdout before touching the test set.

| component | BERT/BIO baseline | LLM v3 | shipped recipe |
|---|---|---|---|
| risk weighted-F1 | 0.745 | **0.861** | LLM label (BERT-ensemble fallback) |
| evidence Phrase-F1 | 0.680 | **0.775** | LLM spans; Indicator→`none`; BIO@0.6 top-3 backfill when LLM empty |
| factors macro-F1 | 0.419 | 0.570 | per-category plan (union default, `llm`/`bert` overrides) → **0.598** |

Validation-sample composite projection: 0.4·0.861 + 0.3·0.775 + 0.3·0.598 ≈ **0.756**
(current leaderboard 0.6429; top-5 threshold 0.739).

## What made the difference

1. **Annotation-convention calibration, not just definitions.** The official
   4-level definitions are one line each; the annotators' real conventions are
   much sharper. Distilled from reading real boundary cases and encoded into the
   rubric + 17 few-shot examples (`data/processed/llm_unified_fewshot.json`):
   - current-stance rule: recovery/advice posts stay Indicator even with
     detailed past-attempt narratives (P01454);
   - euphemisms without explicit death/kill words stay Indicator ("please let
     me go", "it'll all be over soon", "Well Bye");
   - severe self-harm acts count as Attempt ("50 cuts", neck/wrist cutting)
     even with no suicide words; "plan was foiled"/"attempt again" → Attempt;
   - imminence ("now/tonight/this month") upgrades to Behavior, vague "soon"
     does not; idiomatic method-wishes / fantasy / conditional intent stay
     Ideation; poetic 3rd-person self-narratives keep their true level;
   - factor liberality: "anyone to talk?" = coping strategy, "still alive for
     now" = psychological capital, with prevalence hints per category
     (46% … 0.5%).
   Each convention shift was measured: rubric v2→v3 moved risk wF1 0.831→0.861
   and factors-LLM 0.533→0.570 on the same 321 rows.

2. **Metric-aware evidence policy.** Gold evidence is `none` for 96% of
   Indicator rows and 1–3 short spans (median 4 words) otherwise; the official
   Phrase-F1 gives 1.0 for correctly-empty predictions and penalizes long spans
   via the 3× token cap. The LLM extracts short verbatim spans (verified
   substring-of-raw-post, exact casing recovered); predicted-Indicator rows emit
   `none`; BIO tagger backfills only when the LLM finds nothing.

3. **Verify-before-trust fusions.** Measured and REJECTED on validation:
   - BERT+LLM probability blending for risk (never beat pure LLM; in
     disagreements on unsure rows LLM was right 37 vs BERT 22);
   - BIO∪LLM span union (recall up, precision down, F1 −0.7pt);
   - a second adjudication pass on medium/low-confidence rows (fixed 7,
     broke 8 → −0.3pt; the first read is at the κ=0.84 noise ceiling).
   Kept: per-category factor fusion (`data/processed/factor_fusion_plan.json`),
   union default, `llm` override for 10 rare/implicit categories where BERT ≈ 0
   (macro contribution: those categories alone are ~42% of the macro-F1 mass).

4. **Infrastructure**: batched (8 posts/call), cached (JSONL, idempotent
   reruns), concurrent (2–4 workers) headless `claude -p` calls; ~5k-token
   system prompt; spans sanitized (≤12 words, ≤4/post, dedup by containment).
   BERT side upgraded from single fulldata checkpoints to a 6-model ensemble
   per subtask (5 folds + fulldata, `scripts/predict_test_ensemble.py`).

## Holdout confirmation (2026-07-16)

Disjoint 239-row OOF holdout, recipe locked BEFORE these rows were processed:

| component | BERT/BIO | locked recipe | (val sample) |
|---|---|---|---|
| risk wF1 | 0.780 | **0.823** | 0.861 |
| evidence Phrase-F1 | 0.695 | **0.746** | 0.775 |
| factors macro-F1 (locked plan) | 0.411 | **0.606** | 0.598 |

Pooled val+holdout estimate (560 rows): risk 0.845, evidence 0.763, factors
0.602 → composite ≈ **0.747** (S1 ≈ 0.81, S2 ≈ 0.60). Leaderboard top-5
threshold at submission time: 0.7387.

## Final submission (outputs/predictions/RayofHope.csv)

378 rows; risk distribution Indicator/Ideation/Behavior/Attempt =
162/97/95/24; evidence `none` exactly on the 162 Indicator rows; 1.76
spans/post elsewhere (gold ≈1.75), every span a verified verbatim substring
of the RAW test post (smart-quote-safe recovery); factors mean 3.95/post
under the per-category plan. Near-duplicate harmonization triggered 0 flips
(the LLM already labels intra-user near-duplicates consistently).

## Round 2 (2026-07-18) — leaderboard feedback loop

First upload scored composite **0.7212 (rank 11)**: S2 transferred exactly as
validated (0.6142 vs 0.602 projected); S1 landed at 0.767 vs 0.81 OOF — the
hidden test applies roughly the same ~4-point discount the BERT baseline
experienced. Round-2 changes, each measured on the pooled 560-row sample:

- **Factor fusion plan v2** (`data/processed/factor_fusion_plan_v2.json`):
  per-category mode now includes `inter` (hopelessness) and a probability
  rule `p_bert + α·1[LLM] ≥ t` for 4 categories (emotion dysregulation,
  coping strategy, psychological capital, prior self-harm) — adopted only
  where they beat the best set-strategy by ≥0.02 AND survived a split-half
  stability check (3 categories were demoted after failing it).
- **Precision-verification pass** (`scripts/llm_verify_factors.py`): second
  strict read of posts where the unified pass fired the over-firing
  categories; meaning in life 0.110→0.267 F1, cognitive deficits
  0.320→0.400 (validated), applied to meaning-in-life / cognitive-deficits /
  sexual-orientation fires on test.
- Pooled factor macro: **0.6061 → 0.6315**; factors/post 3.95 → 2.91
  (gold 2.99). Expected S2 ≈ 0.63-0.64.
- **Measured and rejected**: LLM self-consistency voting (95% run-to-run
  agreement, disagreements split 3-3); per-class evidence span caps (current
  policy already optimal at 0.7626 pooled); another rubric iteration (error
  review shows the residual Indicator↔Ideation errors sit on gold-label
  self-contradictions — e.g. "wanna sleep forever" is an Ideation span in
  training but Indicator here — i.e. the κ=0.84 annotation-noise ceiling).
- DeBERTa-v3-base side-models: 7/10 fold runs OOM'd on the saturated shared
  A100 (a neighboring 38.5GB job); a patient retrainer is armed to complete
  them when VRAM frees — integrate via `predict_test_ensemble.py`-style
  averaging + re-tuned prob rules if/when available.

Second upload confirmed the S2 gain: composite 0.7212 → **0.7230** (S1
0.7670 unchanged, S2 0.6142 → 0.6201). The stable part of the factor plan v2
transferred (+0.006); the threshold-tuned part shrank as expected.

## Round 3 (2026-07-21) — DeBERTa-v3 risk arbitration

Trained DeBERTa-v3-base 5-fold on both subtasks (OOM-robust retrainer;
`predict_probs_deberta.py` dumps OOF+test probs). Standalone it slightly
beats mental-bert on risk (0.7625 vs 0.7552 OOF) but is much weaker on
factors (0.30 — wrong thresholds), so it is used ONLY for risk.

**Risk arbitration** (`generate_submission_v2.py --deberta_test_probs`): on
LLM-**unsure** rows (self-reported confidence ≠ high) where the
`0.7·mentalbert-ens + 0.3·deberta` risk ensemble is ≥0.8 normalized-confident
AND disagrees with the LLM, take the ensemble label. Mechanism: the LLM's own
confidence flags precisely the rows where a strong discriminative second
opinion pays off (on unsure rows, ensemble-vs-LLM disagreements resolved in
the ensemble's favor).

Split-half validated at fixed conf 0.8 (both halves improve):
- val 0.8613 → 0.8628, holdout 0.8229 → 0.8538; pooled 0.8446 → **0.8588**
  (+0.014 wF1).
- fit-conf-on-one-half → other half: val→hold +0.031, hold→val +0.0045
  (positive both directions — not a single-split artifact).
- A **self-consistency guard** vetoes any flip to a non-Indicator label that
  cannot be grounded in an extractable span (blocks spurious high-confidence
  flips on ultra-short posts like "Am i dying??"); metric-neutral on pooled,
  applied for logical consistency. 28 flips on test (2 vetoed).

Test effect: risk distribution 42.9% → 39.4% Indicator, toward the 37.4%
training prior — the arbitration corrects the LLM's Indicator over-prediction.
Projected composite ≈ 0.726 (risk +0.014 pooled → S1 gain after the hidden-test
discount observed on rounds 1-2). Real gain TBD on upload.

**Still open — decompose S1.** Two A/B files, risk+factors identical to the
main submission, evidence varied: `RayofHope_diagnostic_bioevidence.csv`
(BIO-only evidence) and `RayofHope_variant_unionevidence.csv` (LLM∪BIO).
Uploading either (renamed to RayofHope.csv) isolates the evidence component
on the hidden test: S1_main − S1_bio = 0.3/0.7·(evid_LLM − evid_BIO), telling
us whether the residual S1 gap is in risk or evidence, and whether union
evidence's higher recall helps or (as on the OOF sample) hurts.

## Round 4 (2026-07-23) — dedicated weak-category factor detector

Round 3 (DeBERTa risk arbitration) transferred strongly: **S1 0.7670 → 0.7849**,
composite 0.7230 → 0.7355 (rank 11 → 10). That made S1 top-8 caliber — the
entire remaining gap to top 5 is Subtask 2 (RayofHope S2 0.6201 vs #5 ≈ 0.648;
ALEXIS sits at rank 7 with *worse* S1, on S2 strength alone). Target: S2 ≈ 0.68.

Leverage analysis (pooled-560, per-category F1): 5 categories carry the deficit —
sexual orientation 0.00, meaning-in-life 0.27, exposure 0.29, cognitive 0.40,
physical-health 0.43. Built a **dedicated detector** (`scripts/llm_factor_detector.py`):
each weak category gets its real annotation breadth (learned from all training
positives — e.g. physical-health includes appearance/ugliness + bodily symptoms;
cognitive includes derealization + "can't write coherently"), 3-4 real few-shots,
hard-negative boundaries, and keyword priors for the lexical ones.

Validation was two-stage and caught an over-optimism trap:
- **Targeted set** (`factor_targeted_val.csv`, 811 rows = all weak-category
  positives from the 1635 + 250 sampled negatives, neg-weighted to natural
  prevalence): suggested +0.06 macro across 8 categories.
- **Honest natural pooled-560** (full detector coverage): only 4 of those 8
  survive — physical-health 0.431→0.604, cognitive 0.400→0.500, exposure
  0.286→0.333, sexual-orientation 0.000→0.000-on-pooled (no positives there;
  targeted split-half 0.60/0.67, adopted for pure test upside). The detector
  UNDER-recalls the common categories (psych-capital, interpersonal-difficulty,
  social-support) vs the existing union/prob fusion — kept current there.
  Stressful-life-event won on the enriched targeted set but LOST on the natural
  distribution (0.576→0.524) — dropped. The enriched-set/natural-set gap is the
  key methodological lesson of this round.

Fused factor macro on pooled-560: **0.6315 → 0.6449 (+0.0134)** confirmed, plus
test-only upside from sexual-orientation (fires 5 test rows, ~3 clean gender-identity
posts) and exposure (5 rows) positives that are absent from the pooled sample.
Applied via `generate_submission_v2.py --detector_jsonl ... --detector_adopt ...`.

Honest expectation: S2 ≈ 0.635-0.665, composite ≈ 0.740-0.749 (rank ~7-9, outside
shot at #6). Real progress from #10; a clean top-5 (S2 ~0.68) is beyond this
detector alone — the common mid-categories (0.58-0.64) are the remaining ceiling
and neither approach beats the current fusion there.

## Round 4 result + the transfer law (2026-07-23)

The detector shipped and scored **S2 0.6201 → 0.6197 (−0.0004)** — the +0.0134
pooled gain did NOT transfer. Combined with prior rounds, a clear law emerged:

| change | subtask | pooled Δ | leaderboard Δ | transfer |
|---|---|---|---|---|
| plan v2 + verify | S2 | +0.028 | +0.006 | 21% |
| DeBERTa risk arbitration | S1 | +0.014 | +0.018 | **129%** |
| weak-category detector | S2 | +0.013 | −0.0004 | **−3%** |

**S1/risk changes transfer (even amplify); S2/factor changes do not.** Cause:
the factor plan/detector are tuned on the same 560-row pooled sample used to
validate them (overfit), and macro-F1 over 24 categories with the hidden test's
particular rare-category realizations is high-variance. Risk wF1 is a low-variance
4-class metric that generalizes. Since composite = 0.7·S1 + 0.3·S2, S1 is BOTH
higher-weight and reliable — all further effort goes there. Reaching #5 (0.7535)
via S1 alone needs S1 0.7849 → 0.8108 (top teams sit at 0.80-0.81, so it is not
out of range).

**Round 5 (shipped): arbitration threshold 0.80 → 0.85.** 2D weight×threshold
sweep with split-half: w=0.3/thr=0.85 beats the shipped 0.80 on pooled
(0.8588→0.8623), val (+0.003) and holdout (+0.004) by dropping the 2 lowest-
confidence (net-bad) flips. Detector dropped (factors reverted to plan v2 +
verify, the cleaner 0.6201 config). Small but reliable S1 gain.

**Round 5 (in progress): 3rd arbitration model.** DeBERTa-v3-**large** risk
5-fold training, patient trainer waiting on the contended shared A100. A more
diverse strong discriminative ensemble gives the LLM-unsure rows a better second
opinion — the mechanism that already delivered +0.018. Integrate + re-tune the
arbitration when it lands.

**Highest-value next upload: the S1 decomposition.** `RayofHope_diagnostic_
bioevidence.csv` (risk+factors identical to main, evidence = BIO-only). S1_main −
S1_bio = 0.3/0.7·(evid_LLM − evid_BIO) isolates evidence vs risk on the hidden
test — decides whether the remaining S1 push targets risk (the 3rd model) or
evidence (untouched since round 1, and it is 43% of S1 / 0.3 of composite —
equal weight to ALL of Subtask 2).

## Round 6 (2026-07-25) — 3-model arbitration ensemble

Round 5's threshold bump transferred: **S1 0.7849 → 0.7912, composite 0.7355 →
0.7399, rank 10 → 9.** Confirmed again that S1/risk changes transfer. Gap to #5
(DLL-Lab 0.7541) now +0.0142, entirely S1 (top teams at S1 0.80-0.81).

Added a 3rd architecture to the arbitration ensemble: **RoBERTa-base** risk
5-fold (BERT + DeBERTa + RoBERTa = 3 diverse discriminative encoders). Re-tuned
on OOF with split-half (`scripts/retune_arbitration.py`): ensemble
`(mentalbert-ens + 0.3·deberta + 0.3·roberta)/1.6`, arbitration threshold 0.88.
Pooled risk wF1 **0.8623 → 0.8658** (val 0.875, hold 0.854 — both beat baseline,
robust). Rebuilt `RayofHope.csv` (`--arb_extra rb_test_risk.parquet:rb:0.3`);
19 arbitration flips, fixed the questionable P00708 flip (Behavior→Indicator).
Evidence confirmed near-optimal (trim/union variants all ≤ current 0.7626);
factors stay plan v2 + verify (S2 changes don't transfer). Expected S1 ≈ 0.794-0.796.

4th model (MentalRoBERTa, domain RoBERTa) training for a further increment;
DeBERTa-v3-large queued behind a 49 GB co-tenant job on the shared A100.
Host GPU driver updated mid-run (580.173) — nvidia-smi/NVML broke but CUDA
compute verified working; training unaffected, only memory monitoring lost.

Honest ceiling: incremental base-model additions give ~+0.003-0.005 S1 each with
diminishing returns; reaching #5 needs ~+0.020 S1, realistically only if
DeBERTa-v3-large (much stronger) gets GPU time. Current trajectory lands ~#7-9.

## Round 7 (2026-07-26) — 4-model arbitration ensemble (best)

Scaled the discriminative arbitration ensemble to 4 architectures and re-tuned
per-model weights on OOF with split-half:

| standalone risk wF1 | model |
|---|---|
| 0.755 | mental-bert |
| 0.7625 | deberta-v3-base |
| 0.760 | roberta-base |
| **0.789** | **roberta-large** (strongest) |
| 0.714 | deberta-v3-large (undertrained at lr 2e-5 on 1.3k rows — EXCLUDED) |

Best config: ensemble `(mentalbert-ens + 0.2·deberta-base + 0.2·roberta-base +
0.3·roberta-large)/1.7`, arbitrate LLM-unsure rows at conf ≥ 0.85. Pooled risk
wF1 progression: LLM-alone 0.8446 → 2-model 0.8623 (on leaderboard as S1 0.7912)
→ 3-model 0.8658 → **4-model 0.8710** (val 0.881, hold 0.858, split-half robust).
DeBERTa-v3-large added nothing (weakest model); high-confidence LLM override
tested and rejected (LLM reliably correct when confident). Current `RayofHope.csv`
uses this 4-model config.

Expected: S1 ≈ 0.80 (risk +0.0087 pooled over the 0.7912 leaderboard config, at
the ~100-150% risk transfer rate observed), composite ≈ 0.743-0.748, rank ~7-8.

**Ceiling reached on this architecture.** S1 levers exhausted: evidence
near-optimal (0.7626, all trim/union variants ≤), risk arbitration maxed
(4 models, stronger/more/aggressive all tested). S2 stuck at 0.62 (top teams
0.68) because factor changes don't transfer (overfit the 560-row validation).
Reaching #5 (0.7541) would need either a from-scratch stronger factor model or
a better core risk model than fine-tuning delivers on 1.3k rows/fold. The
robust, reliable gains have been banked: composite 0.6429 (rank 18) → ~0.745
(rank ~7-8) across the campaign.

## Reproduction

```bash
python scripts/predict_test_ensemble.py                     # BERT ensemble test probs
python scripts/llm_unified_pass.py --input data/raw/test.xlsx \
    --output outputs/predictions/llm_test_v3.jsonl --batch_size 8 --workers 2
python scripts/generate_submission_v2.py --llm_jsonl outputs/predictions/llm_test_v3.jsonl
```
