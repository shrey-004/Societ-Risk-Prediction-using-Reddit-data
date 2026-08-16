# Phase 4 — Per-Class Threshold Tuning

Baseline macro F1 (0.5 threshold): 0.2916

Tuned macro F1 (per-class threshold): 0.3723

**Caveat**: tuned on the val set being scored — optimistic estimate, not a fully unbiased generalization number.

## Per-factor thresholds

| Factor | Support | Baseline F1 | Tuned F1 | Threshold |
|---|---|---|---|---|
| mental health issues | 31 | 0.329 | 0.380 | 0.45 |
| physical health/characteristic | 4 | 0.000 | 0.074 | 0.40 |
| substance use | 2 | 0.000 | 0.022 | 0.10 |
| hopelessness | 108 | 0.630 | 0.702 | 0.40 |
| emotion dysregulation | 65 | 0.574 | 0.574 | 0.50 |
| low self-esteem | 71 | 0.451 | 0.537 | 0.35 |
| poor school performance | 1 | 0.000 | 0.083 | 0.30 |
| low socio-economic status | 9 | 0.316 | 0.333 | 0.55 |
| interpersonal violence | 20 | 0.462 | 0.488 | 0.45 |
| prior self-harm or suicidal thought/attempt | 45 | 0.500 | 0.500 | 0.50 |
| poor social support | 104 | 0.497 | 0.657 | 0.25 |
| interpersonal difficulty | 34 | 0.435 | 0.500 | 0.75 |
| dysfunctional family | 28 | 0.538 | 0.538 | 0.50 |
| exposure to others' suicide | 1 | 0.000 | 0.010 | 0.05 |
| stressful life event | 18 | 0.375 | 0.444 | 0.70 |
| traumatic experience | 9 | 0.300 | 0.516 | 0.45 |
| cognitive deficits | 6 | 0.000 | 0.154 | 0.35 |
| suicide means (with access) | 43 | 0.226 | 0.544 | 0.35 |
| sexual orientation related issues | 1 | 0.000 | 0.100 | 0.15 |
| social support | 24 | 0.304 | 0.448 | 0.30 |
| coping strategy | 74 | 0.300 | 0.515 | 0.25 |
| psychological capital | 38 | 0.429 | 0.446 | 0.25 |
| sense of responsibility | 4 | 0.000 | 0.038 | 0.20 |
| meaning in life | 5 | 0.333 | 0.333 | 0.50 |
