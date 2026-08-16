# Phase 1 — Data Cleaning & EDA Report

Total rows: 1635 | Unique users: 153

## Data quality flags
- Length outliers (>1000 words): 4
- Evidence not verbatim substring of post: 103
- Rows with empty factor list: 128

## Risk label distribution (full dataset)
- Indicator: 611 (37.4%)
- Ideation: 519 (31.7%)
- Behavior: 391 (23.9%)
- Attempt: 114 (7.0%)

## Factor frequency (top 10)
- hopelessness: 745
- emotion dysregulation: 542
- poor social support: 501
- low self-esteem: 475
- coping strategy: 443
- suicide means (with access): 292
- psychological capital: 238
- mental health issues: 228
- prior self-harm or suicidal thought/attempt: 215
- interpersonal difficulty: 195

## Split summary
```
Train: 1364 rows / 130 users
Val:   271 rows / 23 users

Risk label distribution (train vs val, %):
  Indicator  train=36.7%  val=40.6%
  Ideation   train=32.8%  val=26.2%
  Behavior   train=24.3%  val=22.1%
  Attempt    train=6.2%  val=11.1%
```

## Sample evidence-mismatch rows (first 10)
```
row_id user_id
P00177   U0127
P00206   U0129
P00216   U0130
P00291   U0135
P00343   U0017
P00407   U0138
P00576   U0031
P00578   U0031
P00579   U0031
P00647   U0146
```