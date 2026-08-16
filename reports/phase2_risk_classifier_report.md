# Phase 2 — Risk Classifier (Subtask 1) Report

Model: mental/mental-bert-base-uncased

Final val metrics: {
  "eval_loss": 0.74672532081604,
  "eval_macro_f1": 0.7527922688763502,
  "eval_weighted_f1": 0.7716465535613636,
  "eval_f1_Indicator": 0.8472906403940886,
  "eval_f1_Ideation": 0.726027397260274,
  "eval_f1_Behavior": 0.7076923076923077,
  "eval_f1_Attempt": 0.7301587301587301,
  "eval_runtime": 0.5857,
  "eval_samples_per_second": 462.701,
  "eval_steps_per_second": 15.366,
  "epoch": 8.0
}

## Classification report
```
              precision    recall  f1-score   support

   Indicator      0.925     0.782     0.847       110
    Ideation      0.707     0.746     0.726        71
    Behavior      0.657     0.767     0.708        60
     Attempt      0.697     0.767     0.730        30

    accuracy                          0.768       271
   macro avg      0.746     0.765     0.753       271
weighted avg      0.783     0.768     0.772       271

```

## Confusion matrix (rows=true, cols=pred)
```
[[86 14 10  0]
 [ 7 53  9  2]
 [ 0  6 46  8]
 [ 0  2  5 23]]
```
