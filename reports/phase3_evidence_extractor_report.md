# Phase 3 — Evidence Span Extraction Report

Model: mental/mental-bert-base-uncased

Rows with at least one taggable span: 1364/1364

Final val metrics: {
  "eval_loss": 0.5882705450057983,
  "eval_evidence_macro_f1": 0.23305588585017833,
  "eval_token_precision": 0.48274002157497303,
  "eval_token_recall": 0.7300163132137031,
  "eval_token_f1": 0.5811688311688312,
  "eval_runtime": 0.6705,
  "eval_samples_per_second": 404.178,
  "eval_steps_per_second": 13.423,
  "epoch": 8.0
}

## seqeval classification report
```
              precision    recall  f1-score   support

        EVID      0.169     0.374     0.233       262

   micro avg      0.169     0.374     0.233       262
   macro avg      0.169     0.374     0.233       262
weighted avg      0.169     0.374     0.233       262

```
