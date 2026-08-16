# Phase 4 — Factor Classifier (Subtask 2) Report

Model: emilyalsentzer/Bio_ClinicalBERT

Final val metrics: {
  "eval_loss": 0.4626193642616272,
  "eval_factors_macro_f1": 0.2979115273057367,
  "eval_factors_micro_f1": 0.44104134762633995,
  "eval_runtime": 0.7368,
  "eval_samples_per_second": 367.826,
  "eval_steps_per_second": 12.216,
  "epoch": 8.0
}

## Per-factor classification report
```
                                             precision    recall  f1-score   support

                       mental health issues      0.371     0.419     0.394        31
             physical health/characteristic      0.000     0.000     0.000         4
                              substance use      0.000     0.000     0.000         2
                               hopelessness      0.697     0.639     0.667       108
                      emotion dysregulation      0.523     0.523     0.523        65
                            low self-esteem      0.507     0.479     0.493        71
                    poor school performance      0.000     0.000     0.000         1
                  low socio-economic status      0.000     0.000     0.000         9
                     interpersonal violence      0.889     0.400     0.552        20
prior self-harm or suicidal thought/attempt      0.565     0.289     0.382        45
                        poor social support      0.674     0.279     0.395       104
                   interpersonal difficulty      0.556     0.441     0.492        34
                       dysfunctional family      0.647     0.393     0.489        28
                exposure to others' suicide      0.000     0.000     0.000         1
                       stressful life event      0.310     0.500     0.383        18
                       traumatic experience      0.667     0.444     0.533         9
                         cognitive deficits      0.250     0.167     0.200         6
                suicide means (with access)      0.556     0.233     0.328        43
          sexual orientation related issues      0.000     0.000     0.000         1
                             social support      0.421     0.333     0.372        24
                            coping strategy      0.396     0.257     0.311        74
                      psychological capital      0.357     0.263     0.303        38
                    sense of responsibility      0.000     0.000     0.000         4
                            meaning in life      1.000     0.200     0.333         5

                                  micro avg      0.513     0.387     0.441       745
                                  macro avg      0.391     0.261     0.298       745
                               weighted avg      0.540     0.387     0.436       745
                                samples avg      0.230     0.222     0.201       745

```
