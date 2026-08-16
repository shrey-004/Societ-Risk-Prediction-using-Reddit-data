"""
Quick diagnostic: check the actual threshold values and actual predicted
probabilities for one real test post, to pinpoint why every factor is
firing.

Usage: python diagnose_factors.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, ".")
from src.data.clean import clean_text
from src.data.dataset_factors import FACTOR_LIST

root = Path(".")
with open(root / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

# 1. print the actual threshold values being used
thresholds_path = root / "data" / "processed" / "factor_thresholds_ensemble.json"
with open(thresholds_path) as f:
    thresholds = json.load(f)
print("=== Thresholds in factor_thresholds_ensemble.json ===")
for name in FACTOR_LIST:
    print(f"  {name:<45} {thresholds.get(name, 'MISSING')}")

# 2. run one seed model on one real test post and print raw probabilities
test_df = pd.read_excel(root / "data" / "raw" / "test.xlsx", sheet_name="Sheet1")
post = clean_text(test_df.iloc[0]["post"])
print(f"\n=== Test post (row 0): {test_df.iloc[0]['row_id']} ===")
print(post[:200])

device = "cuda" if torch.cuda.is_available() else "cpu"
max_len = cfg["model"]["max_seq_len"]

model_dir = root / "outputs" / "checkpoints" / "factor_classifier_best_seed42"
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()

print(f"\nModel config problem_type: {model.config.problem_type}")
print(f"Model config num_labels: {model.config.num_labels}")

enc = tokenizer(post, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
enc = {k: v.to(device) for k, v in enc.items()}
with torch.no_grad():
    logits = model(**enc).logits[0].cpu().numpy()

print(f"\n=== Raw logits (seed42 model, this post) ===")
print(logits)

probs = 1 / (1 + np.exp(-logits))
print(f"\n=== Sigmoid probabilities ===")
for name, p in zip(FACTOR_LIST, probs):
    t = thresholds.get(name, 0.5)
    fires = "FIRES" if p >= t else "no"
    print(f"  {name:<45} prob={p:.4f}  threshold={t:.2f}  -> {fires}")