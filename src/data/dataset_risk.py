"""
Phase 2 — PyTorch Dataset for Subtask 1 (risk classification).
"""
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL2ID = {"Indicator": 0, "Ideation": 1, "Behavior": 2, "Attempt": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class RiskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["post_clean"].tolist()
        self.labels = [LABEL2ID[l] for l in df["risk_label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item