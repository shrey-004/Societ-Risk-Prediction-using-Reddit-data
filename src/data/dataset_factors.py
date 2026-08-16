"""
Phase 4 — PyTorch Dataset for Subtask 2 (multi-label factor classification).

Uses the exact 24-category taxonomy from the official rules doc (verified
to match our extracted factor_vocab.json exactly — 24/24, no mismatches).
"""
import ast

import pandas as pd
import torch
from torch.utils.data import Dataset

# Canonical order per the official competition rules doc
FACTOR_LIST = [
    "mental health issues",
    "physical health/characteristic",
    "substance use",
    "hopelessness",
    "emotion dysregulation",
    "low self-esteem",
    "poor school performance",
    "low socio-economic status",
    "interpersonal violence",
    "prior self-harm or suicidal thought/attempt",
    "poor social support",
    "interpersonal difficulty",
    "dysfunctional family",
    "exposure to others' suicide",
    "stressful life event",
    "traumatic experience",
    "cognitive deficits",
    "suicide means (with access)",
    "sexual orientation related issues",
    "social support",
    "coping strategy",
    "psychological capital",
    "sense of responsibility",
    "meaning in life",
]
FACTOR2ID = {f: i for i, f in enumerate(FACTOR_LIST)}
NUM_FACTORS = len(FACTOR_LIST)


def encode_multihot(factors: list[str]) -> torch.Tensor:
    vec = torch.zeros(NUM_FACTORS, dtype=torch.float)
    for f in factors:
        if f in FACTOR2ID:
            vec[FACTOR2ID[f]] = 1.0
    return vec


class FactorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["post_clean"].tolist()
        self.factors_list = [ast.literal_eval(s) for s in df["factors"].tolist()]
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
        item["labels"] = encode_multihot(self.factors_list[idx])
        return item