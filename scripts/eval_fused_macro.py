"""
Exact fused factor macro-F1 on the natural pooled-560, comparing:
  - shipped plan v2 + verify (current)
  - plan v2 + verify + detector-for-adopted-categories (proposed)
This is the number that predicts the leaderboard S2 move.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset_factors import FACTOR_LIST

recs = {json.loads(l)["row_id"]: json.loads(l) for l in open(ROOT / "outputs/predictions/llm_pooled_v3.jsonl")}
ver = {json.loads(l)["row_id"]: set(json.loads(l)["kept"]) for l in open(ROOT / "outputs/predictions/llm_pooled_verify.jsonl")}
det = {}
for f in ["factor_detector_targeted.jsonl", "factor_detector_pooled.jsonl"]:
    p = ROOT / "outputs/predictions" / f
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            det[r["row_id"]] = set(r["factors_weak"])
adopt = {c for c, v in json.load(open(ROOT / "data/processed/detector_adopt.json")).items() if v}

oof = pd.read_parquet(ROOT / "data/processed/oof_probs.parquet")
df = oof[oof["row_id"].isin(recs)].reset_index(drop=True)
th = json.load(open(ROOT / "data/processed/factor_thresholds_cv_mentalbert.json"))
plan = json.load(open(ROOT / "data/processed/factor_fusion_plan_v2.json"))
probs = df[[f"p_factor_{i}" for i in range(24)]].values
gold = [set(ast.literal_eval(f)) for f in df["factors"]]
det_cov = sum(rid in det for rid in df["row_id"])
print(f"pooled {len(df)} rows | detector coverage {det_cov}/{len(df)} | adopting {sorted(adopt)}\n")


def build(use_detector):
    sets = []
    for i, rid in enumerate(df["row_id"]):
        llm_f = set(recs[rid]["factors"])
        for fac, spec in plan.items():
            if spec.get("verify") and fac in llm_f and rid in ver and fac not in ver[rid]:
                llm_f.discard(fac)
        passing = [(FACTOR_LIST[j], probs[i, j]) for j in range(24) if probs[i, j] >= th.get(FACTOR_LIST[j], 0.5)]
        passing.sort(key=lambda x: -x[1])
        bert = {f for f, _ in passing[:10]}
        det_set = det.get(rid, set())
        s = set()
        for j, fac in enumerate(FACTOR_LIST):
            if use_detector and fac in adopt and rid in det:
                if fac in det_set:
                    s.add(fac)
                continue
            spec = plan[fac]
            m = spec["mode"]
            inb, inl = fac in bert, fac in llm_f
            if m == "bert" and inb: s.add(fac)
            elif m == "llm" and inl: s.add(fac)
            elif m == "union" and (inb or inl): s.add(fac)
            elif m == "inter" and (inb and inl): s.add(fac)
            elif m == "prob" and probs[i, j] + (spec["alpha"] if inl else 0) >= spec["t"]: s.add(fac)
        sets.append(s)
    return sets


def macro(sets, show=False):
    f1s = {}
    for fac in FACTOR_LIST:
        tp = fp = fn = 0
        for p, g in zip(sets, gold):
            if fac in p and fac in g: tp += 1
            elif fac in p: fp += 1
            elif fac in g: fn += 1
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s[fac] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    if show:
        for fac in sorted(f1s, key=f1s.get):
            print(f"  {f1s[fac]:.3f}  {fac}")
    return np.mean(list(f1s.values())), f1s


cur_macro, cur_f1 = macro(build(False))
new_macro, new_f1 = macro(build(True))
print(f"CURRENT (plan v2 + verify):            macro {cur_macro:.4f}  (leaderboard S2 = 0.6201)")
print(f"PROPOSED (+ detector for {len(adopt)} cats):    macro {new_macro:.4f}  (Δ {new_macro-cur_macro:+.4f})")
print("\nadopted-category F1 change:")
for fac in sorted(adopt):
    print(f"  {fac:42s} {cur_f1[fac]:.3f} -> {new_f1[fac]:.3f}  ({new_f1[fac]-cur_f1[fac]:+.3f})")
