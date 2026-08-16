"""
Score the weak-category detector on the targeted validation set with
NEGATIVE-WEIGHTING, so precision/F1 are unbiased estimates at the natural
1635-row prevalence (the targeted set enriches positives, which would
otherwise inflate precision). Recall is prevalence-independent.

Neg-weight = (# pure-negative rows in 1635) / (# sampled negatives) = 4.136:
each sampled clean-negative false-positive counts 4.136x; false-positives on
other-weak-category-positive rows count 1x (those rows are fully present).

Compares detector F1 to the current shipped plan's per-category F1 measured on
the natural pooled-560 sample, and writes the per-category adopt decision.
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
from scripts.llm_factor_detector import CAT_LIST

NEG_WEIGHT = 4.136
WEAK = set(CAT_LIST)


def weighted_prf(ids, predset, gold, is_pure_neg):
    tp = fp = fn = 0.0
    for rid in ids:
        p = rid in predset and predset[rid]
        # handled per-category by caller; this generic not used
    return None


def cat_metrics(ids, det, gold, cat):
    tp = fn = 0
    fp_w = 0.0
    for rid in ids:
        g = cat in gold[rid]
        p = cat in det[rid]
        pure_neg = not (set(gold[rid]) & WEAK)
        if p and g:
            tp += 1
        elif p and not g:
            fp_w += NEG_WEIGHT if pure_neg else 1.0
        elif g and not p:
            fn += 1
    pr = tp / (tp + fp_w) if tp + fp_w else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1, tp, fp_w, fn


def current_plan_pooled_f1():
    """Per-category F1 of the shipped plan v2 + verify on the natural pooled-560."""
    recs = {json.loads(l)["row_id"]: json.loads(l)
            for l in open(ROOT / "outputs/predictions/llm_pooled_v3.jsonl")}
    ver = {json.loads(l)["row_id"]: set(json.loads(l)["kept"])
           for l in open(ROOT / "outputs/predictions/llm_pooled_verify.jsonl")}
    oof = pd.read_parquet(ROOT / "data/processed/oof_probs.parquet")
    df = oof[oof["row_id"].isin(recs)].reset_index(drop=True)
    th = json.load(open(ROOT / "data/processed/factor_thresholds_cv_mentalbert.json"))
    plan = json.load(open(ROOT / "data/processed/factor_fusion_plan_v2.json"))
    probs = df[[f"p_factor_{i}" for i in range(24)]].values
    gold = [set(ast.literal_eval(f)) for f in df["factors"]]
    sets = []
    for i, rid in enumerate(df["row_id"]):
        llm_f = set(recs[rid]["factors"])
        for fac, spec in plan.items():
            if spec.get("verify") and fac in llm_f and rid in ver and fac not in ver[rid]:
                llm_f.discard(fac)
        passing = [(FACTOR_LIST[j], probs[i, j]) for j in range(24) if probs[i, j] >= th.get(FACTOR_LIST[j], 0.5)]
        passing.sort(key=lambda x: -x[1])
        bert = {f for f, _ in passing[:10]}
        s = set()
        for j, fac in enumerate(FACTOR_LIST):
            spec = plan[fac]
            m = spec["mode"]
            inb, inl = fac in bert, fac in llm_f
            if m == "bert" and inb: s.add(fac)
            elif m == "llm" and inl: s.add(fac)
            elif m == "union" and (inb or inl): s.add(fac)
            elif m == "inter" and (inb and inl): s.add(fac)
            elif m == "prob" and probs[i, j] + (spec["alpha"] if inl else 0) >= spec["t"]: s.add(fac)
        sets.append(s)
    out = {}
    for cat in FACTOR_LIST:
        tp = fp = fn = 0
        for p, g in zip(sets, gold):
            if cat in p and cat in g: tp += 1
            elif cat in p: fp += 1
            elif cat in g: fn += 1
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        out[cat] = (2 * pr * rc / (pr + rc) if pr + rc else 0.0, sum(cat in g for g in gold))
    return out


def main():
    gold = json.load(open(ROOT / "data/processed/factor_targeted_gold.json"))
    det = {}
    for line in open(ROOT / "outputs/predictions/factor_detector_targeted.jsonl"):
        r = json.loads(line)
        det[r["row_id"]] = set(r["factors_weak"])
    ids = [rid for rid in gold if rid in det]
    print(f"scoring {len(ids)} targeted rows (detector coverage {len(det)}/{len(gold)})\n")
    cur = current_plan_pooled_f1()

    print(f"{'category':42s} {'nPos':>4s} | {'DETECTOR P/R/F1 (natural)':>26s} | {'CURRENT F1 (pooled)':>18s} | Δ  adopt")
    adopt = {}
    for cat in CAT_LIST:
        pr, rc, f1, tp, fpw, fn = cat_metrics(ids, det, gold, cat)
        cur_f1, cur_n = cur.get(cat, (0.0, 0))
        delta = f1 - cur_f1
        take = f1 >= cur_f1 - 0.005
        adopt[cat] = bool(take)
        npos = sum(cat in gold[rid] for rid in ids)
        print(f"{cat:42s} {npos:4d} | {pr:.2f}/{rc:.2f}/{f1:.3f} (tp{int(tp)} fp{fpw:.0f} fn{fn}) | "
              f"{cur_f1:.3f}            | {delta:+.3f}  {'DET' if take else 'keep'}")

    json.dump(adopt, open(ROOT / "data/processed/detector_adopt.json", "w"), indent=1)
    print("\nwrote data/processed/detector_adopt.json")


if __name__ == "__main__":
    main()


def split_half_check():
    """Detector neg-weighted F1 on two disjoint halves of the targeted set —
    a category's adoption is robust only if it beats current on BOTH halves."""
    import hashlib
    gold = json.load(open(ROOT / "data/processed/factor_targeted_gold.json"))
    det = {}
    for line in open(ROOT / "outputs/predictions/factor_detector_targeted.jsonl"):
        r = json.loads(line)
        det[r["row_id"]] = set(r["factors_weak"])
    ids = [rid for rid in gold if rid in det]
    half = {rid: (int(hashlib.md5(rid.encode()).hexdigest(), 16) % 2) for rid in ids}
    cur = current_plan_pooled_f1()
    print(f"\n{'category':42s} {'F1 half A':>10s} {'F1 half B':>10s} {'current':>8s}  robust?")
    robust = {}
    for cat in CAT_LIST:
        f1s = []
        for h in (0, 1):
            hids = [rid for rid in ids if half[rid] == h]
            _, _, f1, *_ = cat_metrics(hids, det, gold, cat)
            f1s.append(f1)
        cur_f1 = cur.get(cat, (0.0, 0))[0]
        ok = f1s[0] >= cur_f1 - 0.02 and f1s[1] >= cur_f1 - 0.02
        robust[cat] = bool(ok)
        print(f"{cat:42s} {f1s[0]:10.3f} {f1s[1]:10.3f} {cur_f1:8.3f}  {'YES' if ok else 'no'}")
    return robust
