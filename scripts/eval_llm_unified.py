"""
Score the unified LLM pass against gold on an OOF sample, alongside the
BERT/BIO baselines and fusion variants — the decision layer that picks what
actually ships in the submission. Every number here comes from models that
never trained on the scored rows (LLM: few-shots are excluded from the
sample; BERT: out-of-fold predictions).

Usage:
    python scripts/eval_llm_unified.py --llm_jsonl out.jsonl [--sample data/processed/llm_val_sample.csv]
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.metrics import f1_score, confusion_matrix
from src.eval.phrase_f1 import corpus_phrase_f1
from src.data.dataset_factors import FACTOR_LIST

RISK = ["Indicator", "Ideation", "Behavior", "Attempt"]


def load(llm_jsonl, sample_csv):
    sample = pd.read_csv(sample_csv)
    oof = pd.read_parquet(ROOT / "data/processed/oof_probs.parquet")
    df = sample[["row_id"]].merge(oof, on="row_id", how="left")
    llm = {}
    for line in open(llm_jsonl):
        r = json.loads(line)
        llm[r["row_id"]] = r
    df = df[df["row_id"].isin(llm)].reset_index(drop=True)
    df["llm_risk"] = df["row_id"].map(lambda r: llm[r]["risk"])
    df["llm_evidence"] = df["row_id"].map(lambda r: llm[r]["evidence"])
    df["llm_factors"] = df["row_id"].map(lambda r: llm[r]["factors"])
    df["llm_conf"] = df["row_id"].map(lambda r: llm[r].get("confidence", "medium"))
    df["gold_spans"] = df["evidence_spans"].apply(ast.literal_eval)
    df["gold_factors"] = df["factors"].apply(lambda f: set(ast.literal_eval(f)))
    df["bio_spans"] = df["evidence_pred"].apply(
        lambda e: [(s, c) for s, c in json.loads(e)])
    return df


def risk_report(df):
    gold = df["risk_label"].values
    bert_probs = df[[f"p_risk_{l}" for l in RISK]].values
    bert = np.array(RISK)[bert_probs.argmax(1)]
    out = {}
    out["bert"] = f1_score(gold, bert, average="weighted")
    out["llm"] = f1_score(gold, df["llm_risk"], average="weighted")
    onehot = np.zeros_like(bert_probs)
    for i, lab in enumerate(df["llm_risk"]):
        onehot[i, RISK.index(lab)] = 1.0
    for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
        blend = bert_probs + w * onehot
        out[f"blend_w{w}"] = f1_score(gold, np.array(RISK)[blend.argmax(1)], average="weighted")
    print("\n=== RISK weighted F1 ===")
    for k, v in out.items():
        print(f"  {k:12s} {v:.4f}")
    print("\nLLM confusion (rows=gold, cols=pred):")
    print(pd.DataFrame(confusion_matrix(gold, df["llm_risk"], labels=RISK), index=RISK, columns=RISK))
    print("\nper-class F1 (LLM):")
    for lab, f in zip(RISK, f1_score(gold, df["llm_risk"], labels=RISK, average=None)):
        print(f"  {lab:10s} {f:.3f}")
    print("\nLLM self-reported confidence vs accuracy:")
    df["_ok"] = df["llm_risk"] == df["risk_label"]
    print(df.groupby("llm_conf")["_ok"].agg(["mean", "count"]))
    best = max(out, key=out.get)
    return out, best


def evidence_report(df, final_risk_col):
    gold = df["gold_spans"].tolist()
    variants = {}
    variants["bio@0.6"] = [[s for s, c in bs if c >= 0.6] for bs in df["bio_spans"]]
    variants["llm_raw"] = df["llm_evidence"].tolist()

    def gated(llm_sp, bio_sp, risk):
        if risk == "Indicator":
            return []
        sp = list(llm_sp)
        if not sp:
            sp = [s for s, c in bio_sp if c >= 0.6][:3]
        return sp

    variants["llm_gated"] = [gated(l, b, r) for l, b, r in
                             zip(df["llm_evidence"], df["bio_spans"], df[final_risk_col])]

    def union(llm_sp, bio_sp, risk, conf=0.6):
        if risk == "Indicator":
            return []
        sp = list(llm_sp)
        for s, c in bio_sp:
            if c >= conf and not any(s.lower() in x.lower() or x.lower() in s.lower() for x in sp):
                sp.append(s)
        return sp[:4]

    variants["union_gated"] = [union(l, b, r) for l, b, r in
                               zip(df["llm_evidence"], df["bio_spans"], df[final_risk_col])]
    print("\n=== EVIDENCE phrase F1 ===")
    res = {}
    for name, preds in variants.items():
        r = corpus_phrase_f1(preds, gold)
        res[name] = r["phrase_f1"]
        print(f"  {name:12s} F1={r['phrase_f1']:.4f}  P={r['aggregate_precision']:.3f} R={r['aggregate_recall']:.3f}")
    return res


def factor_report(df):
    th = json.load(open(ROOT / "data/processed/factor_thresholds_cv_mentalbert.json"))
    probs = df[[f"p_factor_{i}" for i in range(24)]].values
    gold = df["gold_factors"].tolist()

    bert_sets = []
    for r in range(len(df)):
        passing = [(FACTOR_LIST[i], probs[r, i]) for i in range(24) if probs[r, i] >= th.get(FACTOR_LIST[i], 0.5)]
        passing.sort(key=lambda x: -x[1])
        bert_sets.append({f for f, _ in passing[:10]})
    llm_sets = [set(f) for f in df["llm_factors"]]
    union_sets = [b | l for b, l in zip(bert_sets, llm_sets)]
    inter_sets = [b & l for b, l in zip(bert_sets, llm_sets)]

    def per_cat(pred_sets):
        f1s = {}
        for fac in FACTOR_LIST:
            tp = fp = fn = 0
            for p, g in zip(pred_sets, gold):
                if fac in p and fac in g: tp += 1
                elif fac in p: fp += 1
                elif fac in g: fn += 1
            pr = tp / (tp + fp) if tp + fp else 0
            rc = tp / (tp + fn) if tp + fn else 0
            f1s[fac] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        return f1s

    strat = {"bert": per_cat(bert_sets), "llm": per_cat(llm_sets),
             "union": per_cat(union_sets), "inter": per_cat(inter_sets)}
    print("\n=== FACTORS macro F1 ===")
    for name, f1s in strat.items():
        print(f"  {name:6s} macro={np.mean(list(f1s.values())):.4f}")
    print(f"\n  {'category':45s} {'n':>4s} {'bert':>6s} {'llm':>6s} {'union':>6s} {'inter':>6s} best")
    support = {fac: sum(fac in g for g in gold) for fac in FACTOR_LIST}
    oracle = 0.0
    per_cat_choice = {}
    for fac in sorted(FACTOR_LIST, key=lambda f: -support[f]):
        vals = {k: strat[k][fac] for k in strat}
        best = max(vals, key=vals.get)
        per_cat_choice[fac] = best
        oracle += vals[best]
        print(f"  {fac:45s} {support[fac]:4d} {vals['bert']:6.3f} {vals['llm']:6.3f} "
              f"{vals['union']:6.3f} {vals['inter']:6.3f} {best}")
    print(f"\n  per-category-best (oracle on this sample): {oracle/24:.4f}")
    return strat, per_cat_choice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_jsonl", required=True)
    ap.add_argument("--sample", default=str(ROOT / "data/processed/llm_val_sample.csv"))
    args = ap.parse_args()
    df = load(args.llm_jsonl, args.sample)
    print(f"scoring {len(df)} rows")
    out, best = risk_report(df)
    # use LLM risk for gating (evaluate with the risk source we'd deploy)
    df["final_risk"] = df["llm_risk"]
    evidence_report(df, "final_risk")
    factor_report(df)


if __name__ == "__main__":
    main()
