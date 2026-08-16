"""
Re-tune the risk arbitration ensemble with extra models added.
Ensemble = mentalbert_oof + sum(w_k * extra_k_oof), normalized; arbitration
overrides the LLM on unsure rows where ensemble argmax disagrees and ensemble
normalized-confidence >= thr. Split-half (val/holdout) robust selection.

Extra models given as tag:parquet pairs (parquet has {tag}p_risk_* columns).

Usage:
    python scripts/retune_arbitration.py --extra db:data/processed/deberta_oof_probs.parquet:dp \
                                          rb:data/processed/rb_oof_risk.parquet:rb
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.metrics import f1_score

LAB = ["Indicator", "Ideation", "Behavior", "Attempt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="+", required=True,
                    help="name:parquet:colprefix triples, e.g. db:.../deberta_oof_probs.parquet:dp")
    args = ap.parse_args()

    val_ids = {json.loads(l)["row_id"] for l in open(ROOT / "outputs/predictions/llm_val_v3.jsonl")}
    recs = {json.loads(l)["row_id"]: json.loads(l) for l in open(ROOT / "outputs/predictions/llm_pooled_v3.jsonl")}
    oof = pd.read_parquet(ROOT / "data/processed/oof_probs.parquet")
    df = oof[oof["row_id"].isin(recs)].reset_index(drop=True)

    extras = []
    for spec in args.extra:
        name, path, pref = spec.split(":")
        ex = pd.read_parquet(path if path.startswith("/") else ROOT / path)
        df = df.merge(ex, on="row_id", how="left")
        extras.append((name, pref))

    df["llm"] = df["row_id"].map(lambda r: recs[r]["risk"])
    df["conf"] = df["row_id"].map(lambda r: recs[r].get("confidence", "medium"))
    df["half"] = df["row_id"].map(lambda r: "val" if r in val_ids else "hold")
    gold = df["risk_label"].values
    mv = (df["half"] == "val").values
    mh = (df["half"] == "hold").values

    mb = df[[f"p_risk_{l}" for l in LAB]].values
    exmats = {pref: df[[f"{pref}p_risk_{l}" for l in LAB]].values for _, pref in extras}

    def ens(weights):
        e = mb.copy()
        for (_, pref), w in zip(extras, weights):
            e = e + w * exmats[pref]
        return e / (1 + sum(weights))

    def arb(weights, thr):
        e = ens(weights)
        ea = np.array(LAB)[e.argmax(1)]
        ec = e.max(1) / e.sum(1)
        return np.array([ea[i] if (df["conf"].iloc[i] != "high" and ea[i] != df["llm"].iloc[i] and ec[i] >= thr)
                         else df["llm"].iloc[i] for i in range(len(df))])

    base_v = f1_score(gold[mv], df["llm"].values[mv], average="weighted")
    base_h = f1_score(gold[mh], df["llm"].values[mh], average="weighted")
    base_p = f1_score(gold, df["llm"].values, average="weighted")
    print(f"baseline LLM: pooled {base_p:.4f} val {base_v:.4f} hold {base_h:.4f}")
    print(f"extras: {[n for n,_ in extras]}\n")

    # grid over per-extra weights + threshold
    import itertools
    wgrid = [0.2, 0.3, 0.4] if len(extras) == 1 else [0.2, 0.3]
    results = []
    for weights in itertools.product(wgrid, repeat=len(extras)):
        for thr in [0.85, 0.88, 0.90]:
            p = arb(list(weights), thr)
            po = f1_score(gold, p, average="weighted")
            v = f1_score(gold[mv], p[mv], average="weighted")
            h = f1_score(gold[mh], p[mh], average="weighted")
            robust = v >= base_v and h >= base_h
            results.append((po, list(weights), thr, v, h, robust))
    results.sort(reverse=True)
    print(f"{'pooled':>7s} {'val':>7s} {'hold':>7s}  weights thr  robust")
    for po, w, thr, v, h, rob in results[:10]:
        print(f"{po:.4f} {v:.4f} {h:.4f}  {w} {thr}  {'YES' if rob else ''}")
    best = next((r for r in results if r[5]), results[0])
    print(f"\nBEST robust: weights={best[1]} thr={best[2]} -> pooled {best[0]:.4f} (val {best[3]:.4f} hold {best[4]:.4f})")
    print(f"vs current 2-model (deberta w=0.3 thr=0.85): pooled 0.8623")
    json.dump({"weights": best[1], "thr": best[2], "extras": [n for n, _ in extras]},
              open(ROOT / "data/processed/arbitration_config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
