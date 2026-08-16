"""
Final submission builder v2 — LLM+BERT hybrid.

Sources:
  - data/processed/test_probs_ensemble.parquet  (BERT: 5-fold+fulldata ensembled
    risk softmax, factor sigmoids, BIO evidence spans w/ confidence)
  - outputs/predictions/llm_test.jsonl          (unified LLM pass: risk, evidence,
    factors, confidence — scripts/llm_unified_pass.py on data/raw/test.xlsx)
  - data/processed/test_neardup_clusters.json   (near-identical posts per user)
  - data/processed/factor_fusion_plan.json      (per-category strategy chosen on
    the OOF validation sample by scripts/eval_llm_unified.py analysis)

Fusion rules (each validated on OOF before use — see reports/llm_hybrid_validation.md):
  RISK    : BERT ensemble probs + w * onehot(LLM risk); w from validation.
            Rows without LLM output fall back to BERT argmax.
  EVIDENCE: final risk == Indicator -> "none" (96% of gold Indicator rows have none);
            else LLM spans, backfilled from BIO spans (conf >= 0.6, top-3) when LLM
            returned none. Spans re-verified verbatim against the RAW post text.
  FACTORS : per-category strategy in factor_fusion_plan.json: one of
            bert | llm | union | inter (+ optional threshold override).

Usage:
    python scripts/generate_submission_v2.py --llm_jsonl outputs/predictions/llm_test.jsonl \
        [--risk_llm_weight 1.0] [--out outputs/predictions/RayofHope.csv]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset_factors import FACTOR_LIST

RISK = ["Indicator", "Ideation", "Behavior", "Attempt"]


_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def find_verbatim(span: str, raw_post: str) -> str | None:
    """Return the exact-cased substring of raw_post matching span
    (case-insensitive; smart-quote-normalized and whitespace-flexible
    fallbacks — all 1:1 char translations so indices map back to the
    original raw text), else None."""
    low = raw_post.lower()
    s = span.strip()
    i = low.find(s.lower())
    if i >= 0:
        return raw_post[i:i + len(s)]
    # smart-quote/dash-normalized search (1:1 translation keeps indices valid)
    norm_post = low.translate(_QUOTE_MAP)
    norm_span = s.lower().translate(_QUOTE_MAP)
    i = norm_post.find(norm_span)
    if i >= 0:
        return raw_post[i:i + len(norm_span)]
    # whitespace-flexible regex on the normalized text
    pat = re.escape(re.sub(r"\s+", " ", norm_span)).replace(r"\ ", r"\s+")
    m = re.search(pat, norm_post)
    if m:
        return raw_post[m.start():m.end()]
    return None


# prioritized explicit-signal bank for the last-resort evidence fallback
# (used only when neither LLM nor BIO produced a verifiable span on a
# non-Indicator row — gold non-Indicator rows ALWAYS have spans, so an empty
# prediction there scores a guaranteed 0)
KEYWORD_BANK = [
    r"(?:want(?:ed)?|wanna|going|gonna|plan(?:ning)?|about|ready|need) to (?:kill(?:ing)? my ?self|die|end (?:it|my life|it all)|commit suicide)",
    r"kill(?:ing)? my ?self", r"kms\b", r"commit(?:ting)? suicide", r"suicid\w+",
    r"end(?:ing)? (?:it all|my life|it)\b", r"take my (?:own )?life",
    r"(?:want|wanna|ready|wish) to die", r"wanna die", r"want to die",
    r"(?:i'?d|i would) rather (?:be dead|die)", r"don'?t want to (?:live|be alive|exist)",
    r"(?:overdos\w+|hang(?:ing)? myself|jump(?:ing)? off|slit my|cut(?:ting)? my)",
    r"(?:tired of|done with) (?:living|life|being alive)",
]


def keyword_fallback_span(raw_post: str) -> str | None:
    low = raw_post.lower().translate(_QUOTE_MAP)
    for pat in KEYWORD_BANK:
        m = re.search(pat, low)
        if m:
            return raw_post[m.start():m.end()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_jsonl", required=True)
    ap.add_argument("--risk_llm_weight", type=float, default=1.0)
    ap.add_argument("--bio_backfill_conf", type=float, default=0.6)
    ap.add_argument("--factor_plan", default=str(ROOT / "data/processed/factor_fusion_plan.json"))
    ap.add_argument("--verify_jsonl", default="",
                    help="Optional llm_verify_factors.py output; removes unconfirmed tags for plan entries with verify:true")
    ap.add_argument("--deberta_test_probs", default="",
                    help="Optional data/processed/deberta_test_probs.parquet. When set, enables risk arbitration: "
                         "on LLM-unsure rows where the mentalbert-ens + weighted discriminative ensemble is >= "
                         "--risk_arb_conf normalized-confident and DISAGREES with the LLM, take the ensemble label. "
                         "Split-half validated; mechanism: LLM self-reported confidence flags the rows where a strong "
                         "discriminative second opinion is worth taking.")
    ap.add_argument("--risk_arb_conf", type=float, default=0.88)
    ap.add_argument("--risk_arb_deberta_weight", type=float, default=0.3)
    ap.add_argument("--arb_extra", nargs="*", default=[],
                    help="Extra arbitration models as parquet:colprefix:weight, e.g. "
                         "data/processed/rb_test_risk.parquet:rb:0.3 (adds w*model to the ensemble).")
    ap.add_argument("--detector_jsonl", default="",
                    help="Optional llm_factor_detector.py output; for categories in --detector_adopt, "
                         "replaces the plan-v2 factor decision with the detector's (split-half validated).")
    ap.add_argument("--detector_adopt", default="data/processed/detector_adopt.json")
    ap.add_argument("--out", default=str(ROOT / "outputs/predictions/RayofHope.csv"))
    args = ap.parse_args()

    test_raw = pd.read_excel(ROOT / "data/raw/test.xlsx", sheet_name="Sheet1")
    test_raw["post"] = test_raw["post"].astype(str)
    probs = pd.read_parquet(ROOT / "data/processed/test_probs_ensemble.parquet")
    df = test_raw.merge(probs.drop(columns=["anon_user_id", "post_clean"]), on="row_id")
    assert len(df) == len(test_raw)

    llm = {}
    for line in open(args.llm_jsonl):
        r = json.loads(line)
        llm[r["row_id"]] = r
    print(f"LLM coverage: {sum(df['row_id'].astype(str).isin(llm))}/{len(df)}")

    plan = json.load(open(args.factor_plan))
    th = json.load(open(ROOT / "data/processed/factor_thresholds_cv_mentalbert.json"))

    # ---------- RISK ----------
    # Pure LLM label where available (softmax max < 1 so the +weight onehot
    # always wins), BERT-ensemble argmax as fallback for uncovered rows.
    bert_probs = df[[f"p_risk_{l}" for l in RISK]].values.copy()
    fused = bert_probs.copy()
    llm_risk_col = []
    for i, rid in enumerate(df["row_id"].astype(str)):
        rec = llm.get(rid)
        llm_risk_col.append(rec["risk"] if rec else None)
        if rec:
            fused[i, RISK.index(rec["risk"])] += args.risk_llm_weight
    final_risk = [RISK[k] for k in fused.argmax(1)]
    df["final_risk"] = final_risk
    df["llm_risk"] = llm_risk_col

    # DeBERTa risk arbitration (split-half validated) — override the LLM on
    # unsure rows where a confident, disagreeing discriminative ensemble knows better.
    if args.deberta_test_probs:
        deb = pd.read_parquet(ROOT / args.deberta_test_probs if not str(args.deberta_test_probs).startswith("/")
                              else args.deberta_test_probs)
        deb = df[["row_id"]].merge(deb, on="row_id")
        assert len(deb) == len(df)
        db_probs = deb[[f"dp_risk_{l}" for l in RISK]].values
        w = args.risk_arb_deberta_weight
        # ensemble: mentalbert + w*deberta + sum(extra weights * extra models), then renormalize
        acc = bert_probs + w * db_probs
        wsum = 1.0 + w
        for spec in args.arb_extra:
            path, pref, ew = spec.split(":")
            ew = float(ew)
            ex = pd.read_parquet(ROOT / path if not str(path).startswith("/") else path)
            ex = df[["row_id"]].merge(ex, on="row_id")
            assert len(ex) == len(df), f"{path} row mismatch"
            acc = acc + ew * ex[[f"{pref}p_risk_{l}" for l in RISK]].values
            wsum += ew
            print(f"  arbitration + {pref} (w={ew})")
        bert_ens = acc / wsum
        be_arg = bert_ens.argmax(1)
        be_conf = bert_ens.max(1) / bert_ens.sum(1)
        n_arb = n_guard = 0
        for i, rid in enumerate(df["row_id"].astype(str)):
            rec = llm.get(rid)
            if not rec or rec.get("confidence") == "high":
                continue
            new_lab = RISK[be_arg[i]]
            if new_lab != df.at[i, "final_risk"] and be_conf[i] >= args.risk_arb_conf:
                # self-consistency guard: a non-Indicator label asserts explicit
                # suicidal expression, so refuse the flip if we cannot ground it
                # in any extractable span (blocks spurious high-confidence flips
                # on very short/OOD posts, e.g. "Am i dying??"). Metric-neutral
                # on the pooled sample; pure consistency safeguard.
                if new_lab != "Indicator":
                    raw = df.at[i, "post"]
                    bio = [s for s, c in json.loads(df.at[i, "evidence_pred"]) if c >= 0.5]
                    grounded = any(find_verbatim(s, raw) for s in bio) or keyword_fallback_span(raw) is not None
                    if not grounded:
                        n_guard += 1
                        continue
                df.at[i, "final_risk"] = new_lab
                n_arb += 1
        print(f"risk arbitration: {n_arb} unsure rows overridden by DeBERTa+mentalbert ensemble "
              f"(conf >= {args.risk_arb_conf}); {n_guard} ungroundable flips vetoed by consistency guard")

    # ---------- EVIDENCE ----------
    evidences = []
    n_backfill = n_dropped = 0
    for _, row in df.iterrows():
        rid = str(row["row_id"])
        raw_post = row["post"]
        if row["final_risk"] == "Indicator":
            evidences.append("none")
            continue
        spans = []
        rec = llm.get(rid)
        cand = list(rec["evidence"]) if rec else []
        if not cand:
            bio = [(s, c) for s, c in json.loads(row["evidence_pred"]) if c >= args.bio_backfill_conf]
            bio.sort(key=lambda x: -x[1])
            cand = [s for s, _ in bio[:3]]
            n_backfill += 1
        for s in cand:
            v = find_verbatim(s, raw_post)
            if v is None:
                n_dropped += 1
                continue
            if any(v.lower() in x.lower() or x.lower() in v.lower() for x in spans):
                continue
            spans.append(v)
        if not spans:
            # last resort 1: highest-confidence BIO span at any confidence
            bio = sorted(json.loads(row["evidence_pred"]), key=lambda x: -x[1])
            for s, _ in bio:
                v = find_verbatim(s, raw_post)
                if v:
                    spans = [v]
                    break
        if not spans:
            # last resort 2: explicit-signal keyword bank (non-Indicator gold
            # rows always carry spans; empty prediction = guaranteed 0)
            v = keyword_fallback_span(raw_post)
            if v:
                spans = [v]
        evidences.append("; ".join(spans) if spans else "none")
    df["evidence_out"] = evidences
    print(f"evidence: backfilled from BIO on {n_backfill} rows, dropped {n_dropped} unverifiable spans")

    # ---------- FACTORS ----------
    verify = {}
    if args.verify_jsonl:
        for line in open(args.verify_jsonl):
            r = json.loads(line)
            verify[str(r["row_id"])] = set(r["kept"])
        print(f"verification pass loaded for {len(verify)} rows")

    # Dedicated weak-category detector: for the categories in detector_adopt.json
    # (split-half validated to beat the plan), use the detector's binary output
    # instead of the plan v2 fusion. Untouched for all other categories.
    detector, adopt = {}, {}
    if args.detector_jsonl:
        for line in open(args.detector_jsonl):
            r = json.loads(line)
            detector[str(r["row_id"])] = set(r["factors_weak"])
        adopt = {c for c, v in json.load(open(ROOT / args.detector_adopt
                 if not str(args.detector_adopt).startswith("/") else args.detector_adopt)).items() if v}
        print(f"detector loaded for {len(detector)} rows; adopting {len(adopt)} categories: {sorted(adopt)}")

    fps = df[[f"p_factor_{i}" for i in range(24)]].values
    factors_out = []
    for i, rid in enumerate(df["row_id"].astype(str)):
        rec = llm.get(rid)
        llm_set = set(rec["factors"]) if rec else set()
        passing = [(FACTOR_LIST[j], fps[i, j]) for j in range(24) if fps[i, j] >= th.get(FACTOR_LIST[j], 0.5)]
        passing.sort(key=lambda x: -x[1])
        bert_set = {f for f, _ in passing[:10]}
        det_set = detector.get(rid, set())
        chosen = set()
        for j, fac in enumerate(FACTOR_LIST):
            if fac in adopt:
                # detector owns this category (only if we have a detector row)
                if rid in detector:
                    if fac in det_set:
                        chosen.add(fac)
                    continue
                # else fall through to the plan as backstop
            spec = plan.get(fac, "union")
            if isinstance(spec, str):
                spec = {"mode": spec}
            in_l = fac in llm_set
            if spec.get("verify") and in_l and rid in verify and fac not in verify[rid]:
                in_l = False
            in_b = fac in bert_set
            m = spec["mode"]
            if m == "bert" and in_b: chosen.add(fac)
            elif m == "llm" and in_l: chosen.add(fac)
            elif m == "union" and (in_b or in_l): chosen.add(fac)
            elif m == "inter" and (in_b and in_l): chosen.add(fac)
            elif m == "prob" and fps[i, j] + (float(spec["alpha"]) if in_l else 0.0) >= float(spec["t"]):
                chosen.add(fac)
        if not rec and rid not in detector:
            chosen = bert_set  # no LLM output at all -> pure BERT
        factors_out.append(str(sorted(chosen)))
    df["factors_out"] = factors_out

    # ---------- near-duplicate consistency (report only borderline flips) ----------
    clusters = json.load(open(ROOT / "data/processed/test_neardup_clusters.json"))
    id2idx = {str(r): i for i, r in enumerate(df["row_id"].astype(str))}
    n_flips = 0
    for cl in clusters:
        idxs = [id2idx[r] for r in cl if r in id2idx]
        labs = [df.at[i, "final_risk"] for i in idxs]
        if len(set(labs)) > 1:
            # majority label among cluster, weighted by fused prob margin
            from collections import Counter
            maj = Counter(labs).most_common(1)[0][0]
            for i in idxs:
                if df.at[i, "final_risk"] != maj:
                    p = fused[i] / fused[i].sum()
                    top2 = np.sort(p)[-2:]
                    margin = top2[1] - top2[0]
                    if margin < 0.10:  # only harmonize genuinely borderline rows
                        print(f"  neardup flip {df.at[i,'row_id']}: {df.at[i,'final_risk']} -> {maj} (margin {margin:.3f})")
                        df.at[i, "final_risk"] = maj
                        if maj == "Indicator":
                            df.at[i, "evidence_out"] = "none"
                        n_flips += 1
    print(f"near-dup harmonization: {n_flips} flips")

    # re-apply evidence gate after any flips to non-Indicator (rare): backfill
    for i, row in df.iterrows():
        if row["final_risk"] != "Indicator" and row["evidence_out"] == "none":
            bio = sorted(json.loads(row["evidence_pred"]), key=lambda x: -x[1])
            for s, _ in bio:
                v = find_verbatim(s, row["post"])
                if v:
                    df.at[i, "evidence_out"] = v
                    break
            if df.at[i, "evidence_out"] == "none":
                v = keyword_fallback_span(row["post"])
                if v:
                    df.at[i, "evidence_out"] = v

    sub = pd.DataFrame({
        "row_id": df["row_id"],
        "risk_level": df["final_risk"],
        "evidence": df["evidence_out"],
        "factors": df["factors_out"],
    })
    sub.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")
    print(sub["risk_level"].value_counts())
    sub["_nf"] = sub["factors"].apply(lambda s: len(eval(s)))
    print(f"factors/post mean {sub['_nf'].mean():.2f} median {sub['_nf'].median()}")
    print(f"evidence 'none': {(sub['evidence']=='none').sum()}/{len(sub)}")
    n_llm_bert_disagree = (df["llm_risk"].notna() & (df["llm_risk"] != df["final_risk"])).sum()
    print(f"rows where final risk != LLM risk: {n_llm_bert_disagree}")


if __name__ == "__main__":
    main()
