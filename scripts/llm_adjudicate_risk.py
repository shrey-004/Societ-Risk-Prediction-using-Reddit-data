"""
Risk adjudication pass — second, boundary-focused LLM read for rows where
the unified pass reported medium/low confidence. Candidates are the unified
pass's label plus its ORDINAL NEIGHBORS (the confusion matrix shows errors
are overwhelmingly adjacent-class); the adjudicator picks among them without
being told which was originally chosen (avoids anchoring).

Usage:
    python scripts/llm_adjudicate_risk.py --unified out.jsonl --posts <csv|xlsx> \
        --output adj.jsonl [--batch_size 6] [--workers 2]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.llm_unified_pass import (  # noqa: E402
    RISK_LABELS, FEWSHOT, call_claude, parse_json_response, truncate_post,
)
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

NEIGHBORS = {
    "Indicator": ["Indicator", "Ideation"],
    "Ideation": ["Indicator", "Ideation", "Behavior"],
    "Behavior": ["Ideation", "Behavior", "Attempt"],
    "Attempt": ["Behavior", "Attempt"],
}

BOUNDARY_FEWSHOT_IDS = ["P01454", "P01108", "P00963", "P00908", "P02412", "P00638", "P02249", "P01890"]


def build_system_prompt() -> str:
    L = []
    A = L.append
    A("You are an expert clinical annotator (C-SSRS conventions) re-examining hard suicide-risk classification "
      "cases from r/SuicideWatch. For each post you get a short list of CANDIDATE risk levels — exactly one is "
      "correct. Re-read the post slowly and decide, using this dataset's exact conventions:")
    A("")
    A("- Attempt: the author actually INITIATED an act (past or recent): 'attempted', 'tried to hang myself', "
      "purposeful overdose, aborted-but-started acts ('tied it around my neck'), a plan that was 'foiled', "
      "'attempt again' (implies a prior attempt), or severe self-harm acts (neck/wrist cuts, '50 cuts', heavy "
      "bleeding). Recovery/advice-framed posts describing long-past attempts from a healed stance are Indicator "
      "instead.")
    A("- Behavior: no initiated act, but explicit suicidality PLUS concreteness: a method contemplated for oneself, "
      "means/access, preparations (note written, date set), imminence ('tonight', 'now', 'this month'), asking for "
      "methods, ordinary self-harm ACTS (cutting), standalone 'I want to harm myself' posts, or poetic third-person "
      "narration of one's own suicidality/means.")
    A("- Ideation: explicit first-person suicidal desire/thoughts WITHOUT any concrete element: 'want to die', "
      "'kill myself' (desire), passive wishes ('rather be dead', 'sleep forever'), idiomatic method-wishes "
      "(\"wanna blow my brains out\" while 'too scared'), imagining/fantasy, conditional futures ('if X, I'll end "
      "it'), vague 'soon'. One-line 'I'm about to commit suicide' with nothing else is Ideation.")
    A("- Indicator: NO current explicit suicidal expression: distress only; euphemisms ('let me go', 'it'll all be "
      "over soon', 'Well Bye', 'giving in to the urges'); negated ('I don't want to die'); resisted/overcome urges "
      "('glad I didn't'); deserving-death-as-self-worth; third-person/quotes; recovery/advice posts (even with own "
      "past attempts recounted).")
    A("")
    A("Decision boundaries to check IN ORDER: (1) any initiated act, 'again', or foiled attempt -> Attempt. "
      "(2) else any method/means/timing/preparation/self-harm act -> Behavior. (3) else any explicit first-person "
      "suicidal expression -> Ideation. (4) else -> Indicator.")
    A("")
    A("Calibration examples from the real dataset:")
    fs = {e["row_id"]: e for e in FEWSHOT}
    for rid in BOUNDARY_FEWSHOT_IDS:
        if rid in fs:
            e = fs[rid]
            A(f"POST: \"{e['post'][:700]}\"")
            A(f"LABEL: {e['risk']}")
            A("")
    A("Respond with ONLY a JSON object mapping each row_id to its label string (one of the candidates given for "
      "that post). No prose, no fences.")
    return "\n".join(L)


def build_batch_prompt(rows) -> str:
    L = ["Cases to decide:", ""]
    for rid, post, cands in rows:
        L.append(f"row_id: {rid}")
        L.append(f"candidates: {cands}")
        L.append(f'post: "{truncate_post(post)}"')
        L.append("")
    L.append("ONLY the JSON object {row_id: label}, every row_id present.")
    return "\n".join(L)


def process_batch(batch, system_prompt, model):
    prompt = system_prompt + "\n\n" + build_batch_prompt(batch)
    for attempt in range(3):
        try:
            raw = call_claude(prompt, model)
            parsed = parse_json_response(raw)
            out = {}
            for rid, post, cands in batch:
                v = parsed.get(str(rid))
                out[str(rid)] = v if isinstance(v, str) and v in cands else None
            return out
        except Exception as e:
            if attempt < 2:
                time.sleep(45 * (attempt + 1))
            else:
                print(f"  batch FAILED: {e}", flush=True)
                return {str(rid): None for rid, _, _ in batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True, help="jsonl from llm_unified_pass.py")
    ap.add_argument("--posts", required=True, help="csv/xlsx with row_id + post/post_clean")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    uni = {}
    for line in open(args.unified):
        r = json.loads(line)
        uni[r["row_id"]] = r

    if args.posts.endswith(".xlsx"):
        df = pd.read_excel(args.posts, sheet_name="Sheet1")
        df["__post"] = df["post"].astype(str)
    else:
        df = pd.read_csv(args.posts)
        df["__post"] = df.get("post", df.get("post_clean")).astype(str)
    df["row_id"] = df["row_id"].astype(str)

    todo = []
    for _, row in df.iterrows():
        rec = uni.get(row["row_id"])
        if rec and rec.get("confidence") != "high":
            todo.append((row["row_id"], row["__post"], NEIGHBORS[rec["risk"]]))

    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["row_id"])
            except Exception:
                pass
    todo = [t for t in todo if t[0] not in done]
    print(f"{len(todo)} unsure rows to adjudicate ({len(done)} cached)", flush=True)
    if not todo:
        return

    system_prompt = build_system_prompt()
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    lock = threading.Lock()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(out_path, "a") as fh:
        futs = {pool.submit(process_batch, b, system_prompt, args.model): b for b in batches}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                for rid, lab in res.items():
                    if lab is not None:
                        fh.write(json.dumps({"row_id": rid, "risk_adj": lab}) + "\n")
                fh.flush()
                n += 1
                print(f"  batch {n}/{len(batches)} written", flush=True)
    print(f"done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
