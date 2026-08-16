"""
Precision-verification pass for factor categories where the unified pass
over-fires (pooled-560 precision 0.06-0.19): meaning in life, cognitive
deficits, exposure to others' suicide, sexual orientation related issues.

Only posts where the unified pass ALREADY fired one of these tags are
re-examined (cheap); the verifier applies a STRICT reading with positive
AND negative gold examples, and each tag is kept only if confirmed.
Union-safety does not apply here — this pass can only REMOVE tags, which
is exactly the point (precision repair). Validated on the pooled 560-row
sample before use on test.

Usage:
    python scripts/llm_verify_factors.py --unified in.jsonl --posts <csv|xlsx> --output verified.jsonl
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.llm_unified_pass import call_claude, parse_json_response, truncate_post  # noqa: E402

TARGET = [
    "meaning in life",
    "cognitive deficits",
    "exposure to others' suicide",
    "sexual orientation related issues",
]

GUIDE = """You verify factor tags on r/SuicideWatch posts for a research dataset. A previous pass OVER-TAGS the
four categories below; your job is to KEEP a tag only when the post clearly matches the dataset's strict usage,
and DROP it otherwise. Calibration: in the real data these are rare — "meaning in life" 2.8% of posts,
"cognitive deficits" 2%, "exposure to others' suicide" 0.9%, "sexual orientation related issues" 0.5%.

"meaning in life" — KEEP only when the post genuinely REFLECTS ON life's worth/meaning/reasons to live as a
theme: weighing "the choice between life and death", "hoping to find something or someone worth living for",
living only for someone's sake, philosophical passages about whether existence has a point. DROP when the post
merely expresses misery, hopelessness, "what's the point" as a throwaway venting line, or wanting to die —
those are hopelessness, NOT meaning in life.
  KEEP example: "probably most of us here still and hoping to find something or someone that really worth living for"
  KEEP example: "as i feel like i'm nearing my end, i realized the choice between life and death..."
  DROP example: "everything is pointless, i'm so tired of everything" (hopelessness only)
  DROP example: "there is no reason. I want to die" (ideation + hopelessness only)

"cognitive deficits" — KEEP only for a concrete stated difficulty with thinking/concentration/memory/coherence:
"can't bring myself to concentrate", "thoughts keep circling and I can't think", "can't even write a coherent
post", "my meds make me pass out/foggy", "am I going insane?", broken sleep explicitly damaging the mind. DROP
for ordinary indecision ("idk what to do"), emotional confusion, "I don't know anymore" as venting, or plain
insomnia without a stated effect on thinking.
  KEEP example: "I have a paper due soon but I cant bring myself to concentrate on it. The thoughts keep circling"
  DROP example: "i don't know what to do anymore" (venting, not a cognitive deficit)

"exposure to others' suicide" — KEEP only when ANOTHER PERSON's suicide, suicide attempt, or suicidal thoughts
are mentioned or described (friend/family/acquaintance/celebrity). DROP when the only suicidality in the post is
the author's own, including the author imagining others' reactions to their death.
  KEEP example: "my best friend killed himself last spring"
  DROP example: "The people who find my body will be sad" (author's own death imagined)

"sexual orientation related issues" — KEEP only for the author's own LGBTQ+/gender identity struggles (coming
out, dysphoria, same-sex relationship issues, being rejected for orientation). DROP for ordinary romantic
problems, incel-style complaints, or mentions of others' orientation without it being the author's issue.

Respond with ONLY a JSON object mapping each row_id to the list of tags TO KEEP (subset of that row's candidate
tags; empty list if none survive). No prose, no fences."""


def build_batch_prompt(rows):
    L = ["Cases (each lists its candidate tags to verify):", ""]
    for rid, post, tags in rows:
        L.append(f"row_id: {rid}")
        L.append(f"candidate tags: {json.dumps(tags)}")
        L.append(f'post: "{truncate_post(post, 4000)}"')
        L.append("")
    L.append("ONLY the JSON object {row_id: [kept tags]}, every row_id present.")
    return "\n".join(L)


def process_batch(batch, model):
    prompt = GUIDE + "\n\n" + build_batch_prompt(batch)
    for attempt in range(3):
        try:
            raw = call_claude(prompt, model)
            parsed = parse_json_response(raw)
            out = {}
            for rid, post, tags in batch:
                v = parsed.get(str(rid))
                if isinstance(v, list):
                    out[str(rid)] = [t for t in v if t in tags]
                else:
                    out[str(rid)] = None
            return out
        except Exception as e:
            if attempt < 2:
                time.sleep(45 * (attempt + 1))
            else:
                print(f"  batch FAILED: {e}", flush=True)
                return {str(rid): None for rid, _, _ in batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True)
    ap.add_argument("--posts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=10)
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
        if not rec:
            continue
        fired = [t for t in TARGET if t in rec.get("factors", [])]
        if fired:
            todo.append((row["row_id"], row["__post"], fired))

    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["row_id"])
            except Exception:
                pass
    todo = [t for t in todo if t[0] not in done]
    print(f"{len(todo)} rows with target-category fires to verify ({len(done)} cached)", flush=True)
    if not todo:
        return

    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    lock = threading.Lock()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(out_path, "a") as fh:
        futs = {pool.submit(process_batch, b, args.model): b for b in batches}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                for rid, kept in res.items():
                    if kept is not None:
                        fh.write(json.dumps({"row_id": rid, "kept": kept}) + "\n")
                fh.flush()
                n += 1
                print(f"  batch {n}/{len(batches)} written", flush=True)
    print(f"done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
