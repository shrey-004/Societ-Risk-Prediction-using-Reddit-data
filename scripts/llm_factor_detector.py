"""
Dedicated detector for the WEAK factor categories (the ones dragging Subtask 2
macro-F1: sexual orientation, exposure to others' suicide, cognitive deficits,
physical health, meaning in life, stressful life event, + a few borderline).
The unified pass treats factors as a secondary task in a crowded prompt; here
each weak category gets its real annotation BREADTH (learned from all training
positives) plus 3-4 real few-shots and explicit hard-negative boundaries — the
missing calibration behind the under-recall (physical health, stressful event)
and over-firing (meaning in life, cognitive deficits) failure modes.

Per post it returns the subset of the weak categories that apply. Fused
per-category (validated on the targeted set) into the submission; never touches
the already-strong categories.

Usage:
    python scripts/llm_factor_detector.py --input <csv|xlsx> --output det.jsonl \
        [--batch_size 6] [--workers 2]
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.llm_unified_pass import call_claude, parse_json_response, truncate_post  # noqa: E402

# The weak categories this detector owns, each with its REAL annotation breadth
# (distilled from every training positive) + boundary. Order = output order.
CATEGORIES = {
    "physical health/characteristic":
        "ANY physical body issue: illness/disease, chronic pain, injury, COVID, feeling "
        "sick/nauseous/faint/dizzy, headaches/body aches, sleep-deprivation bodily effects, "
        "disability, AND physical-appearance concerns (ugly, hate my face, asymmetrical, "
        "skin, too fat/thin, height, weight gain). Tag generously — appearance complaints "
        "and bodily symptoms BOTH count. NOT: purely emotional pain with no bodily/appearance element.",
    "cognitive deficits":
        "Any stated difficulty with thinking: can't concentrate/focus, thoughts racing or "
        "circling uncontrollably, brain fog, 'can't think straight', 'losing my mind', "
        "'going insane', feeling 'dumber'/mentally slowed, memory problems, derealization/"
        "feeling disconnected from reality or body, can't write coherently, autistic/processing "
        "difficulty, med- or insomnia-induced mental fog. NOT: ordinary indecision ('idk what to do') "
        "or emotional distress without a thinking/cognition complaint.",
    "exposure to others' suicide":
        "ANOTHER person's suicide, suicide attempt, or suicidal thoughts is mentioned: a "
        "friend/relative/acquaintance who died by, attempted, or is thinking of suicide "
        "('my brother attempted', 'someone I know tried to commit'); celebrities' suicides; "
        "reading/watching about people who died by suicide; pro-suicide forums. NOT: only the "
        "author's OWN suicidality; the author imagining others reacting to THEIR death.",
    "sexual orientation related issues":
        "The author's own gender/sexual identity or sexual-function distress: LGBTQ+ identity "
        "struggles, being gay/lesbian/bi/trans/queer, gender dysphoria, coming out, same-sex "
        "relationship issues, 'gender issues', distressing sexual feelings/dysfunction, wanting "
        "castration. NOT: ordinary heterosexual relationship/romance problems.",
    "meaning in life":
        "Reflection on whether life is worth living / life's meaning / purpose / a reason to "
        "live — POSITIVE or NEGATIVE: 'what's there to live for', 'life has no meaning/point', "
        "'I miss when life had meaning', 'I actually wanna live', weighing life vs death as a "
        "worth question, living for someone's sake, 'destined to' / life-trajectory reflection, "
        "philosophical passages on existence. NOT: a bare 'I want to die' or generic hopelessness "
        "venting with no reflection on life's worth/purpose.",
    "stressful life event":
        "ANY concrete external stressor event/situation (challenging but not necessarily "
        "traumatic): breakup, being ghosted/rejected, job loss or overwork, unemployment, exams/"
        "deadlines/school pressure, moving, financial trouble, a relative's illness, legal issues, "
        "COVID disruption, argument/conflict. Tag generously whenever a concrete stressor is "
        "described — this is under-tagged. NOT: purely internal states with no external event.",
    "sense of responsibility":
        "Staying alive or hesitating BECAUSE of duty to others/oneself: 'can't do it to my "
        "family', 'a way to go and not hurt anyone', 'can't take away my parents' child', "
        "'my dog needs me', 'have things to take care of first', responsibility for one's own "
        "survival. NOT: guilt/self-blame without a protective duty framing.",
    "social support":
        "Someone caring IS present/available: family/friends/partner/therapist/nurse who listens, "
        "helps, checks in, or whom the author can talk to; 'people tell me I deserve help'; a "
        "supportive chat that helped. NOT: only LACK of support (that is a different category).",
    "psychological capital":
        "Any positive psychological resource, however faint: hope, a wish to live deep down "
        "('I actually wanna live', 'still alive for now'), gratitude, pride, interest in "
        "something (art, hobbies), a moment of feeling better, motivation, resilience, seeking "
        "to motivate oneself. NOT: pure despair with zero positive element.",
    "interpersonal difficulty":
        "Difficulty forming/maintaining social connections: can't make friends, social anxiety "
        "in social settings, awkwardness, trouble talking to or keeping people, feeling unable "
        "to connect. NOT: conflict with a specific person (that's other categories), or plain loneliness.",
}
CAT_LIST = list(CATEGORIES.keys())

# lexical priors: if a pattern hits, strongly consider the category (LLM still decides)
KEYWORD_PRIORS = {
    "sexual orientation related issues":
        r"\b(gay|lesbian|bisexual|\bbi\b|trans(?:gender)?|queer|lgbt|nonbinary|non-binary|dysphori|"
        r"come out|came out|closet|same[- ]sex|gender identity|gender issue|castrat|asexual)\b",
    "exposure to others' suicide":
        r"(someone i know|friend|brother|sister|mother|father|mom|dad|cousin|classmate|coworker|"
        r"celebrit|they|he|she) [^.]{0,40}(committed|attempted|died by|killed (?:him|her)self|"
        r"suicide|took (?:his|her) life)|others?'? suicide|people (?:who|that) (?:died|killed)",
    "physical health/characteristic":
        r"\b(sick|nause|puke|vomit|faint|dizzy|headache|migraine|chronic pain|ill\b|illness|covid|"
        r"disease|disabilit|ugly|my face|asymmetr|overweight|obese|underweight|too fat|too thin|"
        r"gained weight|my skin|my appearance|my body|my height)\b",
    "poor school performance": r"\b(failing|failed|flunk|bad grades|dropped out|drop out|exam.{0,15}fail|gpa)\b",
    "substance use": r"\b(drunk|alcohol|drinking|weed|marijuana|cocaine|heroin|meth|pills? to|high|"
                     r"drugs?|smoking|vap(?:e|ing)|overdos)\b",
    "low socio-economic status": r"\b(unemploy|jobless|homeless|poverty|can'?t afford|broke\b|no money|"
                                 r"in debt|bills|evict|fired from)\b",
}


def build_system_prompt(fewshots: dict) -> str:
    L = []
    A = L.append
    A("You are an expert annotator identifying psychological factors in r/SuicideWatch posts for a "
      "research dataset. For each post, decide which of the following categories are clearly supported. "
      "These are the SUBTLE/UNDER-recognized categories — read the definitions and examples carefully, "
      "because the exact annotation conventions are broader (or narrower) than they first appear. Tag a "
      "category when the post genuinely matches its definition; most posts will have only a few of these.")
    A("")
    A("CATEGORIES (with real breadth and boundaries):")
    for name in CAT_LIST:
        A(f'- "{name}": {CATEGORIES[name]}')
    A("")
    A("Real annotated examples (each shows a post that DOES carry the named category):")
    for name in CAT_LIST:
        for rid, txt in fewshots.get(name, [])[:4]:
            A(f'  [{name}] "{txt.strip()[:260]}"')
    A("")
    A("For each post you are also given HINTS: categories whose keywords appeared (they are CANDIDATES "
      "to consider, not automatic — confirm against the definition and reject false matches).")
    A("")
    A("Respond with ONLY a JSON object mapping each row_id to the list of applicable category names "
      "(exact strings above; empty list if none). No prose, no markdown fences.")
    return "\n".join(L)


def compute_hints(text: str) -> list:
    low = text.lower()
    return [cat for cat, pat in KEYWORD_PRIORS.items() if re.search(pat, low)]


def build_batch_prompt(rows) -> str:
    L = ["Posts to annotate:", ""]
    for rid, post in rows:
        hints = compute_hints(str(post))
        L.append(f"row_id: {rid}")
        if hints:
            L.append(f"hints (candidates to verify): {json.dumps(hints)}")
        L.append(f'post: "{truncate_post(post, 4000)}"')
        L.append("")
    L.append("ONLY the JSON object {row_id: [categories]}, every row_id present.")
    return "\n".join(L)


def process_batch(batch, system_prompt, model):
    prompt = system_prompt + "\n\n" + build_batch_prompt(batch)
    for attempt in range(3):
        try:
            raw = call_claude(prompt, model)
            parsed = parse_json_response(raw)
            out = {}
            for rid, post in batch:
                v = parsed.get(str(rid))
                out[str(rid)] = [c for c in v if c in CATEGORIES] if isinstance(v, list) else None
            return out
        except Exception as e:
            if attempt < 2:
                time.sleep(45 * (attempt + 1))
            else:
                print(f"  batch FAILED: {e}", flush=True)
                return {str(rid): None for rid, _ in batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    fewshots = json.load(open(ROOT / "data/processed/factor_detector_fewshots.json"))
    system_prompt = build_system_prompt(fewshots)
    print(f"detector system prompt: {len(system_prompt.split())} words", flush=True)

    if args.input.endswith(".xlsx"):
        df = pd.read_excel(args.input, sheet_name="Sheet1")
        sys.path.insert(0, str(ROOT))
        from src.data.clean import clean_text
        df["__post"] = df["post"].apply(clean_text)
    else:
        df = pd.read_csv(args.input)
        df["__post"] = df.get("post", df.get("post_clean")).astype(str)
    df["row_id"] = df["row_id"].astype(str)

    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["row_id"])
            except Exception:
                pass
    todo = df[~df["row_id"].isin(done)]
    print(f"{len(df)} rows, {len(done)} cached, {len(todo)} to run", flush=True)
    if not len(todo):
        return

    rows = list(zip(todo["row_id"], todo["__post"]))
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    lock = threading.Lock()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(out_path, "a") as fh:
        futs = {pool.submit(process_batch, b, system_prompt, args.model): b for b in batches}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                for rid, cats in res.items():
                    if cats is not None:
                        fh.write(json.dumps({"row_id": rid, "factors_weak": cats}) + "\n")
                fh.flush()
                n += 1
                print(f"  batch {n}/{len(batches)} written", flush=True)
    print(f"done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
