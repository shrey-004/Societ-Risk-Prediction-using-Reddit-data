"""
Unified LLM pass (risk + evidence + factors in ONE call per batch of posts),
via headless `claude -p` — extends the validated llm_factor_boost.py pattern
to all three subtasks with a rubric calibrated on the real training
annotations (see reports/llm_rubric_calibration notes):

  - risk levels follow the annotators' CURRENT-STANCE convention (recovery/
    advice posts with past attempts -> Indicator; future attempt -> Behavior;
    imminence upgrades vague intent to Behavior; negated/resisted urges ->
    Indicator), learned from reading real boundary cases, not just the
    official one-line definitions.
  - evidence spans mimic gold style: 1-3 short verbatim phrases (median 4
    words), [] for Indicator (96% of gold Indicator rows have none).
  - factors grounded in the verbatim Table III definitions + prevalence
    calibration (hopelessness 46% of posts ... sexual orientation 0.5%).

Results are cached per row_id in a JSONL (idempotent reruns), calls run
concurrently (default 4 workers), spans are verified verbatim against the
full post text (case-insensitive locate -> exact original casing kept).

Usage:
    python scripts/llm_unified_pass.py --input <csv|xlsx|parquet> --output out.jsonl \
        [--batch_size 8] [--workers 4] [--model MODEL] [--limit N] [--ids_file f.txt]

Input needs columns: row_id + (post | post_clean).
"""
import argparse
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RISK_LABELS = ["Indicator", "Ideation", "Behavior", "Attempt"]

TAXONOMY = json.load(open(ROOT / "data/processed/taxonomy_definitions.json"))
FEWSHOT = json.load(open(ROOT / "data/processed/llm_unified_fewshot.json"))
FACTOR_NAMES = list(TAXONOMY["risk_factors"].keys()) + list(TAXONOMY["protective_factors"].keys())

# calibration: share of the 1635 labeled posts carrying each factor
PREVALENCE = {
    "hopelessness": "46%", "emotion dysregulation": "33%", "poor social support": "31%",
    "low self-esteem": "29%", "coping strategy": "27%", "suicide means (with access)": "18%",
    "psychological capital": "15%", "mental health issues": "14%",
    "prior self-harm or suicidal thought/attempt": "13%", "interpersonal difficulty": "12%",
    "stressful life event": "11%", "dysfunctional family": "7%", "social support": "7%",
    "interpersonal violence": "5%", "physical health/characteristic": "5%",
    "traumatic experience": "4%", "sense of responsibility": "3.5%",
    "low socio-economic status": "3.3%", "meaning in life": "2.8%",
    "cognitive deficits": "2%", "substance use": "2%", "poor school performance": "1%",
    "exposure to others' suicide": "0.9%", "sexual orientation related issues": "0.5%",
}

FACTOR_HINTS = {
    "mental health issues": "only when a disorder/diagnosis/therapy/medication is explicitly mentioned (depression, BPD, PTSD, anxiety disorder, meds) — NOT mere sadness",
    "hopelessness": "'nothing will change', 'no way out', 'trapped', 'tired of life', 'what's the point' — very common, tag whenever present",
    "emotion dysregulation": "tag LIBERALLY: any crying, anger, rage, panic, anxiety spikes, overwhelm, mood swings, 'I can't handle this', emotional venting/outbursts (CAPS-LOCK rants count)",
    "low self-esteem": "worthless, burden, hate myself, ugly, failure, 'I'm useless'",
    "coping strategy": "tag VERY LIBERALLY — annotators count ANY attempt to manage or reach out: asking someone to talk/pray, posting for help ('anyone to talk?', 'i need help'), therapy, meds, writing it out, distraction, hobbies, 'trying to fix it', giving themselves time, holding on for now",
    "psychological capital": "tag VERY LIBERALLY — ANY glimmer of positivity/hope/will to live, however faint: 'still alive for now', 'I don't want to die', 'hoping to find something worth it', 'help me' (wanting to get better), remembering better times, small wins",
    "suicide means (with access)": "any description of potential means or access to them (owning pills/gun/rope, bridge nearby) — counts even when access is blocked ('they took my pills away')",
    "prior self-harm or suicidal thought/attempt": "PAST/previous episodes: 'used to cut', 'attempted last year', 'my last attempt', 'been suicidal since 14' — distinct from the current state",
    "poor social support": "lonely, no friends, nobody cares, isolated, rejected, abandoned",
    "interpersonal difficulty": "difficulty making friends / socializing / talking to people (social anxiety in social contexts)",
    "interpersonal violence": "bullying, assault, abuse occurring OUTSIDE the home/family",
    "dysfunctional family": "family conflict, abusive/neglectful parents, divorce affecting the author",
    "stressful life event": "tag LIBERALLY: ANY concrete stressor event/situation mentioned — breakup, job loss/overwork, exams, deadlines, moving, relative's illness, covid disruptions, financial pressure",
    "traumatic experience": "abuse, rape, molestation, violent events, death of a loved one — overwhelming events",
    "physical health/characteristic": "illness, chronic pain, disability, obesity/underweight, appearance/height complaints, COVID",
    "low socio-economic status": "unemployment, poverty, homelessness, can't afford things, debt",
    "substance use": "author's own drug/alcohol/tobacco use (uncontrolled or as escape)",
    "poor school performance": "failing tests, bad grades, flunking out",
    "exposure to others' suicide": "someone else's suicide/attempt/suicidal thoughts is mentioned or described",
    "cognitive deficits": "tag LIBERALLY: can't concentrate/focus, racing or circling thoughts, 'going insane', confusion ('idk anymore'), sleep-deprived broken thinking, incoherence, memory problems, med-fogged mind",
    "sexual orientation related issues": "LGBTQ+ identity struggles, gender dysphoria, same-sex relationship issues",
    "social support": "tag when ANYONE caring is mentioned: friends/family/partner/therapist who listens, helps or tells them to get help — even 'people tell me I deserve help'",
    "sense of responsibility": "staying alive / hesitating for others or duties: 'can't do it to my family', 'my dog needs me', 'have things to take care of'",
    "meaning in life": "any reflection on whether life is worth living, reasons to live, life's purpose — positive OR negative: 'find something worth living for', 'the choice between life and death', 'nothing to live for', living for someone's sake, philosophy about existence",
}


def build_system_prompt() -> str:
    L = []
    A = L.append
    A("You are an expert clinical annotator for a suicide-risk research dataset built from r/SuicideWatch posts, "
      "annotated per the Columbia Suicide Severity Rating Scale (C-SSRS). You label each post with (1) a risk level, "
      "(2) verbatim evidence spans, and (3) risk/protective factors. Your labels must reproduce THIS dataset's "
      "annotation conventions, described below with real examples — follow them exactly, even where you might "
      "personally judge differently.")
    A("")
    A("## 1. RISK LEVEL — pick exactly one, using this decision cascade (check in order):")
    A("")
    A("**Attempt** — the author describes having actually INITIATED a suicide attempt (recent or long ago): "
      "'attempted suicide', 'tried to kill myself', overdosed on purpose, tried to hang/jump, or an aborted-but-started "
      "act ('tied it around my neck', 'almost hanged myself today'). A past attempt plus current plan ('planning to "
      "overdose again', \"tonight I'll attempt again\") is still Attempt — the word 'again', or a plan that was "
      "'foiled'/'interrupted'/'failed', implies a prior attempt and makes the post Attempt. ALSO Attempt: severe, "
      "potentially life-threatening self-harm ACTS the author performed — cutting the neck/throat/wrist arteries, "
      "'50 cuts'/'100 cuts', heavy bleeding described — even without explicit suicide words ('I relapsed... cut up my "
      "chest... 50 cuts' is Attempt in this dataset). NOT Attempt: first-time future/planned attempts ('I will attempt "
      "today' with no prior attempt -> Behavior); 'almost' cases where no act was started (-> Behavior); other "
      "people's attempts; posts whose overall stance is recovery/advice-to-others (-> Indicator, see below).")
    A("")
    A("**Behavior** — no actual attempt narrated, but explicit suicidality PLUS any concrete element: a method "
      "contemplated for oneself ('thought about downing Tylenol', 'want to jump off this building'), means/access "
      "('I have a rope', 'sitting near my blades'), preparations ('wrote my suicide note', 'settled on a date'), "
      "timing/imminence ('tonight', 'tomorrow', 'this month', a countdown, 'I will kill myself NOW'), asking how to "
      "kill oneself, or ordinary deliberate self-harm ACTS by the author (cutting arms, burning, 'cut myself everyday' "
      "— acts, not mere urges; severe/life-threatening acts are Attempt instead). Imminence with a concrete time "
      "upgrades vague intent to Behavior even with no method named — but a vague 'soon'/'someday' does NOT "
      "('planning to kill myself pretty soon' with nothing concrete is Ideation).")
    A("")
    A("**Ideation** — explicit first-person suicidal thoughts/wishes WITHOUT method/plan/timing/self-harm: 'I want to "
      "die', 'kill myself' (desire only), 'suicidal thoughts', passive death wishes ('wish I could sleep forever', "
      "'can't wait for death', \"I'd rather be dead\", 'want out', 'wish something could just kill me'), vague intent "
      "without specifics ('I'm going to do it' with no method/time). Wanting to be killed by external means without "
      "one's own plan is Ideation. ALSO Ideation (not Behavior): idiomatic method-wishes without real engagement "
      "('wanna blow my brains out' while 'too scared to kill myself'); imagining/fantasizing a method ('fell asleep "
      "imagining bleeding out'); conditional futures ('if X doesn't change I will end it'); wistful past near-misses "
      "(\"wish I'd finished myself off back then\"); social withdrawal ('cutting everyone off', 'leaving everyone "
      "behind'); and bare one-line 'I'm about to commit suicide' with no other content. Passing self-harm temptation "
      "inside a broader ideation post ('tempted to cut myself' amid suicidal thoughts) stays Ideation.")
    A("")
    A("**Indicator** — none of the above applies. Emotional distress, depression, self-hate, hopelessness WITHOUT any "
      "explicit suicidal expression by the author. ALSO Indicator by convention: negated/denied suicidality "
      "(\"I don't want to die\", 'not suicidal'); urges that are resisted/overcome ('I'm glad I didn't', 'still trying "
      "to fight those urges', 'X cures my suicidal thoughts'); beliefs about deserving death without desire "
      "('I deserve to be dead' as self-worth statement); third-person or hypothetical mentions, quotes, statistics; "
      "and RECOVERY/ADVICE posts addressed to others — even when they recount the author's own past attempts in "
      "detail, if the current stance is supportive/recovered ('That feeling has kept me from trying ever since', "
      "\"Let's hold on together\"). CRITICAL: euphemisms with NO explicit death/suicide/kill word are Indicator — "
      "'please let me go', 'it will all be over soon', 'easier to let go', 'it has to happen, I have no other choice', "
      "'giving in to the urges', a bare 'Well Bye' — the dataset requires an EXPLICIT mention to leave Indicator. "
      "But do NOT use Indicator for poems/lyrics narrating the author's own suicidality in third person ('she took "
      "out her razor, wrote her goodbyes' about oneself -> treat as the author's own, usually Behavior), nor for "
      "standalone explicit self-harm desire posts ('I want to harm myself but I can't' -> Behavior), nor for "
      "sarcastic method references ('I must like getting tortured, because I'm not leaping out a window' -> Behavior).")
    A("")
    A("The dataset's real distribution: Indicator 37.5%, Ideation 31.5%, Behavior 24%, Attempt 7%.")
    A("")
    A("## 2. EVIDENCE SPANS")
    A("- For Ideation/Behavior/Attempt: extract 1-3 short spans (typically 2-8 words, median 4; a 4th only when the "
      "post clearly carries that many DISTINCT signals) copied VERBATIM, character-for-character, from the post — the "
      "decisive phrases expressing the suicidality/plan/attempt that justify your risk level. Each span must be an "
      "exact substring of the post. Pick distinct signals — never two spans saying the same thing; prefer the phrase "
      "carrying the signal over the whole sentence. Order by strength.")
    A("- For Indicator: output an empty list [] (this dataset annotates 'none' for 96% of Indicator posts).")
    A("")
    A("## 3. FACTORS (multi-label; risk factors AND protective factors)")
    A("Tag every category supported by the text. Typical posts carry 2-5 factors; very short posts 0-2. IMPORTANT: "
      "this dataset tags several categories much more liberally than you would guess (marked 'LIBERALLY' below) — "
      "e.g. merely posting 'anyone to talk?' counts as coping strategy, and 'still alive for now' counts as "
      "psychological capital. '(prevalence)' below = share of real posts carrying that factor — calibrate to it.")
    for name in FACTOR_NAMES:
        defn = TAXONOMY["risk_factors"].get(name) or TAXONOMY["protective_factors"].get(name)
        kind = "protective" if name in TAXONOMY["protective_factors"] else "risk"
        A(f"- \"{name}\" ({kind}, {PREVALENCE[name]}): {defn} Hint: {FACTOR_HINTS[name]}.")
    A("")
    A("## 4. EXAMPLES (real annotated posts from this dataset)")
    for ex in FEWSHOT:
        A(f"POST: \"{ex['post']}\"")
        A("ANNOTATION: " + json.dumps(
            {"risk": ex["risk"], "evidence": ex["evidence"], "factors": ex["factors"]},
            ensure_ascii=False))
        A("")
    A("## 5. OUTPUT FORMAT")
    A("You will receive a numbered batch of posts, each with a row_id. Respond with ONLY a JSON object mapping every "
      "row_id to {\"risk\": <one of Indicator|Ideation|Behavior|Attempt>, \"evidence\": [spans], "
      "\"factors\": [category names], \"confidence\": <\"high\"|\"medium\"|\"low\" — your confidence in the risk level>}. "
      "Use the exact category strings. No prose, no markdown fences, no explanations.")
    return "\n".join(L)


def truncate_post(text: str, max_chars: int = 6000) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars]


def build_batch_prompt(rows) -> str:
    L = ["Posts to annotate:", ""]
    for rid, post in rows:
        L.append(f"row_id: {rid}")
        L.append(f'post: "{truncate_post(post)}"')
        L.append("")
    L.append("Remember: ONLY the JSON object, keyed by row_id, every row_id present.")
    return "\n".join(L)


def call_claude(prompt: str, model: str | None, timeout: int = 600) -> str:
    cmd = ["claude", "-p", "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                            timeout=timeout, cwd="/tmp")
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={result.returncode}): "
                           f"stderr={result.stderr[:300]!r} stdout={result.stdout[:300]!r}")
    return result.stdout


def parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob)  # trailing commas
        return json.loads(blob2)


def verify_spans(spans, full_post: str, max_words: int = 12, max_spans: int = 4):
    """Keep spans verbatim-locatable in the full post (case-insensitive find,
    return the post's exact casing). Drop dupes/containments and over-long spans."""
    low = full_post.lower()
    out = []
    for s in spans or []:
        s = str(s).strip().strip('"').strip()
        if not s or len(s.split()) > max_words:
            continue
        i = low.find(s.lower())
        if i < 0:
            s2 = re.sub(r"\s+", " ", s)
            i = low.find(s2.lower())
            if i < 0:
                continue
            s = s2
        exact = full_post[i:i + len(s)]
        if any(exact.lower() in o.lower() or o.lower() in exact.lower() for o in out):
            continue
        out.append(exact)
        if len(out) >= max_spans:
            break
    return out


def sanitize(rec: dict, full_post: str) -> dict | None:
    risk = str(rec.get("risk", "")).strip().capitalize()
    if risk not in RISK_LABELS:
        return None
    return {
        "risk": risk,
        "evidence": verify_spans(rec.get("evidence"), full_post),
        "factors": sorted({f for f in (rec.get("factors") or []) if f in FACTOR_NAMES}),
        "confidence": rec.get("confidence", "medium"),
    }


def process_batch(batch, system_prompt, model):
    """batch: list of (row_id, post). Returns dict row_id -> sanitized rec (or None)."""
    prompt = system_prompt + "\n\n" + build_batch_prompt(batch)
    out = {}
    for attempt in range(3):
        try:
            raw = call_claude(prompt, model)
            parsed = parse_json_response(raw)
            for rid, post in batch:
                rec = parsed.get(str(rid))
                out[str(rid)] = sanitize(rec, str(post)) if isinstance(rec, dict) else None
            missing = [r for r, v in out.items() if v is None]
            if not missing:
                return out
            if attempt < 2:
                time.sleep(5)
                continue
            return out
        except Exception as e:
            if attempt < 2:
                time.sleep(45 * (attempt + 1))
            else:
                print(f"  batch FAILED after retries: {e}", flush=True)
                return {str(rid): None for rid, _ in batch}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids_file", type=str, default="", help="only process these row_ids (one per line)")
    args = p.parse_args()

    if args.input.endswith(".xlsx"):
        df = pd.read_excel(args.input, sheet_name="Sheet1")
        df["__post"] = df["post"].astype(str)
    elif args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
        df["__post"] = df["post_clean"].astype(str)
    else:
        df = pd.read_csv(args.input)
        df["__post"] = df.get("post", df.get("post_clean")).astype(str)

    if args.ids_file:
        keep = {l.strip() for l in open(args.ids_file) if l.strip()}
        df = df[df["row_id"].astype(str).isin(keep)]
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["row_id"])
            except Exception:
                pass
    todo = df[~df["row_id"].astype(str).isin(done)]
    print(f"{len(df)} rows requested, {len(done)} cached, {len(todo)} to run", flush=True)
    if not len(todo):
        return

    system_prompt = build_system_prompt()
    print(f"system prompt: {len(system_prompt.split())} words", flush=True)

    rows = list(zip(todo["row_id"].astype(str), todo["__post"]))
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    lock = threading.Lock()
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(out_path, "a") as fh:
        futs = {pool.submit(process_batch, b, system_prompt, args.model): b for b in batches}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                for rid, rec in res.items():
                    if rec is not None:
                        fh.write(json.dumps({"row_id": rid, **rec}, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                print(f"  batch {n_done}/{len(batches)} written", flush=True)

    n_ok = sum(1 for _ in open(out_path))
    print(f"done. {n_ok} rows in {out_path}", flush=True)


if __name__ == "__main__":
    main()
