"""Draft a clip's labels with Gemini, for a native speaker to check.

    python scripts/prelabel_clip.py hi_01          # one clip
    python scripts/prelabel_clip.py --all          # every clip that has no labels yet
    python scripts/prelabel_clip.py hi_01 --force  # redo (never touches verified rows)

Writes the full draft to data/labels/<clip_id>.json and fills the empty CSV fields.
The row stays verified=no until a person who speaks the language has checked it.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.settings import config, secret  # noqa: E402
from scripts.testset import CLIPS, EMOTIONS, LABELS, read_rows, write_rows  # noqa: E402

LANGUAGE_NAMES = {k: v["name"] for k, v in config()["languages"].items()}

PROMPT = """You are labelling a short {language} film/TV clip for a speech research test set.
Watch and listen to the whole clip, then answer in JSON with exactly these fields:

{{
  "summary": "one sentence: what happens in the scene",
  "setting": "where it takes place",
  "has_music": true/false (background music or score under the dialogue),
  "speakers": [{{"id": "S1", "description": "who they are", "gender": "male/female", "approx_age": "20s"}}],
  "relationships": "how the speakers are related (siblings, friends, boss/employee...) and who is senior",
  "lines": [{{"speaker": "S1", "addressee": "S2", "start": 0.0, "end": 2.1,
             "transcript": "exact words in {script} script, as spoken (keep slang and code-mixing)",
             "emotion": one of {emotions},
             "intensity": "low/medium/high",
             "stressed_words": ["word(s) said with the most emphasis"]}}],
  "overall_emotion": one of {emotions},
  "register_note": "forms of address or honorifics used (e.g. bhai vs bhaiya, tu/tum/aap, anna, dada) and what they signal about the relationship; empty string if none",
  "expected_translation_notes": "how the register and emotion should carry into the other languages ({others}): which address terms fit, e.g. Tamil anna vs machan",
  "confidence": "high/medium/low, and why if not high"
}}

Times are in seconds from the start of the clip. If you cannot make out a word, write [?] rather than guessing."""


def call_gemini(video: bytes, prompt: str) -> dict:
    model = config()["llm"]["fallback_model"]
    body = {
        "contents": [{"parts": [{"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(video).decode()}},
                                {"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }
    for attempt in range(4):
        try:
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                           headers={"x-goog-api-key": secret("GEMINI_API_KEY")}, json=body, timeout=300)
        except httpx.TransportError:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            if "PerDay" in r.text:
                sys.exit("Gemini free daily quota used up; try again tomorrow")
            time.sleep(20 * (attempt + 1))
            continue
        r.raise_for_status()
        return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    sys.exit("Gemini kept failing (rate limit or network)")


def prelabel(row: dict) -> None:
    clip = CLIPS / f"{row['clip_id']}.mp4"
    if not clip.exists():
        print(f"{row['clip_id']}: {clip.name} not in data/clips (download it from the team Drive folder)")
        return
    video = clip.read_bytes()
    if len(video) > 18 * 1024 * 1024:
        print(f"{row['clip_id']}: file over 18 MB, too large to send inline; re-trim it shorter")
        return
    lang = row["language"]
    prompt = PROMPT.format(language=LANGUAGE_NAMES[lang], script=config()["languages"][lang]["script"],
                           emotions="/".join(EMOTIONS),
                           others=", ".join(v for k, v in LANGUAGE_NAMES.items() if k != lang))
    label = call_gemini(video, prompt)
    label["_status"] = "draft from Gemini; check every field, then set verified=yes in testset.csv"

    LABELS.mkdir(parents=True, exist_ok=True)
    (LABELS / f"{row['clip_id']}.json").write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = label.get("lines", [])
    filled = {
        "emotion": label.get("overall_emotion", ""),
        "has_music": "yes" if label.get("has_music") else "no",
        "num_speakers": str(len(label.get("speakers", [])) or ""),
        "register_note": label.get("register_note", ""),
        "transcript": " / ".join(f"{ln.get('speaker', '?')}: {ln.get('transcript', '')}" for ln in lines),
        "expected_translation_notes": label.get("expected_translation_notes", ""),
    }
    for k, v in filled.items():
        if not row.get(k):
            row[k] = v
    print(f"{row['clip_id']}: {row['emotion']}, {row['num_speakers']} speaker(s), music={row['has_music']}, "
          f"confidence {label.get('confidence', '?')}")
    for ln in lines:
        print(f"    {ln.get('speaker')} -> {ln.get('addressee')} [{ln.get('emotion')}]: {ln.get('transcript')}")
    if row["register_note"]:
        print(f"    register: {row['register_note']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip_ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="every clip without labels")
    ap.add_argument("--force", action="store_true", help="redo existing drafts (verified rows are never touched)")
    args = ap.parse_args()
    if not secret("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY missing in .env")

    rows = read_rows()
    by_id = {r["clip_id"]: r for r in rows}
    if args.all:
        targets = [r for r in rows if not (LABELS / f"{r['clip_id']}.json").exists()]
    else:
        missing = [c for c in args.clip_ids if c not in by_id]
        if missing:
            sys.exit(f"not in testset.csv: {', '.join(missing)}")
        targets = [by_id[c] for c in args.clip_ids]

    for row in targets:
        if row.get("verified") == "yes":
            print(f"{row['clip_id']}: already verified, skipped")
            continue
        if (LABELS / f"{row['clip_id']}.json").exists() and not args.force and not args.all:
            print(f"{row['clip_id']}: draft exists (use --force to redo)")
            continue
        if args.force:
            for k in ("emotion", "has_music", "num_speakers", "register_note", "transcript",
                      "expected_translation_notes"):
                row[k] = ""
        prelabel(row)
        write_rows(rows)  # save after each clip so a failure later keeps earlier work


if __name__ == "__main__":
    main()
