"""Redo step 4's scene context and translation for clips already processed, on any machine.

    python scripts/retranslate.py                    # every data/cache/dub/*.json
    python scripts/retranslate.py ta_01.ta-hi        # one
    python scripts/retranslate.py ta_02.ta-hi --to bn mr te   # dub an existing scene into more languages

Reuses each file's step 3 lines, video captions and IndicTrans2 drafts, so no GPU is needed; only
the context and translation calls go to the LLM (main model only: see llm.translate_primary_only).
The previous file is kept as data/cache/dub/before/<name>.json, and a before/after table is printed.
Then rebuild the game library: python scripts/build_library.py
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.llm.client import LLMError  # noqa: E402
from backend.pipeline.dub import draft_engine, process_clip  # noqa: E402
from backend.pipeline.translate.copies import copied_words  # noqa: E402
from backend.pipeline.translate.kinship import kinship_issues  # noqa: E402
from backend.pipeline.translate.written import written_words  # noqa: E402

DUB = ROOT / "data/cache/dub"
ADDED = ("addressee", "emotion_intent", "draft", "dub")   # fields step 4 adds to step 3's lines


def retranslate(path: Path, to: str | None = None) -> None:
    """Redo `path`'s translation; with `to`, translate its lines into that language instead and
    write a new <clip>.<src>-<to>.json (step 3's analysis is language-independent of the target)."""
    old = json.loads(path.read_text(encoding="utf-8"))
    lines = [{k: v for k, v in ln.items() if k not in ADDED} for ln in old["lines"]]
    # IndicTrans2 drafts are reused (no model to load); Sarvam drafts are made fresh (and cached)
    drafts = [ln.get("draft") or "" for ln in old["lines"]] if draft_engine() == "indictrans2" else None
    clip = ROOT / "data/clips" / f"{path.name.split('.')[0]}.mp4"
    src = old["source_language"]
    if to:
        drafts = None                        # the stored drafts are in the old target language
        path = DUB / f"{path.name.split('.')[0]}.{src}-{to}.json"
    new = process_clip(str(clip), lines, src, to or old["target_language"], captions=old.get("captions"), drafts=drafts)
    if not to:
        keep = DUB / "before" / path.name
        keep.parent.mkdir(exist_ok=True)
        if not keep.exists():
            shutil.copy(path, keep)          # the first "before" is the one worth comparing against
    path.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n######## {path.stem}   context: {new['context'].get('_provider')}")
    for s, info in (new["context"].get("speakers") or {}).items():
        print(f"  {s}: {info.get('description')}")
    for o, n in zip(old["lines"], new["lines"]):
        d = n["dub"]
        print(f"\n  {n['text']}")
        print(f"    meaning: {d.get('meaning')}")
        print(f"    before : {o['dub']['translation']}")
        print(f"    after  : {d['translation']}   ({d.get('romanized')})")
        if d.get("reviewed_from"):
            print(f"    (review fixed: {d.get('review_problem')})")
        if d.get("copies_fixed_from"):
            print(f"    (copy fix from: {d['copies_fixed_from']})")
        if d.get("copied_words"):
            print(f"    kept source words: {', '.join(d['copied_words'])}")
        if d.get("written_words"):
            print(f"    still written-style: {', '.join(d['written_words'])}")

    src, tgt = new["source_language"], new["target_language"]
    def rate(doc, check):
        return sum(bool(check(ln, ln["dub"]["translation"])) for ln in doc["lines"]), len(doc["lines"])
    for name, check in [("written-style words", lambda ln, t: written_words(t, tgt)),
                        ("copied source words", lambda ln, t: copied_words(ln["text"], t, src, tgt)),
                        ("wrong family words", lambda ln, t: kinship_issues(ln["text"], t, src, tgt))]:
        (b, n), (a, _) = rate(old, check), rate(new, check)
        print(f"  lines with {name}: before {b}/{n}, after {a}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="dub file names without .json (default: all)")
    ap.add_argument("--to", nargs="+", help="new target languages for the named scenes")
    args = ap.parse_args()
    paths = [DUB / f"{n}.json" for n in args.names] if args.names else sorted(DUB.glob("*.json"))
    for p in paths:
        for to in args.to or [None]:
            try:
                retranslate(p, to)
            except LLMError as e:
                sys.exit(f"{p.stem} -> {to or 'same'}: {e}")


if __name__ == "__main__":
    main()
