"""Run step 4 on a clip and print the delivery script.

    # with step 3's output (python -m backend.pipeline.speech.run data/clips/hi_01.mp4 --lang hi
    # writes data/cache/speech/hi_01.json; "auto" picks that file up)
    python scripts/dub_clip.py data/clips/hi_01.mp4 --speech auto --to ta
    python scripts/dub_clip.py data/clips/hi_01.mp4 --speech path/to/speech.json --to ta

    # without step 3's output: the clip's Gemini pre-label stands in for the speech JSON
    python scripts/dub_clip.py data/clips/hi_01.mp4 --stub --to ta

Writes data/cache/dub/<clip>.<src>-<tgt>.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.dub import lines_from_label, process_clip  # noqa: E402
from backend.settings import config  # noqa: E402
from scripts.testset import LABELS, read_rows  # noqa: E402

SPEECH = ROOT / "data/cache/speech"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--to", required=True, choices=list(config()["languages"]), help="target language")
    ap.add_argument("--from", dest="src", choices=list(config()["languages"]),
                    help="source language (default: from testset.csv)")
    ap.add_argument("--speech", help='step 3 speech JSON for this clip, or "auto" for data/cache/speech/<clip>.json')
    ap.add_argument("--stub", action="store_true", help="use data/labels/<clip>.json as the speech input")
    args = ap.parse_args()

    clip = Path(args.clip)
    row = next((r for r in read_rows() if r["clip_id"] == clip.stem), None)
    if args.speech:
        path = SPEECH / f"{clip.stem}.json" if args.speech == "auto" else Path(args.speech)
        if not path.exists():
            sys.exit(f"{path} missing: run python -m backend.pipeline.speech.run {clip} --lang <lang> first")
        speech = json.loads(path.read_text(encoding="utf-8"))
        lines = speech["lines"] if isinstance(speech, dict) else speech
        meta_lang = speech.get("meta", {}).get("language") if isinstance(speech, dict) else None
        src = args.src or meta_lang or (row and row["language"])
    elif args.stub:
        src = args.src or (row and row["language"])
        label = LABELS / f"{clip.stem}.json"
        if not label.exists():
            sys.exit(f"{label} missing: run scripts/prelabel_clip.py {clip.stem} first")
        lines = lines_from_label(label, src) if src else []
    else:
        sys.exit("pass --speech <json|auto> or --stub")
    if not src:
        sys.exit("source language unknown: pass --from")
    if not lines:
        sys.exit("no spoken lines in the input")

    result = process_clip(str(clip), lines, src, args.to)
    out = ROOT / f"data/cache/dub/{clip.stem}.{src}-{args.to}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    ctx = result["context"]
    print(f"Scene: {ctx.get('summary')}  [{ctx.get('mood')}]")
    for s, info in (ctx.get("speakers") or {}).items():
        print(f"  {s}: {info.get('description')}")
    for rel in ctx.get("relationships") or []:
        print(f"  {' & '.join(rel.get('between', []))}: {rel.get('relation')} (senior: {rel.get('senior')}, "
              f"{rel.get('closeness')})")
    print()
    for ln in result["lines"]:
        ei, dub = ln["emotion_intent"], ln["dub"]
        flag = "  ⚠ voice and words disagree" if ei["mismatch"] else ""
        print(f"{ln['speaker']} → {ln['addressee']}  [{ei['label']} (from {ei['label_source']}), "
              f"{ei['intensity'] or 'unknown'} intensity, {ei['intent']}]{flag}")
        print(f"  source : {ln['text']}")
        print(f"  draft  : {ln['draft']}")
        print(f"  dub    : {dub['translation']}")
        print(f"           {dub.get('romanized', '')}")
        print(f"  deliver: {dub.get('delivery', '')}   "
              f"(syllables {dub['syllables_source']} → {dub['syllables_target']})")
        print()
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
