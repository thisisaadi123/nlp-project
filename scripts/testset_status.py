"""Show what the test set covers and what is still missing.

    python scripts/testset_status.py

Counts use verified rows only (checked by a native speaker); drafts are shown in brackets.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.testset import CLIPS, LABELS, LANGUAGES, read_rows  # noqa: E402

PER_LANGUAGE = 3
REQUIREMENTS = [  # (label, minimum across all languages, test)
    ("angry", 3, lambda r: r["emotion"] == "angry"),
    ("sad / pleading", 3, lambda r: r["emotion"] == "sad"),
    ("comic or sarcastic", 2, lambda r: r["emotion"] in ("comic", "sarcastic")),
    ("calm / neutral", 2, lambda r: r["emotion"] in ("calm", "neutral")),
    ("background music", 3, lambda r: r["has_music"] == "yes"),
    ("two or more speakers", 3, lambda r: r["num_speakers"].isdigit() and int(r["num_speakers"]) >= 2),
    ("register contrast (bhai/bhaiya...)", 2, lambda r: bool(r["register_note"].strip())),
]


def mark(verified: int, total: int, need: int) -> str:
    tick = "✅" if verified >= need else "⬜"
    return f"{tick} {verified}/{need}" + (f"  (+{total - verified} unverified)" if total > verified else "")


def main() -> None:
    rows = read_rows()
    done = [r for r in rows if r.get("verified") == "yes"]
    print(f"{len(rows)} clips, {len(done)} verified\n")

    print("Per language (need 3 each):")
    for lang in LANGUAGES:
        all_l = [r for r in rows if r["language"] == lang]
        ver_l = [r for r in done if r["language"] == lang]
        print(f"  {lang}: {mark(len(ver_l), len(all_l), PER_LANGUAGE)}")

    print("\nCoverage across all languages:")
    for label, need, test in REQUIREMENTS:
        print(f"  {label:36s}{mark(sum(map(test, done)), sum(map(test, rows)), need)}")

    problems = []
    for r in rows:
        if not (CLIPS / f"{r['clip_id']}.mp4").exists():
            problems.append(f"{r['clip_id']}: clip file missing here (get it from the team Drive folder)")
        elif not (LABELS / f"{r['clip_id']}.json").exists():
            problems.append(f"{r['clip_id']}: not pre-labelled yet (scripts/prelabel_clip.py {r['clip_id']})")
        elif r.get("verified") != "yes":
            problems.append(f"{r['clip_id']}: draft labels waiting for a native speaker to check")
    if problems:
        print("\nTo do:")
        for p in problems:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
