"""Build game library entries from step 4's output.

    python scripts/build_library.py                    # every data/cache/dub/*.json with its clip
    python scripts/build_library.py data/cache/dub/ta_02.ta-hi.json

Each entry goes to data/library/<clip>.<src>-<tgt>/ (see backend/game/library.py).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.game.library import build_entry  # noqa: E402
from scripts.testset import read_rows  # noqa: E402


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "data/cache/dub").glob("*.json"))
    titles = {r["clip_id"]: r.get("title") or None for r in read_rows()}
    if not paths:
        sys.exit("No step 4 output in data/cache/dub/. Run scripts/dub_clip.py first.")
    for dub in paths:
        clip = ROOT / "data/clips" / f"{dub.name.split('.')[0]}.mp4"
        if not clip.exists():
            print(f"skip {dub.name}: {clip.relative_to(ROOT)} missing")
            continue
        out = build_entry(dub, clip, title=titles.get(clip.stem))
        entry = json.loads((out / "entry.json").read_text(encoding="utf-8"))
        playable = sum(c["playable"] for c in entry["lines"])
        models = {c["model"] for c in entry["lines"]}
        print(f"{entry['id']}: {playable}/{len(entry['lines'])} lines playable, translated by {', '.join(map(str, models))}")


if __name__ == "__main__":
    main()
