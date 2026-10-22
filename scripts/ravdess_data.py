"""RAVDESS loading shared by train_arousal.py and eval_ravdess.py.

24 actors (odd = male, even = female) say 2 sentences in 8 emotions, 2 takes each.
Download first: scripts/fetch_ravdess.sh
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.scoring.arousal import baseline  # noqa: E402
from backend.scoring.prosody import extract_file  # noqa: E402

DATA = ROOT / "data/datasets/ravdess"
CACHE = ROOT / "data/cache/ravdess_prosody.pkl"
EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
HIGH_AROUSAL = {"angry", "happy", "fearful", "surprised"}
LOW_AROUSAL = {"neutral", "calm", "sad"}   # disgust is ambiguous and left out of arousal tests
N_FOLDS = 4                                # 6 held-out actors per fold


def parse(path: Path) -> dict:
    # 03-01-EE-II-SS-RR-AA.wav: emotion, intensity, statement, repetition, actor
    _, _, e, i, s, r, a = (int(x) for x in path.stem.split("-"))
    return {"path": path, "emotion": EMOTIONS[e], "intensity": i, "statement": s, "rep": r, "actor": a}


def fold_of(actor: int) -> int:
    return actor % N_FOLDS


def _extract(path: Path):
    try:
        return path, extract_file(str(path))
    except Exception as e:  # a handful of clips may be near-silent
        return path, e


def load() -> list[dict]:
    """All clips with prosody (cached) and each clip's speaker baseline.

    The baseline is the mean of the same actor's neutral takes of the *other* sentence,
    like a player's calibration line: same voice, different words, never the clip itself.
    """
    items = [parse(p) for p in sorted(DATA.glob("Actor_*/*.wav"))]
    if not items:
        sys.exit(f"No RAVDESS files in {DATA}. Run scripts/fetch_ravdess.sh first.")
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    todo = [it["path"] for it in items if str(it["path"]) not in cache]
    if todo:
        print(f"Extracting prosody for {len(todo)} clips...")
        with Pool() as pool:
            for path, result in pool.imap_unordered(_extract, todo, chunksize=8):
                cache[str(path)] = result
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(cache))

    out = [{**it, "prosody": cache[str(it["path"])]} for it in items
           if not isinstance(cache.get(str(it["path"])), Exception)]
    if len(out) < len(items):
        print(f"Skipped {len(items) - len(out)} clips with no usable speech")
    neutral = {}
    for it in out:
        if it["emotion"] == "neutral":
            neutral.setdefault((it["actor"], it["statement"]), []).append(it["prosody"])
    for it in out:
        it["baseline"] = baseline(neutral[(it["actor"], 3 - it["statement"])])
    return out


def arousal_xy(items: list[dict], relative: bool) -> tuple[np.ndarray, np.ndarray]:
    from backend.scoring.arousal import features
    rows = [it for it in items if it["emotion"] in HIGH_AROUSAL | LOW_AROUSAL]
    X = np.array([features(it["prosody"].stats, it["baseline"] if relative else None) for it in rows])
    y = np.array([it["emotion"] in HIGH_AROUSAL for it in rows], dtype=float)
    return X, y
