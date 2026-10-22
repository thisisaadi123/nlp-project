"""Evaluate the prosody score on RAVDESS (24 actors, same sentences in 8 emotions, 2 takes each).

    python scripts/eval_ravdess.py              # full report
    python scripts/eval_ravdess.py --fit-scales # also print suggested EXPRESSION_SCALES

A good score ranks same-emotion pairs above different-emotion pairs. We measure that with
AUC: 1.0 = always ranked correctly, 0.5 = no better than chance.

Download the data first: scripts/fetch_ravdess.sh
"""
import argparse
import pickle
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.scoring.compare import EXPRESSION_SCALES, score  # noqa: E402
from backend.scoring.prosody import extract_file  # noqa: E402

DATA = ROOT / "data/datasets/ravdess"
CACHE = ROOT / "data/cache/ravdess_prosody.pkl"
EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
COMPONENTS = ["contour", "timing", "emphasis", "emotion"]


def parse(path: Path) -> dict:
    # 03-01-EE-II-SS-RR-AA.wav: emotion, intensity, statement, repetition, actor
    _, _, e, i, s, r, a = (int(x) for x in path.stem.split("-"))
    return {"path": path, "emotion": EMOTIONS[e], "intensity": i, "statement": s, "rep": r, "actor": a}


def _extract(path: Path):
    try:
        return path, extract_file(str(path))
    except Exception as e:  # a handful of clips may be near-silent
        return path, e


def load_features() -> list[dict]:
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
    failed = [p for p, r in cache.items() if isinstance(r, Exception)]
    if failed:
        print(f"Skipped {len(failed)} clips with no usable speech")
    out = []
    for it in items:
        r = cache.get(str(it["path"]))
        if r is not None and not isinstance(r, Exception):
            out.append({**it, "prosody": r})
    return out


def auc(pos: list[float], neg: list[float]) -> float:
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_pairs(pairs):
    rows = []
    for ref, att, same in pairs:
        r = score(ref["prosody"], att["prosody"])
        rows.append({"same": same, "score": r.score, **r.components,
                     "pair": (ref["emotion"], att["emotion"])})
    return rows


def report(title: str, rows: list[dict]) -> None:
    pos = [r for r in rows if r["same"]]
    neg = [r for r in rows if not r["same"]]
    print(f"\n== {title}  ({len(pos)} same-emotion pairs, {len(neg)} different-emotion pairs)")
    print(f"{'':12s}{'AUC':>7s}{'same':>9s}{'diff':>9s}")
    for k in ["score"] + COMPONENTS:
        scale = 1 if k == "score" else 100
        p = [r[k] * scale for r in pos]
        n = [r[k] * scale for r in neg]
        print(f"{k:12s}{auc(p, n):7.3f}{np.mean(p):9.1f}{np.mean(n):9.1f}")
    dub_same = np.mean([r["score"] >= 70 for r in pos])
    dub_diff = np.mean([r["score"] >= 70 for r in neg])
    print(f"Dub rate (score >= 70): same emotion {dub_same:.0%}, different emotion {dub_diff:.0%}")


def emotion_matrix(rows: list[dict]) -> None:
    cells = defaultdict(list)
    for r in rows:
        cells[r["pair"]].append(r["score"])
    names = list(EMOTIONS.values())
    print("\n== Mean score by emotion (row = original, column = attempt), cross-speaker")
    print(f"{'':10s}" + "".join(f"{n[:7]:>8s}" for n in names))
    for a in names:
        print(f"{a:10s}" + "".join(
            f"{np.mean(cells[(a, b)]):8.1f}" if cells[(a, b)] else f"{'-':>8s}" for b in names))


def demo(items: list[dict], actor: int) -> None:
    """The angry_1 / angry_2 / sad / flat test, on one actor."""
    pick = {(it["emotion"], it["intensity"], it["rep"]): it for it in items
            if it["actor"] == actor and it["statement"] == 1}
    ref = pick[("angry", 2, 1)]
    takes = [("angry, take 2", ("angry", 2, 2)), ("sad", ("sad", 2, 1)), ("neutral (flat)", ("neutral", 1, 1)),
             ("happy", ("happy", 2, 1)), ("calm", ("calm", 2, 1))]
    sex = "male" if actor % 2 else "female"
    print(f"\n== Demo: actor {actor} ({sex}), reference = angry take 1")
    print(f"{'attempt':16s}{'score':>7s}{'contour':>9s}{'timing':>8s}{'emph':>7s}{'emotion':>9s}  verdict")
    for label, key in takes:
        r = score(ref["prosody"], pick[key]["prosody"])
        c = r.components
        print(f"{label:16s}{r.score:7.1f}{c['contour']*100:9.0f}{c['timing']*100:8.0f}"
              f"{c['emphasis']*100:7.0f}{c['emotion']*100:9.0f}  {r.verdict}")


def fit_scales(items: list[dict]) -> None:
    # Spread of each statistic within one actor and emotion, i.e. take-to-take noise.
    groups = defaultdict(list)
    for it in items:
        groups[(it["actor"], it["emotion"], it["intensity"])].append(it["prosody"].stats)
    print("\n== Suggested EXPRESSION_SCALES (2x median within-group std)")
    for k in EXPRESSION_SCALES:
        sds = [np.std([s[k] for s in g]) for g in groups.values() if len(g) > 1]
        print(f"    {k!r}: {2 * np.median(sds):.2f},   (current {EXPRESSION_SCALES[k]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-scales", action="store_true")
    ap.add_argument("--pairs", type=int, default=3000, help="cross-speaker pairs per class")
    args = ap.parse_args()
    random.seed(0)

    items = load_features()
    print(f"{len(items)} clips loaded")

    # A: same actor, same sentence. Positive = same emotion and intensity, other take.
    same_speaker = []
    by_key = defaultdict(list)
    for it in items:
        by_key[(it["actor"], it["statement"])].append(it)
    for group in by_key.values():
        for a in group:
            for b in group:
                if a is b:
                    continue
                if a["emotion"] == b["emotion"] and a["intensity"] == b["intensity"]:
                    same_speaker.append((a, b, True))
                elif a["emotion"] != b["emotion"]:
                    same_speaker.append((a, b, False))
    report("Same speaker", run_pairs(same_speaker))

    # B: different actors (the real game: the user is not the original actor).
    by_statement = defaultdict(list)
    for it in items:
        by_statement[it["statement"]].append(it)
    pos, neg = [], []
    while len(pos) < args.pairs or len(neg) < args.pairs:
        group = by_statement[random.choice([1, 2])]
        a, b = random.sample(group, 2)
        if a["actor"] == b["actor"]:
            continue
        if a["emotion"] == b["emotion"] and len(pos) < args.pairs:
            pos.append((a, b, True))
        elif a["emotion"] != b["emotion"] and len(neg) < args.pairs:
            neg.append((a, b, False))
    cross = run_pairs(pos + neg)
    report("Different speakers", cross)
    report("Different speakers, opposite sex",
           [r for r, (a, b, _) in zip(cross, pos + neg) if a["actor"] % 2 != b["actor"] % 2])
    emotion_matrix(cross)

    demo(items, actor=1)
    demo(items, actor=2)
    if args.fit_scales:
        fit_scales(items)


if __name__ == "__main__":
    main()
