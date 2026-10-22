"""Evaluate the prosody score on RAVDESS.

    python scripts/eval_ravdess.py

A good score ranks matching pairs above non-matching pairs. We measure that with AUC:
1.0 = always ranked correctly, 0.5 = chance. Every pair is scored with an arousal model
that never saw either actor (leave-actors-out folds), so nothing is tested on its own
training data.

Tests:
- same speaker, same vs different emotion      (one person's takes)
- different speakers, same vs different emotion (the game: player is not the original actor)
- different speakers, same vs opposite arousal  (what prosody can judge; emotion2vec adds the rest)
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.scoring.compare import score  # noqa: E402
from scripts.ravdess_data import EMOTIONS, HIGH_AROUSAL, LOW_AROUSAL, N_FOLDS, fold_of, load  # noqa: E402
from scripts.train_arousal import fold_models  # noqa: E402

COMPONENTS = ["emotion", "contour", "timing", "emphasis"]


def auc(pos, neg) -> float:
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(pairs, models, use_baseline: bool) -> list[dict]:
    rows = []
    for a, b, same in pairs:
        r = score(a["prosody"], b["prosody"],
                  a["baseline"] if use_baseline else None, b["baseline"] if use_baseline else None,
                  arousal_model=models[fold_of(a["actor"])])
        rows.append({"same": same, "score": r.score, **r.components, "pair": (a["emotion"], b["emotion"])})
    return rows


def summary(rows) -> dict[str, float]:
    pos = [r for r in rows if r["same"]]
    neg = [r for r in rows if not r["same"]]
    out = {k: auc([r[k] for r in pos], [r[k] for r in neg]) for k in ["score"] + COMPONENTS}
    out["mean_same"] = np.mean([r["score"] for r in pos])
    out["mean_diff"] = np.mean([r["score"] for r in neg])
    out["dub_same"] = np.mean([r["score"] >= 70 for r in pos])
    out["dub_diff"] = np.mean([r["score"] >= 70 for r in neg])
    return out


def print_table(title: str, n_pos: int, n_neg: int, results: dict[str, dict]) -> None:
    print(f"\n== {title}  ({n_pos} matching / {n_neg} non-matching pairs)")
    print(f"{'':16s}{'score':>7s}" + "".join(f"{c:>9s}" for c in COMPONENTS)
          + f"{'mean match':>12s}{'mean other':>12s}{'Dub match':>11s}{'Dub other':>11s}")
    for label, s in results.items():
        print(f"{label:16s}{s['score']:7.3f}" + "".join(f"{s[c]:9.3f}" for c in COMPONENTS)
              + f"{s['mean_same']:12.1f}{s['mean_diff']:12.1f}{s['dub_same']:11.0%}{s['dub_diff']:11.0%}")


def cross_pairs(items, n, match, eligible=lambda it: True):
    """Pairs of different actors from the same held-out fold, saying the same sentence."""
    groups = defaultdict(list)
    for it in items:
        if eligible(it):
            groups[(fold_of(it["actor"]), it["statement"])].append(it)
    keys = list(groups)
    pos, neg = [], []
    while len(pos) < n or len(neg) < n:
        a, b = random.sample(groups[random.choice(keys)], 2)
        if a["actor"] == b["actor"]:
            continue
        m = match(a, b)
        if m and len(pos) < n:
            pos.append((a, b, True))
        elif m is False and len(neg) < n:
            neg.append((a, b, False))
    return pos + neg


def evaluate(title, pairs, models) -> list[dict]:
    rows = {}
    results = {}
    for label, use in (("with baseline", True), ("no baseline", False)):
        rows[label] = run(pairs, models, use)
        results[label] = summary(rows[label])
    n_pos = sum(1 for p in pairs if p[2])
    print_table(title, n_pos, len(pairs) - n_pos, results)
    return rows["with baseline"]


def emotion_matrix(rows) -> None:
    cells = defaultdict(list)
    for r in rows:
        cells[r["pair"]].append(r["score"])
    names = list(EMOTIONS.values())
    print("\n== Mean score, different speakers, with baseline (row = original, column = attempt)")
    print(f"{'':10s}" + "".join(f"{n[:7]:>8s}" for n in names))
    for a in names:
        print(f"{a:10s}" + "".join(
            f"{np.mean(cells[(a, b)]):8.1f}" if cells[(a, b)] else f"{'-':>8s}" for b in names))


def demo(items, models, actor: int) -> None:
    pick = {(it["emotion"], it["intensity"], it["rep"]): it for it in items
            if it["actor"] == actor and it["statement"] == 1}
    ref = pick[("angry", 2, 1)]
    takes = [("angry, take 2", ("angry", 2, 2)), ("sad", ("sad", 2, 1)), ("neutral (flat)", ("neutral", 1, 1)),
             ("happy", ("happy", 2, 1)), ("calm", ("calm", 2, 1))]
    print(f"\n== Demo: actor {actor} ({'male' if actor % 2 else 'female'}), reference = angry take 1, with baseline")
    print(f"{'attempt':16s}{'score':>7s}{'emotion':>9s}{'contour':>9s}{'timing':>8s}{'emph':>7s}"
          f"{'arousal ref/att':>17s}  verdict")
    for label, key in takes:
        att = pick[key]
        r = score(ref["prosody"], att["prosody"], ref["baseline"], att["baseline"],
                  arousal_model=models[fold_of(actor)])
        c, d = r.components, r.details
        print(f"{label:16s}{r.score:7.1f}{c['emotion']*100:9.0f}{c['contour']*100:9.0f}{c['timing']*100:8.0f}"
              f"{c['emphasis']*100:7.0f}{d['ref_arousal']:10.2f} / {d['att_arousal']:.2f}  {r.verdict}")


def main() -> None:
    random.seed(0)
    items = load()
    models = fold_models(items)
    print(f"{len(items)} clips, {N_FOLDS} leave-actors-out folds")

    same_speaker = []
    by_actor = defaultdict(list)
    for it in items:
        by_actor[(it["actor"], it["statement"])].append(it)
    for group in by_actor.values():
        for a in group:
            for b in group:
                if a is b:
                    continue
                if a["emotion"] == b["emotion"] and a["intensity"] == b["intensity"]:
                    same_speaker.append((a, b, True))
                elif a["emotion"] != b["emotion"]:
                    same_speaker.append((a, b, False))
    evaluate("Same speaker: same vs different emotion", same_speaker, models)

    def same_emotion(a, b):
        return a["emotion"] == b["emotion"]
    cross = cross_pairs(items, 3000, same_emotion)
    cross_rows = evaluate("Different speakers: same vs different emotion", cross, models)

    def arousal_group(it):
        return "high" if it["emotion"] in HIGH_AROUSAL else "low"
    cross_ar = cross_pairs(items, 3000, lambda a, b: arousal_group(a) == arousal_group(b),
                           eligible=lambda it: it["emotion"] in HIGH_AROUSAL | LOW_AROUSAL)
    evaluate("Different speakers: same vs opposite arousal", cross_ar, models)

    emotion_matrix(cross_rows)
    demo(items, models, actor=1)
    demo(items, models, actor=2)


if __name__ == "__main__":
    main()
