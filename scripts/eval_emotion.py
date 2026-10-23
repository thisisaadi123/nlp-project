"""THE key step 3 measurement: how well do emotion2vec (which emotion) and our RAVDESS-trained
arousal model (how intense) work on Indic acted speech? Rasa test split, Hindi + Tamil, a balanced
sample of 30 utterances per speaker and emotion. EVALUATION ONLY.

    python scripts/fetch_rasa.py      # once
    python scripts/eval_emotion.py --lang hi ta

Arousal: high = angry/happy/fear/surprise, low = sad/neutral (disgust left out, as for RAVDESS).
"With baseline" uses the same speaker's neutral utterances, leaving out the one being scored.
Embeddings go to data/cache/emotion_rasa_<lang>.npz, predictions to data/cache/emotion_rasa.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.emotion import EMOTIONS, Emotion  # noqa: E402
from backend.scoring.arousal import arousal, baseline  # noqa: E402
from backend.scoring.prosody import extract  # noqa: E402
from scripts.rasa_data import HIGH_AROUSAL, LOW_AROUSAL, load  # noqa: E402


def auc(pos: list[float], neg: list[float]) -> float:
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def confusion(rows: list[dict]) -> str:
    lines = ["true \\ pred " + " ".join(f"{e[:5]:>6}" for e in EMOTIONS) + "  recall"]
    for t in EMOTIONS:
        sel = [r for r in rows if r["emotion"] == t]
        if not sel:
            continue
        counts = [sum(r["pred"] == p for r in sel) for p in EMOTIONS]
        lines.append(f"{t:11} " + " ".join(f"{c:6d}" for c in counts) + f"  {counts[EMOTIONS.index(t)] / len(sel) * 100:5.0f}%")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["hi", "ta"])
    ap.add_argument("--n", type=int, default=30, help="utterances per speaker and emotion")
    args = ap.parse_args()
    model = Emotion()
    out_rows = []
    for lang in args.lang:
        items = load(lang, args.n)
        rows, embs = [], []
        for it in items:
            r = model(it["y"])
            try:
                pros = extract(it["y"])
            except ValueError:
                pros = None
            rows.append({"id": it["id"], "lang": lang, "speaker": it["speaker"], "emotion": it["emotion"],
                         "pred": r.label, "probs": r.probs, "other": r.other, "prosody": pros})
            embs.append(r.embedding)
        np.savez_compressed(ROOT / f"data/cache/emotion_rasa_{lang}.npz", ids=[r["id"] for r in rows],
                            emb=np.stack(embs))

        acc = np.mean([r["pred"] == r["emotion"] for r in rows])
        print(f"\n===== {lang}: emotion2vec on {len(rows)} utterances, accuracy {acc * 100:.1f}% (chance {100 / 7:.0f}%)")
        for spk in sorted({r["speaker"] for r in rows}):
            sel = [r for r in rows if r["speaker"] == spk]
            print(f"  speaker {spk}: {np.mean([r['pred'] == r['emotion'] for r in sel]) * 100:.1f}%")
        print(confusion(rows))
        print(f"  mean 'other' probability: {np.mean([r['other'] for r in rows]):.3f}")

        # arousal transfer
        ar = [r for r in rows if r["prosody"] is not None and r["emotion"] in HIGH_AROUSAL | LOW_AROUSAL]
        for r in ar:
            neutral = [x["prosody"] for x in rows if x["speaker"] == r["speaker"] and x["emotion"] == "neutral"
                       and x["prosody"] is not None and x["id"] != r["id"]]
            r["arousal_raw"] = arousal(r["prosody"])
            r["arousal_rel"] = arousal(r["prosody"], baseline(neutral))
            hi_p = sum(r["probs"][e] for e in HIGH_AROUSAL)
            r["arousal_e2v"] = hi_p
        y = np.array([r["emotion"] in HIGH_AROUSAL for r in ar])
        print(f"  arousal, high vs low, {len(ar)} utterances:")
        for key, name in [("arousal_raw", "prosody, no baseline"), ("arousal_rel", "prosody, with baseline"),
                          ("arousal_e2v", "emotion2vec P(high)")]:
            s = np.array([r[key] for r in ar])
            acc = np.mean((s > 0.5) == y)
            print(f"    {name:24} accuracy {acc * 100:5.1f}%  AUC {auc(s[y], s[~y]):.3f}  "
                  f"mean high {s[y].mean():.2f} / low {s[~y].mean():.2f}")
        for r in rows:
            r.pop("prosody")
        out_rows += rows
    (ROOT / "data/cache/emotion_rasa.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
