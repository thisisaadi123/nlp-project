"""emotion2vec embeddings and its own 7-way probabilities for the emotion head experiments,
cached to data/cache/e2v_<corpus>.npz (GPU, a few minutes per corpus).

    python scripts/emotion_embed.py ravdess rasa_hi rasa_ta

ravdess: training data (24 English actors; calm counts as neutral).
rasa_<lang>: EVALUATION ONLY (see fetch_rasa.py), a balanced sample per speaker and emotion.
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.emotion import EMOTIONS, Emotion  # noqa: E402

CACHE = ROOT / "data/cache"
RAVDESS_MAP = {"neutral": "neutral", "calm": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
               "fearful": "fear", "disgust": "disgust", "surprised": "surprise"}


def ravdess_items() -> list[dict]:
    from scripts.ravdess_data import DATA, parse
    paths = sorted(DATA.glob("Actor_*/*.wav"))
    if not paths:
        sys.exit(f"No RAVDESS files in {DATA}. Run scripts/fetch_ravdess.sh first.")
    items = []
    for p in paths:
        it = parse(p)
        items.append({"id": p.stem, "speaker": f"ravdess_{it['actor']:02d}", "emotion": RAVDESS_MAP[it["emotion"]],
                      "raw": it["emotion"], "intensity": it["intensity"], "y": librosa.load(p, sr=16000)[0]})
    return items


def rasa_items(lang: str, per_class: int) -> list[dict]:
    from scripts.rasa_data import load
    return [{"id": it["id"], "speaker": f"{lang}_{it['speaker']}", "emotion": it["emotion"], "raw": it["emotion"],
             "intensity": 0, "y": it["y"]} for it in load(lang, per_class)]


def embed(corpus: str, per_class: int, model: Emotion) -> Path:
    items = ravdess_items() if corpus == "ravdess" else rasa_items(corpus.split("_", 1)[1], per_class)
    embs, probs = [], []
    for i, it in enumerate(items):
        r = model(it["y"])
        embs.append(r.embedding)
        probs.append([r.probs[e] for e in EMOTIONS])
        if (i + 1) % 200 == 0:
            print(f"  {corpus}: {i + 1}/{len(items)}")
    out = CACHE / f"e2v_{corpus}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=[it["id"] for it in items], speaker=[it["speaker"] for it in items],
                        emotion=[it["emotion"] for it in items], raw=[it["raw"] for it in items],
                        intensity=[it["intensity"] for it in items], emb=np.stack(embs), probs=np.array(probs))
    print(f"{corpus}: {len(items)} utterances, {len({it['speaker'] for it in items})} speakers -> {out.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+", help="ravdess, rasa_hi, rasa_ta, ...")
    ap.add_argument("--per-class", type=int, default=50, help="Rasa utterances per speaker and emotion")
    args = ap.parse_args()
    model = Emotion()
    for corpus in args.corpora:
        embed(corpus, args.per_class, model)


if __name__ == "__main__":
    main()
