"""Rasa loading shared by the step 3 evaluations. EVALUATION ONLY (see fetch_rasa.py).

One female and one male actor per language. Emotional styles are the six Ekman emotions;
"neutral" is the WIKI style (neutral readings of Wikipedia text). The test split already has
~30+ utterances per emotion per speaker, so a balanced sample is drawn from it.
Download first: python scripts/fetch_rasa.py
"""
import io
import random
import sys
from pathlib import Path

import librosa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.settings import config  # noqa: E402

DATA = ROOT / "data/datasets/rasa"
STYLES = {"ANGER": "angry", "DISGUST": "disgust", "FEAR": "fear", "HAPPY": "happy",
          "SAD": "sad", "SURPRISE": "surprise", "WIKI": "neutral"}
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
HIGH_AROUSAL = {"angry", "happy", "fear", "surprise"}
LOW_AROUSAL = {"sad", "neutral"}             # disgust is ambiguous and left out of arousal tests


def load(lang: str, per_class: int = 30, seed: int = 0) -> list[dict]:
    """Balanced sample: up to `per_class` utterances per (speaker, emotion), audio at 16 kHz."""
    folder = DATA / config()["languages"][lang]["name"]
    files = sorted(folder.glob("test-*.parquet"))
    if not files:
        sys.exit(f"No Rasa {lang} in {folder}. Run scripts/fetch_rasa.py first.")
    meta = []
    for f in files:
        for r in pq.read_table(f, columns=["filename", "text", "gender", "style"]).to_pylist():
            if r["style"] in STYLES:
                meta.append({**r, "file": f})
    groups = {}
    for r in meta:
        groups.setdefault((r["gender"], STYLES[r["style"]]), []).append(r)
    rng = random.Random(seed)
    picked = {r["filename"]: r for g in sorted(groups) for r in rng.sample(sorted(groups[g], key=lambda r: r["filename"]),
                                                                           min(per_class, len(groups[g])))}
    items = []
    for f in files:                                   # read audio only for the picked rows
        t = pq.read_table(f, columns=["filename", "audio"])
        for name, audio in zip(t.column("filename").to_pylist(), t.column("audio").to_pylist()):
            if name in picked:
                y, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(1)
                r = picked[name]
                items.append({"id": name, "lang": lang, "speaker": r["gender"][0], "emotion": STYLES[r["style"]],
                              "text": r["text"], "y": librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y})
    return sorted(items, key=lambda it: it["id"])
