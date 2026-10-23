"""Which emotion: emotion2vec+ large through funasr (prosody/arousal says how intense, not which).

emotion2vec's 9 outputs are mapped onto our 7 classes (the six Ekman emotions + neutral, as in
Rasa); "other" and "<unk>" are dropped and the rest renormalised. The 1024-d utterance embedding
is returned too, for later fine-tuning or similarity; it's stored outside the JSON.
"""
import contextlib
import io
import logging
from dataclasses import dataclass

import numpy as np

from backend.settings import config

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
_MAP = {"angry": "angry", "disgusted": "disgust", "fearful": "fear", "happy": "happy",
        "neutral": "neutral", "sad": "sad", "surprised": "surprise"}


@dataclass
class EmotionResult:
    label: str
    probs: dict[str, float]         # over EMOTIONS, sums to 1
    other: float                    # emotion2vec's own "other" probability, before renormalising
    embedding: np.ndarray           # (1024,)


class Emotion:
    def __init__(self, device: str = "cuda"):
        from funasr import AutoModel
        root = logging.getLogger()
        level = root.level
        root.setLevel(logging.WARNING)                             # funasr logs every weight it maps...
        with contextlib.redirect_stdout(io.StringIO()):            # ...and prints the missing ones
            self.model = AutoModel(model=config()["models"]["emotion"], hub="hf", device=f"{device}:0"
                                   if device == "cuda" else device, disable_update=True, log_level="ERROR")
        root.setLevel(level)

    def __call__(self, y: np.ndarray) -> EmotionResult:
        r = self.model.generate(np.ascontiguousarray(y, dtype=np.float32), granularity="utterance",
                                extract_embedding=True, disable_pbar=True)[0]
        raw = {lab.split("/")[-1]: float(s) for lab, s in zip(r["labels"], r["scores"])}
        p = np.array([sum(v for k, v in raw.items() if _MAP.get(k) == e) for e in EMOTIONS])
        p = p / p.sum() if p.sum() > 0 else np.full(len(EMOTIONS), 1 / len(EMOTIONS))
        return EmotionResult(EMOTIONS[int(p.argmax())], {e: round(float(v), 4) for e, v in zip(EMOTIONS, p)},
                             raw.get("other", 0.0), np.asarray(r["feats"], dtype=np.float32))
