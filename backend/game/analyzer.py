"""Analysing a player's recording, in one of two modes (config.yaml game.analyzer):

- "prosody": CPU only, no models. Pitch, loudness, timing, stress and the prosody arousal model.
  Runs anywhere (the Mac, a free Hugging Face Space) but cannot check the words.
- "full": step 3's analyze_line (IndicConformer ASR + word timings + emotion2vec) on top. Adds the
  content score (did they say the line?) and places the stressed word by its timing. Needs the GPU
  stack from requirements-gpu.txt.

A player's recording is one voice into a microphone, the condition step 3's models were tested in
(not a movie soundtrack), so this is where they work well.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from backend.scoring.prosody import Prosody, extract
from backend.settings import config


@dataclass
class Analysis:
    prosody: Prosody | None          # None: no usable voice in the recording
    text: str | None = None          # what ASR heard ("full" mode only)
    words: list[dict] = field(default_factory=list)   # [{"w", "start", "end", ...}] ("full" only)
    arousal_e2v: float | None = None


class ProsodyAnalyzer:
    name = "prosody"

    def analyze(self, y: np.ndarray, lang: str) -> Analysis:
        try:
            return Analysis(extract(y))
        except ValueError:
            return Analysis(None)


class SpeechAnalyzer:
    name = "full"

    def __init__(self):
        from backend.pipeline.speech.models import Models
        self.models = Models()

    def analyze(self, y: np.ndarray, lang: str) -> Analysis:
        from backend.pipeline.speech.run import analyze_line
        r = analyze_line(self.models, y, lang)
        return Analysis(r["_prosody"], r["text"], r["words"], r["emotion"].get("arousal_e2v"))


@lru_cache
def get_analyzer(name: str | None = None):
    name = name or config()["game"]["analyzer"]
    if name == "prosody":
        return ProsodyAnalyzer()
    if name == "full":
        return SpeechAnalyzer()
    raise ValueError(f"unknown analyzer {name!r} (use 'prosody' or 'full')")
