"""VRAM manager for the 8 GB RTX 4060.

Resident (config.yaml vram.resident): ASR, emotion and the word aligners, which the live game
loop needs on every attempt. Everything else (Demucs, pyannote, later IndicTrans2 and Parler-TTS)
loads on demand, one at a time, and is released afterwards:

    models = Models()
    with models.use("separation") as sep:
        vocals = sep(y)
    text = models.asr.transcribe(vocals, "hi")
"""
import gc
from contextlib import contextmanager

import torch

from backend.settings import config


def _load(name: str, lang: str | None = None):
    if name == "asr":
        from backend.pipeline.speech.asr import ASR
        return ASR()
    if name == "emotion":
        from backend.pipeline.speech.emotion import Emotion
        return Emotion()
    if name == "aligner":
        from backend.pipeline.speech.align import CharAligner
        return CharAligner(lang) if CharAligner.available(lang) else None
    if name == "separation":
        from backend.pipeline.speech.separate import Separator
        return Separator()
    if name == "diarization":
        from backend.pipeline.speech.diarize import Diarizer
        return Diarizer()
    if name == "tts":
        from backend.pipeline.tts.parler import ParlerTTS
        return ParlerTTS()
    raise KeyError(name)


def vram_used_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


class Models:
    def __init__(self, preload: bool = True):
        self._resident = {}
        self._aligners = {}
        if preload:
            for name in config()["vram"]["resident"]:
                if name != "aligner":
                    getattr(self, name)

    def _get(self, name):
        if name not in self._resident:
            self._resident[name] = _load(name)
        return self._resident[name]

    @property
    def asr(self):
        return self._get("asr")

    @property
    def emotion(self):
        return self._get("emotion")

    def aligner(self, lang: str):
        """Character aligner for `lang`, or None (then the ASR's own CTC timings are used)."""
        if lang not in self._aligners:
            self._aligners[lang] = _load("aligner", lang)
        return self._aligners[lang]

    @contextmanager
    def use(self, name: str):
        """Load an on-demand model, yield it, and free its VRAM afterwards."""
        model = _load(name)
        try:
            yield model
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
