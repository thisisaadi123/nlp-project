"""The melody line: a compact picture of how a line is delivered, for the website to draw.

Time runs from the first voiced sound of the line (0) to its end. Pitch is in semitones relative
to that speaker's own median, so a player's curve and the original actor's can share one chart
whatever their voices; loudness is 0..1 within the line.
"""
import re

import numpy as np
from scipy.ndimage import median_filter

from backend.scoring.prosody import HOP_S, Prosody

POINTS_PER_S = 20
STEP = round(1 / (POINTS_PER_S * HOP_S))       # prosody frames per point
PITCH_LIMIT = 12.0                             # semitones; beyond this is octave-jump noise
LOUD_RANGE_DB = 35.0


def _clean(word: str) -> str:
    return re.sub(r"[^\w]", "", word)


def shape(p: Prosody, words: list[str] | None = None, stress: list[str] | None = None) -> dict:
    n = len(p.pitch) // STEP
    if n == 0:
        return {"duration": round(p.duration, 2), "step": 1 / POINTS_PER_S, "pitch": [], "loud": [], "voiced": [],
                "stress_t": 0.0, "words": []}

    pitch = median_filter(p.pitch, size=5)[: n * STEP].reshape(n, STEP).mean(axis=1)
    pitch = np.clip(pitch, -PITCH_LIMIT, PITCH_LIMIT)

    top = np.percentile(p.energy, 95)
    loud = np.clip((p.energy - (top - LOUD_RANGE_DB)) / LOUD_RANGE_DB, 0, 1)[: n * STEP].reshape(n, STEP).mean(axis=1)

    t = np.arange(n) * STEP * HOP_S
    voiced = np.ones(n, dtype=bool)
    for a, b in p.pauses:
        voiced &= ~((t >= a) & (t < b))
    voiced &= loud > 0.08

    duration = float(p.duration)
    stress_t = float(np.argmax(p.prominence) * HOP_S)

    word_list = []
    if words:
        targets = {_clean(s) for s in (stress or []) if _clean(s)}
        span = duration / len(words)
        word_list = [{"w": w, "start": round(i * span, 2), "end": round((i + 1) * span, 2),
                      "stress": any(tg in _clean(w) for tg in targets)} for i, w in enumerate(words)]

    return {
        "duration": round(duration, 2),
        "step": 1 / POINTS_PER_S,
        "pitch": [round(float(v), 2) for v in pitch],
        "loud": [round(float(v), 2) for v in loud],
        "voiced": [int(v) for v in voiced],
        "stress_t": round(stress_t, 2),
        "words": word_list,
    }
