"""One round of the game: the player's recording of a translated line -> score, verdict, tips.

The reference is the original actor's line (library.Reference). The player speaks a different
language, so the comparison is about delivery: intensity, pitch shape, timing and stress
placement, plus (with ASR) whether they said the translated line at all.
"""
import re
import unicodedata

import numpy as np

from backend.game.analyzer import ProsodyAnalyzer
from backend.game.library import Reference
from backend.game.shape import shape
from backend.roman import romanize
from backend.scoring.arousal import FEATURES, Baseline, baseline
from backend.scoring.compare import score
from backend.scoring.feedback import tips

NO_VOICE = "We couldn't hear a voice in that recording. Check your microphone and try again."
MIN_SECONDS = 0.5


def _plain(text: str, lang: str) -> str:
    """Romanized, lower-case, letters only: spelling variants (formal vs spoken Tamil) matter less."""
    text = romanize(unicodedata.normalize("NFC", text), lang).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", text)).strip()


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def content_similarity(heard: str, expected: str, lang: str) -> float:
    """1 - character error rate between what was heard and the translated line, romanized."""
    ref, hyp = _plain(expected, lang), _plain(heard, lang)
    if not ref:
        return 1.0
    return max(0.0, 1.0 - _edit_distance(ref, hyp) / len(ref))


def stressed_span(words: list[dict], targets: list[str], lang: str) -> tuple[float, float] | None:
    """Time span of the word that should carry the stress, found in the ASR words."""
    if not words or not targets:
        return None
    best, best_d = None, 0.4  # accept at most 40% character difference
    for target in targets:
        t = _plain(target, lang)
        for w in words:
            h = _plain(w.get("w", ""), lang)
            if t and h:
                d = _edit_distance(t, h) / max(len(t), 1)
                if d < best_d:
                    best, best_d = (float(w["start"]), float(w["end"])), d
    return best


def valid_baseline(data) -> Baseline | None:
    """A baseline sent back by the browser: accept it only if it has every feature as a number."""
    if not isinstance(data, dict):
        return None
    try:
        b = {k: float(data[k]) for k in FEATURES}
    except (KeyError, TypeError, ValueError):
        return None
    return b if all(np.isfinite(v) for v in b.values()) else None


def calibrate(y: np.ndarray) -> dict:
    """The player's neutral voice, from one calm reading of the calibration sentence."""
    if len(y) < MIN_SECONDS * 16000:
        return {"ok": False, "error": "That recording is too short. Read the whole sentence."}
    a = ProsodyAnalyzer().analyze(y, "")
    if a.prosody is None:
        return {"ok": False, "error": NO_VOICE}
    return {"ok": True, "baseline": baseline([a.prosody])}


def play(ref: Reference, y: np.ndarray, player_baseline: Baseline | None, analyzer) -> dict:
    if len(y) < MIN_SECONDS * 16000:
        return {"ok": False, "error": "That recording is too short. Say the whole line."}
    a = analyzer.analyze(y, ref.target_language)
    if a.prosody is None:
        return {"ok": False, "error": NO_VOICE}

    lang = ref.target_language
    content = content_similarity(a.text, ref.card["translation"], lang) if a.text is not None else None
    span = stressed_span(a.words, ref.card.get("stress") or [], lang)
    r = score(ref.prosody, a.prosody, ref.baseline, player_baseline, target_span=span, content=content)

    advice = tips(r)
    if content is not None and content < 0.6:
        advice = [f"Say the line as written. We heard: “{a.text}”"] + advice[:2]
    return {
        "ok": True,
        "score": r.score,
        "verdict": r.verdict,
        "components": {k: round(100 * v) for k, v in r.components.items()},
        "tips": advice,
        "heard": a.text,
        "heard_roman": romanize(a.text, lang) if a.text else None,
        "duration": {"yours": round(a.prosody.duration, 2), "original": round(ref.prosody.duration, 2)},
        "intensity": {"yours": round(r.details["att_arousal"], 2), "original": round(r.details["ref_arousal"], 2)},
        "analyzer": analyzer.name,
        "calibrated": player_baseline is not None,
        "shape": shape(a.prosody),
    }
