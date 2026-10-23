"""One round of the game: the player's recording of a translated line -> score, verdict, tips.

The reference is the original actor's line (library.Reference). The player speaks a different
language, so the comparison is about delivery: intensity, pitch shape, timing and stress
placement, plus (with ASR) whether they said the translated line at all.
"""
import re
import unicodedata

import numpy as np

from backend.game import words
from backend.game.analyzer import ProsodyAnalyzer
from backend.game.library import Reference
from backend.roman import romanize
from backend.scoring.arousal import FEATURES, Baseline, baseline
from backend.scoring.director import level, score_line
from backend.settings import config

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


# Languages close enough that a detector confuses them; never call a take "the wrong language" between these.
_CLOSE = {frozenset({"hi", "mr"})}


def wrong_language_heard(a, lang: str) -> str | None:
    """The language the take was clearly spoken in, when it isn't the line's language."""
    heard = getattr(a, "language", None)
    if not heard or heard == lang or heard not in config()["languages"] or a.language_confidence < 0.8:
        return None
    return None if frozenset({heard, lang}) in _CLOSE else heard


def valid_baseline(data) -> Baseline | None:
    """A baseline sent back by the browser: accept it only if it has every feature as a number."""
    if not isinstance(data, dict):
        return None
    try:
        b = {k: float(data[k]) for k in FEATURES}
        if "energy_level" in data:      # loudness of the calm reading; older baselines lack it
            b["energy_level"] = float(data["energy_level"])
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
    return {"ok": True, "baseline": {**baseline([a.prosody]), "energy_level": level(a.prosody)}}


def play(ref: Reference, y: np.ndarray, player_baseline: Baseline | None, analyzer,
         offset: float | None = None) -> dict:
    """`offset`: the scene time at which the browser started recording (for the timing check)."""
    if len(y) < MIN_SECONDS * 16000:
        return {"ok": False, "error": "That recording is too short. Say the whole line."}
    a = analyzer.analyze(y, ref.target_language)
    if a.prosody is None:
        return {"ok": False, "error": NO_VOICE}

    lang = ref.target_language
    checked = words.check(a.text, ref.card["translation"], lang) if a.text is not None else None
    wrong_language = wrong_language_heard(a, lang)
    content = 0.0 if wrong_language else checked["score"] if checked else None
    r = score_line(a.prosody, ref.prosody, ref.card, player_baseline, offset, content, ref.baseline)
    if checked and "words" in r.parts and not wrong_language:
        missed = sum(not w["ok"] for w in checked["words"])
        part = r.parts["words"]
        part.verdict = "every word" if not missed else f"{missed} word{'s' if missed > 1 else ''} off"
        part.tip = words.tip(checked)

    # the weakest part's tip first: one thing to fix, not five
    advice = [p.tip for _, p in sorted(r.parts.items(), key=lambda kv: kv[1].score) if p.tip]
    if wrong_language:
        names = config()["languages"]
        advice = [f"That sounded like {names[wrong_language]['name']}. Say the line in {names[lang]['name']}."] + advice[:2]
    elif content is not None and content < 0.6:
        advice = [words.tip(checked) or f"Say the line as written. We heard: “{a.text}”"] + \
            [t for t in advice if t != words.tip(checked)][:2]
    return {
        "ok": True,
        "score": r.score,
        "verdict": r.verdict,
        "parts": {k: {"score": round(100 * p.score), "verdict": p.verdict, "tip": p.tip} for k, p in r.parts.items()},
        "components": {k: round(100 * p.score) for k, p in r.parts.items()},
        "tips": advice,
        "heard": a.text,
        "word_check": checked["words"] if checked and not wrong_language else None,
        "heard_roman": romanize(a.text, lang) if a.text else None,
        "duration": {"yours": round(a.prosody.duration, 2), "original": round(ref.prosody.duration, 2)},
        "analyzer": analyzer.name,
        "calibrated": player_baseline is not None,
    }
