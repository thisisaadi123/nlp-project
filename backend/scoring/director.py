"""The game's score: did the player follow the director's note for this line?

The booth tells the player four things, and the score checks exactly those, each against
something measurable in the player's own recording:

- timing: start and stop when the actor does (lip-sync), on the scene's clock
- volume: soft / normal / loud / shouting, measured against the player's own calm calibration
- punch:  the strongest moment lands on the stressed word
- tone:   the voice rises, drops or stays level at the end like the original

The previous score (compare.py) matched the player's prosody to the movie soundtrack directly:
across different voices, music and mixing, most of its parts were close to chance (AUC 0.54-0.57
on RAVDESS cross-actor pairs). Each part here has a plain-language reason that comes with it.
"""
from dataclasses import dataclass, field

import numpy as np

from backend.scoring.arousal import arousal
from backend.scoring.compare import _sim, _stress, ending_slope
from backend.scoring.prosody import Prosody
from backend.settings import config

LEVELS = ["soft", "normal", "loud", "shouting"]
# Bands of (dB louder than calm) + 0.5 x (semitones higher than calm). From RAVDESS, each actor
# against their own neutral takes (median): calm/sad -3; happy/fear/surprise at normal strength
# +6..+8; angry +12; strong anger/fear/happiness +19..+26.
BANDS = [(-99.0, -1.5), (-1.5, 7.0), (7.0, 16.0), (16.0, 99.0)]
INTENSITY_LEVEL = {"low": 0, "medium": 1, "high": 2}


@dataclass
class Part:
    score: float                      # 0..1
    verdict: str                      # one or two words, e.g. "on time", "too soft"
    tip: str                          # what to do next time ("" when it was right)


@dataclass
class Direction:
    score: float                      # 0..100
    verdict: str                      # "Dub" or "Goose"
    parts: dict[str, Part]
    details: dict = field(default_factory=dict)


def level(p: Prosody) -> float:
    """How loud the speech is, dB (90th percentile of the speech frames)."""
    return float(np.percentile(p.energy, 90))


def target_level(intensity: str | None, emotion: str | None) -> int:
    lv = INTENSITY_LEVEL.get(intensity or "", 1)
    return 3 if lv == 2 and emotion in ("angry", "surprised") else lv


def volume(att: Prosody, base: dict | None, intensity: str | None, emotion: str | None,
           ref: Prosody | None = None, ref_base: dict | None = None) -> Part:
    """Half: the level the note asks for (soft..shouting) against the player's calm reading. Half,
    when the original is given: the arousal model's calm-vs-intense for the player vs the actor,
    each against their own baseline (on RAVDESS cross-actor pairs this raised intensity matching
    from AUC 0.70 to 0.76)."""
    want = target_level(intensity, emotion)
    if base and "energy_level" in base:
        x = (level(att) - base["energy_level"]) + 0.5 * (att.stats["pitch_level"] - base["pitch_level"])
    else:
        # no loudness reference: the prosody arousal model, mapped onto the same scale
        x = (arousal(att, base) - 0.35) * 30
    got = next(i for i, (lo, hi) in enumerate(BANDS) if lo <= x < hi)
    lo, hi = BANDS[want]
    miss = max(lo - x, x - hi, 0.0)
    score = _sim(miss, 4.0)
    if ref is not None:
        score = 0.5 * score + 0.5 * _sim(arousal(att, base) - arousal(ref, ref_base), 0.2)
    if got == want:
        return Part(score, LEVELS[got], "")
    more = got < want
    return Part(score, "too soft" if more else "too loud",
                f"You were {LEVELS[got]}; this line needs {LEVELS[want].upper()}. "
                + ("Push your voice more." if more else "Bring it down a notch."))


def timing(att: Prosody, ref_start: float, ref_end: float, offset: float | None) -> Part:
    """`offset`: the scene time at which the recording started. Without it only the length counts."""
    ref_len = max(ref_end - ref_start, 0.2)
    if offset is None:
        r = att.duration / ref_len
        score = _sim(np.log(r), 0.25)
        if 0.8 <= r <= 1.25:
            return Part(score, "right length", "")
        return Part(score, "too long" if r > 1 else "too short",
                    f"Your line ran {r:.1f}x as long as the actor's. " + ("Speed up." if r > 1 else "Slow down a little."))
    start = offset + att.start
    end = start + att.duration
    ds, de = start - ref_start, end - ref_end
    score = 0.5 * _sim(ds, 0.35) + 0.5 * _sim(de, 0.5)
    if abs(ds) <= 0.3 and abs(de) <= 0.45:
        return Part(score, "on time", "")
    if abs(ds) > 0.3 and abs(ds) >= abs(de) * 0.7:
        return Part(score, "late" if ds > 0 else "early",
                    f"You came in {abs(ds):.1f} s {'late' if ds > 0 else 'early'}. "
                    + ("Start as soon as the bar turns red." if ds > 0 else "Wait for the red part of the bar."))
    return Part(score, "ran over" if de > 0 else "stopped early",
                f"You finished {abs(de):.1f} s {'after' if de > 0 else 'before'} the actor. "
                + ("Say it a bit faster." if de > 0 else "Stretch it to fill the line."))


def punch(att: Prosody, target_frac: float | None, word: str | None) -> Part | None:
    if target_frac is None:
        return None
    pos, strength = _stress(att)
    placement = _sim(pos - target_frac, 0.15)
    clear = min(1.0, max(0.0, strength / 1.5))        # a flat delivery has no punch anywhere
    score = 0.7 * placement + 0.3 * clear
    if placement >= 0.6 and clear >= 0.5:
        return Part(score, "on the word", "")
    if clear < 0.5:
        return Part(score, "flat", f"Nothing stood out. Hit {word or 'the gold word'} harder than the rest.")
    where = "early" if pos < target_frac else "late"
    return Part(score, "wrong word", f"Your strongest moment came too {where}. Put it on {word or 'the gold word'}.")


def _movement(slope: float) -> str:
    return "rises" if slope > 4 else "drops" if slope < -4 else "stays level"


def tone(att: Prosody, ref: Prosody) -> Part:
    want, got = ending_slope(ref), ending_slope(att)
    ending = _sim(got - want, 8.0)
    spread = _sim(att.stats["pitch_range"] - ref.stats["pitch_range"], 4.0)
    score = 0.65 * ending + 0.35 * spread
    if _movement(want) == _movement(got) and spread >= 0.5:
        return Part(score, "matched", "")
    if _movement(want) != _movement(got):
        return Part(score, f"{_movement(got)}",
                    f"The actor's voice {_movement(want)} at the end; yours {_movement(got)}.")
    flatter = att.stats["pitch_range"] < ref.stats["pitch_range"]
    return Part(score, "too flat" if flatter else "too wavy",
                "Let your voice move more, like the actor." if flatter else "Keep your voice steadier.")


def score_line(att: Prosody, ref: Prosody, card: dict, base: dict | None, offset: float | None,
               content: float | None = None, ref_base: dict | None = None) -> Direction:
    """`card`: the line's library card (start, onset, shape, intensity, emotion, stress)."""
    cfg = config()["game"]["score"]
    shape = card.get("shape") or {}
    ref_start = card["start"] + card.get("onset", 0.0)
    ref_end = ref_start + (shape.get("duration") or ref.duration)
    stressed = [w for w in shape.get("words", []) if w.get("stress")]
    frac = ((stressed[0]["start"] + stressed[0]["end"]) / 2 / shape["duration"]) if stressed and shape.get("duration") else None
    parts = {"timing": timing(att, ref_start, ref_end, offset),
             "volume": volume(att, base, card.get("intensity"), card.get("emotion"), ref, ref_base),
             "tone": tone(att, ref)}
    p = punch(att, frac, stressed[0]["w"] if stressed else None)
    if p:
        parts["punch"] = p
    if content is not None:
        c = float(min(max(content, 0.0), 1.0))
        parts["words"] = Part(c, "clear" if c >= 0.6 else "unclear", "" if c >= 0.6 else "Say the line as written.")
    w = {k: cfg["weights"][k] for k in parts}
    total = 100 * sum(w[k] * parts[k].score for k in parts) / sum(w.values())
    return Direction(round(total, 1), "Dub" if total >= cfg["dub_threshold"] else "Goose", parts,
                     {"ref_start": ref_start, "ref_end": ref_end, "offset": offset})
