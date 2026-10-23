"""Compare a dub attempt against the original line and score it out of 100.

The original and the dub are in different languages, so nothing here matches sounds or
syllables one-to-one. Each component compares things that survive translation:

- emotion:  how intense the delivery is (arousal), relative to each speaker's own neutral
            voice when a baseline exists. emotion2vec joins this slot in step 3 to judge
            *which* emotion; prosody alone cannot tell angry from happy.
- contour:  overall pitch/loudness shape in quarters of the line, and whether it ends rising or falling
- timing:   line length (matters for lip-sync), share of pauses, speaking rate
- emphasis: how strongly the main stress stands out, and whether it lands in the right place.
            Once ASR word timings exist (step 3), pass the time span of the target word that
            should carry the stress; until then the position is compared in thirds of the line.

Components whose models are not built yet (content needs ASR) are skipped and the
remaining weights are rescaled to 100.

This is the step 2 research score, used by the RAVDESS evaluations (scripts/eval_ravdess.py). The
game scores takes with backend/scoring/director.py, which checks the line's note instead: across
voices, most of these components were close to chance (AUC 0.54-0.57).
"""
from dataclasses import dataclass, field

import numpy as np

from backend.scoring.arousal import Baseline, arousal
from backend.scoring.prosody import HOP_S, Prosody, _zscore
from backend.settings import config

SEGMENTS = 4          # contour is compared per quarter of the line
ENDING_SHARE = 0.25   # final quarter decides rising vs falling


@dataclass
class Result:
    score: float                               # 0..100
    components: dict[str, float]               # similarity 0..1 per component
    verdict: str
    details: dict[str, float] = field(default_factory=dict)  # raw measurements for feedback


def _sim(diff: float, scale: float) -> float:
    """1 when equal, 0.5 at one scale unit apart, falling towards 0."""
    if not np.isfinite(diff):
        return 0.0
    return float(2 ** (-abs(diff) / max(scale, 1e-6)))


def _segment_means(x: np.ndarray, n: int = SEGMENTS) -> np.ndarray:
    return np.array([part.mean() for part in np.array_split(x, n)])


def ending_slope(p: Prosody) -> float:
    """Pitch slope over the last quarter of the line, semitones per second (+ rising, - falling)."""
    k = max(int(len(p.pitch) * ENDING_SHARE), 3)
    tail = p.pitch[-k:]
    return float(np.polyfit(np.arange(k) * HOP_S, tail, 1)[0])


def _stress(p: Prosody) -> tuple[float, float]:
    """(position of the strongest stress as a fraction of the line, how far it stands out)."""
    i = int(np.argmax(p.prominence))
    return i / max(len(p.prominence) - 1, 1), float(p.prominence[i] - np.median(p.prominence))


def emotion(ref_arousal: float, att_arousal: float) -> float:
    return _sim(att_arousal - ref_arousal, 0.2)


def contour(ref: Prosody, att: Prosody) -> float:
    shape = np.mean([
        _sim(np.abs(_segment_means(_zscore(ref.pitch)) - _segment_means(_zscore(att.pitch))).mean(), 0.5),
        _sim(np.abs(_segment_means(_zscore(ref.energy)) - _segment_means(_zscore(att.energy))).mean(), 0.5),
    ])
    ending = _sim(ending_slope(att) - ending_slope(ref), 8.0)
    return float(0.6 * shape + 0.4 * ending)


def timing(ref: Prosody, att: Prosody) -> float:
    duration = _sim(np.log(att.duration / ref.duration), 0.15)
    pauses = _sim(att.stats["pause_ratio"] - ref.stats["pause_ratio"], 0.08)
    # Languages differ in syllables per second, so rate gets a loose scale and a small weight.
    rate = _sim(np.log(max(att.stats["rate"], 0.1) / max(ref.stats["rate"], 0.1)), 0.3)
    return float(0.6 * duration + 0.2 * pauses + 0.2 * rate)


def emphasis(ref: Prosody, att: Prosody, target_span: tuple[float, float] | None = None) -> float:
    """`target_span`: (start, end) seconds in the attempt's audio of the word that should be stressed."""
    ref_pos, ref_strength = _stress(ref)
    att_pos, att_strength = _stress(att)
    strength = _sim(att_strength - ref_strength, 1.0)
    if target_span:
        peak_t = att.start + att_pos * att.duration
        lo, hi = target_span
        miss = max(lo - peak_t, peak_t - hi, 0.0)
        placement = _sim(miss, 0.1)
    else:
        placement = _sim(att_pos - ref_pos, 0.25)
    return float(0.5 * strength + 0.5 * placement)


def score(ref: Prosody, att: Prosody,
          ref_baseline: Baseline | None = None, att_baseline: Baseline | None = None,
          target_span: tuple[float, float] | None = None, arousal_model: dict | None = None,
          content: float | None = None) -> Result:
    """Score `att` (the dub) against `ref` (the original line).

    Baselines are each speaker's neutral voice (see arousal.baseline): for the player, their
    calibration line; for the original, the same speaker's other lines in the clip.
    `content` (0..1) is how closely the player's words match the translated line; it needs ASR,
    so callers without it leave it out and the other weights are rescaled.
    """
    cfg = config()["scoring"]
    ref_ar = arousal(ref, ref_baseline, arousal_model)
    att_ar = arousal(att, att_baseline, arousal_model)
    components = {
        "emotion": emotion(ref_ar, att_ar),
        "contour": contour(ref, att),
        "timing": timing(ref, att),
        "emphasis": emphasis(ref, att, target_span),
    }
    if content is not None:
        components["content"] = float(min(max(content, 0.0), 1.0))
    weights = {k: cfg["weights"][k] for k in components}
    total = 100 * sum(weights[k] * components[k] for k in weights) / sum(weights.values())
    ref_pos, ref_strength = _stress(ref)
    att_pos, att_strength = _stress(att)
    details = {
        "ref_arousal": ref_ar, "att_arousal": att_ar,
        "duration_ratio": att.duration / ref.duration,
        "ref_ending_slope": ending_slope(ref), "att_ending_slope": ending_slope(att),
        "ref_stress_pos": ref_pos, "att_stress_pos": att_pos,
        "ref_stress_strength": ref_strength, "att_stress_strength": att_strength,
    }
    if ref_baseline and att_baseline:
        details["pitch_raise_ref"] = ref.stats["pitch_level"] - ref_baseline["pitch_level"]
        details["pitch_raise_att"] = att.stats["pitch_level"] - att_baseline["pitch_level"]
    verdict = "Dub" if total >= cfg["dub_threshold"] else "Goose"
    return Result(round(total, 1), components, verdict, details)
