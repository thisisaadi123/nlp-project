"""Compare a dub attempt against the original line and score it out of 100.

Each component gives a similarity in [0, 1]. Components that need models not built yet
(content needs ASR) are skipped and the remaining weights are rescaled to 100. Until
emotion2vec is in, the emotion slot is filled by `expression`: global prosodic statistics
that track arousal (how intense the delivery is) but not valence (happy vs angry).
"""
from dataclasses import dataclass

import librosa
import numpy as np

from backend.scoring.prosody import HOP_S, Prosody, _zscore
from backend.settings import config

# Typical spread of each statistic across RAVDESS recordings; a difference of one scale
# unit halves that feature's similarity. Re-fit with scripts/eval_ravdess.py --fit-scales.
EXPRESSION_SCALES = {
    "pitch_range": 3.0,
    "pitch_sd": 1.0,
    "energy_range": 4.0,
    "spectral_tilt": 3.0,
    "hnr": 2.5,
}


@dataclass
class Result:
    score: float
    components: dict[str, float]     # similarity 0..1 per component
    verdict: str


def _sim(diff: float, scale: float) -> float:
    """1 when equal, 0.5 at one scale unit apart, falling towards 0."""
    return float(2 ** (-abs(diff) / scale))


def contour(ref: Prosody, att: Prosody) -> float:
    """Shape of the pitch and loudness curves, aligned in time with DTW."""
    step = 2  # compare on a 20 ms grid; plenty for intonation and keeps DTW fast
    x = np.vstack([_zscore(ref.pitch[::step]), _zscore(ref.energy[::step])])
    y = np.vstack([_zscore(att.pitch[::step]), _zscore(att.energy[::step])])
    cost, path = librosa.sequence.dtw(X=x, Y=y, metric="euclidean",
                                      global_constraints=True, band_rad=0.3)
    mean_cost = cost[-1, -1] / len(path)
    return _sim(mean_cost, 0.6)


def timing(ref: Prosody, att: Prosody) -> float:
    """Overall length, speaking rate and how much of the line is pause."""
    dur = _sim(np.log(att.duration / ref.duration), 0.15)
    rate = _sim(np.log(max(att.stats["rate"], 0.1) / max(ref.stats["rate"], 0.1)), 0.2)
    pause = _sim(att.stats["pause_ratio"] - ref.stats["pause_ratio"], 0.08)
    return 0.5 * dur + 0.25 * rate + 0.25 * pause


def emphasis(ref: Prosody, att: Prosody) -> float:
    """Is the strongest stress in the same part of the line, and as strong?

    Without word alignment this uses relative position in time; once ASR word timings
    exist this should compare the stressed word against the mapped target word.
    """
    def strongest(p: Prosody) -> tuple[float, float]:
        i = int(np.argmax(p.prominence))
        return i / max(len(p.prominence) - 1, 1), float(p.prominence[i])

    ref_pos, ref_height = strongest(ref)
    att_pos, att_height = strongest(att)
    return _sim(att_pos - ref_pos, 0.12) * _sim(att_height - ref_height, 1.0)


def expression(ref: Prosody, att: Prosody) -> float:
    """Intensity of delivery: pitch range, loudness range, vocal effort, voice quality."""
    sims = [_sim(att.stats[k] - ref.stats[k], s) for k, s in EXPRESSION_SCALES.items()]
    return float(np.mean(sims))


def score(ref: Prosody, att: Prosody) -> Result:
    cfg = config()["scoring"]
    weights = cfg["weights"]
    components = {
        "contour": contour(ref, att),
        "timing": timing(ref, att),
        "emphasis": emphasis(ref, att),
        "emotion": expression(ref, att),
    }
    used = {k: weights[k] for k in components}
    total = 100 * sum(used[k] * components[k] for k in used) / sum(used.values())
    verdict = "Dub" if total >= cfg["dub_threshold"] else "Goose"
    return Result(round(total, 1), components, verdict)
