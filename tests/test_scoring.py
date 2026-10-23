"""Regression tests for step 2's scoring core, on synthetic voices (no recordings needed)."""
import numpy as np
import pytest

from backend.scoring.arousal import baseline
from backend.scoring.compare import _sim, score
from backend.scoring.feedback import tips
from backend.scoring.prosody import SR, extract


def voice(f0_start: float, f0_end: float, seconds: float, loud: float = 0.3, seed: int = 0) -> np.ndarray:
    """A buzzy voice-like tone with a pitch glide and syllable-like loudness bumps."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    f0 = np.linspace(f0_start, f0_end, len(t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    wave = sum(np.sin(k * phase) / k for k in range(1, 8))
    envelope = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 2.5 * t))
    pad = np.zeros(int(0.3 * SR))
    sig = loud * envelope * wave / 2 + 0.001 * rng.standard_normal(len(t))
    return np.concatenate([pad, sig, pad]).astype(np.float32)


def test_sim():
    assert _sim(0, 1) == 1.0
    assert _sim(1, 1) == 0.5
    assert _sim(float("nan"), 1) == 0.0


def test_extract_finds_the_line_and_its_pitch():
    p = extract(voice(150, 150, 1.5))
    assert abs(p.duration - 1.5) < 0.15
    assert abs(p.start - 0.3) < 0.1
    assert abs(p.stats["pitch_level"] - 12 * np.log2(150 / 100)) < 0.5


def test_ending_direction_is_measured():
    from backend.scoring.compare import ending_slope
    assert ending_slope(extract(voice(120, 240, 1.5))) > 4    # an octave rise, like a question
    assert ending_slope(extract(voice(240, 120, 1.5))) < -4


def test_identical_takes_score_as_a_dub():
    p = extract(voice(160, 120, 1.6))
    r = score(p, p)
    assert r.score > 95 and r.verdict == "Dub"


def test_much_longer_attempt_scores_lower_and_gets_a_timing_tip():
    ref = extract(voice(160, 120, 1.5))
    slow = extract(voice(160, 120, 3.0, seed=1))
    r = score(ref, slow)
    assert r.components["timing"] < 0.5
    assert any("long" in t for t in tips(r))


def test_baselines_are_accepted():
    ref, att = extract(voice(150, 150, 1.5)), extract(voice(220, 220, 1.5, seed=2))
    r = score(ref, att, baseline([ref]), baseline([att]))
    assert 0 <= r.score <= 100 and "pitch_raise_att" in r.details


def test_silence_is_rejected():
    with pytest.raises(ValueError):
        extract(np.zeros(SR, dtype=np.float32))
