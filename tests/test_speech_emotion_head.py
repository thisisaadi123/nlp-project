import numpy as np
import pytest

from backend.pipeline.speech.emotion import EMOTIONS, HEAD_PATH, head_probs


def test_head_is_shipped_and_matches_the_labels():
    d = np.load(HEAD_PATH)
    assert list(d["emotions"]) == EMOTIONS and d["W"].shape == (len(EMOTIONS), 1024)


def test_head_probs_is_a_distribution():
    rng = np.random.default_rng(0)
    p = head_probs(rng.normal(size=1024), [rng.normal(size=1024), rng.normal(size=1024)])
    assert set(p) == set(EMOTIONS) and abs(sum(p.values()) - 1) < 1e-3


def test_own_calm_reference_reads_as_neutral():
    """A recording that is exactly the speaker's calm reference has nothing left after centring."""
    calm = np.random.default_rng(1).normal(size=1024)
    p = head_probs(calm, [calm])
    assert max(p, key=p.get) == "neutral"


def test_head_probs_needs_a_reference():
    with pytest.raises(ValueError):
        head_probs(np.zeros(1024), [])
