"""Arousal: how intense a delivery is (shouting, excited) vs calm (soft, flat), as a probability 0..1.

Prosody cannot tell angry from happy, but it reliably tells intense from calm across
different speakers (84% on held-out RAVDESS actors with a speaker baseline, 80% without).
Two logistic models are trained by scripts/train_arousal.py and stored in arousal_model.json:

- "relative": features minus the speaker's own neutral baseline (preferred)
- "raw": absolute features, used when no baseline recording exists for that speaker
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from backend.scoring.prosody import Prosody

FEATURES = ["pitch_level", "pitch_range", "pitch_sd", "energy_range", "spectral_tilt", "hnr", "rate"]
MODEL_PATH = Path(__file__).with_name("arousal_model.json")

Baseline = dict[str, float]


def baseline(neutral: list[Prosody]) -> Baseline:
    """A speaker's neutral voice: mean statistics over one or more calm, normal recordings."""
    return {k: float(np.mean([p.stats[k] for p in neutral])) for k in FEATURES}


def features(stats: dict, base: Baseline | None = None) -> np.ndarray:
    x = np.array([stats[k] for k in FEATURES])
    return x - np.array([base[k] for k in FEATURES]) if base else x


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> dict:
    """L2-regularised logistic regression on standardised features."""
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / std

    def loss(w):
        z = Z @ w[1:] + w[0]
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[1:] ** 2)

    w = minimize(loss, np.zeros(Z.shape[1] + 1), method="L-BFGS-B").x
    return {"mean": mean.tolist(), "std": std.tolist(), "bias": float(w[0]), "weights": w[1:].tolist()}


def predict(params: dict, x: np.ndarray) -> float:
    z = (x - np.array(params["mean"])) / np.array(params["std"])
    return float(1 / (1 + np.exp(-(z @ np.array(params["weights"]) + params["bias"]))))


@lru_cache
def load_model(path: str = str(MODEL_PATH)) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/train_arousal.py")
    return json.loads(p.read_text())


def arousal(p: Prosody, base: Baseline | None = None, model: dict | None = None) -> float:
    model = model or load_model()
    if base:
        return predict(model["relative"], features(p.stats, base))
    return predict(model["raw"], features(p.stats))
