"""Train the arousal models on RAVDESS and save backend/scoring/arousal_model.json.

    python scripts/train_arousal.py

Accuracy is measured with leave-actors-out cross-validation (the model never sees the
test actors), then the final models are fit on all actors.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.scoring.arousal import FEATURES, MODEL_PATH, fit, predict  # noqa: E402
from scripts.ravdess_data import N_FOLDS, arousal_xy, fold_of, load  # noqa: E402


def fold_models(items: list[dict]) -> dict[int, dict]:
    """One model pair per fold, trained without that fold's actors. Used by eval_ravdess.py."""
    models = {}
    for f in range(N_FOLDS):
        train = [it for it in items if fold_of(it["actor"]) != f]
        models[f] = {kind: fit(*arousal_xy(train, kind == "relative")) for kind in ("relative", "raw")}
    return models


def main() -> None:
    items = load()
    models = fold_models(items)
    cv = {}
    for kind in ("relative", "raw"):
        correct = total = 0
        for f in range(N_FOLDS):
            X, y = arousal_xy([it for it in items if fold_of(it["actor"]) == f], kind == "relative")
            p = np.array([predict(models[f][kind], x) for x in X])
            correct += int(np.sum((p > 0.5) == y))
            total += len(y)
        cv[kind] = round(correct / total, 4)
        print(f"{kind:8s} held-out-actor accuracy: {cv[kind]:.1%}")

    final = {kind: fit(*arousal_xy(items, kind == "relative")) for kind in ("relative", "raw")}
    final["info"] = {
        "trained_on": "RAVDESS speech, 24 actors; high = angry/happy/fearful/surprised, low = neutral/calm/sad",
        "features": FEATURES,
        "cv_accuracy_leave_actors_out": cv,
        "date": date.today().isoformat(),
    }
    MODEL_PATH.write_text(json.dumps(final, indent=2))
    print(f"Saved {MODEL_PATH.relative_to(MODEL_PATH.parents[2])}")
    w = dict(zip(FEATURES, np.round(final["relative"]["weights"], 2)))
    print("relative-model weights (standardised):", w)


if __name__ == "__main__":
    main()
