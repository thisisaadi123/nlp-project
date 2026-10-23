"""Can a small head on emotion2vec embeddings beat emotion2vec's own 7-way output on Indic speech?

Trained on RAVDESS only (English actors); tested on Rasa Hindi + Tamil, which is EVALUATION ONLY.
Run scripts/emotion_embed.py first (GPU); this part is CPU and takes seconds.

    python scripts/train_emotion_head.py            # compare the methods
    python scripts/train_emotion_head.py --save     # also save the head to backend/pipeline/speech/

Methods, all scored on the same Rasa utterances:
  e2v            emotion2vec's own head (what step 3 uses today)
  head           logistic regression on the embeddings
  head+calm k    the same, on embeddings minus the speaker's calm reference: the mean of k of their
                 neutral lines (other than the one scored). This is the saved head; a player's
                 calm reference is the mic-check reading.
  speaker avg k  embeddings minus the mean of k of the speaker's own lines, any emotion. Strong when
                 the lines are mixed, but it fails when they share an emotion (a character angry for
                 the whole scene): "same emotion" shows that case. Reported, not used.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.emotion import EMOTIONS  # noqa: E402

CACHE = ROOT / "data/cache"
HEAD_PATH = ROOT / "backend/pipeline/speech/emotion_head.npz"
HIGH = {"angry", "happy", "fear", "surprise"}
LOW = {"sad", "neutral"}          # disgust is ambiguous and left out of arousal, as before
N_FOLDS = 4


def load(corpus: str) -> dict:
    p = CACHE / f"e2v_{corpus}.npz"
    if not p.exists():
        sys.exit(f"{p} missing: run scripts/emotion_embed.py {corpus}")
    d = dict(np.load(p))
    d["y"] = np.array([EMOTIONS.index(e) for e in d["emotion"]])
    return d


def by_speaker(X: np.ndarray, speakers: np.ndarray, fn) -> np.ndarray:
    out = np.empty_like(X)
    for s in np.unique(speakers):
        m = speakers == s
        out[m] = fn(X[m])
    return out


def centre(X, speakers):
    return by_speaker(X, speakers, lambda x: x - x.mean(0))


NEUTRAL = EMOTIONS.index("neutral")


def minus_reference(d: dict, k: int | None, rng, pool: str = "calm") -> np.ndarray:
    """Each embedding minus the mean of k other lines of the same speaker (all of them if k is None):
    calm = their neutral lines, mixed = any lines, same = lines with the same emotion as this one."""
    X, n = d["emb"], len(d["y"])
    out = np.empty_like(X)
    for i in range(n):
        m = (d["speaker"] == d["speaker"][i]) & (np.arange(n) != i)
        if pool == "calm":
            m &= d["y"] == NEUTRAL
        elif pool == "same":
            m &= d["y"] == d["y"][i]
        idx = np.flatnonzero(m)
        if pool != "calm":                 # a speaker's own average includes the line itself
            idx = np.append(rng.choice(idx, k - 1, replace=False) if k else idx, i)
        elif k:
            idx = rng.choice(idx, k, replace=False)
        out[i] = X[i] - X[idx].mean(0)
    return out


def fit(X: np.ndarray, y: np.ndarray, l2: float) -> dict:
    """Multinomial logistic regression on standardised features, classes weighted to balance."""
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / std
    k, d = len(EMOTIONS), Z.shape[1]
    Y = np.eye(k)[y]
    w_cls = len(y) / (k * np.maximum(Y.sum(0), 1))
    sw = w_cls[y]

    def loss(w):
        W, b = w[:k * d].reshape(k, d), w[k * d:]
        logits = Z @ W.T + b
        logits -= logits.max(1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        p = np.exp(logp)
        g = (p - Y) * sw[:, None]
        val = -(sw * (logp * Y).sum(1)).sum() + l2 * (W ** 2).sum()
        return val, np.concatenate([(g.T @ Z + 2 * l2 * W).ravel(), g.sum(0)])

    w = minimize(loss, np.zeros(k * d + k), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    return {"mean": mean, "std": std, "W": w[:k * d].reshape(k, d), "b": w[k * d:]}


def predict(m: dict, X: np.ndarray) -> np.ndarray:
    logits = ((X - m["mean"]) / m["std"]) @ m["W"].T + m["b"]
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def auc(pos, neg) -> float:
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def scores(P: np.ndarray, d: dict) -> dict:
    pred = P.argmax(1)
    recall = [float((pred[d["y"] == c] == c).mean()) for c in range(len(EMOTIONS)) if (d["y"] == c).any()]
    hi = P[:, [EMOTIONS.index(e) for e in HIGH]].sum(1)
    is_hi = np.array([e in HIGH for e in d["emotion"]])
    sel = np.array([e in HIGH | LOW for e in d["emotion"]])
    return {"acc": float((pred == d["y"]).mean()), "macro": float(np.mean(recall)),
            "arousal_acc": float(((hi > 0.5) == is_hi)[sel].mean()),
            "arousal_auc": auc(hi[sel & is_hi], hi[sel & ~is_hi])}


def confusion(P, d) -> str:
    pred = P.argmax(1)
    rows = ["  true \\ pred " + " ".join(f"{e[:5]:>6}" for e in EMOTIONS) + "  recall"]
    for c, e in enumerate(EMOTIONS):
        m = d["y"] == c
        if m.any():
            counts = [int((pred[m] == p).sum()) for p in range(len(EMOTIONS))]
            rows.append(f"  {e:11} " + " ".join(f"{n:6d}" for n in counts) + f"  {counts[c] / m.sum() * 100:5.0f}%")
    return "\n".join(rows)


def pick_l2(train: dict, X: np.ndarray) -> float:
    """Leave-actors-out on RAVDESS only, so Rasa never influences the head."""
    actors = np.array([int(s.split("_")[1]) for s in train["speaker"]])
    best = None
    for l2 in (1, 10, 100, 1000):
        accs = []
        for f in range(N_FOLDS):
            tr, te = actors % N_FOLDS != f, actors % N_FOLDS == f
            m = fit(X[tr], train["y"][tr], l2)
            accs.append((predict(m, X[te]).argmax(1) == train["y"][te]).mean())
        print(f"  RAVDESS held-out actors, l2={l2:<5} accuracy {np.mean(accs) * 100:5.1f}%")
        if best is None or np.mean(accs) > best[1]:
            best = (l2, np.mean(accs))
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", default=["rasa_hi", "rasa_ta"])
    ap.add_argument("--reps", type=int, default=5, help="random draws of the k reference lines")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    train = load("ravdess")
    print(f"Training on RAVDESS: {len(train['y'])} clips, {len(set(train['speaker']))} actors")
    X_calm = minus_reference(train, None, rng)          # each actor's neutral + calm clips
    l2 = pick_l2(train, X_calm)
    head = fit(train["emb"], train["y"], l2)
    head_calm = fit(X_calm, train["y"], l2)
    head_avg = fit(centre(train["emb"], train["speaker"]), train["y"], l2)

    def avg(model, k, pool):
        runs = [scores(predict(model, minus_reference(d, k, rng, pool)), d) for _ in range(args.reps)]
        return {m: float(np.mean([r[m] for r in runs])) for m in runs[0]}

    results = {}
    for corpus in args.test:
        d = load(corpus)
        R = {"e2v": scores(d["probs"], d), "head": scores(predict(head, d["emb"]), d)}
        for k in (1, 3, 5):
            R[f"head+calm {k}"] = avg(head_calm, k, "calm")
        for k in (5, 20):
            R[f"speaker avg {k}"] = avg(head_avg, k, "mixed")
            R[f"  same emotion {k}"] = avg(head_avg, k, "same")
        results[corpus] = R
        print(f"\n===== {corpus}: {len(d['y'])} utterances, {len(set(d['speaker']))} speakers (chance 14%)")
        print(f"  {'method':18} {'7-way':>7} {'balanced':>9} {'arousal':>8} {'AUC':>6}")
        for k, s in R.items():
            print(f"  {k:18} {s['acc'] * 100:6.1f}% {s['macro'] * 100:8.1f}% {s['arousal_acc'] * 100:7.1f}% {s['arousal_auc']:6.3f}")
        print(f"\n  confusion, e2v (today):\n{confusion(d['probs'], d)}")
        print(f"\n  confusion, head+calm 3:\n{confusion(predict(head_calm, minus_reference(d, 3, rng)), d)}")

    (CACHE / "emotion_head_results.json").write_text(json.dumps(results, indent=1))
    if args.save:
        np.savez_compressed(HEAD_PATH, W=head_calm["W"].astype(np.float32), b=head_calm["b"].astype(np.float32),
                            mean=head_calm["mean"].astype(np.float32), std=head_calm["std"].astype(np.float32),
                            emotions=EMOTIONS, info=json.dumps({
                                "trained_on": "RAVDESS 24 actors; emotion2vec+ large embeddings minus the actor's "
                                              "neutral+calm mean", "l2": l2, "date": date.today().isoformat(),
                                "rasa_eval": results}))
        print(f"\nSaved {HEAD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
