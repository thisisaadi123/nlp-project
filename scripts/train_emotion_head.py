"""Can a small head on emotion2vec embeddings beat emotion2vec's own 7-way output on Indic speech?

Trained on RAVDESS only (English actors); tested on Rasa Hindi + Tamil, which is EVALUATION ONLY.
Run scripts/emotion_embed.py first (GPU); this part is CPU and takes seconds.

    python scripts/train_emotion_head.py            # compare the methods
    python scripts/train_emotion_head.py --save     # also save the head to backend/pipeline/speech/

Methods, all scored on the same Rasa utterances:
  e2v           emotion2vec's own head (what step 3 uses today)
  e2v+speaker   the same probabilities divided by the speaker's average probabilities: if a voice
                always sounds "neutral" to the model, that bias is removed. Needs no labels, only
                several lines from the same speaker (a clip's lines, or a player's takes).
  head          logistic regression on the embeddings
  head+speaker  the same, on embeddings minus the speaker's average embedding
  blend         e2v+speaker and head+speaker averaged in log space
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


def debias(P, speakers):
    """Divide by the speaker's mean probabilities, then renormalise."""
    def f(p):
        q = p / (p.mean(0) + 1e-6)
        return q / q.sum(1, keepdims=True)
    return by_speaker(P, speakers, f)


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


def blend(*Ps):
    logp = np.mean([np.log(P + 1e-6) for P in Ps], axis=0)
    p = np.exp(logp - logp.max(1, keepdims=True))
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


def pick_l2(train: dict) -> float:
    """Leave-actors-out on RAVDESS only, so Rasa never influences the head."""
    actors = np.array([int(s.split("_")[1]) for s in train["speaker"]])
    X = centre(train["emb"], train["speaker"])
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
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    train = load("ravdess")
    print(f"Training on RAVDESS: {len(train['y'])} clips, {len(set(train['speaker']))} actors")
    l2 = pick_l2(train)
    head = fit(train["emb"], train["y"], l2)
    head_spk = fit(centre(train["emb"], train["speaker"]), train["y"], l2)

    results = {}
    for corpus in args.test:
        d = load(corpus)
        P = {"e2v": d["probs"], "e2v+speaker": debias(d["probs"], d["speaker"]),
             "head": predict(head, d["emb"]), "head+speaker": predict(head_spk, centre(d["emb"], d["speaker"]))}
        P["blend"] = blend(P["e2v+speaker"], P["head+speaker"])
        results[corpus] = {k: scores(v, d) for k, v in P.items()}
        print(f"\n===== {corpus}: {len(d['y'])} utterances, {len(set(d['speaker']))} speakers (chance 14%)")
        print(f"  {'method':14} {'7-way':>7} {'balanced':>9} {'arousal':>8} {'AUC':>6}")
        for k, s in results[corpus].items():
            print(f"  {k:14} {s['acc'] * 100:6.1f}% {s['macro'] * 100:8.1f}% {s['arousal_acc'] * 100:7.1f}% {s['arousal_auc']:6.3f}")
        best = max(results[corpus], key=lambda k: results[corpus][k]["macro"])
        print(f"\n  confusion, e2v (today):\n{confusion(P['e2v'], d)}")
        if best != "e2v":
            print(f"\n  confusion, {best}:\n{confusion(P[best], d)}")

    (CACHE / "emotion_head_results.json").write_text(json.dumps(results, indent=1))
    if args.save:
        np.savez_compressed(HEAD_PATH, W=head_spk["W"].astype(np.float32), b=head_spk["b"].astype(np.float32),
                            mean=head_spk["mean"].astype(np.float32), std=head_spk["std"].astype(np.float32),
                            emotions=EMOTIONS, info=json.dumps({"trained_on": "RAVDESS, speaker-centred emotion2vec+ "
                                                                "large embeddings", "l2": l2, "date": date.today().isoformat(),
                                                                "rasa_eval": results}))
        print(f"\nSaved {HEAD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
