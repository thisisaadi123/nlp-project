"""A word-timing test with exact ground truth (nobody records anything, so we build one).

Each word of a FLEURS sentence is synthesised separately with Indic Parler-TTS, checked with
ASR (TTS can mis-speak), trimmed to its audible part and joined to the next with no gap. Every
word boundary is therefore known exactly. 0.5 s of quiet noise before and after tests where a
line starts and ends. Output: data/cache/align_synth/<lang>_<k>.wav + .json (git-ignored).

    python scripts/make_align_test.py --lang hi ta --n 8
"""
import argparse
import json
import sys
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.asr import ASR, SR  # noqa: E402
from backend.pipeline.tts.parler import ParlerTTS, describe, trim  # noqa: E402
from scripts.eval_asr import fleurs_items, normalize  # noqa: E402

OUT = ROOT / "data/cache/align_synth"
VOICE = {"hi": "Rohit", "ta": "Jaya"}
EDGE_S = 0.5
MAX_WORDS = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["hi", "ta"])
    ap.add_argument("--n", type=int, default=8, help="sentences per language")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tts, asr = ParlerTTS(), ASR()
    rng = np.random.default_rng(0)
    for lang in args.lang:
        for k, it in enumerate(fleurs_items(lang, args.n, seed=2)):
            words = normalize(it["ref"]).split()[:MAX_WORDS]
            pieces, kept = [], []
            for w in words:
                for _ in range(3):                                   # TTS can mis-speak; retry
                    y = trim(tts.say(w, describe(VOICE[lang])))
                    if len(y) > 0.08 * SR and jiwer.cer(w, normalize(asr.transcribe(y, lang).text) or "-") <= 0.5:
                        pieces.append(y); kept.append(w); break
            if len(kept) < 3:
                continue
            level = max(np.abs(np.concatenate(pieces)).max(), 1e-3)
            noise = lambda n: (rng.standard_normal(n) * level * 10 ** (-60 / 20)).astype(np.float32)
            t, truth = EDGE_S, []
            for w, y in zip(kept, pieces):
                truth.append({"w": w, "start": round(t, 4), "end": round(t + len(y) / SR, 4)})
                t += len(y) / SR
            y = np.concatenate([noise(int(EDGE_S * SR)), *pieces, noise(int(EDGE_S * SR))])
            y = y + noise(len(y)) * 0.1
            name = f"{lang}_{k:02d}"
            sf.write(OUT / f"{name}.wav", y, SR)
            (OUT / f"{name}.json").write_text(json.dumps({"lang": lang, "source": it["id"], "words": truth},
                                                          ensure_ascii=False, indent=1))
            print(f"{name}: {len(kept)}/{len(words)} words, {len(y)/SR:.1f}s")


if __name__ == "__main__":
    main()
