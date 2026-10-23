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
import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.asr import ASR, SR  # noqa: E402
from scripts.eval_asr import fleurs_items, normalize  # noqa: E402

OUT = ROOT / "data/cache/align_synth"
VOICE = {"hi": "Rohit", "ta": "Jaya"}
DESC = "{} speaks at a moderate pace in a clear, neutral voice. The recording is very clear with no background noise."
EDGE_S = 0.5
MAX_WORDS = 8


class TTS:
    def __init__(self):
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        name = "ai4bharat/indic-parler-tts"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sr = self.model.config.sampling_rate

    def say(self, text: str, desc: str) -> np.ndarray:
        d = self.desc_tok(desc, return_tensors="pt").to("cuda")
        p = self.tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            audio = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                        prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        y = np.atleast_1d(audio.float().cpu().numpy().squeeze())
        return librosa.resample(y, orig_sr=self.sr, target_sr=SR) if y.size > 1000 else np.zeros(0, np.float32)


def trim(y: np.ndarray, below_peak_db: float = 35) -> np.ndarray:
    if len(y) < 800:
        return y[:0]
    db =20 * np.log10(librosa.feature.rms(y=y, frame_length=160, hop_length=80)[0] + 1e-8)
    on = np.flatnonzero(db > db.max() - below_peak_db)
    return y[on[0] * 80: (on[-1] + 2) * 80] if len(on) else y[:0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["hi", "ta"])
    ap.add_argument("--n", type=int, default=8, help="sentences per language")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tts, asr = TTS(), ASR()
    rng = np.random.default_rng(0)
    for lang in args.lang:
        for k, it in enumerate(fleurs_items(lang, args.n, seed=2)):
            words = normalize(it["ref"]).split()[:MAX_WORDS]
            pieces, kept = [], []
            for w in words:
                for _ in range(3):                                   # TTS can mis-speak; retry
                    y = trim(tts.say(w, DESC.format(VOICE[lang])))
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
