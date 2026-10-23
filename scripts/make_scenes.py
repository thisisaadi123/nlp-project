"""Generated two-speaker scenes: exact ground truth for diarization and music separation.

Dialogue comes from data/scene_scripts.json (two named Indic Parler-TTS voices per language,
emotion set per line through the style description). Every line is checked with ASR, because
TTS can mis-speak or read its description aloud; a line is retried up to 3 times and dropped if
it never reaches CER <= 0.2. Each scene is saved clean and with background music mixed in at
0, -5 and -10 dB music-to-speech (RFM CC BY 3.0 / public-domain tracks, scripts/fetch_music.py).

    python scripts/fetch_music.py                     # once
    python scripts/make_scenes.py --lang hi ta

Output (git-ignored): data/clips/scenes/<lang>_<nn>_{clean,m0,m-5,m-10}.wav + <lang>_<nn>.json
"""
import argparse
import json
import sys
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.asr import ASR, SR  # noqa: E402
from backend.pipeline.tts.parler import ParlerTTS, describe, trim  # noqa: E402
from scripts.eval_asr import normalize  # noqa: E402

SCRIPTS = ROOT / "data/scene_scripts.json"
MUSIC = ROOT / "data/datasets/music"
OUT = ROOT / "data/clips/scenes"
LEVELS = [0, -5, -10]                 # music-to-speech ratio, dB
MAX_CER = 0.2
LICENSES = {"rfm": "CC BY 3.0 (Kevin MacLeod, incompetech.com)", "fma-western-art": "public domain"}


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2)) + 1e-9)


def music_bed(n: int, rng) -> tuple[np.ndarray, dict]:
    tracks = sorted(MUSIC.glob("*/*.wav"))
    if not tracks:
        sys.exit(f"No music in {MUSIC}. Run scripts/fetch_music.py first.")
    path = tracks[rng.integers(len(tracks))]
    y, sr = sf.read(path, dtype="float32")
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    start = int(rng.integers(0, max(1, len(y) - n)))
    y = np.tile(y, int(np.ceil((start + n) / len(y))))[start:start + n]
    return y, {"track": path.name, "source": path.parent.name, "license": LICENSES[path.parent.name],
               "offset_s": round(start / SR, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["hi", "ta"])
    args = ap.parse_args()
    scripts = json.loads(SCRIPTS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    tts, asr = ParlerTTS(), ASR()
    rng = np.random.default_rng(0)
    for lang in args.lang:
        voices = scripts["voices"][lang]
        for k, scene in enumerate(scripts[lang]):
            name = f"{lang}_{k:02d}"
            pieces, lines, dropped = [], [], []
            for spk, emo, text in scene["lines"]:
                best = None
                for _ in range(3):
                    y = trim(tts.say(text, describe(voices[spk], emo)))
                    if not len(y):
                        continue
                    cer = jiwer.cer(normalize(text), normalize(asr.transcribe(y, lang).text) or "-")
                    if best is None or cer < best[1]:
                        best = (y, cer)
                    if cer <= MAX_CER:
                        break
                if best is None or best[1] > MAX_CER:
                    dropped.append(text); continue
                pieces.append(best[0]); lines.append({"speaker": spk, "emotion": emo, "text": text,
                                                      "asr_cer": round(best[1], 3)})
            if len(lines) < 3 or len({l["speaker"] for l in lines}) < 2:
                print(f"{name}: skipped ({len(lines)} usable lines)"); continue

            t, parts = 0.6, [np.zeros(int(0.6 * SR), np.float32)]
            for i, (y, line) in enumerate(zip(pieces, lines)):
                line.update(line_id=i, start=round(t, 3), end=round(t + len(y) / SR, 3))
                parts.append(y); t += len(y) / SR
                gap = rng.uniform(0.25, 0.7) if i + 1 < len(lines) else 0.6
                parts.append(np.zeros(int(gap * SR), np.float32)); t += gap
            clean = np.concatenate(parts)
            clean = clean / max(np.abs(clean).max(), 1e-6) * 0.5
            speech_rms = rms(np.concatenate([clean[int(l["start"] * SR):int(l["end"] * SR)] for l in lines]))
            clean_out = clean + rng.standard_normal(len(clean)).astype(np.float32) * speech_rms * 10 ** (-55 / 20)
            sf.write(OUT / f"{name}_clean.wav", clean_out, SR)

            bed, music = music_bed(len(clean), rng)
            for level in LEVELS:
                mix = clean_out + bed * (speech_rms / rms(bed)) * 10 ** (level / 20)
                sf.write(OUT / f"{name}_m{level}.wav", mix / max(1.0, np.abs(mix).max()), SR)

            meta = {"lang": lang, "title": scene["title"], "voices": voices, "duration": round(len(clean) / SR, 2),
                    "lines": lines, "dropped_lines": dropped, "music": music, "levels_db": LEVELS}
            (OUT / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{name}: {len(lines)}/{len(scene['lines'])} lines, {meta['duration']}s, music {music['track']}")


if __name__ == "__main__":
    main()
