"""Generate Hindi test takes with Gemini TTS: one line in several emotions, two voices.

    python scripts/tts_takes.py          # writes data/clips/step2/tts/<voice>_<take>.wav

Synthetic acting is weaker than a real actor, so this is a smoke test for Indic speech,
not an accuracy measurement. Real emotional Indic speech for evaluation comes from datasets (step 3).

TTS models sometimes read the style instruction aloud ("Say angrily, भैया..."), which
silently ruins the take. Every take is transcribed with Groq Whisper and regenerated if
any English came out. gemini-2.5-flash-preview-tts with short "Say ...:" prompts was the
only combination that reliably spoke just the line (gemini-3.8-flash-tts read prompts out).
"""
import base64
import io
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.settings import secret  # noqa: E402

OUT = ROOT / "data/clips/step2/tts"
LINE = "भैया, ये क्या कर दिया?"
CALIBRATION = "आज मौसम काफ़ी अच्छा है, और मैं शाम को बाज़ार जाऊँगा।"
VOICES = {"male": "Puck", "female": "Kore"}
# take: (text, adverb for "Say ...:" prompts, tag for "[...] text" prompts)
TAKES = {
    "angry_1": (LINE, "angrily", "angry, shouting"),
    "angry_2": (LINE, "angrily", "angry, shouting"),
    "sad": (LINE, "sadly", "sad, soft"),
    "flat": (LINE, "in a flat monotone", "flat, monotone"),
    "neutral_calibration": (CALIBRATION, "calmly", "calm"),
}
# Each model has its own free daily quota; when one runs out the next is used.
MODELS = [
    ("gemini-2.5-flash-preview-tts", lambda text, adverb, tag: f"Say {adverb}: {text}"),
    ("gemini-3.8-flash-tts", lambda text, adverb, tag: f"[{tag}] {text}"),
]


class DailyQuotaExhausted(Exception):
    pass


def spoken_text(audio: np.ndarray, rate: int) -> str:
    from groq import Groq
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV")
    client = Groq(api_key=secret("GROQ_API_KEY"))
    return client.audio.transcriptions.create(file=("take.wav", buf.getvalue()), model="whisper-large-v3").text.strip()


def synthesize(model: str, prompt: str, voice: str) -> tuple[np.ndarray, int]:
    for attempt in range(5):
        try:
            r = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": secret("GEMINI_API_KEY")}, timeout=120,
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"responseModalities": ["AUDIO"],
                                           "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}},
            )
        except httpx.TransportError:  # network blip
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            if "PerDay" in r.text:
                raise DailyQuotaExhausted(model)
            time.sleep(15 * (attempt + 1))  # per-minute limit: back off and retry
            continue
        r.raise_for_status()
        part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
        raw = base64.b64decode(part["data"])
        if "wav" in part["mimeType"]:
            return sf.read(io.BytesIO(raw), dtype="float32")
        rate = int(part["mimeType"].split("rate=")[1].split(";")[0])  # audio/L16;codec=pcm;rate=24000
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768, rate
    raise RuntimeError("Gemini TTS failed 5 times (rate limit or network)")


def clean_take(text: str, adverb: str, tag: str, voice: str, tries: int = 3) -> tuple[np.ndarray, int, str]:
    """Synthesize until the audio contains only the Hindi line, no spoken instruction."""
    for model, make_prompt in MODELS:
        prompt = make_prompt(text, adverb, tag)
        try:
            for _ in range(tries):
                audio, rate = synthesize(model, prompt, voice)
                heard = spoken_text(audio, rate)
                if not re.search(r"[A-Za-z]{2,}", heard):
                    return audio, rate, f"{heard}  [{model}]"
                print(f"  rejected (instruction read aloud): {heard}")
        except DailyQuotaExhausted:
            print(f"  {model}: free daily quota used up, trying the next model")
    raise RuntimeError(f"No TTS model produced a clean take for: {text}")


def main() -> None:
    if not secret("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY missing in .env")
    OUT.mkdir(parents=True, exist_ok=True)
    for sex, voice in VOICES.items():
        for take, (line, adverb, tag) in TAKES.items():
            path = OUT / f"{sex}_{take}.wav"
            if path.exists():
                continue
            audio, rate, text = clean_take(line, adverb, tag, voice)
            sf.write(path, audio, rate)
            print(f"wrote {path.relative_to(ROOT)} ({len(audio) / rate:.1f} s): {text}")


if __name__ == "__main__":
    main()
