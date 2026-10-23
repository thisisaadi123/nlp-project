"""A spoken example of each dubbed line in the player's language (Sarvam Bulbul text-to-speech).

The player hears the original in the other language; this lets them hear their own line once
before recording. Made once when the library is built and cached in data/cache/tts/, so players
cost nothing. Each character gets their own voice of the right gender. When the natural reading
is much longer than the actor's line, it is read again faster so it fits the scene.
"""
import base64
import hashlib
import io
import json
import time

import httpx
import numpy as np
import soundfile as sf

from backend import sarvam_http
from backend.settings import ROOT, config
CACHE = ROOT / "data/cache/tts"


class VoiceError(RuntimeError):
    pass


def available() -> bool:
    return sarvam_http.available() and config()["game"].get("voice", {}).get("enabled", False)


def cast(speakers: dict, order: list[str]) -> dict[str, str]:
    """speaker id -> Bulbul voice: distinct voices per character, by gender, in order of appearance."""
    cfg = config()["game"]["voice"]
    pools = {"male": list(cfg["male"]), "female": list(cfg["female"])}
    used = {"male": 0, "female": 0}
    voices = {}
    for sid in order:
        gender = ((speakers.get(sid) or {}).get("gender") or "male").lower()
        gender = gender if gender in pools else "male"
        voices[sid] = pools[gender][used[gender] % len(pools[gender])]
        used[gender] += 1
    return voices


def _tts(text: str, lang: str, speaker: str, pace: float) -> bytes:
    cfg = config()["game"]["voice"]
    body = {"text": text, "target_language_code": f"{lang}-IN", "model": cfg.get("model", "bulbul:v3"),
            "speaker": speaker, "pace": round(pace, 2)}
    key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = CACHE / f"{key}.wav"
    if path.exists():
        return path.read_bytes()
    for attempt in range(6):
        try:
            r = sarvam_http.post("/text-to-speech", json=body, timeout=60)
        except sarvam_http.NoSarvamKey as e:
            raise VoiceError(str(e)) from e
        except httpx.HTTPError as e:
            r, err = None, str(e)
        if r is not None and r.status_code == 200:
            break
        if r is not None and r.status_code not in (429,) and r.status_code < 500:
            raise VoiceError(f"Sarvam TTS HTTP {r.status_code}: {r.text[:200]}")
        err = err if r is None else f"HTTP {r.status_code}"
        time.sleep(2 * (attempt + 1))        # rate limit or a hiccup: wait and try again
    else:
        raise VoiceError(f"Sarvam TTS kept failing ({err})")
    wav = base64.b64decode(r.json()["audios"][0])
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav)
    return wav


def _seconds(wav: bytes) -> float:
    info = sf.info(io.BytesIO(wav))
    return info.frames / info.samplerate


def speak(text: str, lang: str, speaker: str, target_seconds: float | None = None) -> bytes:
    """WAV bytes of `text` read by `speaker`; re-read faster if it runs well past the actor's line."""
    wav = _tts(text, lang, speaker, 1.0)
    if target_seconds:
        took = _seconds(wav)
        if took > 1.3 * target_seconds:
            wav = _tts(text, lang, speaker, float(np.clip(took / target_seconds, 1.0, 2.0)))
    return wav
