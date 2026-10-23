"""Speech-to-text for players' takes through Sarvam's API (Indian languages, spoken forms)."""
import io
import logging

import httpx
import numpy as np
import soundfile as sf

from backend import sarvam_http
from backend.settings import config

MAX_SECONDS = 30            # the real-time endpoint's limit
log = logging.getLogger(__name__)


def hear(y: np.ndarray, sr: int = 16000) -> dict | None:
    """What the player said and in which language, detected rather than assumed:
    {"text", "language" (e.g. "hi"), "confidence"}, or None when it can't be checked."""
    r = _call(y, "unknown", sr)
    if r is None:
        return None
    code = str(r.get("language_code") or "")
    return {"text": str(r.get("transcript", "")).strip(), "language": code[:2] or None,
            "confidence": float(r.get("language_probability") or 0.0)}


def _call(y: np.ndarray, language_code: str, sr: int) -> dict | None:
    if not sarvam_http.available():
        return None
    buf = io.BytesIO()
    sf.write(buf, y[: MAX_SECONDS * sr], sr, format="WAV", subtype="PCM_16")
    try:
        r = sarvam_http.post("/speech-to-text", timeout=30, files={"file": ("take.wav", buf.getvalue(), "audio/wav")},
                             data={"model": config()["game"].get("stt_model", "saarika:v2.5"), "language_code": language_code})
    except (httpx.HTTPError, sarvam_http.NoSarvamKey) as e:
        log.warning("Sarvam speech-to-text failed: %s", e)
        return None
    if r.status_code != 200:
        log.warning("Sarvam speech-to-text HTTP %s: %s", r.status_code, r.text[:200])
        return None
    return r.json()
