"""English-letter spelling of a dubbed line, for players who speak the language but can't read
its script. Sarvam's transliteration reads more naturally than ours for Hindi and Marathi
("pocket", "nahi" vs "paॉket", "naheen"), but for Tamil, Bengali and Telugu it changed words
instead of spelling them ("intiki" -> "ghar-ku"), so only Devanagari targets use it.
Cached in data/cache/translit/.
"""
import hashlib
import json

import httpx

from backend import sarvam_http
from backend.settings import ROOT
CACHE = ROOT / "data/cache/translit"
USE_FOR = {"hi", "mr"}


def sarvam_roman(text: str, lang: str) -> str | None:
    """Sarvam's romanization of `text`, or None (other language, no key, or an API error)."""
    if lang not in USE_FOR or not sarvam_http.available() or not text.strip():
        return None
    body = {"input": text, "source_language_code": f"{lang}-IN", "target_language_code": "en-IN"}
    path = CACHE / f"{hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["transliterated_text"]
    try:
        r = sarvam_http.post("/transliterate", json=body, timeout=30)
    except (httpx.HTTPError, sarvam_http.NoSarvamKey):
        return None
    if r.status_code != 200 or not r.json().get("transliterated_text"):
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r.json(), ensure_ascii=False), encoding="utf-8")
    return r.json()["transliterated_text"]
