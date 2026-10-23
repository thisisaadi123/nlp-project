"""First-draft translation with Sarvam's Mayura model in its spoken ("modern-colloquial") mode.

IndicTrans2 learned from written text, so its drafts are bookish, and it can't read spoken Tamil
(கெளம்புறேன்டா, "I'm leaving", came out as a Hindi-script copy). Mayura translates between Indian
languages into the way people talk: on Hindi->Tamil it gave "நைட் எப்படி போச்சு?" where
IndicTrans2 gave "இரவு எப்படி கழித்தாய்". It still misses context (address terms, some meanings),
which the LLM refine step fixes. Needs SARVAM_API_KEY; answers are cached in data/cache/sarvam/.
"""
import hashlib
import json
import time

import httpx

from backend import sarvam_http
from backend.settings import ROOT, config

CACHE = ROOT / "data/cache/sarvam"
MAX_CHARS = 1000            # Mayura's input limit per request


class SarvamError(RuntimeError):
    pass


def available() -> bool:
    return sarvam_http.available()


def _code(lang: str) -> str:
    return f"{lang}-IN"


def translate(text: str, src: str, tgt: str, gender: str | None = None) -> str:
    cfg = config()["llm"]
    body = {"input": text[:MAX_CHARS], "source_language_code": _code(src), "target_language_code": _code(tgt),
            "model": cfg.get("sarvam_model", "mayura:v1"), "mode": cfg.get("sarvam_mode", "modern-colloquial"),
            "output_script": "spoken-form-in-native"}
    if gender in ("male", "female"):
        body["speaker_gender"] = gender.capitalize()
    key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["translated_text"]
    for attempt in range(4):
        try:
            r = sarvam_http.post("/translate", json=body, timeout=60)
        except sarvam_http.NoSarvamKey as e:
            raise SarvamError(str(e)) from e
        except httpx.TransportError:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise SarvamError(f"Sarvam HTTP {r.status_code}: {r.text[:200]}")
        out = r.json()
        if not str(out.get("translated_text", "")).strip():
            raise SarvamError("Sarvam returned an empty translation")
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out["translated_text"]
    raise SarvamError("Sarvam kept failing")


def drafts(lines: list[dict], src: str, tgt: str, people: dict | None = None) -> list[str]:
    """One spoken-style draft per line; "" for a line Sarvam couldn't do (the refine step then
    translates it from the source)."""
    out = []
    for ln in lines:
        gender = ((people or {}).get(ln.get("speaker")) or {}).get("gender")
        try:
            out.append(translate(ln["text"], src, tgt, gender))
        except SarvamError:
            out.append("")
    return out
