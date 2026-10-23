"""Every call to Sarvam's API goes through here, so one place handles the keys.

SARVAM_API_KEY is used first, then SARVAM_API_KEY_2 ... _5 (teammates' accounts) once a key is out
of credits or refused. A spent key is skipped for the rest of the process; nothing else changes for
the callers, which still decide how to retry rate limits and server errors.
"""
import httpx

from backend.settings import secrets

BASE = "https://api.sarvam.ai"
_spent: set[str] = set()


class NoSarvamKey(RuntimeError):
    """No key set, or every key is out of credits."""


def available() -> bool:
    return any(k not in _spent for k in secrets("SARVAM_API_KEY"))


def _out_of_credits(r: httpx.Response) -> bool:
    if r.status_code in (401, 402, 403):
        return True
    text = r.text.lower()
    return r.status_code == 429 and any(w in text for w in ("credit", "balance", "quota", "insufficient"))


def post(path: str, **kwargs) -> httpx.Response:
    """POST to Sarvam with the first key that still works. Raises NoSarvamKey when none does;
    network errors (httpx.HTTPError) pass through for the caller to retry."""
    extra = kwargs.pop("headers", {})
    for key in secrets("SARVAM_API_KEY"):
        if key in _spent:
            continue
        r = httpx.post(f"{BASE}{path}", headers={**extra, "api-subscription-key": key,
                                                  "Authorization": f"Bearer {key}"}, **kwargs)
        if _out_of_credits(r):
            _spent.add(key)
            continue
        return r
    raise NoSarvamKey("no Sarvam key with credits left (SARVAM_API_KEY, SARVAM_API_KEY_2...)")
