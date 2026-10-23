"""LLM calls with JSON output, retries, an on-disk cache and fallbacks.

Text goes to gpt-oss-120b on Groq, then the same model on Cerebras (when CEREBRAS_API_KEY is
set), then the smaller gpt-oss-20b, then Gemini. Free tiers have per-minute and per-day limits:
per-minute limits are waited out, a daily limit moves on to the next provider. A request too big
for Groq's per-minute budget goes straight to Cerebras. Identical calls are answered from
data/cache/llm/, so replaying a clip costs nothing.
"""
import base64
import hashlib
import json
import re
import time

import httpx

from backend import sarvam_http
from backend.settings import ROOT, config, secret, secrets

CACHE_DIR = ROOT / "data/cache/llm"


class LLMError(RuntimeError):
    pass


class TruncatedReply(Exception):
    """The model ran out of output tokens; the JSON is incomplete."""


GEMINI_MAX_TOKENS = 8192  # Gemini's free tier has no tight per-request output cap


def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _cached(payload: dict, call, validate=None, primary: set[str] | None = None) -> dict:
    """Answer from the cache, or call and store. Replies that fail `validate` are retried and
    never stored, so one malformed answer cannot be replayed forever. When `primary` is given,
    only answers from that provider are stored: a fallback model's answer is used once, and the
    next run asks the main model again."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(payload)}.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if validate is None or validate(result):
            return result
        path.unlink()
    for _ in range(2):
        result = call()
        if validate is None or validate(result):
            if primary is None or result.get("_provider") in primary:
                path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return result
    raise LLMError("the model's reply did not have the expected structure, twice")


def _parse(text: str) -> dict:
    """JSON from a reply that may wrap it in <think> blocks, code fences or prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object in reply", text, 0)
    return json.loads(text[start:end + 1])


_LESS = {"high": "medium", "medium": "low"}


class DailyLimit(LLMError):
    """This key's daily quota is spent; the next key (if any) may still have some."""


def _groq(model: str, messages: list[dict], max_tokens: int, effort: str | None, json_mode: bool) -> dict:
    """Groq with GROQ_API_KEY, then the backup keys (GROQ_API_KEY_2, ...) once a key's daily limit is hit."""
    keys = secrets("GROQ_API_KEY")
    for i, key in enumerate(keys):
        try:
            return _groq_key(key, model, messages, max_tokens, effort, json_mode)
        except DailyLimit:
            if i == len(keys) - 1:
                raise
    raise LLMError("no GROQ_API_KEY set")


def _groq_key(key: str, model: str, messages: list[dict], max_tokens: int, effort: str | None, json_mode: bool) -> dict:
    from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

    client = Groq(api_key=key, max_retries=0)
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens,
              "temperature": config()["llm"]["temperature"]}
    if effort:
        kwargs["reasoning_effort"] = effort
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    attempt = waits = 0
    while attempt < 4:
        attempt += 1
        try:
            r = client.chat.completions.create(**kwargs)
            if r.choices[0].finish_reason == "length":
                # Retrying the same request would be cut off again; let the caller fall back.
                raise TruncatedReply(f"{model} hit max_tokens={max_tokens}")
            content = r.choices[0].message.content or ""
            if not content.strip():
                raise LLMError("empty reply (reasoning used the whole token budget?)")
            return _parse(content)
        except RateLimitError as e:
            msg = str(e)
            if "per day" in msg.lower() or "PerDay" in msg:
                raise DailyLimit(f"Groq daily limit for {model}") from e
            wait = float(e.response.headers.get("retry-after", 10 * attempt))
            time.sleep(min(wait, 60))
            waits += 1
            if waits <= 6:          # a per-minute limit only means "wait": it is not a failed attempt
                attempt -= 1
        except TruncatedReply as e:
            raise LLMError(str(e)) from e
        except APIStatusError as e:
            # JSON mode sometimes rejects the model's own output ("json_validate_failed"), and
            # 5xx errors are transient: both are worth another try. Anything else (bad request,
            # auth, model gone) will not get better by retrying: fall back.
            if e.status_code >= 500 or "json_validate_failed" in str(e):
                # An empty failed generation means the reasoning used the whole token budget:
                # the same request would fail again, so think less on the next try.
                if "'failed_generation': ''" in str(e) and kwargs.get("reasoning_effort") in _LESS:
                    kwargs["reasoning_effort"] = _LESS[kwargs["reasoning_effort"]]
                time.sleep(2 * (attempt + 1))
                continue
            raise LLMError(f"Groq {model}: HTTP {e.status_code}") from e
        except (json.JSONDecodeError, LLMError, httpx.TransportError, APIConnectionError):
            time.sleep(2 * (attempt + 1))
    raise LLMError(f"Groq {model} kept failing")




def _sarvam_chat(model: str, messages: list[dict], max_tokens: int, effort: str | None = None) -> dict:
    """Sarvam's own LLMs (sarvam-105b reasons first; sarvam-105b-conversations answers directly).
    Built for Indian languages: they write spoken Tamil that gpt-oss could not."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": config()["llm"]["temperature"]}
    if effort and not model.endswith("conversations"):
        body["reasoning_effort"] = effort
    for attempt in range(4):
        try:
            r = sarvam_http.post("/v1/chat/completions", json=body, timeout=300)
        except sarvam_http.NoSarvamKey as e:
            raise LLMError(str(e)) from e
        except httpx.TransportError:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 400 and "reasoning_effort" in body and "reasoning" in r.text:
            body.pop("reasoning_effort")           # not every model takes it
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(float(r.headers.get("retry-after", 5 * (attempt + 1))), 60))
            continue
        if r.status_code != 200:
            raise LLMError(f"Sarvam HTTP {r.status_code}: {r.text[:200]}")
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMError(f"Sarvam {model} hit max_tokens={max_tokens}")
        try:
            return _parse(choice["message"].get("content") or "")
        except json.JSONDecodeError:
            time.sleep(1)
    raise LLMError(f"Sarvam {model} kept failing")


CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"


def _cerebras(model: str, messages: list[dict], max_tokens: int, effort: str | None) -> dict:
    """The same gpt-oss model on Cerebras (OpenAI-compatible API)."""
    body = {"model": model, "messages": messages, "max_completion_tokens": max_tokens,
            "temperature": config()["llm"]["temperature"], "response_format": {"type": "json_object"}}
    if effort:
        body["reasoning_effort"] = effort
    for attempt in range(4):
        try:
            r = httpx.post(CEREBRAS_URL, headers={"Authorization": f"Bearer {secret('CEREBRAS_API_KEY')}"},
                           json=body, timeout=300)
        except httpx.TransportError:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 429:
            if "day" in r.text.lower():
                raise LLMError(f"Cerebras daily limit for {model}")
            time.sleep(min(float(r.headers.get("retry-after", 10 * (attempt + 1))), 60))
            continue
        if r.status_code >= 500:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise LLMError(f"Cerebras HTTP {r.status_code}: {r.text[:200]}")
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMError(f"Cerebras {model} hit max_tokens={max_tokens}")
        try:
            return _parse(choice["message"].get("content") or "")
        except json.JSONDecodeError:
            time.sleep(2)
    raise LLMError(f"Cerebras {model} kept failing")


def _gemini(messages: list[dict], max_tokens: int, images: list[bytes] = (), model: str | None = None,
            video: bytes | None = None) -> dict:
    """Gemini with GEMINI_API_KEY, then the backup keys once a key's daily limit is hit."""
    keys = secrets("GEMINI_API_KEY")
    for i, key in enumerate(keys):
        try:
            return _gemini_key(key, messages, max_tokens, images, model, video)
        except DailyLimit:
            if i == len(keys) - 1:
                raise
    raise LLMError("no GEMINI_API_KEY set")


def _gemini_key(key: str, messages: list[dict], max_tokens: int, images: list[bytes] = (),
                model: str | None = None, video: bytes | None = None) -> dict:
    model = model or config()["llm"]["fallback_model"]
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img).decode()}} for img in images]
    if video:
        parts.append({"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(video).decode()}})
    parts += [{"text": m["content"]} for m in messages if m["role"] == "user"]
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": max_tokens,
                                 "temperature": config()["llm"]["temperature"]}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    for attempt in range(4):
        try:
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                           headers={"x-goog-api-key": key}, json=body, timeout=180)
        except httpx.TransportError:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            if "PerDay" in r.text:
                raise DailyLimit("Gemini daily limit")
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code >= 500:  # Gemini is "overloaded" now and then; it passes
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
        try:
            return _parse(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, json.JSONDecodeError):
            time.sleep(2)
    raise LLMError("Gemini kept failing")


def _estimated_tokens(messages: list[dict]) -> int:
    """Rough input size, erring a little high: Groq rejects a request whose input plus max_tokens
    exceeds the per-minute limit. English/JSON runs ~4 characters per token; Indic scripts take
    far more tokens per character (about 1 per 1.5 characters)."""
    text = json.dumps(messages, ensure_ascii=False)
    ascii_chars = sum(c < "\x80" for c in text)
    return int(ascii_chars / 3.5 + (len(text) - ascii_chars) / 1.3) + 50


def chat_json(messages: list[dict], max_tokens: int = 4000, effort: str = "medium", cache: bool = True,
              validate=None, primary_only: bool = False, prefer: str | None = None) -> dict:
    """Text model (gpt-oss) -> parsed JSON dict. Adds "_provider" saying which service answered.

    `messages` use the OpenAI format; ask for JSON in the prompt itself. `validate(result)` checks
    the reply's structure; failing replies are retried and never cached. `primary_only` raises
    instead of falling back to the smaller model or Gemini (translation: a weak answer is worse
    than waiting for the quota); Cerebras serves the same model, so it still counts.
    `prefer` = "sarvam:<model>" or "gemini:<model>" asks that model first (llm.translator: both
    write spoken Indian languages far better than gpt-oss); with primary_only, its failure is final.
    """
    llm = config()["llm"]
    model, backup = llm["text_model"], llm["text_backup_model"]
    payload = {"kind": "text", "model": prefer or model, "messages": messages, "max_tokens": max_tokens, "effort": effort}
    # Groq counts max_tokens against the tokens-per-minute limit, so the budget must fit beside the input.
    room = llm["groq_tokens_per_minute"] - _estimated_tokens(messages) - 300
    budget = min(max_tokens, room)

    cerebras = llm.get("cerebras_model") if secret("CEREBRAS_API_KEY") else None

    def call():
        errors = []
        if prefer:
            kind, _, name = prefer.partition(":")
            try:
                if kind == "sarvam":
                    # a reasoning model thinks before answering: leave room for both; the
                    # conversations model answers directly and caps output at 8192
                    budget_s = 8192 if name.endswith("conversations") else max(max_tokens * 3, 16000)
                    return {**_sarvam_chat(name, messages, budget_s, effort), "_provider": prefer}
                return {**_gemini(messages, max(max_tokens, GEMINI_MAX_TOKENS), model=name), "_provider": prefer}
            except LLMError as e:
                if primary_only:
                    raise
                errors.append(f"{prefer}: {e}")
        if room < min(max_tokens, 1500):  # not enough room beside the input for a useful answer
            errors.append("Groq: request too large for the free per-minute limit")
        else:
            try:
                # High effort needs room to reason; in a small budget it leaves an empty answer.
                groq_effort = effort if budget >= 6000 else _LESS.get(effort, effort)
                return {**_groq(model, messages, budget, groq_effort, json_mode=True), "_provider": f"groq:{model}"}
            except LLMError as e:
                errors.append(f"Groq {model}: {e}")
        if cerebras:
            try:
                return {**_cerebras(cerebras, messages, max_tokens, effort), "_provider": f"cerebras:{cerebras}"}
            except LLMError as e:
                errors.append(str(e))
        if primary_only:
            raise LLMError("; ".join(errors) + " (main model only: try again when a quota resets)")
        if room >= min(max_tokens, 1500):
            try:
                # The smaller model fails Groq's JSON validation, so it runs without JSON mode, and
                # at low effort (at medium or high it often reasons past its token budget).
                return {**_groq(backup, messages, budget, "low", json_mode=False), "_provider": f"groq:{backup}"}
            except LLMError as e:
                errors.append(f"Groq {backup}: {e}")
        try:
            return {**_gemini(messages, GEMINI_MAX_TOKENS), "_provider": "gemini"}
        except LLMError as e:
            errors.append(f"Gemini: {e}")
        raise LLMError("; ".join(errors))

    main = {f"groq:{model}"} | ({f"cerebras:{cerebras}"} if cerebras else set()) | ({prefer} if prefer else set())
    return _cached(payload, call, validate, primary=main) if cache else call()


def video_json(prompt: str, video: bytes, cache: bool = True, validate=None, model: str | None = None) -> dict:
    """Gemini watching a short video (inline, keep it a few MB) -> parsed JSON dict."""
    digest = hashlib.sha256(video).hexdigest()
    payload = {"kind": "video", "model": model or config()["llm"]["fallback_model"], "prompt": prompt, "video": digest}

    def call():
        return {**_gemini([{"role": "user", "content": prompt}], GEMINI_MAX_TOKENS, model=model, video=video),
                "_provider": "gemini"}

    return _cached(payload, call, validate) if cache else call()


def vision_json(prompt: str, images: list[bytes], cache: bool = True, validate=None) -> dict:
    """Vision model (qwen, at most 3 JPEG images per request) -> parsed JSON dict."""
    llm = config()["llm"]
    if len(images) > 3:
        raise ValueError("the vision model accepts at most 3 images per request; tile frames first")
    model = llm["vision_model"]
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(i).decode()}"}}
                for i in images]
    messages = [{"role": "user", "content": content}]
    payload = {"kind": "vision", "model": model, "prompt": prompt,
               "images": [hashlib.sha256(i).hexdigest() for i in images]}

    def call():
        try:
            return {**_groq(model, messages, llm["vision_max_tokens"], None, json_mode=False),
                    "_provider": f"groq:{model}"}
        except LLMError as groq_error:
            try:
                return {**_gemini([{"role": "user", "content": prompt}], GEMINI_MAX_TOKENS, images),
                        "_provider": "gemini"}
            except LLMError as e:
                raise LLMError(f"Groq: {groq_error}; Gemini: {e}") from e

    return _cached(payload, call, validate) if cache else call()
