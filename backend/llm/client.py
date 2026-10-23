"""LLM calls with JSON output, retries, an on-disk cache and a Gemini fallback.

Groq's free tier has per-minute and per-day limits. Per-minute limits are waited out; when a
daily limit is hit, or Groq keeps failing, the call goes to Gemini instead. Identical calls
are answered from data/cache/llm/, so replaying a clip costs nothing.
"""
import base64
import hashlib
import json
import re
import time

import httpx

from backend.settings import ROOT, config, secret

CACHE_DIR = ROOT / "data/cache/llm"


class LLMError(RuntimeError):
    pass


class TruncatedReply(Exception):
    """The model ran out of output tokens; the JSON is incomplete."""


GEMINI_MAX_TOKENS = 8192  # Gemini's free tier has no tight per-request output cap


def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _cached(payload: dict, call, validate=None, primary: str | None = None) -> dict:
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
            if primary is None or result.get("_provider") == primary:
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


def _groq(model: str, messages: list[dict], max_tokens: int, effort: str | None, json_mode: bool) -> dict:
    from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

    client = Groq(api_key=secret("GROQ_API_KEY"), max_retries=0)
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens,
              "temperature": config()["llm"]["temperature"]}
    if effort:
        kwargs["reasoning_effort"] = effort
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(4):
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
                raise LLMError(f"Groq daily limit for {model}") from e
            wait = float(e.response.headers.get("retry-after", 10 * (attempt + 1)))
            time.sleep(min(wait, 60))
        except TruncatedReply as e:
            raise LLMError(str(e)) from e
        except APIStatusError as e:
            # JSON mode sometimes rejects the model's own output ("json_validate_failed"), and
            # 5xx errors are transient: both are worth another try. Anything else (bad request,
            # auth, model gone) will not get better by retrying: fall back.
            if e.status_code >= 500 or "json_validate_failed" in str(e):
                time.sleep(2 * (attempt + 1))
                continue
            raise LLMError(f"Groq {model}: HTTP {e.status_code}") from e
        except (json.JSONDecodeError, LLMError, httpx.TransportError, APIConnectionError):
            time.sleep(2 * (attempt + 1))
    raise LLMError(f"Groq {model} kept failing")


def _gemini(messages: list[dict], max_tokens: int, images: list[bytes] = ()) -> dict:
    model = config()["llm"]["fallback_model"]
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img).decode()}} for img in images]
    parts += [{"text": m["content"]} for m in messages if m["role"] == "user"]
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": max_tokens,
                                 "temperature": config()["llm"]["temperature"]}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    for attempt in range(4):
        try:
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                           headers={"x-goog-api-key": secret("GEMINI_API_KEY")}, json=body, timeout=180)
        except httpx.TransportError:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            if "PerDay" in r.text:
                raise LLMError("Gemini daily limit")
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
    """Rough input size. Indic scripts take far more tokens per character than English, so this
    errs high: Groq rejects a request whose input plus max_tokens exceeds the per-minute limit."""
    return len(json.dumps(messages, ensure_ascii=False)) // 2


def chat_json(messages: list[dict], max_tokens: int = 4000, effort: str = "medium", cache: bool = True,
              validate=None) -> dict:
    """Text model (gpt-oss) -> parsed JSON dict. Adds "_provider" saying which service answered.

    `messages` use the OpenAI format; ask for JSON in the prompt itself. `validate(result)` checks
    the reply's structure; failing replies are retried and never cached.
    """
    llm = config()["llm"]
    model, backup = llm["text_model"], llm["text_backup_model"]
    payload = {"kind": "text", "model": model, "messages": messages, "max_tokens": max_tokens, "effort": effort}
    # Groq counts max_tokens against the tokens-per-minute limit, so the budget must fit beside the input.
    room = llm["groq_tokens_per_minute"] - _estimated_tokens(messages) - 300
    budget = min(max_tokens, room)

    def call():
        errors = []
        if room < min(max_tokens, 1500):  # not enough room beside the input for a useful answer
            errors.append("Groq: request too large for the free per-minute limit")
        else:
            try:
                return {**_groq(model, messages, budget, effort, json_mode=True), "_provider": f"groq:{model}"}
            except LLMError as e:
                errors.append(f"Groq {model}: {e}")
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

    return _cached(payload, call, validate, primary=f"groq:{model}") if cache else call()


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
