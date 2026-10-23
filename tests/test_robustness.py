"""Failure modes found while running step 4 on real clips, each pinned by a test."""
import json

import httpx
import pytest

from backend.llm import client
from backend.pipeline.translate import refine as R
from backend.pipeline.translate.draft import degenerate


# --- IndicTrans2 sometimes gets stuck repeating a word -----------------------------------------

@pytest.mark.parametrize("text, stuck", [
    ("এখন ছেলেও আপনাদের দুজনের কাঁচা কাঁচা কাঁচা কাঁচা কাঁচা", True),
    ("भाई, भाई, भाई, ऐ! भाई!", False),       # three in a row is real dialogue
    ("आजा बेटा आजा आजा", False),
    ("", False),
])
def test_degenerate(text, stuck):
    assert degenerate(text) is stuck


# --- a malformed reply must never be cached --------------------------------------------------------

def test_invalid_cached_reply_is_discarded_and_redone(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)
    msgs = [{"role": "user", "content": "q"}]
    payload = {"kind": "text", "model": "openai/gpt-oss-120b", "messages": msgs, "max_tokens": 4000, "effort": "medium"}
    (tmp_path / f"{client._cache_key(payload)}.json").write_text('{"lines": "a string, not a list"}')
    monkeypatch.setattr(client, "_groq", lambda *a, **k: {"lines": [{"line_id": 1}]})
    result = client.chat_json(msgs, validate=lambda r: isinstance(r["lines"], list))
    assert result["lines"] == [{"line_id": 1}]
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["lines"] == [{"line_id": 1}]


def test_reply_failing_validation_twice_raises_and_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(client, "_groq", lambda *a, **k: {"lines": "bad"})
    with pytest.raises(client.LLMError):
        client.chat_json([{"role": "user", "content": "q"}], validate=lambda r: isinstance(r["lines"], list))
    assert list(tmp_path.glob("*.json")) == []


# --- Groq counts max_tokens against its per-minute limit ------------------------------------------

def test_oversized_request_skips_groq_and_uses_gemini(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)
    groq_calls = []
    monkeypatch.setattr(client, "_groq", lambda *a, **k: groq_calls.append(1) or {})  # must not be called
    monkeypatch.setattr(client, "_gemini", lambda messages, max_tokens, images=(): {"ok": True})
    huge = [{"role": "user", "content": "भ" * 20000}]
    assert client.chat_json(huge, cache=False) == {"ok": True, "_provider": "gemini"}
    assert groq_calls == []


def test_groq_budget_shrinks_to_fit_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)
    seen = {}

    def fake_groq(model, messages, max_tokens, effort, json_mode):
        seen["max_tokens"] = max_tokens
        return {}

    monkeypatch.setattr(client, "_groq", fake_groq)
    msgs = [{"role": "user", "content": "x" * 6000}]   # ~3000 estimated tokens
    client.chat_json(msgs, max_tokens=6000, cache=False)
    assert seen["max_tokens"] + client._estimated_tokens(msgs) <= 8000


# --- Gemini returns 503 when overloaded -------------------------------------------------------------

def test_gemini_retries_server_errors(monkeypatch):
    replies = [httpx.Response(503, text="overloaded"),
               httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}]})]
    monkeypatch.setattr(client.httpx, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    assert client._gemini([{"role": "user", "content": "q"}], 100) == {"ok": 1}


def test_both_providers_failing_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)

    def fail(*a, **k):
        raise client.LLMError("down")

    monkeypatch.setattr(client, "_groq", fail)
    monkeypatch.setattr(client, "_gemini", fail)
    with pytest.raises(client.LLMError, match="gpt-oss-120b: down; Groq openai/gpt-oss-20b: down; Gemini: down"):
        client.chat_json([{"role": "user", "content": "q"}], cache=False)


# --- translation goes in batches that see earlier lines ---------------------------------------------

def test_refine_batches_lines_and_passes_earlier_translations(monkeypatch):
    requests = []

    def fake_chat_json(messages, **kwargs):
        body = json.loads(messages[1]["content"].split("\n\nReturn JSON")[0])
        requests.append(body)
        return {"lines": [{"line_id": ln["line_id"], "translation": f"T{ln['line_id']}"} for ln in body["lines"]],
                "_provider": "fake"}

    monkeypatch.setattr(R, "chat_json", fake_chat_json)
    monkeypatch.setattr(R, "fit_lengths", lambda lines, out, tgt: out)
    lines = [{"line_id": i, "text": f"line {i}"} for i in range(1, 7)]
    out = R.refine(lines, [""] * 6, "hi", "ta", scene={"setting": "x"})
    assert [o["translation"] for o in out] == ["T1", "T2", "T3", "T4", "T5", "T6"]
    assert [len(r["lines"]) for r in requests] == [3, 3]          # config translate_batch: 3
    assert "already_translated" not in requests[0]
    assert [a["translation"] for a in requests[1]["already_translated"]] == ["T1", "T2", "T3"]
    assert requests[0]["lines"][0]["draft"].startswith("(no usable draft")


# --- context and captions degrade instead of crashing -----------------------------------------------

def test_context_failure_degrades_to_empty_context(monkeypatch):
    from backend.pipeline.context import build

    def fail(*a, **k):
        raise client.LLMError("down")

    monkeypatch.setattr(build, "chat_json", fail)
    ctx = build.build_context([{"line_id": 1, "text": "x"}], None, "hi")
    assert ctx["lines"] == [] and ctx["confidence"].startswith("unavailable")


def test_small_request_with_small_max_tokens_still_uses_groq(tmp_path, monkeypatch):
    """A 300-token request must not be treated as too large (bug found by the network test)."""
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)
    seen = {}

    def fake_groq(model, messages, max_tokens, effort, json_mode):
        seen["max_tokens"] = max_tokens
        return {"ok": True}

    monkeypatch.setattr(client, "_groq", fake_groq)
    result = client.chat_json([{"role": "user", "content": "q"}], max_tokens=300, cache=False)
    assert result["_provider"] == "groq:openai/gpt-oss-120b" and seen["max_tokens"] == 300
