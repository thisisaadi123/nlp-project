import json

import pytest

from backend.llm import client


@pytest.mark.parametrize("reply", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '<think>reasoning {not json}</think>\n{"a": 1}',
    'Here you go: {"a": 1} hope that helps',
])
def test_parse_extracts_json(reply):
    assert client._parse(reply) == {"a": 1}


def test_parse_rejects_reply_without_json():
    with pytest.raises(json.JSONDecodeError):
        client._parse("no json here")


def test_cache_answers_repeat_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)
    calls = []

    def fake_groq(*args, **kwargs):
        calls.append(1)
        return {"answer": 42}

    monkeypatch.setattr(client, "_groq", fake_groq)
    msgs = [{"role": "user", "content": "q"}]
    first = client.chat_json(msgs)
    second = client.chat_json(msgs)
    assert first == second == {"answer": 42, "_provider": "groq:openai/gpt-oss-120b"}
    assert len(calls) == 1


def test_falls_back_to_gemini_when_groq_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)

    def failing_groq(*args, **kwargs):
        raise client.LLMError("Groq daily limit")

    monkeypatch.setattr(client, "_groq", failing_groq)
    monkeypatch.setattr(client, "_gemini", lambda messages, max_tokens, images=(): {"answer": "from gemini"})
    result = client.chat_json([{"role": "user", "content": "q"}], cache=False)
    assert result == {"answer": "from gemini", "_provider": "gemini"}


def test_truncated_groq_reply_falls_back_without_retrying(tmp_path, monkeypatch):
    """A reply cut off at max_tokens is not retried: each Groq model is tried once, then Gemini."""
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)
    attempts = []

    class Choice:
        finish_reason = "length"
        message = type("M", (), {"content": '{"partial": '})

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kwargs):
            attempts.append(1)
            return type("R", (), {"choices": [Choice()]})()

    import groq
    monkeypatch.setattr(groq, "Groq", FakeGroq)
    monkeypatch.setattr(client, "_gemini", lambda messages, max_tokens, images=(): {"ok": max_tokens})
    result = client.chat_json([{"role": "user", "content": "q"}], cache=False)
    assert len(attempts) == 2          # gpt-oss-120b once, gpt-oss-20b once
    assert result == {"ok": client.GEMINI_MAX_TOKENS, "_provider": "gemini"}


def test_vision_rejects_more_than_three_images():
    with pytest.raises(ValueError):
        client.vision_json("describe", [b"x"] * 4)


@pytest.mark.network
def test_groq_json_round_trip():
    result = client.chat_json([{"role": "user", "content": 'Return JSON {"ok": true}'}], max_tokens=300,
                              effort="low", cache=False)
    assert result["ok"] is True


def _fake_groq(monkeypatch, outcomes):
    """Groq client whose create() raises or returns the given outcomes in order."""
    import groq
    import httpx

    def status_error(code, message):
        resp = httpx.Response(code, request=httpx.Request("POST", "https://api.groq.com"))
        cls = groq.BadRequestError if code == 400 else groq.AuthenticationError if code == 401 else groq.InternalServerError
        return cls(message, response=resp, body=None)

    calls = []

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kwargs):
            outcome = outcomes[len(calls)]
            calls.append(1)
            if isinstance(outcome, tuple):
                raise status_error(*outcome)
            choice = type("Ch", (), {"finish_reason": "stop", "message": type("M", (), {"content": outcome})})
            return type("R", (), {"choices": [choice]})()

    monkeypatch.setattr(groq, "Groq", FakeGroq)
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    return calls


def test_json_validation_failure_is_retried(monkeypatch):
    calls = _fake_groq(monkeypatch, [(400, "json_validate_failed"), '{"ok": 1}'])
    assert client._groq("m", [], 100, None, json_mode=True) == {"ok": 1}
    assert len(calls) == 2


def test_server_error_is_retried(monkeypatch):
    calls = _fake_groq(monkeypatch, [(503, "unavailable"), '{"ok": 2}'])
    assert client._groq("m", [], 100, None, json_mode=True) == {"ok": 2}
    assert len(calls) == 2


def test_auth_error_falls_back_without_retrying(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)
    calls = _fake_groq(monkeypatch, [(401, "invalid api key"), (401, "invalid api key")])
    monkeypatch.setattr(client, "_gemini", lambda messages, max_tokens, images=(): {"ok": 3})
    assert client.chat_json([{"role": "user", "content": "q"}], cache=False) == {"ok": 3, "_provider": "gemini"}
    assert len(calls) == 2             # one attempt per Groq model, no retries


def test_backup_model_answers_but_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)

    def fake_groq(model, messages, max_tokens, effort, json_mode):
        if model == "openai/gpt-oss-120b":
            raise client.LLMError("daily limit")
        assert json_mode is False and effort == "low"
        return {"answer": "from 20b"}

    monkeypatch.setattr(client, "_groq", fake_groq)
    result = client.chat_json([{"role": "user", "content": "q"}])
    assert result == {"answer": "from 20b", "_provider": "groq:openai/gpt-oss-20b"}
    assert list(tmp_path.glob("*.json")) == []   # next run asks 120b again


def test_cerebras_serves_the_main_model_when_groq_is_out(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("CEREBRAS_API_KEY", "test")
    def out(*a, **k):
        raise client.LLMError("Groq daily limit")
    monkeypatch.setattr(client, "_groq", out)
    monkeypatch.setattr(client, "_cerebras", lambda model, messages, max_tokens, effort: {"answer": model})
    monkeypatch.setattr(client, "_gemini", lambda *a, **k: pytest.fail("must not reach Gemini"))
    r = client.chat_json([{"role": "user", "content": "x"}], primary_only=True)
    assert r["_provider"].startswith("cerebras:") and len(list(tmp_path.glob("*.json"))) == 1   # cached


def test_sarvam_moves_to_the_next_key_when_one_is_out_of_credits(monkeypatch):
    import httpx
    from backend import sarvam_http
    monkeypatch.setattr(sarvam_http, "_spent", set())
    monkeypatch.setattr(sarvam_http, "secrets", lambda name: ["k1", "k2"])
    used = []
    def fake_post(url, headers, **kw):
        used.append(headers["api-subscription-key"])
        status = 402 if headers["api-subscription-key"] == "k1" else 200
        return httpx.Response(status, json={"ok": True}, request=httpx.Request("POST", url))
    monkeypatch.setattr(sarvam_http.httpx, "post", fake_post)
    assert sarvam_http.post("/translate", json={}).status_code == 200 and used == ["k1", "k2"]
    sarvam_http.post("/translate", json={})
    assert used == ["k1", "k2", "k2"]                 # the spent key is skipped from then on
    monkeypatch.setattr(sarvam_http, "secrets", lambda name: ["k1"])
    with pytest.raises(sarvam_http.NoSarvamKey):
        sarvam_http.post("/translate", json={})
