import pytest


@pytest.fixture(autouse=True)
def no_paid_voice_calls(monkeypatch):
    """Library builds in tests never call Sarvam text-to-speech (it costs credits)."""
    from backend.game import voice
    monkeypatch.setattr(voice, "available", lambda: False)


@pytest.fixture(autouse=True)
def no_paid_transliteration(monkeypatch):
    """Translation tests never call Sarvam's transliteration (it costs credits)."""
    from backend.pipeline.translate import refine
    monkeypatch.setattr(refine, "sarvam_roman", lambda text, lang: None)
