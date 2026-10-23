import pytest

from backend.pipeline.translate.refine import _line_payload, syllables


@pytest.mark.parametrize("text, expected", [
    ("भैया", 2),                      # भै + या
    ("क्या", 1),                      # क्या is one cluster
    ("अण्णा", 2),                     # अ + ण्णा
    ("அண்ணா", 2),                     # அ + ண்ணா (Tamil pulli)
    ("நீ என்ன பண்ணிட்ட?", 6),          # நீ | எ ன்ன | ப ண்ணி ட்ட = 1 + 2 + 3
    ("ক্ষমা", 2),                      # Bengali conjunct counts once
    ("hello", 0),                      # Latin text is not counted
])
def test_syllables(text, expected):
    assert syllables(text) == expected


def test_line_payload_maps_arousal_to_intensity():
    line = {"line_id": 1, "text": "x", "emotion": {"label": "angry", "arousal": 0.9}, "start": 1.0, "end": 2.5}
    payload = _line_payload(line, "draft")
    assert payload["intensity"] == "high"
    assert payload["emotion"] == "angry"
    assert payload["duration"] == 1.5
    assert payload["draft"] == "draft"


def test_line_payload_passes_subtext_and_visual_cue():
    line = {"line_id": 2, "text": "x", "subtext": "sarcasm", "visual_cue": "smirks"}
    payload = _line_payload(line, "d")
    assert payload["subtext"] == "sarcasm"
    assert payload["visual_cue"] == "smirks"
    assert "intensity" not in payload


@pytest.mark.network
def test_draft_translation_runs():
    from backend.pipeline.translate.draft import translator
    out = translator().translate(["आप कैसे हैं?"], "hi", "ta")
    assert len(out) == 1 and syllables(out[0]) > 0
