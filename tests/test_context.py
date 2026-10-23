import json

from backend.pipeline.context.fusion import fuse
from backend.pipeline.dub import lines_from_label, translation_input


def test_label_comes_from_context_not_the_voice():
    """emotion2vec calls most angry Indic lines "happy" (step 3); the context decides the label."""
    f = fuse({"line_id": 1, "text": "x", "emotion": {"label": "happy", "arousal": 0.2, "arousal_e2v": 0.9}},
             {"intent": "accusation", "text_emotion": "angry"})
    assert f["label"] == "angry" and f["label_source"] == "context"
    assert f["voice_emotion"] == "happy"


def test_intensity_prefers_emotion2vec_over_prosody():
    f = fuse({"line_id": 1, "text": "x", "emotion": {"label": "happy", "arousal": 0.2, "arousal_e2v": 0.9}}, None)
    assert f["intensity"] == "high" and f["arousal"] == 0.9 and f["arousal_source"] == "emotion2vec"


def test_intensity_falls_back_to_prosody():
    f = fuse({"line_id": 1, "text": "x", "emotion": {"label": "angry", "arousal": 0.8}}, None)
    assert f["intensity"] == "high" and f["arousal_source"] == "prosody"


def test_voice_label_used_only_without_context():
    f = fuse({"line_id": 1, "text": "x", "emotion": {"label": "fear"}}, None)
    assert f["label"] == "fearful" and f["label_source"] == "voice"   # emotion2vec name mapped


def test_intense_voice_on_calm_words_is_flagged():
    f = fuse({"line_id": 1, "text": "x", "emotion": {"arousal_e2v": 0.9}}, {"intent": "comfort", "text_emotion": "calm"})
    assert f["mismatch"] is True


def test_flat_voice_on_angry_words_is_flagged():
    f = fuse({"line_id": 1, "text": "x", "emotion": {"arousal_e2v": 0.1}}, {"intent": "threat", "text_emotion": "angry"})
    assert f["mismatch"] is True


def test_matching_voice_and_words_not_flagged():
    f = fuse({"line_id": 1, "text": "x", "emotion": {"arousal_e2v": 0.85}}, {"intent": "threat", "text_emotion": "angry"})
    assert f["mismatch"] is False


def test_fusion_marks_sarcasm():
    f = fuse({"line_id": 1, "text": "x", "emotion": {"label": "happy"}}, {"intent": "sarcasm", "text_emotion": "happy"})
    assert f["label"] == "sarcastic" and f["mismatch"] is True


def test_fusion_takes_stress_from_word_prosody():
    words = [{"w": "भैया", "stress": 2.4}, {"w": "ये", "stress": 0.2}]
    f = fuse({"line_id": 1, "text": "भैया ये", "words": words}, None)
    assert f["stressed_words"] == ["भैया"]


def test_fusion_without_any_information():
    f = fuse({"line_id": 1, "text": "x"}, None)
    assert f["label"] == "neutral" and f["label_source"] == "default"
    assert f["intensity"] is None and f["mismatch"] is False


# A line exactly as step 3's backend/pipeline/speech/run.py writes it.
STEP3_LINE = {
    "line_id": 0, "speaker": "SPEAKER_01", "start": 1.92, "end": 4.31, "language": "ta",
    "text": "அண்ணா நீ என்ன பண்ணிட்ட", "text_roman": "anna nee enna pannitta",
    "words": [{"w": "அண்ணா", "start": 1.95, "end": 2.41, "stress": 2.31, "pitch": 3.2, "energy": -14.1},
              {"w": "நீ", "start": 2.41, "end": 2.6, "stress": None, "pitch": None, "energy": None},
              {"w": "என்ன", "start": 2.6, "end": 2.98, "stress": 0.4, "pitch": 0.1, "energy": -18.0},
              {"w": "பண்ணிட்ட", "start": 2.98, "end": 3.9, "stress": -0.6, "pitch": -2.0, "energy": -21.3}],
    "pauses": [[2.41, 2.6]], "prosody": {"pitch_level": 8.1, "pitch_range": 6.2},
    "emotion": {"label": "happy", "probs": {"angry": 0.21, "happy": 0.52, "neutral": 0.1, "sad": 0.05,
                                            "fear": 0.04, "surprise": 0.06, "disgust": 0.02},
                "arousal": 0.31, "arousal_e2v": 0.83},
}


def test_step3_line_with_unvoiced_word_does_not_crash():
    f = fuse(STEP3_LINE, {"line_id": 0, "intent": "accusation", "text_emotion": "angry"})
    assert f["stressed_words"] == ["அண்ணா"]
    assert f["label"] == "angry" and f["intensity"] == "high" and f["mismatch"] is False


def test_step3_line_where_no_word_is_voiced():
    line = {**STEP3_LINE, "words": [{"w": "ம்", "start": 0, "end": 0.2, "stress": None}]}
    assert fuse(line, None)["stressed_words"] == []


def test_context_sees_voice_intensity_not_emotion2vec_label():
    from backend.pipeline.context.build import _transcript
    item = _transcript([STEP3_LINE])[0]
    assert item["voice_intensity"] == "high" and "voice_emotion" not in item


def test_context_gets_stub_label_when_there_is_no_measurement():
    from backend.pipeline.context.build import _transcript
    item = _transcript([{"line_id": 1, "text": "x", "emotion": {"label": "angry"}}])[0]
    assert item["voice_emotion"] == "angry" and "voice_intensity" not in item


def test_step3_line_flows_into_translation_input():
    f = fuse(STEP3_LINE, {"line_id": 0, "intent": "accusation", "text_emotion": "angry", "addressee": "SPEAKER_00"})
    item = translation_input(STEP3_LINE, f, {"addressee": "SPEAKER_00"})
    assert item["emotion"] == {"label": "angry", "arousal": 0.83}
    assert item["stressed_words"] == ["அண்ணா"] and item["addressee"] == "SPEAKER_00"


def test_translation_input_adds_subtext_on_mismatch():
    line = {"line_id": 3, "speaker": "S1", "text": "x", "start": 0.0, "end": 1.0}
    fused = {"label": "sarcastic", "arousal": None, "intent": "sarcasm", "stressed_words": [],
             "visual_cue": None, "mismatch": True}
    item = translation_input(line, fused, {"addressee": "S2"})
    assert item["addressee"] == "S2"
    assert "subtext" in item
    assert "arousal" not in item["emotion"]


def test_lines_from_label_drops_addressee(tmp_path):
    label = {"lines": [{"speaker": "S1", "addressee": "S2", "start": 0, "end": 1, "transcript": "अण्णा",
                        "emotion": "angry", "stressed_words": ["अण्णा"]}]}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
    lines = lines_from_label(path, "mr")
    assert lines == [{"line_id": 1, "speaker": "S1", "start": 0, "end": 1, "language": "mr", "text": "अण्णा",
                      "emotion": {"label": "angry"}, "stressed_words": ["अण्णा"]}]


# --- context step: step 3's speaker ids, fixed vocabularies ---------------------------------------

def test_build_context_uses_short_aliases_and_maps_them_back(monkeypatch):
    from backend.pipeline.context import build
    sent = {}

    def fake_chat_json(messages, **kwargs):
        sent["body"] = messages[1]["content"]
        return {"speakers": {"S1": {"description": "younger brother"}, "S2": {"description": "elder brother"}},
                "relationships": [{"between": ["S1", "S2"], "senior": "S2", "relation": "siblings"}],
                "lines": [{"line_id": 0, "addressee": "S2", "intent": "Plea/Command", "text_emotion": "frustrated, angry"},
                          {"line_id": 1, "addressee": "S1", "intent": "acknowledgement", "text_emotion": "dismissive"}]}

    monkeypatch.setattr(build, "chat_json", fake_chat_json)
    lines = [{**STEP3_LINE, "line_id": 0, "speaker": "SPEAKER_01"},
             {**STEP3_LINE, "line_id": 1, "speaker": "SPEAKER_00"}]
    ctx = build.build_context(lines, None, "ta")
    assert "SPEAKER_0" not in sent["body"] and '"speaker": "S1"' in sent["body"]
    assert set(ctx["speakers"]) == {"SPEAKER_01", "SPEAKER_00"}
    assert ctx["relationships"][0]["between"] == ["SPEAKER_01", "SPEAKER_00"]
    assert ctx["relationships"][0]["senior"] == "SPEAKER_00"
    assert [ln["addressee"] for ln in ctx["lines"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert [ln["intent"] for ln in ctx["lines"]] == ["plea", "other"]
    assert [ln["text_emotion"] for ln in ctx["lines"]] == ["angry", "neutral"]


def test_unknown_ids_like_audience_are_kept():
    from backend.pipeline.context.build import normalize
    ctx = normalize({"lines": [{"line_id": 0, "addressee": "audience"}, {"line_id": 1, "addressee": None}]},
                    {"SPEAKER_00": "S1"})
    assert [ln["addressee"] for ln in ctx["lines"]] == ["audience", None]


def test_tone_reaches_the_translator():
    f = fuse(STEP3_LINE, {"line_id": 0, "intent": "threat", "text_emotion": "angry", "tone": "cold"})
    assert translation_input(STEP3_LINE, f, {})["tone"] == "cold"
