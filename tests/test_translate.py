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


def test_copied_words_catches_source_words_in_target_script():
    from backend.pipeline.translate.copies import copied_words
    assert copied_words("நான் கெளம்புறேன்டா.", "मैं गुम्बुरेंटा हूँ।", "ta", "hi") == ["गुम्बुरेंटा"]
    assert copied_words("அது ஒன்னும் இல்லைங்க அண்ணா.", "कुछ भी नहीं, अन्ना।", "ta", "hi") == ["अन्ना"]
    assert "पोनना" in copied_words("அதான் அத்தை பொண்ணா இருந்தா", "वही अंटी पोनना ने किया", "ta", "hi")


def test_copied_words_passes_real_translations():
    from backend.pipeline.translate.copies import copied_words
    assert copied_words("சரிங்க அண்ணி.", "ठीक है, भाभी।", "ta", "hi") == []
    assert copied_words("வாங்கண்ணே வீட்டுக்குப் போலாம்.", "चलो, भैया, घर चलें।", "ta", "hi") == []
    assert copied_words("और कौन से गांव का है रे?", "எந்த ஊர்க்காரன் டா நீ?", "hi", "ta") == []


def test_line_payload_gives_genders_and_warns_about_a_copying_draft():
    line = {"line_id": 1, "speaker": "S1", "addressee": "S2", "text": "நான் கெளம்புறேன்டா."}
    people = {"S1": {"gender": "male"}, "S2": {"gender": "female"}}
    p = _line_payload(line, "मैं गुम्बुरेंटा हूँ।", people, "ta", "hi")
    assert p["speaker_gender"] == "male" and p["addressee_gender"] == "female"
    assert "गुम्बुरेंटा" in p["draft_warning"]


def test_fix_copies_keeps_names_and_replaces_copies(monkeypatch):
    from backend.pipeline.translate import refine as r
    lines = [{"line_id": 1, "text": "அண்ணே, இவன்தானே என் பிரெண்டு வேல்முருகன்."}]
    out = [{"line_id": 1, "translation": "भैया, इवण्टाने मेरा दोस्त वेलमुरुगन है।"}]
    monkeypatch.setattr(r, "chat_json", lambda *a, **k: {"lines": [
        {"line_id": 1, "translation": "भैया, यही है मेरा दोस्त वेलमुरुगन।", "kept": ["वेलमुरुगन"]}]})
    fixed = r.fix_copies(lines, out, "ta", "hi")[0]
    assert fixed["translation"] == "भैया, यही है मेरा दोस्त वेलमुरुगन।"
    assert fixed["copies_fixed_from"].startswith("भैया, इवण्टाने") and fixed["copied_words"] == ["वेलमुरुगन"]


def test_fix_copies_rejects_a_repair_that_still_copies(monkeypatch):
    from backend.pipeline.translate import refine as r
    lines = [{"line_id": 1, "text": "நான் கெளம்புறேன்டா."}]
    out = [{"line_id": 1, "translation": "मैं गुम्बुरेंटा हूँ।"}]
    monkeypatch.setattr(r, "chat_json", lambda *a, **k: {"lines": [
        {"line_id": 1, "translation": "मैं गुम्बुरेंटा रहा हूँ।", "kept": []}]})
    fixed = r.fix_copies(lines, out, "ta", "hi")[0]
    assert fixed["translation"] == "मैं गुम्बुरेंटा हूँ।" and fixed["copied_words"] == ["गुम्बुरेंटा"]


def test_primary_only_does_not_fall_back(monkeypatch):
    from backend.llm import client
    calls = []
    def fail(model, *a, **k):
        calls.append(model)
        raise client.LLMError("quota")
    monkeypatch.setattr(client, "_groq", fail)
    monkeypatch.setattr(client, "_gemini", lambda *a, **k: calls.append("gemini") or {"ok": 1})
    with pytest.raises(client.LLMError):
        client.chat_json([{"role": "user", "content": "x"}], cache=False, primary_only=True)
    assert calls == [client.config()["llm"]["text_model"]]


def test_review_applies_fixes_and_keeps_ok_lines(monkeypatch):
    from backend.pipeline.translate import refine as r
    lines = [{"line_id": 1, "text": "சரிங்க அண்ணி."}, {"line_id": 2, "text": "வணக்கம் அண்ணே."}]
    out = [{"line_id": 1, "translation": "ठीक है, भैया।"}, {"line_id": 2, "translation": "नमस्ते भैया।"}]
    monkeypatch.setattr(r, "chat_json", lambda *a, **k: {"lines": [
        {"line_id": 1, "verdict": "fixed", "problem": "anni is bhabhi", "translation": "ठीक है, भाभी।"},
        {"line_id": 2, "verdict": "ok", "translation": "नमस्ते भैया।"}]})
    a, b = r.review(lines, out, ["", ""], "ta", "hi")
    assert a["translation"] == "ठीक है, भाभी।" and a["reviewed_from"] == "ठीक है, भैया।"
    assert b["translation"] == "नमस्ते भैया।" and "reviewed_from" not in b


def test_written_words_finds_bookish_tamil_and_passes_spoken():
    from backend.pipeline.translate.written import written_words
    assert written_words("இன்றிரவு இந்தியா டிவி பார்க்கவும். வரலாறு மாறப் போகிறது.", "ta") == ["இன்றிரவு", "பார்க்கவும்", "போகிறது"]
    assert written_words("அரே, காமதேவ் வந்திருக்கிறான். அவன் நின்றான்.", "ta") == ["வந்திருக்கிறான்", "நின்றான்"]
    assert written_words("நைட் எப்படி போச்சு? வா டா வா. சொல்லுங்க அண்ணா, நான் போறேன்", "ta") == []
    assert written_words("कृपया अनुमति दीजिए।", "hi") == ["कृपया", "अनुमति"]
    assert written_words("चल यार, घर चलते हैं।", "hi") == []


def test_fix_written_takes_only_a_more_spoken_rewrite(monkeypatch):
    from backend.pipeline.translate import refine as r
    lines = [{"line_id": 1, "text": "देख लेना"}, {"line_id": 2, "text": "रात कैसी कटी"}]
    out = [{"line_id": 1, "translation": "டிவி பார்க்கவும்.", "register": "casual"},
           {"line_id": 2, "translation": "இரவு எப்படி போகிறது?", "register": "casual"}]
    monkeypatch.setattr(r, "chat_json", lambda *a, **k: {"lines": [
        {"line_id": 1, "translation": "டிவி பாத்துக்கோ."},
        {"line_id": 2, "translation": "இரவு எப்படி போகிறது? சொல்லுங்கள்"}]})      # worse: refused
    a, b = r.fix_written(lines, out, "ta")
    assert a["translation"] == "டிவி பாத்துக்கோ." and a["written_fixed_from"] == "டிவி பார்க்கவும்."
    assert b["translation"] == "இரவு எப்படி போகிறது?" and b["written_words"] == ["போகிறது"]


def test_sarvam_drafts_fall_back_to_empty_on_errors(monkeypatch):
    from backend.pipeline.translate import sarvam
    calls = []
    def fake(text, src, tgt, gender=None):
        calls.append(gender)
        if text == "bad":
            raise sarvam.SarvamError("x")
        return f"<{text}>"
    monkeypatch.setattr(sarvam, "translate", fake)
    out = sarvam.drafts([{"text": "a", "speaker": "S1"}, {"text": "bad", "speaker": "S2"}], "hi", "ta",
                        {"S1": {"gender": "female"}})
    assert out == ["<a>", ""] and calls == ["female", None]


def test_fix_copies_ignores_edits_when_only_names_were_flagged(monkeypatch):
    from backend.pipeline.translate import refine as r
    lines = [{"line_id": 1, "text": "राम भाई, रात कैसी कटी?"}]
    out = [{"line_id": 1, "translation": "ராம் டா, நைட் எப்படி போச்சு?"}]
    monkeypatch.setattr(r, "chat_json", lambda *a, **k: {"lines": [
        {"line_id": 1, "translation": "ராம் டா, இரவு எப்படி போச்சு?", "kept": ["ராம்"]}]})
    assert r.fix_copies(lines, out, "hi", "ta")[0]["translation"] == "ராம் டா, நைட் எப்படி போச்சு?"


def test_romanization_that_no_longer_matches_the_line_is_replaced():
    from backend.pipeline.translate.refine import checked_romanization
    assert checked_romanization("Theek hai bhabhi.", "ठीक है भाभी।", "hi") == "Theek hai bhabhi."
    assert checked_romanization("Dekho, chheh sau de do na bhaiya", "ठीक है भाभी।", "hi") == "theek hai bhaabhi."
    assert checked_romanization(None, "எந்த ஊருடா நீ?", "ta") == "endha ooruda nee?"


def test_kinship_check_finds_wrong_relation_words_and_particles():
    from backend.pipeline.translate.kinship import kinship_issues, relation_hints
    assert kinship_issues("சரிங்க அண்ணி.", "సరే అక్కా.", "ta", "te") == ["elder brother's wife: use వదిన"]
    assert kinship_issues("சரிங்க அண்ணி.", "ঠিক আছে দিদি।", "ta", "bn") == ["elder brother's wife: use বৌদি"]
    assert kinship_issues("சரிங்க அண்ணி.", "ठीक है, भाभी।", "ta", "hi") == []
    assert kinship_issues("राम भाई, रात कैसी कटी?", "ராம் மச்சான், நைட் எப்படி போச்சு?", "hi", "ta") == []
    assert "use 'రా'" in kinship_issues("நான் கெளம்புறேன்டா.", "నేను బయలుదేరుతున్నా, డా.", "ta", "te")[0]
    assert relation_hints("சரிங்க அண்ணி.", "ta", "mr") == ["elder brother's wife: वहिनी"]


def test_fix_kinship_accepts_only_a_corrected_line(monkeypatch):
    from backend.pipeline.translate import refine as r
    lines = [{"line_id": 1, "text": "சரிங்க அண்ணி."}, {"line_id": 2, "text": "சரிங்க அண்ணி."}]
    out = [{"line_id": 1, "translation": "సరే అక్కా."}, {"line_id": 2, "translation": "సరే అక్కా."}]
    monkeypatch.setattr(r, "chat_json", lambda *a, **k: {"lines": [
        {"line_id": 1, "translation": "సరే వదినా."}, {"line_id": 2, "translation": "సరే అక్కయ్యా."}]})
    a, b = r.fix_kinship(lines, out, "ta", "te")
    assert a["translation"] == "సరే వదినా." and "kinship_problems" not in a
    assert b["translation"] == "సరే అక్కా." and b["kinship_problems"]


def test_a_romanization_that_changes_words_is_not_used():
    from backend.pipeline.translate.refine import checked_romanization as c
    # Sarvam turned Marathi into Hindi: refused, falls back to ours
    assert c(None, "आत ठेव.", "mr", "Aat rakh.") == "aat thev."
    assert c(None, "ठीक आहे वहिनी.", "mr", "Theek hai Vahini.") == "theek aahe vahinee."
    # a faithful one is kept
    assert c(None, "घरी गेलास का?", "mr", "Ghari gelaas ka?") == "Ghari gelaas ka?"


def test_spoken_aunt_counts_but_not_inside_other_words():
    from backend.pipeline.translate.kinship import kinship_issues
    assert kinship_issues("ஆயிருந்தா அத்த பொண்ணா இருந்தா", "যদি ওই আন্টি হতেন।", "ta", "bn") == ["father's sister: use পিসি"]
    assert kinship_issues("அத்தனை பேரு வந்தாங்க", "এত লোক এসেছে।", "ta", "bn") == []
