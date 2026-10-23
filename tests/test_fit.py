from backend.pipeline.translate import refine as R

SOURCE = {"line_id": 1, "text": "भैया, ये क्या कर दिया?"}  # 8 syllables -> target range 6-10


def item(translation):
    return {"line_id": 1, "translation": translation, "romanized": "",
            "address_terms": [{"source": "भैया", "target": "அண்ணா"}],
            "emphasis": [{"source_word": "भैया", "target_word": "அண்ணா"}]}


def fake_rewrite(monkeypatch, translation):
    calls = []

    def chat_json(messages, **kwargs):
        calls.append(messages)
        return {"lines": [{"line_id": 1, "translation": translation, "romanized": "r"}]}

    monkeypatch.setattr(R, "chat_json", chat_json)
    return calls


def test_line_in_range_is_left_alone(monkeypatch):
    calls = fake_rewrite(monkeypatch, "unused")
    out = R.fit_lengths([SOURCE], [item("அண்ணா, என்ன பண்ணிட்ட?")], "ta")  # 7 syllables
    assert calls == [] and "fitted_from" not in out[0] and out[0]["length_ratio"] == 0.88


def test_closer_rewrite_that_keeps_address_term_is_accepted(monkeypatch):
    long = "அண்ணா, நீ இப்போது என்ன காரியம் பண்ணிவிட்டாய் என்று சொல்லு?"
    fake_rewrite(monkeypatch, "அண்ணா, நீ என்ன பண்ணிட்ட?")
    out = R.fit_lengths([SOURCE], [item(long)], "ta")
    assert out[0]["translation"] == "அண்ணா, நீ என்ன பண்ணிட்ட?"
    assert out[0]["fitted_from"] == long


def test_rewrite_that_drops_address_term_is_rejected(monkeypatch):
    long = "அண்ணா, நீ இப்போது என்ன காரியம் பண்ணிவிட்டாய் என்று சொல்லு?"
    fake_rewrite(monkeypatch, "நீ என்ன பண்ணிட்ட?")  # shorter, but lost அண்ணா
    out = R.fit_lengths([SOURCE], [item(long)], "ta")
    assert out[0]["translation"] == long and "fitted_from" not in out[0]


def test_rewrite_that_changes_register_is_rejected(monkeypatch):
    long = "அண்ணா, நீ இப்போது என்ன காரியம் பண்ணிவிட்டாய் என்று சொல்லு?"
    old = item(long) | {"register": "intimate"}

    def chat_json(messages, **kwargs):
        return {"lines": [{"line_id": 1, "translation": "அண்ணா, நீங்க என்ன பண்ணீங்க?", "register": "respectful"}]}

    monkeypatch.setattr(R, "chat_json", chat_json)
    out = R.fit_lengths([SOURCE], [old], "ta")
    assert out[0]["translation"] == long


def test_rewrite_that_pads_by_repeating_a_name_is_rejected():
    old = item("அண்ணா, இது என்ன")                              # far too short
    padded = {"translation": "அண்ணா, அண்ணா, இது என்ன ஆச்சு"}     # repeats அண்ணா
    assert not R._acceptable(old, padded, SOURCE["text"])


def test_slightly_short_line_is_not_rewritten(monkeypatch):
    calls = fake_rewrite(monkeypatch, "unused")
    R.fit_lengths([SOURCE], [item("அண்ணா, என்ன ஆச்சு?")], "ta")   # 6/8 = 0.75: short but above 0.6
    assert calls == []
