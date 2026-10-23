import pytest

from scripts.eval_register import check
from scripts.testset import next_clip_id, parse_time


@pytest.mark.parametrize("text, seconds", [("75", 75), ("1:15", 75), ("01:01:15", 3675), ("1:15.5", 75.5)])
def test_parse_time(text, seconds):
    assert parse_time(text) == seconds


@pytest.mark.parametrize("bad", ["1m15", "", "1:2:3:4", "-5"])
def test_parse_time_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_time(bad)


def test_next_clip_id_skips_taken_ids():
    rows = [{"clip_id": "ta_01"}, {"clip_id": "ta_02"}, {"clip_id": "hi_01"}]
    assert next_clip_id("ta", rows) == "ta_03"
    assert next_clip_id("bn", rows) == "bn_01"


CASE = {"expect": [["அண்ணா", "அண்ணே"], ["நீ"]], "forbid": ["சகோதர"]}


def test_register_check_passes_with_every_group_matched():
    assert check("அண்ணா, நீ என்ன பண்ணிட்ட?", CASE) == (True, "")


def test_register_check_reports_missing_group():
    ok, why = check("அண்ணா, என்ன ஆச்சு?", CASE)
    assert not ok and "நீ" in why


def test_register_check_reports_forbidden_term():
    ok, why = check("சகோதரரே, நீ என்ன செய்தாய்? அண்ணா", CASE)
    assert not ok and "சகோதர" in why


def test_whole_word_terms_do_not_match_inside_other_words():
    from scripts.eval_register import has
    assert has("வா டா போலாம்", "=டா")
    assert has("டா, போ!", "=டா")
    assert not has("கவலைப்பட வேண்டாம்.", "=டா")   # டா inside வேண்டாம்
    assert has("வேண்டாம்", "டா")                    # plain terms still match substrings


def test_forbidden_whole_word_reported_without_marker():
    ok, why = check("அண்ணா, நீ வா", {"expect": [], "forbid": ["=நீ"]})
    assert not ok and why == "contains நீ"
