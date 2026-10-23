"""Step 5 backend: library, one round of play, and the web API, on a synthetic clip."""
import io
import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from backend.game import library as L
from backend.game.analyzer import Analysis, ProsodyAnalyzer
from backend.game.play import calibrate, content_similarity, play, stressed_span, valid_baseline
from backend.scoring.arousal import FEATURES
from backend.scoring.prosody import SR, extract
from tests.test_scoring import voice

LINES = [(0.5, 2.0, 180, 130), (2.6, 4.1, 150, 150), (4.7, 6.2, 200, 120)]  # start, end, f0 from, f0 to


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A 7 s video whose audio has three voice-like lines from one speaker."""
    d = tmp_path_factory.mktemp("clip")
    y = np.zeros(int(7 * SR), np.float32)
    for i, (a, b, f1, f2) in enumerate(LINES):
        v = voice(f1, f2, b - a, seed=i)[int(0.3 * SR):-int(0.3 * SR)]
        y[int(a * SR): int(a * SR) + len(v)] = v
    sf.write(d / "audio.wav", y, SR)
    path = d / "zz_01.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:d=7",
                    "-i", str(d / "audio.wav"), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(path)], check=True)
    return path


def dub_json(tmp, src="hi", tgt="ta"):
    lines = []
    for i, (a, b, _, _) in enumerate(LINES):
        lines.append({"line_id": i, "speaker": "SPEAKER_00", "addressee": "SPEAKER_01", "start": a, "end": b,
                      "text": "भैया ये क्या कर दिया", "text_roman": "bhaiya ye kya kar diya",
                      "emotion_intent": {"label": "angry", "intensity": "high", "tone": "furious", "mismatch": False},
                      "dub": {"translation": "அண்ணா நீ என்ன பண்ணிட்ட", "romanized": "anna nee enna pannitta",
                              "register": "intimate", "delivery": "Shout it", "_provider": "groq:openai/gpt-oss-120b",
                              "emphasis": [{"source_word": "भैया", "target_word": "அண்ணா"}]}})
    path = tmp / f"zz_01.{src}-{tgt}.json"
    context = {"summary": "a fight", "mood": "tense",
               "speakers": {"SPEAKER_00": {"description": "younger brother, 19, furious", "gender": "male"},
                            "SPEAKER_01": {"description": "elder brother (28)", "gender": "male"}},
               "relationships": [{"between": ["SPEAKER_00", "SPEAKER_01"], "relation": "siblings", "closeness": "close"}]}
    path.write_text(json.dumps({"source_language": src, "target_language": tgt,
                                "context": context, "lines": lines},
                               ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def lib(clip, tmp_path_factory):
    root = tmp_path_factory.mktemp("library")
    L.build_entry(dub_json(tmp_path_factory.mktemp("dub")), clip, out_root=root, title="Test Film")
    return L.Library(root)


# --- library ---------------------------------------------------------------------------------------

def test_entry_has_cards_audio_and_a_speaker_baseline(lib):
    e = lib.entry("zz_01.hi-ta")
    assert [c["playable"] for c in e["lines"]] == [True, True, True]
    c = e["lines"][0]
    assert c["translation"] == "அண்ணா நீ என்ன பண்ணிட்ட" and c["stress"] == ["அண்ணா"]
    assert c["emotion"] == "angry" and c["duration"] == 1.5 and c["model"] == "groq:openai/gpt-oss-120b"
    assert e["calibration_text"].startswith("இன்று")
    ref = lib.reference("zz_01.hi-ta", 1)
    assert ref.baseline is not None and set(ref.baseline) == set(FEATURES)
    assert abs(ref.prosody.duration - 1.5) < 0.3
    assert lib.line_audio("zz_01.hi-ta", 2).exists()
    assert lib.entries()[0] == {"id": "zz_01.hi-ta", "clip": "zz_01", "title": "Test Film", "source_language": "hi",
                                "target_language": "ta", "scene": "a fight", "mood": "tense", "difficulty": "easy",
                                "lines": 3, "characters": ["younger brother", "elder brother"]}
    assert lib.entries("ta")[0]["id"] == "zz_01.hi-ta" and lib.entries("hi") == []


def test_cards_carry_names_coaching_and_a_melody_line(lib):
    e = lib.entry("zz_01.hi-ta")
    c = e["lines"][0]
    assert c["speaker_name"] == "younger brother" and c["addressee_name"] == "elder brother"
    assert c["coach"] == "Angry, loud. Hit அண்ணா hard. About 2 seconds."
    sh = c["shape"]
    assert abs(sh["duration"] - 1.5) < 0.3 and sh["step"] == 0.05
    assert len(sh["pitch"]) == len(sh["loud"]) == len(sh["voiced"]) > 20
    assert all(-12 <= v <= 12 for v in sh["pitch"]) and all(0 <= v <= 1 for v in sh["loud"])
    assert [w["w"] for w in sh["words"]] == "அண்ணா நீ என்ன பண்ணிட்ட".split()
    assert [w["stress"] for w in sh["words"]] == [True, False, False, False]
    assert sh["words"][-1]["end"] == sh["duration"]
    assert -0.2 < c["onset"] < 0.5          # the voice may start a moment before the labelled line start
    assert e["characters"][0]["name"] == "younger brother" and e["relationships"][0]["relation"] == "siblings"
    assert e["title"] == "Test Film" and e["difficulty"] == "easy"


def test_coach_sentences():
    from backend.game.coach import coach
    assert coach({"emotion": "sad", "intensity": "low", "stress": [], "duration": 3.2}) == "Sad, soft. About 3 seconds."
    assert coach({"emotion": "neutral", "intensity": None, "stress": ["भैया"], "duration": 1.0}) == \
        "Matter-of-fact. Hit भैया hard. About 1 second."
    assert coach({"emotion": None, "duration": 2}) == "Natural. About 2 seconds."


@pytest.mark.parametrize("bad", ["../zz_01.hi-ta", "zz_01.hi-ta/../..", "nope", "", "/etc"])
def test_library_refuses_unknown_or_escaping_ids(lib, bad):
    with pytest.raises(KeyError):
        lib.entry(bad)


def test_unknown_line_raises(lib):
    with pytest.raises(KeyError):
        lib.reference("zz_01.hi-ta", 99)


# --- one round -------------------------------------------------------------------------------------

def test_saying_it_like_the_original_is_a_dub(lib):
    ref = lib.reference("zz_01.hi-ta", 0)
    y = voice(180, 130, 1.5, seed=0)
    r = play(ref, y, None, ProsodyAnalyzer())
    assert r["ok"] and r["verdict"] == "Dub" and r["score"] >= 70
    assert "content" not in r["components"] and r["heard"] is None and r["analyzer"] == "prosody"


def test_a_much_slower_flat_attempt_scores_lower(lib):
    ref = lib.reference("zz_01.hi-ta", 0)
    good = play(ref, voice(180, 130, 1.5, seed=0), None, ProsodyAnalyzer())
    bad = play(ref, voice(140, 140, 3.4, loud=0.1, seed=5), None, ProsodyAnalyzer())
    assert bad["score"] < good["score"] - 15
    assert any("long" in t for t in bad["tips"])


def test_silence_and_too_short_are_rejected(lib):
    ref = lib.reference("zz_01.hi-ta", 0)
    assert play(ref, np.zeros(SR, np.float32), None, ProsodyAnalyzer())["ok"] is False
    assert "too short" in play(ref, np.zeros(SR // 4, np.float32), None, ProsodyAnalyzer())["error"]


class FakeASR:
    """Stands in for the "full" analyzer (step 3's ASR) with a fixed transcript."""
    name = "full"

    def __init__(self, text, words=()):
        self.text, self.words = text, list(words)

    def analyze(self, y, lang):
        return Analysis(extract(y), self.text, self.words, 0.9)


def test_full_mode_adds_content_and_finds_the_stressed_word(lib):
    ref = lib.reference("zz_01.hi-ta", 0)
    words = [{"w": "அண்ணா", "start": 0.3, "end": 0.8}, {"w": "நீ", "start": 0.8, "end": 1.0}]
    right = play(ref, voice(180, 130, 1.5), None, FakeASR("அண்ணா நீ என்ன பண்ணிட்ட", words))
    wrong = play(ref, voice(180, 130, 1.5), None, FakeASR("வணக்கம் எல்லாருக்கும்"))
    assert right["components"]["content"] == 100 and right["heard_roman"].startswith("anna")
    assert wrong["components"]["content"] < 50 and wrong["score"] < right["score"]
    assert wrong["tips"][0].startswith("Say the line as written")


def test_content_similarity_tolerates_spelling_variants():
    assert content_similarity("அண்ணா நீ என்ன பண்ணிட்ட", "அண்ணா, நீ என்ன பண்ணிட்ட?", "ta") == 1.0
    assert content_similarity("அண்ணா நீ என்ன பண்ணிட்டே", "அண்ணா நீ என்ன பண்ணிட்ட", "ta") > 0.85
    assert content_similarity("ஹலோ", "அண்ணா நீ என்ன பண்ணிட்ட", "ta") < 0.4


def test_stressed_span():
    words = [{"w": "अरे", "start": 0.1, "end": 0.3}, {"w": "भैया", "start": 0.3, "end": 0.7}]
    assert stressed_span(words, ["भैया"], "hi") == (0.3, 0.7)
    assert stressed_span(words, ["दीदी"], "hi") is None
    assert stressed_span([], ["भैया"], "hi") is None


def test_calibration_and_baseline_validation():
    r = calibrate(voice(150, 150, 2.0))
    assert r["ok"] and set(r["baseline"]) == set(FEATURES)
    assert valid_baseline(r["baseline"]) == r["baseline"]
    assert valid_baseline({"pitch_level": 1}) is None
    assert valid_baseline({k: "x" for k in FEATURES}) is None
    assert valid_baseline({k: float("nan") for k in FEATURES}) is None
    assert valid_baseline("not a dict") is None


# --- web API ---------------------------------------------------------------------------------------

@pytest.fixture
def api(lib, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api import app as A
    monkeypatch.setattr(A, "library", lib)
    return TestClient(A.app)


def wav_bytes(y):
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV")
    return buf.getvalue()


def test_api_thumbnail_subtitles_and_filter(api):
    t = api.get("/api/clips/zz_01.hi-ta/thumbnail")
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg" and t.content[:2] == b"\xff\xd8"
    v = api.get("/api/clips/zz_01.hi-ta/subtitles.vtt")
    assert v.status_code == 200 and v.text.startswith("WEBVTT") and "00:00:00.500 --> 00:00:02.000" in v.text
    assert "அண்ணா நீ என்ன பண்ணிட்ட" in v.text
    assert [c["id"] for c in api.get("/api/clips?lang=ta").json()] == ["zz_01.hi-ta"]
    assert api.get("/api/clips?lang=bn").json() == []
    assert api.get("/api/clips/nope/thumbnail").status_code == 404


def test_api_premiere_mixes_takes_into_the_scene(api):
    takes = [("takes", ("0.wav", wav_bytes(voice(180, 130, 1.5)), "audio/wav")),
             ("takes", ("2.wav", wav_bytes(voice(200, 120, 1.5)), "audio/wav"))]
    r = api.post("/api/premiere", data={"clip": "zz_01.hi-ta", "starts": json.dumps({"0": 0.2, "2": 4.4})}, files=takes)
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and len(r.content) > 10_000
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", "-"],
                           input=r.content, capture_output=True)
    assert set(probe.stdout.decode().split()) == {"video", "audio"}


def test_api_premiere_rejects_bad_input(api):
    one = [("takes", ("0.wav", wav_bytes(voice(180, 130, 1.5)), "audio/wav"))]
    post = lambda data, files=one: api.post("/api/premiere", data=data, files=files).status_code  # noqa: E731
    assert post({"clip": "nope", "starts": "{}"}) == 404
    assert post({"clip": "zz_01.hi-ta", "starts": "not json"}) == 400
    assert post({"clip": "zz_01.hi-ta", "starts": json.dumps({"9": 1.0})},
                [("takes", ("9.wav", wav_bytes(voice(180, 130, 1.0)), "audio/wav"))]) == 400
    assert post({"clip": "zz_01.hi-ta", "starts": json.dumps({"0": 1.0})},
                [("takes", ("take.wav", wav_bytes(voice(180, 130, 1.0)), "audio/wav"))]) == 400
    assert post({"clip": "zz_01.hi-ta", "starts": json.dumps({"0": 1.0})},
                [("takes", ("0.webm", b"not audio", "audio/webm"))]) == 400


def test_api_score_returns_the_players_melody_line(api):
    r = api.post("/api/score", data={"clip": "zz_01.hi-ta", "line": "0"},
                 files={"audio": ("take.wav", wav_bytes(voice(180, 130, 1.5)), "audio/wav")}).json()
    assert len(r["shape"]["pitch"]) > 20 and r["shape"]["words"] == []


def test_api_lists_and_serves_clips(api):
    assert api.get("/api/clips").json()[0]["id"] == "zz_01.hi-ta"
    assert len(api.get("/api/clips/zz_01.hi-ta").json()["lines"]) == 3
    v = api.get("/api/clips/zz_01.hi-ta/video")
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4"
    assert api.get("/api/clips/zz_01.hi-ta/lines/1/audio").headers["content-type"] == "audio/wav"
    assert api.get("/api/clips/nope").status_code == 404
    assert api.get("/api/clips/..%2F..%2Fetc").status_code == 404
    assert api.get("/api/clips/zz_01.hi-ta/lines/42/audio").status_code == 404
    assert "இன்று" in api.get("/api/calibration/ta").json()["text"]
    assert api.get("/api/calibration/xx").status_code == 404


def test_api_calibrate_then_score(api):
    cal = api.post("/api/calibrate", files={"audio": ("cal.wav", wav_bytes(voice(150, 150, 2.0)), "audio/wav")})
    assert cal.status_code == 200
    r = api.post("/api/score", data={"clip": "zz_01.hi-ta", "line": "0", "baseline": json.dumps(cal.json()["baseline"])},
                 files={"audio": ("take.wav", wav_bytes(voice(180, 130, 1.5)), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["calibrated"] is True and body["verdict"] in ("Dub", "Goose") and 0 <= body["score"] <= 100


def test_api_rejects_bad_input(api):
    silence = wav_bytes(np.zeros(SR, np.float32))
    assert api.post("/api/score", data={"clip": "zz_01.hi-ta", "line": "0"},
                    files={"audio": ("s.wav", silence, "audio/wav")}).status_code == 422
    assert api.post("/api/score", data={"clip": "zz_01.hi-ta", "line": "0"},
                    files={"audio": ("x.webm", b"not audio", "audio/webm")}).status_code == 400
    assert api.post("/api/score", data={"clip": "nope", "line": "0"},
                    files={"audio": ("t.wav", wav_bytes(voice(150, 150, 1.0)), "audio/wav")}).status_code == 404
    assert api.post("/api/score", data={"clip": "zz_01.hi-ta", "line": "7"},
                    files={"audio": ("t.wav", wav_bytes(voice(150, 150, 1.0)), "audio/wav")}).status_code == 404
    garbage_baseline = api.post("/api/score", data={"clip": "zz_01.hi-ta", "line": "0", "baseline": "{oops"},
                                files={"audio": ("t.wav", wav_bytes(voice(180, 130, 1.5)), "audio/wav")})
    assert garbage_baseline.status_code == 200 and garbage_baseline.json()["calibrated"] is False
    too_long = wav_bytes(voice(150, 150, 31.0))
    assert api.post("/api/calibrate", files={"audio": ("l.wav", too_long, "audio/wav")}).status_code == 400
