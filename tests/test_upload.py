"""Players' own clips: checks, the background job, and the API, with every paid or networked step stubbed."""
import json
import subprocess

import pytest

from backend.game import library as L
from backend.game import stt, upload
from tests.test_game import LINES, clip, dub_json  # noqa: F401  (clip is a fixture)


class Now:
    """Runs a queued job straight away, so tests see the finished status."""
    def submit(self, fn, *args):
        fn(*args)


def label(safe=True, lang="hi"):
    return {"language_spoken": lang, "family_safe": {"ok": safe, "reason": "" if safe else "a slap"},
            "summary": "a fight", "speakers": [{"id": "S1", "description": "younger brother", "gender": "male"}],
            "lines": [{"speaker": "S1", "start": a, "end": b, "transcript": "भैया ये क्या कर दिया",
                       "emotion": "angry", "intensity": "high", "stressed_words": ["भैया"]} for a, b, _, _ in LINES]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(upload, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(upload, "library_dir", lambda: lib)
    monkeypatch.setattr(L, "library_dir", lambda: lib)
    monkeypatch.setattr(upload, "_worker", Now())
    monkeypatch.setattr(upload, "video_json", lambda prompt, video, **k: label())   # noqa: E731
    monkeypatch.setattr(stt, "_call", lambda y, code, sr: {"transcript": "भैया ये क्या कर दिया"})
    from backend.pipeline import dub as D
    dubbed = json.loads(dub_json(tmp_path).read_text())
    monkeypatch.setattr(D, "process_clip", lambda clip, lines, src, tgt: {**dubbed, "source_language": src,
                                                                       "target_language": tgt})
    return lib


def test_a_clip_becomes_a_playable_scene_marked_as_the_players(env, clip):  # noqa: F811
    job = upload.start(clip, "ta", title="My fight scene")
    s = upload.status(job)
    assert s["state"] == "ready" and s["entry"] == f"{job}.hi-ta" and s["source"] == "hi"
    entry = json.loads((env / s["entry"] / "entry.json").read_text())
    assert entry["uploaded"] is True and entry["title"] == "My fight scene"
    assert any(e["uploaded"] for e in L.Library(env).entries("ta"))
    upload.remove(s["entry"])
    assert not (env / s["entry"]).exists()


def test_unsafe_clips_and_other_languages_are_turned_away(env, clip, monkeypatch):  # noqa: F811
    monkeypatch.setattr(upload, "video_json", lambda prompt, video, **k: label(safe=False))
    s = upload.status(upload.start(clip, "ta"))
    assert s["state"] == "failed" and "a slap" in s["error"] and "family-friendly" in s["error"]
    monkeypatch.setattr(upload, "video_json", lambda prompt, video, **k: label(lang="other"))
    assert "couldn't hear" in upload.status(upload.start(clip, "ta"))["error"]


def test_checks_before_queueing(env, clip, tmp_path, monkeypatch):  # noqa: F811
    with pytest.raises(upload.UploadError, match="already in that language"):
        upload.start(clip, "hi", source="hi")
    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:d=4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)], check=True)
    with pytest.raises(upload.UploadError, match="no sound"):
        upload.start(silent, "ta")
    monkeypatch.setitem(upload.config()["game"]["upload"], "max_seconds", 5)
    with pytest.raises(upload.UploadError, match="3-5 seconds"):
        upload.start(clip, "ta")
    monkeypatch.setitem(upload.config()["game"]["upload"], "max_seconds", 60)
    monkeypatch.setattr(upload, "started_today", lambda: 99)
    with pytest.raises(upload.UploadError, match="Try again tomorrow"):
        upload.start(clip, "ta")


def test_unknown_or_odd_job_ids_are_not_found(env):
    for bad in ("nope", "../etc", "up..x"):
        with pytest.raises(KeyError):
            upload.status(bad)


def test_upload_api_needs_consent_and_reports_problems(env, clip):  # noqa: F811
    from fastapi.testclient import TestClient
    from backend.api import app as A
    client = TestClient(A.app)
    files = {"video": ("clip.mp4", clip.read_bytes(), "video/mp4")}
    assert client.post("/api/uploads", data={"target": "ta"}, files=files).status_code == 400
    assert client.post("/api/uploads", data={"target": "hi", "source": "hi", "consent": "true"}, files=files).status_code == 422
    r = client.post("/api/uploads", data={"target": "ta", "consent": "true"}, files=files)
    assert r.status_code == 200 and client.get(f"/api/uploads/{r.json()['id']}").json()["state"] == "ready"


def test_language_names_and_codes_are_understood():
    assert [upload.language_code(x) for x in ("hi", "hin", "Hindi", "TAMIL", "tel", "other", None)] == \
        ["hi", "hi", "hi", "ta", "te", None, None]
