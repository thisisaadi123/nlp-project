"""A player's own movie clip -> a playable scene in the library, without a GPU.

    check -> watch -> hear -> translate -> voices

- check: 3-45 s with an audio track (game.upload.max_seconds); re-encoded to 720p with timestamps from 0 (like add_clip.py).
- watch: Gemini watches the clip and labels who says what, when and how (the same labels the built-in
  scenes were made from), which language is spoken, and whether it is family-friendly. Clips with
  swearing, violence, blood, sexual content, substances or frightening scenes are turned away.
- hear: Sarvam speech-to-text on each line's audio for the exact words (Gemini's are approximate).
- translate: step 4 as for every scene (captions, context, Sarvam translation and the checks).
- voices: the library entry is built, voice lines included, and marked as the player's.

Jobs run one at a time in a background thread; the website polls the status file. About ₹2.5-3 of
Sarvam credits per clip; game.upload.daily_limit caps how many start per day.
"""
import json
import secrets as token
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from backend.game import stt
from backend.game.library import build_entry, library_dir
from backend.llm.client import LLMError, video_json
from backend.scoring.prosody import SR, load_audio
from backend.settings import ROOT, config

UPLOADS = ROOT / "data/uploads"
STEPS = [("check", "Checking your clip"), ("watch", "Watching the scene"), ("hear", "Hearing every line"),
         ("translate", "Understanding and translating"), ("voices", "Recording voice lines")]
EMOTIONS = ["angry", "sad", "happy", "fearful", "surprised", "disgust", "neutral", "calm", "sarcastic", "comic"]
STALE_S = 15 * 60          # a job that hasn't moved in this long died with the server
_worker = ThreadPoolExecutor(max_workers=1)

PROMPT = """You are preparing a short film clip for a family-friendly dubbing game played by children
and adults. Watch and listen to the whole clip, then answer in JSON with exactly these fields:

{{
  "language_spoken": "hi" | "ta" | "bn" | "mr" | "te" | "other" (the main language of the dialogue),
  "family_safe": {{"ok": true/false, "reason": "if not ok: what makes it unsuitable, in a few words"}},
  "summary": "one sentence: what happens in the scene",
  "setting": "where it takes place",
  "speakers": [{{"id": "S1", "description": "who they are", "gender": "male/female", "approx_age": "20s"}}],
  "relationships": "how the speakers are related (siblings, friends, boss/employee...) and who is senior",
  "lines": [{{"speaker": "S1", "start": 0.0, "end": 2.1,
             "transcript": "exact words in the language's own script, as spoken (keep slang and code-mixing)",
             "emotion": one of {emotions},
             "intensity": "low/medium/high",
             "stressed_words": ["word(s) said with the most emphasis"]}}]
}}

family_safe is false for ANY swearing or insults, violence (even a slap or a shove), blood or injury,
sexual content or kissing, alcohol, smoking or drugs, or frightening scenes. Times are seconds from the
start of the clip; a line is one person's continuous speech. Skip songs and background chatter."""


class UploadError(ValueError):
    """Something the player can fix (wrong length, no audio, daily limit): the message says what."""


def _cfg() -> dict:
    return config()["game"]["upload"]


def _probe(path: Path) -> tuple[float, bool]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                          "-of", "json", str(path)], capture_output=True, text=True)
    if out.returncode != 0:
        raise UploadError("That file isn't a video we can read.")
    info = json.loads(out.stdout)
    duration = float(info.get("format", {}).get("duration") or 0)
    return duration, any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


def _write(job: Path, **fields) -> dict:
    path = job / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(fields, updated=time.time())
    path.write_text(json.dumps(status, ensure_ascii=False, indent=1))
    return status


def started_today() -> int:
    today = date.today().isoformat()
    return sum(json.loads(p.read_text()).get("created", "").startswith(today) for p in UPLOADS.glob("*/status.json"))


def start(video: Path, target: str, source: str | None = None, title: str | None = None) -> str:
    """Check the clip and queue it; returns the job id. Raises UploadError with a message for the player."""
    cfg, langs = _cfg(), config()["languages"]
    if target not in langs or (source and source not in langs):
        raise UploadError("Pick one of the game's languages.")
    if source == target:
        raise UploadError("The clip is already in that language. Pick another language to dub into.")
    if started_today() >= cfg["daily_limit"]:
        raise UploadError(f"Today's {cfg['daily_limit']} new scenes are made. Try again tomorrow.")
    duration, has_audio = _probe(video)
    if not has_audio:
        raise UploadError("That clip has no sound. Upload a scene with dialogue.")
    if not 3 <= duration <= cfg["max_seconds"]:
        raise UploadError(f"Clips must be 3-{cfg['max_seconds']} seconds long (this one is {duration:.0f} s). Trim it and try again.")
    job_id = f"up{date.today().strftime('%m%d')}{token.token_hex(3)}"
    job = UPLOADS / job_id
    job.mkdir(parents=True)
    # 720p max, timestamps from 0 so the website's clock matches the audio (see scripts/add_clip.py)
    _ffmpeg("-i", str(video), "-vf", "scale=-2:'min(720,ih)'", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
            str(job / f"{job_id}.mp4"))
    _write(job, id=job_id, state="working", step="watch", steps=[k for k, _ in STEPS], target=target,
           source=source, title=(title or "").strip()[:60] or "Your scene", created=date.today().isoformat())
    _worker.submit(_run, job_id)
    return job_id


def status(job_id: str) -> dict:
    path = UPLOADS / job_id / "status.json"
    if not job_id.isalnum() or not path.exists():
        raise KeyError(job_id)
    s = json.loads(path.read_text())
    if s.get("state") == "working" and time.time() - s.get("updated", 0) > STALE_S:
        s = _write(path.parent, state="failed", error="Making this scene was interrupted. Try uploading it again.")
    return {k: v for k, v in s.items() if k != "updated"} | {"labels": dict(STEPS)}


def _label(job: Path, clip: Path) -> dict:
    small = job / "small.mp4"      # a few MB is plenty to judge the scene and keeps the request small
    _ffmpeg("-i", str(clip), "-vf", "scale=-2:360", "-c:v", "libx264", "-crf", "32", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "64k", "-ac", "1", str(small))
    label = video_json(PROMPT.format(emotions="/".join(EMOTIONS)), small.read_bytes(),
                       validate=lambda r: isinstance(r.get("lines"), list) and isinstance(r.get("family_safe"), dict))
    (job / "label.json").write_text(json.dumps(label, ensure_ascii=False, indent=1))
    return label


def _hear(clip: Path, label_lines: list[dict], src: str, duration: float) -> list[dict]:
    """Step 3's line format from the labels, with Sarvam's words where it heard any."""
    y = load_audio(str(clip))
    lines = []
    for ln in label_lines:
        start, end = float(ln.get("start") or 0), float(ln.get("end") or 0)
        start, end = max(0.0, min(start, duration)), max(0.0, min(end, duration))
        if end - start < 0.3:
            continue
        seg = y[int(max(0.0, start - 0.15) * SR): int(min(duration, end + 0.15) * SR)]
        heard = stt._call(np.asarray(seg, dtype=np.float32), f"{src}-IN", SR) or {}
        text = str(heard.get("transcript") or "").strip() or str(ln.get("transcript") or "").strip()
        if not text:
            continue
        lines.append({"line_id": len(lines) + 1, "speaker": ln.get("speaker"), "start": round(start, 2),
                      "end": round(end, 2), "language": src, "text": text,
                      "emotion": {"label": ln.get("emotion")}, "stressed_words": ln.get("stressed_words") or []})
    return lines


def _run(job_id: str) -> None:
    job = UPLOADS / job_id
    s = json.loads((job / "status.json").read_text())
    clip = job / f"{job_id}.mp4"
    try:
        label = _label(job, clip)
        safe = label.get("family_safe") or {}
        if not safe.get("ok", False):
            raise UploadError(f"This clip has {safe.get('reason') or 'content'} the game can't use. "
                              "Pick a family-friendly scene.")
        src = s.get("source") or label.get("language_spoken")
        if src not in config()["languages"]:
            raise UploadError("We couldn't hear Hindi, Tamil, Bengali, Marathi or Telugu in this clip.")
        if src == s["target"]:
            raise UploadError("The clip is already in that language. Pick another language to dub into.")
        _write(job, step="hear", source=src)
        duration, _ = _probe(clip)
        lines = _hear(clip, label.get("lines") or [], src, duration)
        if not lines:
            raise UploadError("We couldn't find any dialogue in this clip.")
        _write(job, step="translate")
        from backend.pipeline.dub import process_clip
        dub = process_clip(str(clip), lines, src, s["target"])
        (job / "dub.json").write_text(json.dumps(dub, ensure_ascii=False, indent=1))
        _write(job, step="voices")
        out = build_entry(job / "dub.json", clip, title=s["title"])
        entry = json.loads((out / "entry.json").read_text())
        entry["uploaded"] = True
        (out / "entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1))
        if not any(c["playable"] for c in entry["lines"]):
            raise UploadError("None of the lines in this clip had clear enough speech to play.")
        _write(job, state="ready", step="done", entry=out.name)
    except UploadError as e:
        _write(job, state="failed", error=str(e))
    except (LLMError, subprocess.CalledProcessError, OSError, KeyError, ValueError) as e:
        _write(job, state="failed", error="Something went wrong making this scene. Try again in a minute.",
               detail=str(e)[:300])


def remove(entry_id: str) -> None:
    """Delete a player's scene from the library (only uploaded ones)."""
    path = library_dir() / entry_id / "entry.json"
    if not path.exists() or not json.loads(path.read_text()).get("uploaded"):
        raise KeyError(entry_id)
    shutil.rmtree(path.parent)
