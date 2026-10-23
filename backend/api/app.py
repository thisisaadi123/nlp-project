"""The game's web API, and the website itself (the built frontend, frontend/dist, is served at /).

    uvicorn backend.api.app:app --reload --port 8000

GET  /api/clips?lang=ta                  library entries (one clip in one language direction), easiest first
GET  /api/clips/{id}                     the entry with a delivery card per line
GET  /api/clips/{id}/video               the original scene
GET  /api/clips/{id}/thumbnail           a still frame for the scene picker
GET  /api/clips/{id}/subtitles.vtt       original line + translation, for the scene briefing
GET  /api/clips/{id}/lines/{line}/audio  the original actor's line, as heard in the scene
GET  /api/calibration/{lang}             the sentence a player reads once, calmly
POST /api/calibrate                      form: audio -> the player's neutral-voice baseline
POST /api/score                          form: audio, clip, line, [baseline] -> score, verdict, tips
POST /api/premiere                       form: clip, starts {line: s}, takes (one file per line) -> mp4

The server keeps no sessions and stores no audio: /api/calibrate returns the player's baseline to
the browser, which sends it back with each recording; uploads live only in temporary files.
"""
import json
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles

from backend.game.analyzer import get_analyzer
from backend.game.library import Library
from backend.game.play import calibrate, play, valid_baseline
from backend.game.premiere import MAX_TAKES, premiere
from backend.scoring.prosody import SR, load_audio
from backend.settings import ROOT, config

MAX_UPLOAD_BYTES = 10 * 2**20
MAX_SECONDS = 30

app = FastAPI(title="Dub-Dub-Goose")
library = Library()


def _decode(upload: UploadFile) -> "np.ndarray":  # noqa: F821
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Recording is too large (10 MB max).")
    if not data:
        raise HTTPException(400, "The recording is empty.")
    with tempfile.NamedTemporaryFile(suffix=Path(upload.filename or "rec").suffix or ".webm") as f:
        f.write(data)
        f.flush()
        try:
            y = load_audio(f.name)
        except subprocess.CalledProcessError:
            raise HTTPException(400, "Couldn't read that audio. Try recording again.")
    if len(y) > MAX_SECONDS * SR:
        raise HTTPException(400, f"Recordings can be at most {MAX_SECONDS} seconds.")
    return y


def _entry_or_404(entry_id: str) -> dict:
    try:
        return library.entry(entry_id)
    except KeyError:
        raise HTTPException(404, f"No clip {entry_id!r}.")


@app.get("/api/clips")
def clips(lang: str | None = None):
    return library.entries(lang)


@app.get("/api/clips/{entry_id}")
def clip(entry_id: str):
    return _entry_or_404(entry_id)


@app.get("/api/clips/{entry_id}/video")
def clip_video(entry_id: str):
    _entry_or_404(entry_id)
    return FileResponse(library.clip_path(entry_id), media_type="video/mp4")


@app.get("/api/clips/{entry_id}/thumbnail")
def clip_thumbnail(entry_id: str):
    _entry_or_404(entry_id)
    try:
        return FileResponse(library.thumbnail_path(entry_id), media_type="image/jpeg")
    except KeyError:
        raise HTTPException(404, "No thumbnail for this scene.")


@app.get("/api/clips/{entry_id}/subtitles.vtt")
def clip_subtitles(entry_id: str):
    _entry_or_404(entry_id)
    return PlainTextResponse(library.subtitles(entry_id), media_type="text/vtt; charset=utf-8")


@app.get("/api/clips/{entry_id}/lines/{line_id}/audio")
def line_audio(entry_id: str, line_id: int):
    _entry_or_404(entry_id)
    try:
        return FileResponse(library.line_audio(entry_id, line_id), media_type="audio/wav")
    except KeyError:
        raise HTTPException(404, f"No line {line_id} in {entry_id!r}.")


@app.get("/api/calibration/{lang}")
def calibration_text(lang: str):
    text = config()["game"]["calibration"].get(lang)
    if not text:
        raise HTTPException(404, f"No calibration sentence for {lang!r}.")
    return {"language": lang, "text": text}


@app.post("/api/calibrate")
def calibrate_endpoint(audio: UploadFile = File(...)):
    result = calibrate(_decode(audio))
    if not result["ok"]:
        raise HTTPException(422, result["error"])
    return result


@app.post("/api/score")
def score_endpoint(audio: UploadFile = File(...), clip: str = Form(...), line: int = Form(...),
                   baseline: str | None = Form(None)):
    _entry_or_404(clip)
    try:
        ref = library.reference(clip, line)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    player_baseline = None
    if baseline:
        try:
            player_baseline = valid_baseline(json.loads(baseline))
        except json.JSONDecodeError:
            player_baseline = None
    result = play(ref, _decode(audio), player_baseline, get_analyzer())
    if not result["ok"]:
        raise HTTPException(422, result["error"])
    return result


@app.post("/api/premiere")
def premiere_endpoint(clip: str = Form(...), starts: str = Form(...), takes: list[UploadFile] = File(...)):
    """The scene with the player's takes mixed in. Each take's filename is its line id
    (e.g. "3.webm"); `starts` maps line id -> scene time when that take's recording began."""
    entry = _entry_or_404(clip)
    if len(takes) > MAX_TAKES:
        raise HTTPException(400, f"At most {MAX_TAKES} takes.")
    try:
        begin = {int(k): float(v) for k, v in json.loads(starts).items()}
    except (ValueError, AttributeError, json.JSONDecodeError):
        raise HTTPException(400, "Bad take start times.")
    ends = {c["line_id"]: c["end"] for c in entry["lines"]}
    tmp = tempfile.TemporaryDirectory()
    parts = []
    for upload in takes:
        try:
            line_id = int(Path(upload.filename or "").stem)
        except ValueError:
            raise HTTPException(400, "Each take must be named after its line id.")
        if line_id not in ends or line_id not in begin:
            raise HTTPException(400, f"No line {line_id} in this scene.")
        data = upload.file.read(MAX_UPLOAD_BYTES + 1)
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, "A take is empty or too large.")
        path = Path(tmp.name) / f"{line_id}{Path(upload.filename).suffix or '.webm'}"
        path.write_bytes(data)
        parts.append((path, begin[line_id], max(ends[line_id], begin[line_id]) + 1.0))
    out = Path(tmp.name) / "premiere.mp4"
    try:
        premiere(library.clip_path(clip), sorted(parts, key=lambda t: t[1]), out)
    except subprocess.CalledProcessError:
        tmp.cleanup()
        raise HTTPException(400, "Couldn't mix those takes. Try recording them again.")
    return FileResponse(out, media_type="video/mp4", background=BackgroundTask(tmp.cleanup))


@app.get("/api/health")
def health():
    return {"ok": True, "analyzer": config()["game"]["analyzer"], "clips": len(library.entries())}


# Built with `npm run build` in frontend/. During development run `npm run dev` instead (it proxies /api here).
frontend = ROOT / "frontend" / "dist"
if (frontend / "index.html").exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
