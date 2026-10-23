"""The clip library: everything a round of the game needs, prepared once per clip.

An entry is one clip in one language direction (e.g. ta_02 dubbed into Hindi):

    data/library/<clip>.<src>-<tgt>/
        entry.json        clip info + one delivery card per line
        clip.mp4          the original scene (the player watches it before dubbing)
        thumb.jpg         a still frame for the scene picker
        lines/<id>.wav    each original line's audio, 16 kHz mono
        prosody.pkl       each original line's measured prosody + its speaker's baseline

Built from step 4's output (data/cache/dub/<clip>.<src>-<tgt>.json) by scripts/build_library.py.
The original actor's line is the reference the player's delivery is compared against: different
language and words, same emotion, intensity, timing and stress placement.
"""
import json
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import soundfile as sf

from backend.game.coach import coach
from backend.game.shape import shape
from backend.scoring.arousal import Baseline, baseline
from backend.scoring.prosody import SR, Prosody, extract, load_audio
from backend.settings import ROOT, config

PAD_S = 0.1  # audio kept around each line's start/end


def library_dir() -> Path:
    return ROOT / config()["game"]["library_dir"]


def short(description: str | None) -> str | None:
    """'younger brother, 19, furious at ...' -> 'younger brother' (for one-line labels)."""
    if not description:
        return None
    first = description.split(",")[0].split(";")[0].split("(")[0].strip()
    return first[:40]


def card(line: dict, names: dict[str, str] | None = None) -> dict:
    """What the player sees for one line: the delivery card."""
    dub, ei = line["dub"], line["emotion_intent"]
    stress_targets = [e.get("target_word") for e in dub.get("emphasis") or [] if e.get("target_word")]
    names = names or {}
    return {
        "line_id": line["line_id"],
        "speaker": line.get("speaker"),
        "addressee": line.get("addressee"),
        "speaker_name": names.get(line.get("speaker")),
        "addressee_name": names.get(line.get("addressee")),
        "start": line["start"],
        "end": line["end"],
        "duration": round(line["end"] - line["start"], 2),
        "source_text": line["text"],
        "source_roman": line.get("text_roman") or "",
        "translation": dub["translation"],
        "romanized": dub.get("romanized") or "",
        "emotion": ei.get("label"),
        "intensity": ei.get("intensity"),
        "tone": ei.get("tone"),
        "stress": stress_targets,
        "delivery": dub.get("delivery") or "",
        "register": dub.get("register"),
        "mismatch": ei.get("mismatch", False),
        "model": dub.get("_provider"),
    }


def difficulty(cards: list[dict]) -> str:
    playable = [c for c in cards if c.get("playable")]
    if not playable:
        return "hard"
    avg = sum(c["duration"] for c in playable) / len(playable)
    if len(playable) <= 6 and avg <= 2.5:
        return "easy"
    if len(playable) > 12 or avg > 4.0:
        return "hard"
    return "medium"


def audio_start(clip_path: Path) -> float:
    """When the audio track starts on the video's clock. Line times come from the decoded audio
    (which starts at 0), browsers count from the file's timeline: this converts between them."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_time",
                          "-of", "csv=p=0", str(clip_path)], capture_output=True, text=True).stdout.strip()
    try:
        return max(0.0, float(out))
    except ValueError:
        return 0.0


def thumbnail(clip_path: Path, at: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{max(0.0, at):.2f}", "-i", str(clip_path),
                    "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(out)], check=True)


def build_entry(dub_json: Path, clip_path: Path, out_root: Path | None = None, title: str | None = None) -> Path:
    """Build one library entry from step 4's output and the clip it came from."""
    dub = json.loads(dub_json.read_text(encoding="utf-8"))
    context = dub.get("context") or {}
    speakers = context.get("speakers") or {}
    names = {sid: short(info.get("description")) for sid, info in speakers.items() if isinstance(info, dict)}
    src, tgt = dub["source_language"], dub["target_language"]
    entry_id = f"{clip_path.stem}.{src}-{tgt}"
    out = (out_root or library_dir()) / entry_id
    if out.exists():
        shutil.rmtree(out)
    (out / "lines").mkdir(parents=True)
    shutil.copy(clip_path, out / "clip.mp4")

    y = load_audio(str(clip_path))
    shift = audio_start(clip_path)
    prosodies: dict[int, Prosody | None] = {}
    cards = []
    for line in dub["lines"]:
        a = max(0.0, line["start"] - PAD_S)
        b = min(len(y) / SR, line["end"] + PAD_S)
        seg = y[int(a * SR): int(b * SR)]
        sf.write(out / "lines" / f"{line['line_id']}.wav", seg, SR)
        try:
            p = extract(seg)
        except ValueError:  # no voiced speech in this line's audio: it can't be a reference
            p = None
        prosodies[line["line_id"]] = p
        c = card(line, names)
        c["start"], c["end"] = round(line["start"] + shift, 3), round(line["end"] + shift, 3)  # video clock
        c["playable"] = p is not None
        c["coach"] = coach(c)
        # where the voice starts, relative to the line's start (the melody line's time 0)
        c["onset"] = round((a - line["start"]) + p.start, 2) if p else 0.0
        c["shape"] = shape(p, c["translation"].split(), c["stress"]) if p else None
        cards.append(c)

    # Each speaker's baseline: their own lines in this clip (as in step 3), when there are 2+.
    baselines: dict[str, Baseline] = {}
    for spk in {c["speaker"] for c in cards}:
        own = [prosodies[c["line_id"]] for c in cards if c["speaker"] == spk and prosodies[c["line_id"]]]
        if len(own) >= 2:
            baselines[spk] = baseline(own)

    with open(out / "prosody.pkl", "wb") as f:  # built by us from our own clips
        pickle.dump({"prosody": prosodies, "baselines": baselines}, f)
    first = next((c for c in cards if c["playable"]), cards[0] if cards else None)
    thumbnail(clip_path, (first["start"] + first["end"]) / 2 if first else 1.0, out / "thumb.jpg")
    characters = [{"id": sid, "name": names.get(sid), "description": info.get("description"),
                   "gender": info.get("gender"), "age": info.get("approx_age")}
                  for sid, info in speakers.items() if isinstance(info, dict)]
    relationships = [{"between": [names.get(x) or x for x in r.get("between") or []], "relation": r.get("relation"),
                      "closeness": r.get("closeness")}
                     for r in context.get("relationships") or [] if isinstance(r, dict)]
    entry = {
        "id": entry_id,
        "clip": clip_path.stem,
        "title": title or clip_path.stem,
        "source_language": src,
        "target_language": tgt,
        "scene": context.get("summary"),
        "mood": context.get("mood"),
        "characters": characters,
        "relationships": relationships,
        "difficulty": difficulty(cards),
        "calibration_text": config()["game"]["calibration"][tgt],
        "lines": cards,
    }
    (out / "entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


@dataclass
class Reference:
    """The original line a player's attempt is compared with."""
    card: dict
    prosody: Prosody
    baseline: Baseline | None
    target_language: str


class Library:
    def __init__(self, root: Path | None = None):
        self.root = root or library_dir()

    def entries(self, lang: str | None = None) -> list[dict]:
        """Scene summaries for the picker, easiest first; `lang` keeps only scenes dubbed into it."""
        order = {"easy": 0, "medium": 1, "hard": 2}
        out = []
        for p in sorted(self.root.glob("*/entry.json")):
            e = json.loads(p.read_text(encoding="utf-8"))
            if lang and e["target_language"] != lang:
                continue
            out.append({k: e.get(k) for k in ("id", "clip", "title", "source_language", "target_language",
                                              "scene", "mood", "difficulty")}
                       | {"lines": sum(c["playable"] for c in e["lines"]),
                          "characters": [c["name"] for c in e.get("characters", []) if c.get("name")][:3]})
        return sorted(out, key=lambda e: (order.get(e["difficulty"], 1), e["id"]))

    def entry(self, entry_id: str) -> dict:
        path = self._dir(entry_id) / "entry.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def clip_path(self, entry_id: str) -> Path:
        return self._dir(entry_id) / "clip.mp4"

    def thumbnail_path(self, entry_id: str) -> Path:
        path = self._dir(entry_id) / "thumb.jpg"
        if not path.exists():
            raise KeyError(f"no thumbnail for {entry_id}")
        return path

    def subtitles(self, entry_id: str) -> str:
        """WebVTT: each line's original words with the translation underneath."""
        def ts(t: float) -> str:
            t = max(0.0, t)
            return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"
        cues = ["WEBVTT", ""]
        for c in self.entry(entry_id)["lines"]:
            cues += [f"{ts(c['start'])} --> {ts(c['end'])}", c["source_text"], c["translation"], ""]
        return "\n".join(cues)

    def line_audio(self, entry_id: str, line_id: int) -> Path:
        path = self._dir(entry_id) / "lines" / f"{int(line_id)}.wav"
        if not path.exists():
            raise KeyError(f"no line {line_id} in {entry_id}")
        return path

    def reference(self, entry_id: str, line_id: int) -> Reference:
        e = self.entry(entry_id)
        c = next((c for c in e["lines"] if c["line_id"] == line_id), None)
        if c is None:
            raise KeyError(f"no line {line_id} in {entry_id}")
        data = _load_prosody(str(self._dir(entry_id) / "prosody.pkl"))
        p = data["prosody"].get(line_id)
        if p is None:
            raise KeyError(f"line {line_id} in {entry_id} has no usable reference audio")
        return Reference(c, p, data["baselines"].get(c["speaker"]), e["target_language"])

    def _dir(self, entry_id: str) -> Path:
        # entry ids come from URLs: refuse anything that is not a plain directory name in the library
        path = (self.root / entry_id).resolve()
        if path.parent != self.root.resolve() or not (path / "entry.json").exists():
            raise KeyError(f"no library entry {entry_id!r}")
        return path


@lru_cache(maxsize=64)
def _load_prosody(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

