"""Shared helpers for the test set: data/testset.csv plus clips in data/clips/.

Clips are git-ignored (copyright, size) and shared through the team Drive folder;
testset.csv and the label JSONs in data/labels/ are committed.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/testset.csv"
CLIPS = ROOT / "data/clips"
LABELS = ROOT / "data/labels"

COLUMNS = ["clip_id", "language", "source", "start", "end", "emotion", "has_music", "num_speakers",
           "register_note", "transcript", "expected_translation_notes", "verified", "title"]
LANGUAGES = ["hi", "bn", "mr", "te", "ta"]
EMOTIONS = ["angry", "sad", "happy", "fearful", "surprised", "disgust", "neutral", "calm", "sarcastic", "comic"]


def read_rows() -> list[dict]:
    if not CSV_PATH.exists() or not CSV_PATH.read_text().strip():
        return []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict]) -> None:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def next_clip_id(lang: str, rows: list[dict]) -> str:
    taken = {r["clip_id"] for r in rows} | {p.stem for p in CLIPS.glob(f"{lang}_*.mp4")}
    n = 1
    while f"{lang}_{n:02d}" in taken:
        n += 1
    return f"{lang}_{n:02d}"


def parse_time(t: str) -> float:
    """'75', '1:15', '01:01:15' or '1:15.5' -> seconds."""
    if not re.fullmatch(r"\d+(:\d{1,2}){0,2}(\.\d+)?", t):
        raise ValueError(f"bad time '{t}' (use seconds, mm:ss or hh:mm:ss)")
    seconds = 0.0
    for part in t.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds
