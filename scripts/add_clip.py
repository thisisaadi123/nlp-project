"""Add a clip to the test set: trim it, save it as data/clips/<lang>_<nn>.mp4, add a CSV row.

    # from YouTube (only the requested section is downloaded)
    python scripts/add_clip.py "https://www.youtube.com/watch?v=..." --lang hi --start 1:02:15 --end 1:02:31

    # from a file on disk
    python scripts/add_clip.py ~/Downloads/scene.mp4 --lang ta --start 0:05 --end 0:22

Then run scripts/prelabel_clip.py <clip_id> to draft its labels.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.settings import config  # noqa: E402
from scripts.testset import CLIPS, LANGUAGES, next_clip_id, parse_time, read_rows, write_rows  # noqa: E402

MIN_SECONDS = 3


def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def has_video(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return bool(out.strip())


def cut(src: Path, seek: float, end: float, out: Path) -> None:
    """Cut [seek, end] of `src` into a 720p-max .mp4 whose timestamps start at 0.

    Sections downloaded from YouTube keep their original timestamps (a clip could start at
    3.6 s); browsers count video time from that, while ffmpeg counts audio from its first
    sample, so the website's clock would be off by seconds. make_zero shifts the earliest
    timestamp to 0 and keeps audio and video in sync with each other.
    """
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{seek:.3f}", "-to", f"{end:.3f}", "-i", str(src)]
    if has_video(src):
        cmd += ["-vf", "scale=-2:'min(720,ih)'", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast"]
    else:  # audio-only recording: add a black frame track so every clip is an .mp4 video
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360",
               "-ss", f"{seek:.3f}", "-to", f"{end:.3f}", "-i", str(src), "-map", "0:v", "-map", "1:a",
               "-shortest", "-c:v", "libx264", "-tune", "stillimage", "-crf", "30"]
    cmd += ["-c:a", "aac", "-b:a", "128k", "-ac", "1", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def download_section(url: str, start: float, end: float, workdir: Path) -> Path:
    """Download only [start, end] of a video, at most 720p."""
    pad = 2.0  # keyframes: grab a little extra, trim exactly afterwards
    cmd = [sys.executable, "-m", "yt_dlp", "--quiet", "--no-playlist",
           "-f", "bv*[height<=720]+ba/b[height<=720]/b",
           "--download-sections", f"*{max(start - pad, 0)}-{end + pad}",
           "-o", str(workdir / "source.%(ext)s"), url]
    subprocess.run(cmd, check=True)
    files = [p for p in workdir.glob("source.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        sys.exit("yt-dlp produced no file")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="YouTube/web URL or a local audio/video file")
    ap.add_argument("--lang", required=True, choices=LANGUAGES)
    ap.add_argument("--start", help="clip start (seconds, mm:ss or hh:mm:ss); default: beginning")
    ap.add_argument("--end", help="clip end; default: end of file")
    args = ap.parse_args()

    max_seconds = config()["clip"]["max_seconds"]
    start = parse_time(args.start) if args.start else 0.0
    is_url = args.source.startswith(("http://", "https://"))
    if is_url and args.end is None:
        sys.exit("--end is required for URLs, so only the scene is downloaded")

    rows = read_rows()
    clip_id = next_clip_id(args.lang, rows)
    CLIPS.mkdir(parents=True, exist_ok=True)
    out = CLIPS / f"{clip_id}.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        if is_url:
            src = download_section(args.source, start, parse_time(args.end), Path(tmp))
            # the downloaded file starts up to 2 s before `start`
            offset = start - max(start - 2.0, 0)
            seek, end = offset, offset + parse_time(args.end) - start
        else:
            src = Path(args.source).expanduser()
            if not src.exists():
                sys.exit(f"{src} not found")
            seek = start
            end = parse_time(args.end) if args.end else probe_duration(src)

        length = end - seek
        if not MIN_SECONDS <= length <= max_seconds:
            sys.exit(f"clip is {length:.1f} s; keep it between {MIN_SECONDS} and {max_seconds} s")

        cut(src, seek, end, out)

    source = args.source if is_url else f"file:{Path(args.source).name}"
    rows.append({"clip_id": clip_id, "language": args.lang, "source": source,
                 "start": f"{start:.1f}", "end": f"{start + length:.1f}", "verified": "no"})
    write_rows(rows)
    print(f"Added {clip_id} ({length:.1f} s) -> {out.relative_to(CLIPS.parent.parent)}")
    print(f"Next: python scripts/prelabel_clip.py {clip_id}")


if __name__ == "__main__":
    main()
