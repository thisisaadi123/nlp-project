"""Keyframes from a clip, tiled into 2x2 grids for the vision model.

The vision model takes at most 3 images per request, so up to 8 frames are packed four to an
image: one request describes the whole clip. Frames are spread evenly through the clip, with
scene cuts taking priority so every shot is seen.
"""
import io
import re
import subprocess

from PIL import Image

from backend.settings import config

MAX_FRAMES = 8   # 2 sheets: ~4.3k input tokens, inside Qwen's 7k-per-minute free limit
TILE_WIDTH = 512
GRID = 2  # 2x2 frames per image


def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def scene_cuts(path: str, threshold: float = 0.3) -> list[float]:
    """Times (s) where the picture changes sharply."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-vf", f"select='gt(scene,{threshold})',showinfo",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    return [float(t) for t in re.findall(r"pts_time:([\d.]+)", out)]


def pick_times(length: float, cuts: list[float], fps: float, limit: int = MAX_FRAMES) -> list[float]:
    n = max(1, min(limit, int(length * fps)))
    times = [(i + 0.5) * length / n for i in range(n)]
    for cut in cuts:  # replace the nearest evenly-spaced frame with a frame just after the cut
        t = min(cut + 0.2, length - 0.05)
        nearest = min(range(len(times)), key=lambda i: abs(times[i] - t))
        times[nearest] = t
    return sorted(set(round(t, 2) for t in times))


def frame_at(path: str, t: float) -> Image.Image:
    jpg = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", path,
                          "-frames:v", "1", "-vf", f"scale={TILE_WIDTH}:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                         capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(jpg)).convert("RGB")


def tile(frames: list[Image.Image]) -> bytes:
    """Up to 4 frames in reading order (top-left, top-right, bottom-left, bottom-right) -> one JPEG."""
    h = max(f.height for f in frames)
    sheet = Image.new("RGB", (TILE_WIDTH * GRID, h * GRID), "black")
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % GRID) * TILE_WIDTH, (i // GRID) * h))
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def keyframe_sheets(path: str) -> tuple[list[float], list[bytes]]:
    """Frame times and the tiled JPEG sheets (at most 3, each holding up to 4 frames)."""
    times = pick_times(duration(path), scene_cuts(path), config()["clip"]["keyframe_fps"])
    frames = [frame_at(path, t) for t in times]
    sheets = [tile(frames[i:i + GRID * GRID]) for i in range(0, len(frames), GRID * GRID)]
    return times, sheets
