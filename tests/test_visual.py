import io
import subprocess

import pytest
from PIL import Image

from backend.pipeline.visual.keyframes import GRID, MAX_FRAMES, TILE_WIDTH, keyframe_sheets, pick_times, tile


def test_pick_times_spreads_frames_and_caps_count():
    times = pick_times(60, [], fps=1.5)
    assert len(times) == MAX_FRAMES
    assert times == sorted(times) and 0 < times[0] < times[-1] < 60


def test_pick_times_includes_a_frame_after_each_cut():
    times = pick_times(30, [10.0, 20.4], fps=1.5)
    assert 10.2 in times and 20.6 in times


def test_pick_times_short_clip():
    assert len(pick_times(0.5, [], fps=1.5)) == 1


def test_tile_packs_four_frames_in_reading_order():
    colours = ["red", "green", "blue", "yellow"]
    frames = [Image.new("RGB", (TILE_WIDTH, 288), c) for c in colours]
    sheet = Image.open(io.BytesIO(tile(frames)))
    assert sheet.size == (TILE_WIDTH * GRID, 288 * GRID)
    top_right = sheet.getpixel((TILE_WIDTH + 10, 10))
    bottom_left = sheet.getpixel((10, 288 + 10))
    assert top_right[1] > 100 and top_right[0] < 100      # green
    assert bottom_left[2] > 150 and bottom_left[0] < 100  # blue


@pytest.fixture
def two_shot_video(tmp_path):
    path = tmp_path / "shots.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:d=3", "-f", "lavfi", "-i", "smptebars=s=320x180:d=3",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def test_keyframe_sheets_on_a_real_video(two_shot_video):
    times, sheets = keyframe_sheets(two_shot_video)
    assert len(times) == 8                     # 6 s at 1.5 fps = 9, capped at MAX_FRAMES
    assert any(3.0 < t < 3.5 for t in times)   # a frame right after the cut at 3 s
    assert len(sheets) == 2                    # 4 + 4 frames
    assert all(s[:2] == b"\xff\xd8" for s in sheets)  # JPEG
