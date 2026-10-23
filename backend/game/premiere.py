"""The premiere: the scene played back with the player's takes in place of the original voices.

The original soundtrack is kept but turned down inside each dubbed line's window, and each take
is laid in at the moment its recording started (the video time the browser sends with it).
"""
import subprocess
from pathlib import Path

DUCK = 0.15        # original soundtrack volume while a take plays
TAKE_GAIN = 1.6
MAX_TAKES = 15


def premiere(clip: Path, takes: list[tuple[Path, float, float]], out: Path) -> Path:
    """`takes`: (audio file, start time in the scene, end of the line's window) per dubbed line."""
    if not takes:
        raise ValueError("no takes to mix")
    if len(takes) > MAX_TAKES:
        raise ValueError(f"at most {MAX_TAKES} takes")
    windows = "+".join(f"between(t,{max(0.0, s):.2f},{e:.2f})" for _, s, e in takes)
    graph = [f"[0:a]volume={DUCK}:enable='{windows}'[bg]"]
    mix = ["[bg]"]
    for i, (_, start, _) in enumerate(takes, 1):
        ms = int(max(0.0, start) * 1000)
        graph.append(f"[{i}:a]aresample=48000,volume={TAKE_GAIN},adelay={ms}|{ms}[t{i}]")
        mix.append(f"[t{i}]")
    graph.append(f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0[aout]")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(clip)]
    for path, _, _ in takes:
        cmd += ["-i", str(path)]
    cmd += ["-filter_complex", ";".join(graph), "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out
