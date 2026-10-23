"""Diarization and music separation, measured on the generated scenes (scripts/make_scenes.py).

- Diarization error rate (DER) on clean scenes and on music mixes (with and without Demucs),
  standard 0.25 s collar and no collar, with the number of speakers given or not.
- ASR word/character error rate per line at each music level, with and without Demucs. Lines are
  cut at their true times so only the audio quality differs. Demucs is kept only if it helps.

    python scripts/eval_scenes.py --lang hi ta
"""
import argparse
import gc
import itertools
import json
import sys
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.asr import ASR, SR  # noqa: E402
from scripts.eval_asr import normalize  # noqa: E402

SCENES = ROOT / "data/clips/scenes"


def free():
    gc.collect(); torch.cuda.empty_cache()


def load_scenes(langs):
    out = []
    for f in sorted(SCENES.glob("*.json")):
        meta = json.loads(f.read_text(encoding="utf-8"))
        if meta["lang"] in langs:
            meta["name"] = f.stem
            meta["audio"] = {v: sf.read(SCENES / f"{f.stem}_{v}.wav", dtype="float32")[0]
                             for v in ["clean"] + [f"m{l}" for l in meta["levels_db"]]}
            out.append(meta)
    if not out:
        sys.exit(f"No scenes in {SCENES}. Run scripts/make_scenes.py first.")
    return out


def speech_reference(s):
    """Reference speaker turns with in-line pauses (>= 0.15 s) removed; exact, the clean signal is known.
    (Marking whole lines as speech would count pyannote's correct pause detection as missed speech.)"""
    from pyannote.core import Annotation, Segment
    y = s["audio"]["clean"]
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0] + 1e-9)
    active = db > db.max() - 45
    ref = Annotation()
    for l in s["lines"]:
        a = int(l["start"] / 0.01)
        seg = active[a: int(l["end"] / 0.01)]
        k = 0
        while k < len(seg):
            if not seg[k]:
                k += 1; continue
            j = k
            while j < len(seg) and (seg[j] or seg[j: j + 15].any()):
                j += 1
            ref[Segment((a + k) * 0.01, (a + j) * 0.01)] = l["speaker"]
            k = j
    return ref


def der(scenes, hyps, collar):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate(collar=collar)
    for s, turns in zip(scenes, hyps):
        hyp = Annotation()
        for t in turns:
            hyp[Segment(t.start, t.end)] = t.speaker
        metric(speech_reference(s), hyp)
    return abs(metric)


def lines_right(s, turns) -> int:
    """Lines whose most-overlapping hypothesis speaker is right, under the best speaker mapping."""
    labels = sorted({t.speaker for t in turns})
    best = 0
    for perm in itertools.permutations(labels + [None, None], 2):
        mapping = dict(zip(("A", "B"), perm))
        ok = 0
        for l in s["lines"]:
            overlap = {}
            for t in turns:
                o = min(l["end"], t.end) - max(l["start"], t.start)
                if o > 0:
                    overlap[t.speaker] = overlap.get(t.speaker, 0) + o
            ok += bool(overlap) and max(overlap, key=overlap.get) == mapping[l["speaker"]]
        best = max(best, ok)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["hi", "ta"])
    args = ap.parse_args()
    scenes = load_scenes(args.lang)
    variants = [v for v in scenes[0]["audio"] if v != "clean"]
    print(f"{len(scenes)} scenes, {sum(len(s['lines']) for s in scenes)} lines, "
          f"{sum(s['duration'] for s in scenes):.0f} s; music levels {variants}")

    from backend.pipeline.speech.separate import Separator
    sep = Separator()
    for s in scenes:
        for v in list(s["audio"]):
            s["audio"][f"{v}+demucs"] = sep(s["audio"][v])
    del sep; free()

    from backend.pipeline.speech.diarize import Diarizer
    dia = Diarizer()
    n_lines = sum(len(s["lines"]) for s in scenes)
    print("\nDiarization (DER vs a speech-only reference; lower is better)")
    print(f"  {'audio':28} {'DER 0.25s collar':>16} {'no collar':>10} {'2 speakers found':>17} {'lines right':>12}")
    for v in ["clean", "clean+demucs"] + [x for v in variants for x in (v, f"{v}+demucs")]:
        for given in ((False, True) if v == "clean" else (False,)):
            hyps = [dia(s["audio"][v], num_speakers=2 if given else None) for s in scenes]
            n_ok = sum(len({t.speaker for t in h}) == 2 for h in hyps)
            right = sum(lines_right(s, h) for s, h in zip(scenes, hyps))
            label = v + (" (n=2 given)" if given else "")
            print(f"  {label:28} {der(scenes, hyps, 0.25) * 100:15.1f}% {der(scenes, hyps, 0.0) * 100:9.1f}%"
                  f" {n_ok:10d}/{len(scenes)} {right * 100 / n_lines:11.0f}%")
    del dia; free()

    asr = ASR()
    print("\nASR on each line, cut at its true time")
    print(f"  {'audio':16} {'WER':>6} {'CER':>6}")
    rows = {}
    for v in ["clean", "clean+demucs"] + [x for v in variants for x in (v, f"{v}+demucs")]:
        refs, hyps = [], []
        for s in scenes:
            y = s["audio"][v]
            for l in s["lines"]:
                seg = y[max(0, int((l["start"] - 0.1) * SR)): int((l["end"] + 0.1) * SR)]
                refs.append(normalize(l["text"])); hyps.append(normalize(asr.transcribe(seg, s["lang"]).text) or "-")
        rows[v] = (jiwer.wer(refs, hyps), jiwer.cer(refs, hyps))
        print(f"  {v:16} {rows[v][0] * 100:5.1f}% {rows[v][1] * 100:5.1f}%")
    out = ROOT / "data/cache/eval_scenes.json"
    out.write_text(json.dumps({k: {"wer": a, "cer": b} for k, (a, b) in rows.items()}, indent=1))


if __name__ == "__main__":
    main()
