"""The whole speech branch (backend/pipeline/speech/run.py) against the generated scenes,
plus the step 3 VRAM and speed targets.

- Per scene and music level: lines found, speaker right per line, text error of each line,
  and whether "auto" Demucs switched on.
- Targets: ASR + emotion + aligners resident <= ~3.5 GB; a ~30 s clip through all stages <= ~45 s;
  live loop (one 5 s recording -> ASR + word timings + prosody + emotion) <= 3 s.

    python scripts/eval_pipeline.py --lang hi ta
"""
import argparse
import itertools
import json
import sys
import threading
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.models import Models, vram_used_gb  # noqa: E402
from backend.pipeline.speech.run import SR, analyze_line, process  # noqa: E402
from scripts.eval_asr import normalize  # noqa: E402

SCENES = ROOT / "data/clips/scenes"


class PeakVRAM:
    """Samples whole-GPU memory use in the background (ONNX Runtime memory is invisible to torch)."""
    def __enter__(self):
        self.peak, self.run = vram_used_gb(), True
        def poll():
            while self.run:
                self.peak = max(self.peak, vram_used_gb()); time.sleep(0.05)
        self.t = threading.Thread(target=poll, daemon=True); self.t.start()
        return self

    def __exit__(self, *a):
        self.run = False; self.t.join()


def score(truth: dict, out: dict) -> dict:
    """Match each true line to the output line it overlaps most; best speaker mapping."""
    hyp = out["lines"]
    labels = sorted({l["speaker"] for l in hyp})
    best = None
    for perm in itertools.permutations(labels + [None, None], 2):
        mapping = dict(zip(("A", "B"), perm))
        right, refs, hyps = 0, [], []
        for t in truth["lines"]:
            ov = [(min(t["end"], h["end"]) - max(t["start"], h["start"]), h) for h in hyp]
            ov = [x for x in ov if x[0] > 0]
            h = max(ov, key=lambda x: x[0])[1] if ov else None
            right += bool(h) and h["speaker"] == mapping[t["speaker"]]
            refs.append(normalize(t["text"])); hyps.append(normalize(h["text"]) if h else "-")
        if best is None or right > best[0]:
            best = (right, refs, hyps)
    # text is scored per scene on all lines in time order, so a diarization merge or split doesn't
    # count words twice or drop them; speaker errors are counted separately above
    ref_text = " ".join(normalize(t["text"]) for t in truth["lines"])
    hyp_text = " ".join(normalize(h["text"]) for h in sorted(hyp, key=lambda h: h["start"])) or "-"
    return {"lines_true": len(truth["lines"]), "lines_found": len(hyp), "speaker_right": best[0],
            "refs": [ref_text], "hyps": [hyp_text]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["hi", "ta"])
    args = ap.parse_args()
    torch.cuda.empty_cache()
    base = vram_used_gb()
    t = time.time()
    models = Models()
    print(f"resident ASR + emotion: {vram_used_gb() - base:.2f} GB (over {base:.2f} GB already in use)")
    for lang in args.lang:
        models.aligner(lang)
        print(f"  + aligner {lang}: {vram_used_gb() - base:.2f} GB")
    print(f"loaded in {time.time() - t:.0f} s")

    results = {}
    for lang in args.lang:
        scenes = sorted(f for f in SCENES.glob(f"{lang}_*.json"))
        print(f"\n{lang}: {'audio':8} {'lines found':>12} {'speaker right':>14} {'WER':>6} {'CER':>6} "
              f"{'Demucs on':>10} {'music est.':>11}")
        results[lang] = run_variants(models, scenes)
    timing_tests(models, args.lang)
    (ROOT / "data/cache/eval_pipeline.json").write_text(json.dumps(results, indent=1))


def run_variants(models, scenes) -> dict:
    results = {}
    for variant in ["clean", "m-10", "m-5", "m0"]:
        agg = {"lines_true": 0, "lines_found": 0, "speaker_right": 0}; refs, hyps = [], []; demucs_on = 0; mus = []
        for f in scenes:
            truth = json.loads(f.read_text(encoding="utf-8"))
            out, _ = process(str(SCENES / f"{f.stem}_{variant}.wav"), truth["lang"], models, "auto")
            s = score(truth, out)
            for k in agg:
                agg[k] += s[k]
            refs += s["refs"]; hyps += s["hyps"]
            demucs_on += out["meta"]["demucs_used"]; mus.append(out["meta"]["music_db"])
        n = agg["lines_true"]
        wer, cer = jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)
        results[variant] = {**agg, "wer": wer, "cer": cer, "demucs_on": demucs_on, "music_db_median": float(np.median(mus))}
        print(f"    {variant:8} {agg['lines_found']:5d} / {n:<5d} {agg['speaker_right'] * 100 / n:12.0f}% {wer * 100:5.1f}% "
              f"{cer * 100:5.1f}% {demucs_on:5d}/{len(scenes):<4d} {np.median(mus):8.1f} dB")
    return results


def timing_tests(models, langs):
    # ~30 s clip: three scenes of one language back to back, with music at -10 dB
    lang = langs[0]
    parts = [sf.read(SCENES / f"{lang}_{k:02d}_m-10.wav", dtype="float32")[0] for k in range(3)]
    clip = ROOT / "data/cache/clip30.wav"
    sf.write(clip, np.concatenate(parts), SR)
    for mode in ("auto", "on"):
        with PeakVRAM() as pv:
            out, _ = process(str(clip), lang, models, mode)
        m = out["meta"]
        print(f"\n~30 s clip ({m['duration']} s, Demucs {mode}): {m['timings_s']['total']} s total "
              f"{m['timings_s']}; peak VRAM {pv.peak:.2f} GB; {len(out['lines'])} lines")

    # live loop: one ~5 s recording
    for lang in langs:
        y = sf.read(SCENES / f"{lang}_00_clean.wav", dtype="float32")[0]
        rec = y[int(0.4 * SR): int(5.4 * SR)]
        times = []
        for _ in range(6):
            t = time.time(); analyze_line(models, rec, lang); times.append(time.time() - t)
        print(f"live loop {lang}: 5 s recording -> {np.median(times[1:]):.2f} s (median of 5 warm runs, "
              f"first {times[0]:.2f} s)")


if __name__ == "__main__":
    main()
