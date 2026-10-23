"""Speech branch: a clip goes in, a JSON with every spoken line comes out.

    python -m backend.pipeline.speech.run <clip> --lang hi [--demucs on|off] [--out file.json]

Stages: Demucs (music removal, optional) -> pyannote (who spoke when) -> per line: IndicConformer
ASR -> word timings (character aligner) -> prosody (backend/scoring/prosody.py) -> emotion2vec.
analyze_line() is also the live game loop: the same analysis on one recorded attempt.
Embeddings are saved next to the JSON (.npz), not inside it.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from backend.pipeline.speech.align import align as ctc_align
from backend.pipeline.speech.models import Models, vram_used_gb
from backend.roman import romanize
from backend.scoring.arousal import arousal, baseline
from backend.scoring.prosody import HOP_S, extract, load_audio
from backend.settings import ROOT, config

SR = 16000
PAD_S = 0.15            # audio kept around each diarized turn
MERGE_GAP_S = 1.0       # same-speaker turns closer than this form one line (pyannote splits at pauses)
FLICKER_S = 0.6         # shorter turns are absorbed by a longer neighbour instead of standing alone
MIN_LINE_S = 0.25


def _merge_same_speaker(lines: list[dict]) -> list[dict]:
    out = []
    for l in lines:
        if out and out[-1]["speaker"] == l["speaker"] and l["start"] - out[-1]["end"] < MERGE_GAP_S:
            out[-1]["end"] = max(out[-1]["end"], l["end"])
        else:
            out.append(dict(l))
    return out


def lines_from_turns(turns) -> list[dict]:
    """Diarized turns -> lines. pyannote sometimes flickers between speakers in 0.1-0.2 s pieces at
    a line's start; dropping those would lose the line, so short pieces join the longer neighbour."""
    lines = _merge_same_speaker([{"speaker": t.speaker, "start": t.start, "end": t.end} for t in turns])
    changed = True
    while changed and len(lines) > 1:
        changed = False
        for i, l in enumerate(lines):
            if l["end"] - l["start"] >= FLICKER_S:
                continue
            near = [j for j in (i - 1, i + 1) if 0 <= j < len(lines)
                    and min(abs(lines[j]["start"] - l["end"]), abs(l["start"] - lines[j]["end"])) < MERGE_GAP_S]
            if not near:
                continue
            j = max(near, key=lambda k: lines[k]["end"] - lines[k]["start"])
            if lines[j]["end"] - lines[j]["start"] <= l["end"] - l["start"]:
                continue
            lines[j]["start"], lines[j]["end"] = min(lines[j]["start"], l["start"]), max(lines[j]["end"], l["end"])
            del lines[i]
            lines = _merge_same_speaker(lines)
            changed = True
            break
    return [l for l in lines if l["end"] - l["start"] >= MIN_LINE_S]


def word_prosody(p, words, offset: float) -> list[dict]:
    """Slice the line's 10 ms prosody arrays at each word's span."""
    out = []
    for w in words:
        d = {"w": w.w, "start": round(w.start, 3), "end": round(w.end, 3)}
        i0 = i1 = 0
        if p is not None:                      # clip to the voiced span the arrays cover
            i0 = max(0, int(round((w.start - offset - p.start) / HOP_S)))
            i1 = min(len(p.pitch), int(round((w.end - offset - p.start) / HOP_S)))
        if i1 > i0:
            d.update(stress=round(float(p.prominence[i0:i1].max()), 2), pitch=round(float(p.pitch[i0:i1].mean()), 2),
                     energy=round(float(p.energy[i0:i1].mean()), 1))
        else:                                  # outside the voiced span (e.g. a whispered word)
            d.update(stress=None, pitch=None, energy=None)
        out.append(d)
    return out


def analyze_line(models: Models, seg: np.ndarray, lang: str, offset: float = 0.0, base=None) -> dict:
    """ASR + word timings + prosody + emotion for one line (or one recorded attempt)."""
    tr = models.asr.transcribe(seg, lang)
    aligner = models.aligner(lang)
    words = aligner.align(seg, tr.text.split()) if aligner else ctc_align(tr, y=seg)
    for w in words:
        w.start, w.end = w.start + offset, w.end + offset
    try:
        p = extract(seg)
    except ValueError:                         # too little voiced speech
        p = None
    emo = models.emotion(seg)
    return {
        "text": tr.text, "text_roman": romanize(tr.text, lang),
        "words": word_prosody(p, words, offset),
        "pauses": [[round(offset + p.start + a, 3), round(offset + p.start + b, 3)] for a, b in p.pauses] if p else [],
        "prosody": {k: round(v, 3) for k, v in p.stats.items()} if p else {},
        "emotion": {"label": emo.label, "probs": emo.probs, "arousal": round(arousal(p, base), 3) if p else None,
                    "arousal_e2v": round(sum(emo.probs[e] for e in ("angry", "happy", "fear", "surprise")), 3)},
        "_prosody": p, "_embedding": emo.embedding,
    }


def music_db(y: np.ndarray, vocals: np.ndarray) -> float:
    """How loud the removed part (music) was relative to the kept vocals, dB."""
    n = min(len(y), len(vocals))
    rest = y[:n] - vocals[:n]
    return float(10 * np.log10((np.mean(rest ** 2) + 1e-12) / (np.mean(vocals[:n] ** 2) + 1e-12)))


def process(path: str, lang: str, models: Models, demucs: str = "auto") -> tuple[dict, np.ndarray]:
    timings = {}
    t0 = time.time()
    y = load_audio(path)
    timings["load"] = time.time() - t0
    y_speech, music = y, None
    if demucs != "off":
        t = time.time()
        with models.use("separation") as sep:
            vocals = sep(y)
        music = music_db(y, vocals)
        if demucs == "on" or music >= config()["speech"]["demucs_min_music_db"]:
            y_speech = vocals
        timings["separation"] = time.time() - t
    t = time.time()
    with models.use("diarization") as dia:
        turns = dia(y_speech)
    timings["diarization"] = time.time() - t

    t = time.time()
    lines = []
    for i, l in enumerate(lines_from_turns(turns)):
        a, b = max(0.0, l["start"] - PAD_S), min(len(y_speech) / SR, l["end"] + PAD_S)
        res = analyze_line(models, y_speech[int(a * SR): int(b * SR)], lang, offset=a)
        if not res["text"]:
            continue
        lines.append({"line_id": len(lines), "speaker": l["speaker"], "start": round(l["start"], 3),
                      "end": round(l["end"], 3), "language": lang, **res})
    # arousal relative to each speaker's own voice in this clip (needs 2+ lines from that speaker)
    for spk in {l["speaker"] for l in lines}:
        own = [l for l in lines if l["speaker"] == spk and l["_prosody"] is not None]
        if len(own) >= 2:
            base = baseline([l["_prosody"] for l in own])
            for l in own:
                l["emotion"]["arousal"] = round(arousal(l["_prosody"], base), 3)
    timings["lines"] = time.time() - t
    timings["total"] = time.time() - t0
    emb = np.stack([l.pop("_embedding") for l in lines]) if lines else np.zeros((0, 1024), np.float32)
    for l in lines:
        l.pop("_prosody")
    meta = {"clip": str(path), "language": lang, "duration": round(len(y) / SR, 2),
            "music_db": round(music, 1) if music is not None else None, "demucs_used": y_speech is not y,
            "speakers": sorted({l["speaker"] for l in lines}), "timings_s": {k: round(v, 2) for k, v in timings.items()},
            "vram_used_gb": round(vram_used_gb(), 2)}
    return {"meta": meta, "lines": lines}, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--lang", required=True, choices=list(config()["languages"]))
    ap.add_argument("--demucs", choices=["auto", "on", "off"], default=None,
                    help="default: config.yaml speech.demucs")
    ap.add_argument("--out", default=None, help="default: data/cache/speech/<clip name>.json")
    args = ap.parse_args()
    demucs = args.demucs or config()["speech"]["demucs"]
    out = Path(args.out) if args.out else ROOT / "data/cache/speech" / (Path(args.clip).stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    result, emb = process(args.clip, args.lang, Models(), demucs)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".npz"), line_id=np.arange(len(emb)), embedding=emb)
    m = result["meta"]
    print(f"{len(result['lines'])} lines, {len(m['speakers'])} speakers, {m['duration']} s audio in "
          f"{m['timings_s']['total']} s -> {out}")


if __name__ == "__main__":
    main()
