"""How accurate are the CTC word timings?

1. Exact test: word chains built by scripts/make_align_test.py (TTS words joined with no gaps,
   so every boundary is known). Aligned with the true text, so ASR errors don't count.
2. Pause check on real speech (FLEURS): at every clear pause (>= 150 ms of near-silence found by
   backend/scoring/prosody.py) the word before should end where the pause starts and the word
   after should start where it ends. The silence constraint uses the same loudness rule, so
   this check flatters "masked"; trust test 1 for comparing modes.
3. A listening page (data/cache/align_check.html): one button per word plays just that word.

    python scripts/check_alignment.py --lang hi ta --n 60 --listen 5
"""
import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.align import CharAligner, align  # noqa: E402
from backend.pipeline.speech.asr import ASR  # noqa: E402
from backend.roman import romanize  # noqa: E402
from backend.scoring.prosody import extract  # noqa: E402
from scripts.eval_asr import fleurs_items  # noqa: E402

SYNTH = ROOT / "data/cache/align_synth"
MODES = {"raw": {}, "masked": {"use_audio": True}, "masked+pad": {"use_audio": True, "pad_frames": 1.0},
         "w2v": {"char": True, "mask": False, "close": False}, "w2v+mask": {"char": True, "mask": True, "close": False},
         "w2v+mask+close": {"char": True, "mask": True, "close": True}}
_char = {}


def run_align(tr, y, mode, text=None, lang=None):
    opts = MODES[mode]
    if opts.get("char"):
        if lang not in _char:
            _char[lang] = CharAligner(lang) if CharAligner.available(lang) else None
        if _char[lang] is None:
            return []
        return _char[lang].align(y, (text if text is not None else tr.text).split(), mask_silence=opts["mask"],
                                 close_gaps=opts["close"])
    return align(tr, text=text, y=y if opts.get("use_audio") else None, pad_frames=opts.get("pad_frames", 0.0))


def summary(name: str, errs: list[float]) -> str:
    e = np.array(errs) * 1000
    if not len(e):
        return f"{name:22} n=0"
    return (f"{name:22} n={len(e):3d}  median {np.median(e):+5.0f} ms  mean |err| {np.mean(np.abs(e)):4.0f} ms  "
            f"within 100 ms {np.mean(np.abs(e) <= 100) * 100:3.0f}%")


def exact_test(asr: ASR, langs: list[str]):
    files = sorted(SYNTH.glob("*.json"))
    if not files:
        print("No word chains; run scripts/make_align_test.py first."); return
    for lang in langs:
        chains = [json.loads(f.read_text()) | {"wav": f.with_suffix(".wav")} for f in files]
        chains = [c for c in chains if c["lang"] == lang]
        if not chains:
            continue
        print(f"\n{lang}: exact test, {len(chains)} word chains, {sum(len(c['words']) for c in chains)} words "
              f"(+ = later than the truth)")
        for mode in MODES:
            inner, onset, offset = [], [], []
            for c in chains:
                y, _ = sf.read(c["wav"], dtype="float32")
                truth = c["words"]
                words = run_align(asr.transcribe(y, lang), y, mode, text=" ".join(w["w"] for w in truth), lang=lang)
                if len(words) != len(truth):
                    continue
                onset.append(words[0].start - truth[0]["start"])
                offset.append(words[-1].end - truth[-1]["end"])
                for i in range(len(truth) - 1):
                    inner.append(words[i].end - truth[i]["end"])
                    inner.append(words[i + 1].start - truth[i + 1]["start"])
            print(f"  {mode}")
            print("    " + summary("line start", onset))
            print("    " + summary("line end", offset))
            print("    " + summary("between words", inner))


def boundary_errors(words, pros) -> tuple[list[float], list[float]]:
    """(word end - pause start), (next word start - pause end) for each interior pause, seconds."""
    ends, starts = [], []
    for ps, pe in pros.pauses:
        ps, pe = pros.start + ps, pros.start + pe
        before = [w for w in words if w.start < ps]
        after = [w for w in words if w.end > pe]
        if not before or not after:
            continue
        ends.append(before[-1].end - ps)
        starts.append(after[0].start - pe)
    return ends, starts


def pause_check(asr: ASR, langs: list[str], n: int):
    for lang in langs:
        res = {m: ([], []) for m in MODES}
        for it in fleurs_items(lang, n, seed=1):
            y, _ = sf.read(it["path"], dtype="float32")
            tr = asr.transcribe(y, lang)
            try:
                pros = extract(y)
            except ValueError:
                continue
            for mode in MODES:
                e, s = boundary_errors(run_align(tr, y, mode, lang=lang), pros)
                res[mode][0].extend(e); res[mode][1].extend(s)
        print(f"\n{lang}: pause check on FLEURS (+ = word edge later than the pause edge)")
        for mode, (e, s) in res.items():
            print("  " + summary(f"{mode} end", e))
            print("  " + summary(f"{mode} start", s))


def listening_page(asr: ASR, langs: list[str], k: int, out: Path):
    parts = ["<!doctype html><meta charset=utf-8><title>Word timing check</title>",
             "<style>body{font:16px system-ui;max-width:900px;margin:2em auto;padding:0 16px}"
             "button{margin:3px;padding:6px 9px;font-size:16px}small{color:#666}</style>",
             "<h1>Word timing check</h1><p>Click a word: it should play that whole word and nothing of "
             "its neighbours. Note words that sound cut off or include the next word.</p>"]
    n = 0
    for lang in langs:
        for it in fleurs_items(lang, k, seed=1)[: max(1, k // len(langs))]:
            y, _ = sf.read(it["path"], dtype="float32")
            words = run_align(asr.transcribe(y, lang), y, "w2v+mask+close" if CharAligner.available(lang) else "masked",
                              lang=lang)
            buf = io.BytesIO(); sf.write(buf, y, 16000, format="WAV")
            src = base64.b64encode(buf.getvalue()).decode()
            parts.append(f"<h3>{n + 1}. {lang} <small>{html.escape(it['id'])}</small></h3>"
                         f"<audio id=a{n} controls src='data:audio/wav;base64,{src}'></audio><div>")
            for w in words:
                parts.append(f"<button onclick='play({n},{w.start},{w.end})' title='{w.start:.2f}-{w.end:.2f}s'>"
                             f"{html.escape(w.w)}<br><small>{html.escape(romanize(w.w, lang))}</small></button>")
            parts.append("</div>")
            n += 1
    parts.append("<script>let t;function play(k,s,e){const a=document.getElementById('a'+k);clearTimeout(t);"
                 "a.currentTime=s;a.play();t=setTimeout(()=>a.pause(),(e-s)*1000)}</script>")
    out.write_text("".join(parts), encoding="utf-8")
    print(f"\nlistening page: {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["hi", "ta"])
    ap.add_argument("--n", type=int, default=60, help="FLEURS utterances per language for the pause check")
    ap.add_argument("--listen", type=int, default=6, help="utterances on the listening page")
    args = ap.parse_args()
    asr = ASR()
    exact_test(asr, args.lang)
    pause_check(asr, args.lang, args.n)
    listening_page(asr, args.lang, args.listen, ROOT / "data/cache/align_check.html")


if __name__ == "__main__":
    main()
