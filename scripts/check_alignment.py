"""How accurate are the CTC word timings?

1. Objective check, no listening needed: at every clear pause (>= 150 ms of near-silence found
   by backend/scoring/prosody.py), the word before should end where the pause starts and the
   word after should start where it ends. Reports the errors for raw CTC spans and padded ones.
2. A listening page (data/cache/align_check.html): one button per word plays just that word,
   so a person can hear whether the boundaries cut words off.

    python scripts/check_alignment.py --lang hi ta --n 60 --listen 5
"""
import argparse
import base64
import html
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.align import align  # noqa: E402
from backend.pipeline.speech.asr import ASR  # noqa: E402
from backend.roman import romanize  # noqa: E402
from backend.scoring.prosody import extract  # noqa: E402
from scripts.eval_asr import fleurs_items  # noqa: E402


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


def summary(name: str, errs: list[float]) -> str:
    e = np.array(errs) * 1000
    return (f"{name:10} n={len(e):3d}  median {np.median(e):+5.0f} ms  mean |err| {np.mean(np.abs(e)):4.0f} ms  "
            f"within 100 ms {np.mean(np.abs(e) <= 100) * 100:3.0f}%")


def listening_page(entries: list[dict], out: Path):
    parts = ["<!doctype html><meta charset=utf-8><title>Word timing check</title>",
             "<style>body{font:16px system-ui;max-width:900px;margin:2em auto;padding:0 16px}"
             "button{margin:3px;padding:6px 9px;font-size:16px}small{color:#666}</style>",
             "<h1>Word timing check</h1><p>Click a word: it should play that whole word and nothing of "
             "its neighbours. Note words that sound cut off or include the next word.</p>"]
    for k, e in enumerate(entries):
        buf = io.BytesIO(); sf.write(buf, e["y"], 16000, format="WAV")
        src = base64.b64encode(buf.getvalue()).decode()
        parts.append(f"<h3>{k + 1}. {e['lang']} <small>{html.escape(e['id'])}</small></h3>"
                     f"<audio id=a{k} controls src='data:audio/wav;base64,{src}'></audio><div>")
        for w in e["words"]:
            parts.append(f"<button onclick='play({k},{w.start},{w.end})' title='{w.start:.2f}-{w.end:.2f}s'>"
                         f"{html.escape(w.w)}<br><small>{html.escape(romanize(w.w, e['lang']))}</small></button>")
        parts.append("</div>")
    parts.append("<script>let t;function play(k,s,e){const a=document.getElementById('a'+k);clearTimeout(t);"
                 "a.currentTime=s;a.play();t=setTimeout(()=>a.pause(),(e-s)*1000)}</script>")
    out.write_text("".join(parts), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["hi", "ta"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--listen", type=int, default=5, help="utterances on the listening page")
    args = ap.parse_args()

    asr = ASR()
    listen = []
    for lang in args.lang:
        res = {"raw": ([], []), "padded": ([], [])}
        for k, it in enumerate(fleurs_items(lang, args.n, seed=1)):
            y, _ = sf.read(it["path"], dtype="float32")
            tr = asr.transcribe(y, lang)
            try:
                pros = extract(y)
            except ValueError:
                continue
            for mode in res:
                words = align(tr, pad=mode == "padded")
                e, s = boundary_errors(words, pros)
                res[mode][0].extend(e); res[mode][1].extend(s)
            if k < args.listen and (lang == args.lang[0] or len(listen) < args.listen):
                listen.append({"lang": lang, "id": it["id"], "y": y, "words": align(tr)})
        print(f"\n{lang}: word edges vs pause edges (+ = word edge later than the pause edge)")
        for mode, (e, s) in res.items():
            print("  " + summary(f"{mode} end", e))
            print("  " + summary(f"{mode} start", s))
    out = ROOT / "data/cache/align_check.html"
    listening_page(listen[: args.listen], out)
    print(f"\nlistening page: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
