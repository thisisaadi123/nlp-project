"""ASR baseline: word and character error rate of IndicConformer (CTC) per language.

    python scripts/fetch_fleurs.py            # once
    python scripts/eval_asr.py                # FLEURS dev, 20 distinct sentences per language
    python scripts/eval_asr.py --n 50 --lang hi ta

Both sides are normalised the same way before scoring: Unicode NFC, punctuation removed
(including the danda), whitespace collapsed. Per-utterance results go to data/cache/.
"""
import argparse
import csv
import json
import random
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.speech.asr import ASR  # noqa: E402
from backend.settings import config  # noqa: E402

FLEURS = ROOT / "data/datasets/fleurs"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def fleurs_items(lang: str, n: int, seed: int = 0) -> list[dict]:
    """n utterances of n distinct sentences (FLEURS has several speakers per sentence)."""
    d = FLEURS / lang / "dev"
    if not d.is_dir():
        sys.exit(f"No FLEURS {lang} in {d}. Run scripts/fetch_fleurs.py first.")
    # columns: id, file, raw transcription, normalised transcription, chars, samples, gender
    rows = list(csv.reader(open(d / "transcripts.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    first = {}
    for r in rows:
        if (d / r[1]).exists():
            first.setdefault(r[0], r)
    picked = random.Random(seed).sample(sorted(first.values()), min(n, len(first)))
    return [{"id": r[1], "path": d / r[1], "ref": r[2]} for r in picked]


def rasa_items(lang: str, per_class: int) -> list[dict]:
    from scripts.rasa_data import load
    return [{**it, "ref": it["text"]} for it in load(lang, per_class)]


def evaluate(asr: ASR, lang: str, items: list[dict]) -> dict:
    refs, hyps, rows, audio_s, asr_s = [], [], [], 0.0, 0.0
    for it in items:
        y = it["y"] if "y" in it else sf.read(it["path"], dtype="float32")[0]
        t = time.time()
        hyp = asr.transcribe(y, lang).text
        asr_s += time.time() - t
        audio_s += len(y) / 16000
        ref_n, hyp_n = normalize(it["ref"]), normalize(hyp)
        refs.append(ref_n); hyps.append(hyp_n)
        rows.append({"id": it["id"], "emotion": it.get("emotion"), "ref": ref_n, "hyp": hyp_n,
                     "wer": jiwer.wer(ref_n, hyp_n)})
    # CER with spaces removed: Tamil/Telugu WER is inflated by word-splitting conventions
    # (ஒருநாள் vs ஒரு நாள்), which don't matter for scoring a dub
    nospace = [r.replace(" ", "") for r in refs], [h.replace(" ", "") for h in hyps]
    by_emotion = {}
    for e in sorted({r["emotion"] for r in rows if r["emotion"]}):
        sel = [k for k, r in enumerate(rows) if r["emotion"] == e]
        by_emotion[e] = jiwer.wer([refs[k] for k in sel], [hyps[k] for k in sel])
    return {"lang": lang, "n": len(items), "wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps),
            "cer_nospace": jiwer.cer(*nospace), "audio_s": audio_s, "rtf": asr_s / audio_s,
            "wer_by_emotion": by_emotion, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fleurs", choices=["fleurs", "rasa"])
    ap.add_argument("--lang", nargs="+", default=None, help="default: all 5 (FLEURS) or hi ta (Rasa)")
    ap.add_argument("--n", type=int, default=20, help="FLEURS: sentences per language; Rasa: per speaker+emotion")
    args = ap.parse_args()
    langs = args.lang or (list(config()["languages"]) if args.dataset == "fleurs" else ["hi", "ta"])
    get_items = fleurs_items if args.dataset == "fleurs" else rasa_items

    asr = ASR()
    warm = fleurs_items("hi", 1)[0]
    asr.transcribe(sf.read(warm["path"], dtype="float32")[0], "hi")  # first call allocates GPU memory
    results = []
    print(f"{'lang':5} {'n':>3} {'WER':>6} {'CER':>6} {'CER-ns':>6} {'audio':>7} {'RTF':>6}")
    for lang in langs:
        r = evaluate(asr, lang, get_items(lang, args.n))
        results.append(r)
        print(f"{lang:5} {r['n']:3d} {r['wer']*100:5.1f}% {r['cer']*100:5.1f}% {r['cer_nospace']*100:5.1f}% "
              f"{r['audio_s']:6.0f}s {r['rtf']:6.3f}")
        if r["wer_by_emotion"]:
            print("      WER by emotion: " + ", ".join(f"{e} {w*100:.0f}%" for e, w in r["wer_by_emotion"].items()))
    out = ROOT / f"data/cache/asr_eval_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"per-utterance results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
