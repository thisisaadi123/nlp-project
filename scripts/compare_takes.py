"""Score recordings against a reference line and print the breakdown.

    # one reference, several attempts
    python scripts/compare_takes.py original.wav take1.m4a take2.m4a

    # with neutral calibration recordings for each speaker (more accurate)
    python scripts/compare_takes.py original.wav take1.m4a \
        --ref-baseline original_speaker_neutral.wav --att-baseline my_neutral.wav

    # all takes from one speaker: --baseline applies to every file
    python scripts/compare_takes.py angry_1.m4a angry_2.m4a sad.m4a flat.m4a --baseline calm_read.m4a

    # every file against every other file
    python scripts/compare_takes.py data/clips/step2/*.wav --matrix

Accepts any audio or video format ffmpeg can read.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.scoring.arousal import baseline  # noqa: E402
from backend.scoring.compare import score  # noqa: E402
from backend.scoring.feedback import tips  # noqa: E402
from backend.scoring.prosody import extract_file  # noqa: E402


def load_baseline(paths):
    return baseline([extract_file(p) for p in paths]) if paths else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--baseline", nargs="+", help="neutral recordings used for every file (one speaker)")
    ap.add_argument("--ref-baseline", nargs="+", help="neutral recordings of the reference speaker")
    ap.add_argument("--att-baseline", nargs="+", help="neutral recordings of the attempt speaker")
    ap.add_argument("--matrix", action="store_true", help="score every file against every other file")
    args = ap.parse_args()

    shared = load_baseline(args.baseline)
    ref_base = load_baseline(args.ref_baseline) or shared
    att_base = load_baseline(args.att_baseline) or shared
    files = [args.reference] + args.attempts
    prosody = {f: extract_file(f) for f in files}
    names = {f: Path(f).stem for f in files}

    if args.matrix:
        w = max(len(n) for n in names.values()) + 2
        print("Score matrix (row = reference, column = attempt)\n")
        print(" " * w + "".join(f"{names[f]:>{w}s}" for f in files))
        for r in files:
            row = [score(prosody[r], prosody[a], ref_base, att_base).score if a != r else None for a in files]
            print(f"{names[r]:{w}s}" + "".join(f"{'-':>{w}s}" if s is None else f"{s:>{w}.1f}" for s in row))
        return

    ref = prosody[args.reference]
    print(f"Reference: {names[args.reference]} ({ref.duration:.2f} s)"
          f"   baselines: reference {'yes' if ref_base else 'no'}, attempt {'yes' if att_base else 'no'}\n")
    print(f"{'attempt':20s}{'score':>7s}{'emotion':>9s}{'contour':>9s}{'timing':>8s}{'emph':>7s}"
          f"{'intensity ref/att':>19s}  verdict")
    results = []
    for f in args.attempts:
        r = score(ref, prosody[f], ref_base, att_base)
        results.append((f, r))
        c, d = r.components, r.details
        print(f"{names[f]:20s}{r.score:7.1f}{c['emotion']*100:9.0f}{c['contour']*100:9.0f}"
              f"{c['timing']*100:8.0f}{c['emphasis']*100:7.0f}"
              f"{d['ref_arousal']:12.2f} / {d['att_arousal']:.2f}  {r.verdict}")
    for f, r in results:
        print(f"\n{names[f]}:")
        for t in tips(r):
            print(f"  - {t}")


if __name__ == "__main__":
    main()
