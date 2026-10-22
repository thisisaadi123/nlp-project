"""Downloads Google FLEURS (CC BY 4.0) read speech for our languages into data/datasets/fleurs.

Used only to measure ASR word error rate; never commit it. Defaults to the dev split
(~130-280 MB per language). Usage:
    python scripts/fetch_fleurs.py                 # all 5 languages, dev split
    python scripts/fetch_fleurs.py --lang hi ta --split test
"""
import argparse
import shutil
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.settings import config  # noqa: E402

DEST = ROOT / "data/datasets/fleurs"
REPO = "google/fleurs"


def fetch(lang: str, split: str) -> Path:
    out = DEST / lang / split
    if out.is_dir() and any(out.glob("*.wav")):
        print(f"{lang}/{split}: already in {out}")
        return out
    code = f"{lang}_in"
    tsv = hf_hub_download(REPO, f"data/{code}/{split}.tsv", repo_type="dataset")
    tgz = hf_hub_download(REPO, f"data/{code}/audio/{split}.tar.gz", repo_type="dataset")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(tsv, out / "transcripts.tsv")
    with tarfile.open(tgz) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".wav"):
                m.name = Path(m.name).name          # flatten
                tar.extract(m, out, filter="data")
    print(f"{lang}/{split}: {len(list(out.glob('*.wav')))} files in {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=list(config()["languages"]))
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    args = ap.parse_args()
    for lang in args.lang:
        fetch(lang, args.split)


if __name__ == "__main__":
    main()
