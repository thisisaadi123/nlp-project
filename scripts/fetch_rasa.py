"""Downloads the test split of AI4Bharat Rasa (expressive acted speech) into data/datasets/rasa.

EVALUATION ONLY. The dataset card says CC BY 4.0; the team treats it as evaluation-only anyway:
no modified copies are published and no model is trained on it. Never commit it.
Hindi test ~1.6 GB, Tamil test ~1.9 GB (2 speakers each, one female and one male).

    python scripts/fetch_rasa.py                  # Hindi + Tamil
    python scripts/fetch_rasa.py --lang hi
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.settings import config  # noqa: E402

DEST = ROOT / "data/datasets/rasa"


def fetch(lang: str) -> Path:
    name = config()["languages"][lang]["name"]          # Rasa folders use language names
    snapshot_download("ai4bharat/Rasa", repo_type="dataset", local_dir=DEST,
                      allow_patterns=[f"{name}/test-*.parquet"])
    files = sorted((DEST / name).glob("test-*.parquet"))
    print(f"{lang}: {len(files)} parquet files, {sum(f.stat().st_size for f in files)/2**30:.2f} GB in {DEST / name}")
    return DEST / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["hi", "ta"])
    for lang in ap.parse_args().lang:
        fetch(lang)


if __name__ == "__main__":
    main()
