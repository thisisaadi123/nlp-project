"""Background music for the generated test scenes: part of MUSAN (OpenSLR 17, corypaik/musan).

Only two sources are used, both free to mix:
  - rfm: Kevin MacLeod / incompetech royalty-free music, CC BY 3.0
  - fma-western-art: public domain
Share-alike (fma, hd-classical, jamendo) and no-derivatives (some jamendo) tracks are skipped.
Two of the ten music shards (~0.9 GB) are downloaded; tracks go to data/datasets/music/<source>/
with each source's LICENSE (credits). Never commit them.
"""
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data/datasets/music"
REPO = "corypaik/musan"
SHARDS = ["music/train-00002-of-00010.parquet", "music/train-00008-of-00010.parquet"]
SOURCES = {"rfm", "fma-western-art"}


def main():
    for src in SOURCES:
        (DEST / src).mkdir(parents=True, exist_ok=True)
        lic = hf_hub_download(REPO, f"licenses/music/{src}/LICENSE", repo_type="dataset")
        (DEST / src / "LICENSE").write_bytes(Path(lic).read_bytes())
    n = 0
    for shard in SHARDS:
        t = pq.read_table(hf_hub_download(REPO, shard, repo_type="dataset"), columns=["source", "path", "audio"])
        for src, path, audio in zip(*(t.column(c).to_pylist() for c in ("source", "path", "audio"))):
            if src not in SOURCES:
                continue
            out = DEST / src / Path(path).name
            if not out.exists():
                y, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                sf.write(out, y if y.ndim == 1 else y.mean(1), sr)
            n += 1
    total = sum(f.stat().st_size for f in DEST.rglob("*.wav")) / 2**30
    print(f"{n} tracks in {DEST.relative_to(ROOT)} ({total:.2f} GB)")
    sys.exit(0 if n else 1)


if __name__ == "__main__":
    main()
