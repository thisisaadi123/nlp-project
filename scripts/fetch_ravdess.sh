#!/usr/bin/env bash
# Downloads the RAVDESS speech audio (~200 MB, CC BY-NC-SA 4.0) into data/datasets/ravdess.
# Used only to test the prosody score; never commit it.
set -euo pipefail
cd "$(dirname "$0")/.."
dest=data/datasets/ravdess
if [ -d "$dest/Actor_01" ]; then echo "RAVDESS already in $dest"; exit 0; fi
mkdir -p "$dest"
curl -fSL -o "$dest/ravdess.zip" "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
unzip -q "$dest/ravdess.zip" -d "$dest"
rm "$dest/ravdess.zip"
echo "$(find "$dest" -name '*.wav' | wc -l) files in $dest"
