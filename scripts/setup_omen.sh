#!/usr/bin/env bash
# Step 0 setup for the Omen, run INSIDE WSL Ubuntu from the repo root:
#   bash scripts/setup_omen.sh
# Prerequisites (on Windows): latest NVIDIA driver, WSL2 + Ubuntu 22.04. See docs/omen-setup.md.
set -euo pipefail

if ! grep -qi microsoft /proc/version; then
  echo "This script is meant for WSL on the Omen." >&2
  exit 1
fi

case "$(pwd)" in
  /mnt/*) echo "Clone the repo inside WSL (e.g. ~/dub-dub-goose), not under /mnt/c — file access there is very slow." >&2; exit 1 ;;
esac

echo "==> Checking GPU passthrough"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi failed. Update the NVIDIA driver on Windows (do not install one inside WSL), then run 'wsl --shutdown' in PowerShell and retry." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y build-essential git ffmpeg sox libsndfile1 curl

echo "==> uv + Python 3.11"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate

echo "==> Python packages (light stack + torch with CUDA)"
uv pip install -r requirements-mac.txt
# Linux wheels from PyPI bundle CUDA, so no special index is needed for the 4060.
uv pip install torch torchaudio

[ -f .env ] || { cp .env.example .env; echo "Created .env — fill in your keys, then re-run the check."; }

echo "==> Running checks"
python scripts/check_setup.py
