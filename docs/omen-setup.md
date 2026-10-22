# Omen setup (Windows → WSL2 + CUDA)

The Omen (RTX 4060, 8 GB) runs the backend and all GPU models inside WSL2 Ubuntu.

## 1. On Windows

1. **Update the NVIDIA driver** (Game Ready or Studio) from nvidia.com or the NVIDIA app.
   This driver also provides CUDA to WSL. **Never install an NVIDIA driver inside WSL.**
2. **Install WSL2 + Ubuntu.** In PowerShell as administrator:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
   Reboot, open "Ubuntu 22.04" from the Start menu, create a Linux username and password.
3. **Power:** keep the laptop plugged in, set power mode to *Best performance*, and turn off
   sleep while it is serving the game.
4. Keep **~50 GB free** on the drive for models, environments and clips.

## 2. Inside Ubuntu (WSL)

```bash
nvidia-smi                       # must list the RTX 4060
sudo apt update && sudo apt install -y git
gh auth login || true            # or use an HTTPS token / SSH key for GitHub
cd ~
git clone https://github.com/<owner>/dub-dub-goose.git
cd dub-dub-goose
bash scripts/setup_omen.sh
```

Then fill in `.env` (copy the same keys as on the Mac) and re-run:

```bash
source .venv/bin/activate
python scripts/check_setup.py
```

Step 0 is done on the Omen when the GPU line shows
`PASS GPU — NVIDIA GeForce RTX 4060 Laptop GPU, ~8 GB VRAM`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi` not found in WSL | Update the Windows NVIDIA driver, then `wsl --shutdown` in PowerShell and reopen Ubuntu |
| `torch.cuda.is_available()` is False | Same as above; also check `wsl --version` shows WSL 2 |
| Everything is slow | The repo is under `/mnt/c/...`; clone it into `~` instead |
| `wsl --install` says virtualization is off | Enable Intel VT-x / AMD-V (SVM) in the BIOS |

## Later: edit on the Mac, run on the Omen

Once GPU stages are in progress, install Tailscale on both machines and enable SSH into WSL,
then use VS Code Remote-SSH from the Mac. Until then, push from the Mac and pull on the Omen.
