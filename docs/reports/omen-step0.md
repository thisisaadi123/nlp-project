# Omen Step 0 report

Date: 2026-10-23. Result: **done**. `check_setup.py` gives 14 passed, 0 warnings, 0 failed.

## Machine

| | |
|---|---|
| WSL | WSL 2 (wsl.exe 3.0.3.0, kernel 6.18.54.1-1) |
| Distro | Ubuntu **24.04.3**, not 22.04 as `omen-setup.md` asks. That is fine because uv supplies Python 3.11. |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |
| Driver | 592.82 on Windows (nvidia-smi reports CUDA 13.1); no driver is installed inside WSL |
| torch | 2.14.1+cu130; torch sees CUDA 13.0; `torch.cuda.is_available()` = True |
| Python | 3.11.17 in `.venv` (uv) |
| Repo | `~/nlp-project` inside WSL, not under `/mnt/c` |
| Disk | ~950 GB free in WSL |

## GPU smoke test

10× 4096×4096 fp32 matmul on CUDA: **0.27 s**. Free VRAM: **6.65 / 8.00 GiB**. About 1 GB is held by
Windows apps; close games and launchers while the Omen is serving, so the resident models (ASR and emotion)
have more headroom.

## check_setup.py

```
[PASS] Python — 3.11.17
[PASS] ffmpeg — ffmpeg version 6.1.1-3ubuntu5
[PASS] config.yaml — languages: Hindi, Bengali, Marathi, Telugu, Tamil
[PASS] Groq text_model — openai/gpt-oss-120b
[PASS] Groq vision_model — qwen/qwen3.8-27b
[PASS] Groq test call — அண்ணா
        Live Groq models: allam-2-7b, canopylabs/orpheus-arabic-saudi, canopylabs/orpheus-v1-english, meta-llama/llama-prompt-guard-2-22m, meta-llama/llama-prompt-guard-2-86m, openai/gpt-oss-120b, openai/gpt-oss-20b, openai/gpt-oss-safeguard-20b, qwen/qwen3.8-27b, whisper-large-v3, whisper-large-v3-turbo
[PASS] Hugging Face — logged in as epicbluedude
[PASS] HF access pyannote/speaker-diarization-3.1
[PASS] HF access pyannote/segmentation-3.0
[PASS] HF access ai4bharat/indic-conformer-600m-multilingual
[PASS] HF access ai4bharat/indictrans2-indic-indic-dist-320M
[PASS] HF access ai4bharat/indic-parler-tts
[PASS] Gemini (backup) — 50 models visible
[PASS] GPU — NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GB VRAM, torch 2.14.1+cu130

14 passed, 0 warnings, 0 failed
```

## Notes

- The Omen uses its own Groq, Hugging Face and Gemini keys. All 5 gated models are accepted on the
  `epicbluedude` HF account.
- The first HF check failed on all 5 gated models. The fix was accepting the terms and/or using a **Read**
  token: a fine-grained token cannot open gated repos unless that permission is ticked.
- The `sudo apt-get` step had to be run by hand in an Ubuntu terminal, because it needs the user's password.
- Claude Code runs on Windows and drives WSL through `wsl.exe`. It is not installed inside WSL.
- The heavy stack in `requirements-gpu.txt` is not installed yet; that is Step 3.
- No repo files were changed during setup.
