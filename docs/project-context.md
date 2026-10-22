# Project context

Living summary of decisions and status. Read this before working on the project.

## What we are building

**Dub-Dub-Goose**: an emotion- and context-aware speech-to-speech dubbing system for Indian
languages, turned into a social game.

1. A user uploads a short video clip and picks a target language.
2. The system transcribes it, works out the scene context (who is speaking to whom, the setting,
   the mood) and how each line is delivered (emotion, stress, pitch, pauses, timing).
3. It translates each line **directly** between Indian languages (no English text in between),
   choosing words by context: *bhai* (peer, casual) and *bhaiya* (elder, respectful) must come out
   differently, e.g. Tamil *machan / nanba* vs *anna*.
4. It shows the user a **delivery card**: the translated line annotated with emotion, stressed
   words, pauses and pitch direction, plus a synthesized reference dub. "bhaiya…" (pleading) and
   "BHAIYA!" (angry) must be conveyed differently.
5. The user records their own dub. It is analysed with the same feature extractors as the
   original and scored out of 100, with short AI-written tips.
6. Score ≥ 70 earns a **Dub**; below that is a **Goose**.

## Languages

Hindi, Bengali, Marathi, Telugu, Tamil, any pair (20 directions). Build and test Hindi ↔ one other
first; the code supports all five through `config.yaml`.

## Machines

| Machine | Role |
|---|---|
| Mac (Aadi) | development, prosody + scoring code, API-only stages |
| Omen laptop, RTX 4060 8 GB, Windows + **WSL2 Ubuntu 22.04** | runs the backend and all GPU models |
| Google Colab Pro | fine-tuning speech emotion, score calibration |
| Groq free API | captioning, scene context, translation refinement, insights |
| Gemini free API | backup LLM |

## Pipeline

**A. Clip processing — once per upload, async, ~20–60 s**

1. Upload clip (≤ 45 s), pick target language
2. ffmpeg: 16 kHz mono audio + keyframes (scene changes + ~1.5 fps)
3. In parallel:
   - **Visual:** keyframes → Groq vision model captions (setting, people, expressions, actions)
   - **Speech:** Demucs (remove music) → pyannote (who spoke when) → IndicConformer ASR with CTC
     word timestamps → Parselmouth/librosa prosody per word → emotion2vec emotion + valence/arousal/dominance
4. Scene context (Groq LLM): captions + diarized transcript → speakers, relationships, mood, addressee per line
5. Emotion-intent fusion per line: text intent + acoustic emotion + visual cue + word tags
   (e.g. `BHAIYA[stress, pitch-spike] ⏸0.6s`); mismatches flagged (possible sarcasm)
6. IndicTrans2 draft → Groq LLM refines (register, emphasis mapping, fit to original duration).
   If Groq is unavailable or rate-limited, the IndicTrans2 draft is used as is.
7. Delivery card + Indic Parler-TTS reference dub (style prompt from the emotion vector)
8. Cache everything per clip

**B. Game loop — every attempt, target < 5 s**

1. User picks a line, studies the card, records a dub
2. Same speech analysis on the user audio (ASR + emotion models stay loaded in VRAM)
3. Score: content 20 (user ASR vs translated line), emotion 25, emphasis 15, contour 20
   (pitch z-scored per speaker, DTW), timing 20
4. Groq writes 2–3 tips (template fallback if rate-limited) → Dub / Goose

## Models (all free)

| Stage | Model | Runs on |
|---|---|---|
| Music removal | Demucs (htdemucs) | Omen GPU |
| Diarization | pyannote/speaker-diarization-3.1 | Omen GPU |
| ASR | ai4bharat/indic-conformer-600m-multilingual | Omen GPU |
| Prosody | Parselmouth + librosa | CPU |
| Speech emotion | emotion2vec (to be fine-tuned on Indic data in Colab) | Omen GPU |
| Translation draft | ai4bharat/indictrans2-indic-indic-dist-320M | Omen GPU |
| Captioning | Groq `qwen/qwen3.8-27b` | Groq |
| Context, translation refine, insights | Groq `openai/gpt-oss-120b` | Groq |
| Reference TTS | ai4bharat/indic-parler-tts | Omen GPU |
| Backup LLM | Gemini `gemini-flash-latest` | Google |

VRAM plan for 8 GB: ASR + emotion stay resident; the others load one at a time and unload.

## Findings so far

- Groq retired Llama 3.3 70B and Llama 4 Scout. `gpt-oss-120b` kept bhai/bhaiya apart in Tamil
  (*நண்பா* vs *அண்ணா*); `qwen3.8-27b` turned both into *பிரோ* ("bro"), so Qwen is used only for
  image captioning.
- `gpt-oss-120b` is a reasoning model: it spends output tokens thinking, so never set a tiny `max_tokens`.
- Qwen's free tier has a low output-tokens-per-minute cap; keep captioning calls short (`vision_max_tokens`).
- MMS forced-aligner weights are non-commercial (CC-BY-NC), so word timestamps come from IndicConformer's CTC output.
- emotion2vec is trained mostly on English/Chinese speech; its accuracy on Indic speech must be checked in step 3.

## Status

| Step | State |
|---|---|
| 0. Setup | **Mac done** (all checks pass). **Omen pending.** |
| 1. Test set: 15 clips (3 per language) labelled in `data/testset.csv` | not started |
| 2. Prosody + scoring core (Mac) | next; needs Aadi's 4 test recordings in `data/clips/step2/` |
| 3. Speech branch on the Omen | after Omen setup |
| 4. Visual branch + context + translation | after step 2 |
| 5. Delivery card + TTS + game loop (MVP) | — |
| 6. Train + calibrate (Colab) | — |
| 7. Polish + demo | — |

## Ground rules

- Secrets live only in `.env` (git-ignored). Never print, commit or paste keys.
- Video clips and model files are git-ignored; never commit them.
- Model names live in `config.yaml`; `python scripts/check_setup.py` verifies them against what is live.
- Aadi handles git pushes to GitHub; ask before committing or pushing.
- Repo: https://github.com/thisisaadi123/nlp-project (public)
