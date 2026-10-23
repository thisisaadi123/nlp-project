# Step 3 report: speech branch on the Omen

Date: 2026-10-23. Runs on the Omen (RTX 4060 Laptop 8 GB, WSL2 Ubuntu 24.04, driver 592.82).
A clip goes in; a JSON with every spoken line comes out (speaker, words with timings and prosody,
emotion): `python -m backend.pipeline.speech.run <clip> --lang hi`.

## What was built

| File | What it does |
|---|---|
| `backend/pipeline/speech/asr.py` | IndicConformer CTC on onnxruntime-gpu (no NeMo); keeps the CTC log-probs |
| `backend/pipeline/speech/align.py` | Word timings: character-level wav2vec2 CTC aligner (Vakyansh, MIT) for hi/ta; IndicConformer CTC fallback |
| `backend/pipeline/speech/emotion.py` | emotion2vec+ large via funasr, mapped to our 7 classes; 1024-d embedding |
| `backend/pipeline/speech/diarize.py` | pyannote speaker-diarization-3.1 (pyannote.audio 4) |
| `backend/pipeline/speech/separate.py` | Demucs htdemucs, vocals only |
| `backend/pipeline/speech/models.py` | VRAM manager: ASR, emotion, aligners resident; Demucs, pyannote (later TTS, IndicTrans2) on demand |
| `backend/pipeline/speech/run.py` | The CLI; `analyze_line()` is also the live game loop |
| `backend/pipeline/tts/parler.py` | Indic Parler-TTS wrapper (used for test data now, reference dubs in step 5) |
| `backend/roman.py` | Readable romanization: "anna, enna panra?", "bhaiya, ye kya kar diya?" |
| `scripts/fetch_{fleurs,rasa,music}.py` | Dataset downloads (licenses below) |
| `scripts/eval_asr.py`, `check_alignment.py`, `eval_emotion.py`, `eval_scenes.py`, `eval_pipeline.py` | The measurements in this report |
| `scripts/make_align_test.py`, `make_scenes.py`, `data/scene_scripts.json` | Generated test data with exact ground truth |

## Environment (3.0)

Everything runs in one venv; `requirements-gpu.txt` is pinned, install with
`uv pip install -r requirements-gpu.txt --override overrides-gpu.txt`.

| Package | Version | Note |
|---|---|---|
| torch / torchaudio / torchcodec | 2.14.1 / 2.11.0 / 0.17.0 (cu130) | no downgrade needed |
| onnxruntime-gpu | 1.31.0 | IndicConformer ships as ONNX, so NeMo is not needed |
| transformers | 4.46.1 | parler-tts pins exactly this; IndicTransToolkit also needs < 5 |
| pyannote.audio | 4.0.7 | needs access to `pyannote/speaker-diarization-community-1` too (PLDA) |
| funasr | 1.4.16 | emotion2vec |
| demucs | 4.1.0 | |
| IndicTransToolkit / parler-tts | 1.1.1 / git d108732 | step 4 / 5, smoke-tested only |
| protobuf | 7.36.2 (override) | descript-audiotools (via parler-tts) declares < 5, pyannote 4 needs >= 5; Parler works with 7 |

## ASR (3.1)

IndicConformer 600M, CTC decoding. Text normalised the same way on both sides (NFC, punctuation removed).

| | hi | bn | mr | te | ta |
|---|---|---|---|---|---|
| FLEURS dev, 100 sentences: WER | **12.0%** | 15.1% | 19.0% | 24.7% | **27.1%** |
| FLEURS CER | 4.9% | 3.9% | 5.2% | 7.7% | 9.5% |
| Rasa (acted, 6 emotions + neutral, ~420 utterances): WER | **7.3%** | | | | **31.0%** |
| Rasa CER | 1.7% | | | | 5.7% |

Real-time factor ~0.01 (10 s of audio in ~0.1 s). Hindi holds up on emotional speech (WER 6-9% for
every emotion). Tamil emotional lines are worse (27-39%) than neutral (20%), mostly because Rasa's
emotional scripts are **spoken Tamil** and the ASR writes formal spelling: *irukkara* -> *irukkira*,
*anga* -> *ange*, *illa* -> *illai*, *ammaava* -> *ammaavai*. The meaning survives (CER 5.7%).
**Step 5:** the content score should compare characters or romanized text, not exact words.

## Word timings (3.2)

IndicConformer's own CTC timings are poor: its output is not peaky and it emits a line's first word
~0.3-0.4 s before the sound (the time axis itself is right). A character-level wav2vec2 CTC model
(Vakyansh, MIT, 95M parameters, 20 ms frames) now places IndicConformer's words in time. Silent
frames are forbidden to every character, and two words in continuous speech share one boundary.

Exact test: TTS words joined with no gaps, so every boundary is known (`make_align_test.py`,
8 sentences per language, 48 hi / 62 ta words).

| Within 100 ms of the truth | line start / end (hi, ta) | between words hi | between words ta |
|---|---|---|---|
| IndicConformer CTC | 25-75% | 38% | 39% |
| + silence mask | 50-88% | 38% | 40% |
| **wav2vec2 aligner + silence mask + gap closing** | **100% (~20 ms error)** | **68%** | **62%** |

On real FLEURS speech, word edges at pauses are within 100 ms 63-78% of the time for Hindi (was
36-47%) and 45-59% for Tamil, where the ASR's own word errors get aligned too. Bengali, Marathi and
Telugu use the IndicConformer fallback for now (Vakyansh has models for them, but their license
fields are empty). **Not done: the listening spot-check** (no one has listened yet);
`check_alignment.py` writes `data/cache/align_check.html` with one button per word.

## Emotion (3.3): the key measurement

emotion2vec+ large on Rasa test, 30 utterances per speaker and emotion (one female and one male
actor per language). Chance is 14%.

| Recall | angry | disgust | fear | happy | neutral | sad | surprise | **accuracy** |
|---|---|---|---|---|---|---|---|---|
| Hindi | 15% | 17% | 7% | 90% | 78% | 10% | 50% | **38.2%** |
| Tamil | 42% | 2% | 8% | 83% | 73% | 5% | 30% | **34.8%** |

Confusions: most angry, disgust and surprise lines are called **happy**; Hindi sad lines are called
neutral, Tamil sad lines happy. "Other" is almost never predicted (mean probability 0.01).

Arousal, high (angry/happy/fear/surprise) vs low (sad/neutral), 360 utterances per language:

| | Hindi AUC | Tamil AUC | accuracy at 0.5 |
|---|---|---|---|
| prosody arousal model, no baseline | 0.66 | 0.64 | 40-41% |
| prosody arousal model, speaker baseline | 0.77 | 0.69 | 53-60% |
| **emotion2vec: P(high-arousal emotions)** | **0.85** | **0.81** | **82%** |
| both (mean of z-scores) | 0.87 | 0.82 | |

1. **emotion2vec reliably tells how intense a line is, not which emotion it is** on Indic speech.
   The emotion slot of the score should rest on intensity (emotion2vec + prosody) for now, and the
   delivery card's emotion label should come from step 4's context fusion, not audio alone.
2. **The RAVDESS-trained arousal model ranks Indic speech in the right order but is mis-calibrated**
   (intense lines average 0.22-0.47). The speaker baseline helps a lot. `arousal_model.json` is unchanged.
3. Fine-tuning on Indic emotional speech (step 6) is clearly needed. Rasa's dataset card says
   **CC BY 4.0** (`docs/project-context.md` says CC BY-ND); it was used for evaluation only here.
   Whether it may be used for training is a team decision; if so, a different held-out set is needed.

Predictions: `data/cache/emotion_rasa.json`; embeddings: `data/cache/emotion_rasa_{hi,ta}.npz`.

## Generated scenes (3.6)

Nobody records, so `make_scenes.py` builds them from `data/scene_scripts.json`: 10 Hindi (Rohit,
Divya) and 10 Tamil (Arjun, Jaya) two-speaker scenes, 4 lines each, ~10-12 s, all seven emotions,
with register contrasts (*bhaiya/bhai*, *anna/machan*, *aap/tum*). Indic Parler-TTS lists only female
Tamil voices; Arjun (listed for another language) speaks Tamil with 0.8% CER. Every line was checked
with ASR (CER <= 0.2, up to 3 tries); all 80 passed. Each scene is also mixed with music at 0, -5 and
-10 dB music-to-speech. Music: MUSAN (OpenSLR 17) **RFM tracks (Kevin MacLeod, incompetech.com,
CC BY 3.0)** and **FMA western-art tracks (public domain)** only; share-alike and no-derivatives
tracks were skipped. Each scene's JSON names its track.

## Diarization and Demucs (3.4, 3.5)

20 scenes, 80 lines, 229 s (`eval_scenes.py`):

| Audio | DER (0.25 s collar) | DER (no collar) | 2 speakers found | lines with the right speaker |
|---|---|---|---|---|
| clean | **10.7%** | 16.1% | 19/20 | **95%** (hi 100%, ta 90%) |
| clean, number of speakers given | 10.7% | 16.1% | 19/20 | 95% |
| clean + Demucs | 10.7% | 16.6% | 19/20 | 95% |
| music -10 dB | 13.4% | 21.3% | 19/20 | 95% |
| music -10 dB + Demucs | **10.5%** | 17.9% | 19/20 | 96% |
| music -5 dB | 17.7% | 27.1% | 20/20 | 94% |
| music -5 dB + Demucs | **14.9%** | 22.4% | 17/20 | 91% |
| music 0 dB | 26.9% | 37.1% | 18/20 | 85% |
| music 0 dB + Demucs | **16.6%** | 24.4% | 18/20 | 91% |

DER is measured against a speech-only reference (pauses inside a line removed, exact because the
clean signal is known); marking whole lines as speech counts pyannote's correct pause detection as
missed speech (17.7% DER on clean scenes). Demucs mostly removes false alarms from the music. The remaining confusion is in 3 Tamil scenes: in one,
Arjun and Jaya are merged into a single speaker; in two, one line gets the other speaker. Lowering
the clustering threshold (0.70 -> 0.45) or `min_cluster_size` (12 -> 2) did not fix them and starts
splitting voices, so pyannote's defaults are kept.

ASR on each line cut at its true time:

| Music | WER without Demucs | WER with Demucs | CER without | CER with |
|---|---|---|---|---|
| none | **16.4%** | 17.9% | **5.7%** | 6.1% |
| -10 dB | 24.9% | **23.4%** | 8.9% | **8.8%** |
| -5 dB | 33.8% | **30.6%** | 16.4% | **13.7%** |
| 0 dB | 45.5% | **38.7%** | 28.6% | **22.6%** |

**Demucs helps whenever music is present and slightly hurts clean speech**, so it runs in **auto**
mode (`config.yaml speech.demucs`): the vocals are kept only when the removed part was at least
-20 dB relative to them. Its estimate is accurate (median -37.0 dB on clean scenes, -9.4 / -4.2 /
+1.1 dB at -10 / -5 / 0 dB), and it switched on for 60/60 music mixes and 0/20 clean scenes.

## The whole pipeline (3.7)

`run.py` on the scenes, auto Demucs. Text is scored per scene on all lines in time order, so a
diarization merge doesn't count words twice.

| | lines found | speaker right | WER | CER |
|---|---|---|---|---|
| Hindi clean | 40/40 | 100% | 8.9% | 3.0% |
| Hindi music -10 dB | 40/40 | 100% | 17.4% | 6.7% |
| Hindi music -5 dB | 39/40 | 98% | 26.0% | 14.2% |
| Hindi music 0 dB | 38/40 | 95% | 38.8% | 26.4% |
| Tamil clean | 35/40 | 90% | 30.2% | 7.4% |
| Tamil music -10 dB | 35/40 | 90% | 31.2% | 9.6% |
| Tamil music -5 dB | 31/40 | 85% | 42.9% | 16.8% |
| Tamil music 0 dB | 35/40 | 85% | 50.3% | 26.5% |

Lines are built from diarized turns: same-speaker turns less than 1.0 s apart form one line
(pyannote splits at pauses), and pieces under 0.6 s join their longer neighbour (pyannote sometimes
flickers between speakers at a line's start; dropping those pieces lost whole lines).

Each output line has `line_id, speaker, start, end, language, text, text_roman, words` (each word:
`w, start, end, stress, pitch, energy`), `pauses, prosody` (the step 2 stats) and `emotion`
(`label, probs, arousal` from prosody with the speaker's own baseline, plus `arousal_e2v`).
Embeddings go to a `.npz` next to the JSON.

## VRAM and speed (3.8)

| Target | Result |
|---|---|
| ASR + emotion resident <= ~3.5 GB | **2.97 GB**; **3.36 GB** with one language's aligner (3.75 GB with both) |
| ~30 s clip through all stages <= ~45 s | **7.5 s** warm (36 s clip, 12 lines; Demucs 2.4, pyannote 3.2, lines 1.9 s); ~20 s cold |
| live loop (5 s recording) <= 3 s | **0.13 s** warm, 0.2 s first call |

Peak whole-GPU memory while a clip runs is 7.5 GB of 8 (Windows itself uses ~1.1 GB), with Demucs
and pyannote released after use. Loading only the target language's aligner keeps the resident set
under target. Close games and launchers on Windows while serving.

## Problems and fixes

| Problem | Fix |
|---|---|
| onnxruntime 1.31 refuses IndicConformer's external weights in the HF cache (symlinks into `blobs/`) | model copied as plain files into `models/` |
| parler-tts pins transformers 4.46.1; IndicTransToolkit breaks on transformers 5 | whole venv on 4.46.1 |
| protobuf: descript-audiotools < 5 vs pyannote 4 >= 5 | `overrides-gpu.txt` (protobuf 7.36.2), tested |
| the `onnx` package needs protobuf >= 5 | not installed (only onnxruntime is needed) |
| pyannote.audio 4 needs access to a second gated repo | accepted on the HF account; added to `check_setup.py` |
| IndicConformer CTC word timings ~200 ms off | wav2vec2 character aligner (above) |
| `forced_align` needs contiguous log-probs (numpy fancy indexing returns Fortran order) | `np.ascontiguousarray` |
| stock transliteration of Tamil is unreadable ("ghurabhbhadhdhadhu") | `backend/roman.py` with positional voicing |
| diarization flicker dropped whole lines | short pieces join their neighbour |
| WSL shuts down when no session is attached, killing detached jobs | run long jobs as tracked tasks |

## Open items

- **Listening check of word timings** (`data/cache/align_check.html`): needs a person.
- **Gemini TTS smoke takes** from the team Drive folder: the folder looked empty from the Omen's
  account, so the Hindi TTS smoke test was not run (Rasa covers the same ground).
- **Emotion:** fine-tune emotion2vec on Indic data (step 6); decide whether Rasa may be used for training.
- **Arousal model:** recalibrate on Indic speech (scores are compressed low).
- **Tamil diarization** on short scenes with similar voices: re-check on the step 4 movie clips.
- **Romanization** of Bengali is still rough.
- **Aligners for bn, mr, te:** pick license-clear models.
- IndicConformer's model card now recommends a newer model (*Indic-Transcribe*); not tried.
