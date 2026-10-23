# Dub-Dub-Goose

A dubbing game for Indian languages. Pick a movie scene, hear each line, then say it yourself in your
language over the muted scene. You get a score out of 100 and a **Dub** (55+) or a **Goose**.

Behind the game is a direct Indic-to-Indic dubbing pipeline that keeps context and emotion: it knows who
is talking to whom, so Tamil *anni* becomes Hindi *bhabhi* (not "sister"), Hindi *bhai* between friends
becomes Tamil *machan* (not *anna*), and lines come out the way people talk, not the way they write.

**Languages:** Hindi, Tamil, Bengali, Marathi, Telugu. The library has 10 scenes (2+ per language).

## How it works

```
movie clip ──► step 3: speech analysis ──► step 4: context + translation ──► game library ──► the game
               who spoke when, words,       scene context (video captions      cards, voice      listen, say it,
               pitch/loudness, emotion      + dialogue), Sarvam draft,         lines, reference  get scored
                                            translation with checks            prosody
```

**Translation (step 4).** Sarvam does the language work; code checks what models get wrong:

1. *Draft:* Sarvam Mayura in spoken ("modern-colloquial") mode. IndicTrans2 is the offline fallback.
2. *Scene context:* who the characters are, how they are related, who each line is said to.
3. *Translation:* Sarvam's chat model writes each line from its meaning, in the target language's spoken
   style (a style guide per language), with the speaker's and listener's gender.
4. *Review:* the same model re-reads the whole scene as a native dialogue writer.
5. *Checks in code,* each followed by a targeted repair that is only accepted if it fixes the problem:
   - **family words** (`translate/kinship.py`): the source's relation must come out as the target's word
     (anni → bhabhi / vadina / boudi / vahini);
   - **written-style words** (`translate/written.py`): forms people read but never say (Tamil
     பார்க்கவும், போகிறது; Hindi कृपया);
   - **copied words** (`translate/copies.py`): source words left in the target script instead of being
     translated (Tamil கெளம்புறேன்டா once came out as Hindi "गुम्बुरेंटा").
6. *Fit:* lines that run much longer than the actor's line are shortened.
7. *Romanization* for players who speak the language but can't read its script, checked word by word.

**Scoring (`backend/scoring/director.py`).** Each take is checked against what the screen asked for:

| Part | What it checks |
|---|---|
| Timing | you started and stopped when the actor did (on the scene's clock) |
| Volume | soft / normal / loud / shouting, measured against your own calm mic-check reading |
| Punch | your strongest moment lands on the gold word |
| Tone | your voice rises, drops or stays level at the end like the actor's |
| Words | Sarvam speech-to-text: did you say the line, and in the right language? |

Each part comes with a plain verdict ("0.8 s late", "too soft: needs LOUD"). Players can rate each score
👍/👎; ratings (numbers only) go to `data/feedback/ratings.jsonl` for setting the Dub line from real plays.

## Setup (Mac)

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-mac.txt
cp .env.example .env              # at least SARVAM_API_KEY; see the comments in the file
python scripts/check_setup.py
cd frontend && npm install && cd ..
```

Step 3 (speech analysis of new clips) needs a GPU: run `notebooks/colab_setup.ipynb` on Colab
(Python 3.11, `requirements-gpu.txt`).

## Run the game

```bash
python scripts/build_library.py                   # step 4 output -> data/library/ (makes the voice lines once)
cd frontend && npm run build && cd ..             # the website -> frontend/dist
uvicorn backend.api.app:app --port 8000           # open http://localhost:8000
```

Restart the server after backend changes. For frontend work, `npm run dev` in `frontend/` serves on
http://localhost:5173 and forwards /api to :8000.

## Translate scenes

```bash
python scripts/dub_clip.py data/clips/ta_02.mp4 --speech auto --to hi   # step 4 from step 3's output
python scripts/retranslate.py                     # redo context + translation for every scene (no GPU)
python scripts/retranslate.py ta_02.ta-hi --to bn mr te   # dub an existing scene into more languages
python scripts/build_library.py                   # then rebuild the game library
```

`retranslate.py` prints every line before and after, with counts of lines that still have written-style,
copied or wrong family words. Sarvam answers are cached (`data/cache/`), so re-running unchanged steps is free.

## Tests and evaluations

```bash
python -m pytest                       # offline tests (no paid API calls)
python scripts/eval_register.py        # forms of address with vs without scene context (24 cases)
python scripts/judge_translations.py   # blind comparison of draft / previous / current translations (Gemini)
python scripts/eval_ravdess.py         # step 2 research score on RAVDESS
python scripts/train_emotion_head.py   # emotion head on emotion2vec embeddings (see docs/reports/step3.md)
```

Reports: [step 2](docs/reports/step2.md) (prosody and scoring), [step 3](docs/reports/step3.md) (speech
analysis, emotion), [step 4](docs/reports/step4.md) (context-aware translation).

## Keys and costs

| Service | Used for | Cost |
|---|---|---|
| Sarvam | drafts, translator model, voice lines, transliteration, players' speech-to-text | about ₹1 per scene to translate, ₹1–2 per scene for voices, ₹0.04 per recorded take |
| Groq | video captions (and gpt-oss if chosen as translator) | free tier |
| Gemini | backup LLM, pre-labels, translation judge | free tier |
| Hugging Face | step 3 models, Rasa dataset | free |

Translation and voices are made once per scene when the library is built; only speech-to-text runs per take.
Backup keys (`SARVAM_API_KEY_2`, `GROQ_API_KEY_2`, ...) take over when a key runs out.

## Layout

```
backend/pipeline/speech/     step 3: ASR, alignment, diarization, Demucs, emotion
backend/pipeline/visual/     video captions
backend/pipeline/context/    scene context and emotion-intent fusion
backend/pipeline/translate/  drafts (Sarvam, IndicTrans2), translation + review, the code checks
backend/game/                library build, voice lines, players' speech-to-text, scoring a take
backend/scoring/             prosody, arousal model, the game score (director.py), step 2 research score
backend/llm/                 LLM client: Groq, Cerebras, Sarvam, Gemini, cache, fallbacks
backend/sarvam_http.py       every Sarvam call, with backup keys
backend/api/app.py           FastAPI app (also serves the built website)
frontend/                    the game (React, Vite, Tailwind)
scripts/                     pipeline runs, evaluations, dataset fetchers
notebooks/                   Colab: setup, step 3+4 on clips, emotion head
config.yaml                  languages, models, score weights, thresholds
data/                        clips, labels, caches, library (only the CSV and labels are committed)
```

Movie clips are excerpts used for research and are not committed; the library is built locally.
