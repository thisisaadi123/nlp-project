# Step 2 report: prosody + scoring core

Date: 2026-10-23. Runs on the Mac (CPU only, ~0.1 s per clip).

## What was built

| File | What it does |
|---|---|
| `backend/scoring/prosody.py` | Pitch (semitones vs the speaker's median), loudness, stress curve, pauses, syllable rate, spectral tilt, HNR, pitch level. Trims breaths: the line runs from the first to the last voiced sound. |
| `backend/scoring/arousal.py` + `arousal_model.json` | Logistic model: how intense a delivery is (0..1). A "relative" model uses the speaker's neutral baseline; a "raw" model is used when no baseline exists. |
| `backend/scoring/compare.py` | Score out of 100 from four components that survive translation: emotion (arousal match), contour (pitch/loudness shape per quarter + rising/falling ending), timing (length, pauses, rate), emphasis (stress strength + placement; accepts the target word's time span once ASR exists). |
| `backend/scoring/feedback.py` | Plain-language tips from the measurements: the fallback when the LLM is rate-limited, and the facts the LLM writes insights from. |
| `scripts/compare_takes.py` | Score any recordings against a reference, with optional baselines, or as a matrix. |
| `scripts/train_arousal.py` | Trains the arousal models on RAVDESS with leave-actors-out validation. |
| `scripts/eval_ravdess.py` | The evaluation below. |
| `scripts/tts_takes.py` | Hindi test takes from Gemini TTS, each verified by Whisper. |
| `scripts/fetch_ravdess.sh`, `scripts/ravdess_data.py` | Dataset download and loading. |

## Data

RAVDESS (CC BY-NC-SA 4.0): 24 actors, 2 sentences, 8 emotions, 2 takes. English, but every feature
here is language-independent (pitch, loudness, timing). Not committed.

Every pair is scored by an arousal model that never saw either actor (4 folds, 6 actors held out each).
Baselines are always the same actor's neutral takes of the *other* sentence, like a player's calibration line.

## Results

AUC = how often a matching pair outscores a non-matching pair (1.0 perfect, 0.5 chance).

| Test | First version | Final, no baseline | Final, with baseline |
|---|---|---|---|
| Same speaker, same vs different emotion | 0.768 | 0.829 | **0.845** |
| Different speakers, same vs different emotion | 0.562 | 0.617 | **0.643** |
| Different speakers, same vs opposite arousal | – | 0.678 | **0.710** |

Arousal classifier on unseen actors: **82.8%** with baseline, 80.5% without.

Component AUCs, different speakers, same vs different emotion, with baseline: emotion 0.638,
contour 0.568, timing 0.548, emphasis 0.545.

Demo (reference = angry take 1, with baseline):

| Attempt | Actor 1 (male) | Actor 2 (female) |
|---|---|---|
| angry, take 2 | **72.1 Dub** | **84.7 Dub** |
| sad | 52.3 Goose | 53.3 Goose |
| neutral | 43.0 Goose | 38.4 Goose |
| happy | 47.9 Goose | 53.2 Goose |
| calm | 49.1 Goose | 35.3 Goose |

### Hindi check (Gemini TTS, "भैया, ये क्या कर दिया?")

| | angry 2 | sad | flat |
|---|---|---|---|
| male vs male angry 1 (baseline) | **67.5** | 41.3 | 41.1 |
| female vs female angry 1 (no baseline) | 45.6 | 32.4 | 35.9 |
| female attempts vs male angry 1 | 55.0 / 57.8 | 36.8 | 45.1 |

The ranking is right in every row. The female "angry 2" take came out much calmer than "angry 1"
(intensity 0.22 vs 0.80), most likely a weak TTS take. Synthetic acting is a smoke test only.

## Findings

1. **Raw prosody distances can't match emotions across speakers** (AUC 0.56). Each voice has its own
   pitch, loudness and timbre, so the same emotion looks different in absolute numbers.
2. **Prosody reliably measures intensity (arousal), not which emotion.** Angry, happy, fearful and
   surprised all read as intense; it cannot tell angry from happy. Telling those apart is
   emotion2vec's job (step 3).
3. **A neutral baseline per speaker helps everywhere** (+0.02 to +0.03 AUC, +2.3 points of classifier
   accuracy). The game should ask each player to read one calibration sentence at the start.
4. **Contour, timing and emphasis add little to emotion matching** but give specific, correct
   feedback (ending direction, line length, stress placement).
5. **TTS models can read the style instruction aloud.** `gemini-3.8-flash-tts` said "Say angrily…"
   as part of the take, which ruined the first batch without any error. Every synthetic take is now
   checked with Whisper. This also applies to the reference dub in step 5.
6. **Free Gemini TTS quota is ~10 requests per model per day.** Not usable for live reference dubs;
   Indic Parler-TTS on the Omen stays the plan.

## Open items (for step 6 calibration)

- **Dub threshold:** at 70, only 51% of same-speaker true matches pass. Fit the threshold and the
  component weights on human-rated dubs.
  *Update (October 2026):* across speakers it was worse: at raw 70 only 12% of correct attempts
  (different RAVDESS actors, same sentence and emotion) earned a Dub. Players now see the score
  through a display curve that moved the Dub line to raw 55 (49% of correct attempts, 29% of
  wrong-emotion ones). That curve was later replaced by the game's own score
  (`backend/scoring/director.py`: timing, volume, punch, tone and words, each checked against the
  line's note); this research score stays for the RAVDESS evaluations.
- **Emotion slot:** combine emotion2vec (which emotion) with arousal (how intense) once step 3
  measures emotion2vec on Indic speech.
- **Arousal model on Indic speech:** trained on English actors; it rated Hindi TTS anger as milder
  (0.55) than RAVDESS anger (0.9+). Retrain or recalibrate with Indic data.
- **Emphasis placement** currently uses thirds of the line; switch to the target word's time span
  from ASR word timings (step 3) and the translation's emphasis map (step 4).
- **Female calibration take** for the TTS test was not generated (daily quota).
  Run `python scripts/tts_takes.py` again after the quota resets to fill it in.
