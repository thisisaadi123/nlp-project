# Step 4 report: scene context + context-aware translation

Date: 2026-10-23. Runs on the Mac (IndicTrans2 on CPU; Groq and Gemini through their free APIs).

## What was built

| File | What it does |
|---|---|
| `backend/llm/client.py` | JSON calls with retries, an on-disk cache and a fallback chain: gpt-oss-120b → gpt-oss-20b → Gemini. Replies are validated before caching; only main-model answers are cached. Request size is fitted to Groq's per-minute token limit. |
| `backend/pipeline/visual/keyframes.py`, `caption.py` | Up to 8 keyframes (scene cuts first), tiled 2×2 so one Qwen vision request sees the whole clip (Qwen takes at most 3 images). |
| `backend/pipeline/context/build.py` | Scene context from captions + transcript: speakers, relationships (who is senior, how close), and per line the addressee, intent, text emotion and visual cue. |
| `backend/pipeline/context/fusion.py` | Emotion-Intent Vector per line: voice emotion + what the words say + what the face shows. Flags lines where voice and words disagree (sarcasm, held-back emotion). |
| `backend/pipeline/translate/draft.py` | IndicTrans2 (distilled 320M, direct Indic → Indic) draft, in batches of 4 with a length cap; drafts where the model got stuck repeating a word are dropped. |
| `backend/pipeline/translate/refine.py` | The dubbing translation: register by relationship, emotion, stressed-word mapping, romanization, a delivery note, then a length-fit pass. Lines go 3 per call; later batches see earlier translations so forms of address stay consistent. |
| `backend/pipeline/dub.py`, `scripts/dub_clip.py` | Step 4 end to end for one clip. Takes step 3's speech JSON; `--stub` uses the clip's Gemini pre-label until step 3 is ready. |
| `data/register_cases.jsonl`, `scripts/eval_register.py` | Register benchmark: 24 scenes (12 Hindi→Tamil, 12 Tamil→Hindi) whose correct form of address depends on the relationship. |
| `tests/` | 73 offline unit tests + 2 network tests (see Testing). |

Test clips added with `scripts/add_clip.py` and pre-labelled with `scripts/prelabel_clip.py`
(labels are drafts until a native speaker checks them):

| Clip | Source (official channel) | What it tests |
|---|---|---|
| hi_01 | Panchayat, Prime Video India, 0:28–0:56 | condescending *beta*, *tu*, a mocking *chacha*; 6 speakers, music |
| hi_02 | Ram-Leela, Eros, 0:30–1:10 | *Ram bhai* / *Kanji bhai* as friendly titles, Gujarati-flavoured Hindi, comedy |
| ta_01 | Veyil, Flywheel, 2:30–3:08 | *anne* (respectful elder brother) answered with intimate *-da*; emotional reunion |
| ta_02 | Samudhiram, KTV, 0:29–1:06 | *anni* (brother's wife) asked politely, answered rudely (*veliya po*) |

## Results

### Register benchmark (24 cases)

| Setup | All | hi→ta | ta→hi |
|---|---|---|---|
| A. IndicTrans2 draft only (the "Google Translate" baseline) | 25% | 25% | 25% |
| B. LLM refinement, no scene context | 71% | 83% | 58% |
| **C. LLM refinement with scene context (step 4)** | **92%** | 83% | **100%** |

- Scene context adds 21 points over the same LLM without context. In Tamil→Hindi it is what turned
  *anna* to an elder brother into *bhaiya*, *machi* between female friends into *yaar*, and *amma*
  to an elderly stranger into *aunty* rather than *maa*.
- Two real misses remain in C: a thug's hostile *ae bhai* came out as *paiya* (a Tamil film thug says
  *dei*), and an elder brother's *chhote* came out as *chinna* (*thambi* is the natural word).
- **How the checks changed:** the first run scored C at 75%. Reviewing every failure showed five
  were the checker's fault, not the translation's: respect carried by a Tamil verb ending with the
  pronoun dropped (*kaelunga*, *kavalaippadaadheergal*), *bhai ji* as a polite address, Hindi
  children's natural *tum* to a grandfather, and the particle டா (*da*) matching inside வேண்டாம்
  (*vendaam*). Those checks were widened, and short particles now match whole words only. Every
  change is marked in `data/register_cases.jsonl`. All outputs, with romanization, are in
  `docs/reports/step4-register-review.md` for a native speaker to confirm.
- **Length fit:** 58% of lines were within 80–130% of the source's syllable count; the fit pass
  brings it to **71%** with **no loss of register** (92% before and after). A first version reached
  96% but damaged register (dropping a son's respectful verb ending to his father, padding lines by
  repeating names), so rewrites are now rejected if the register label changes or any address term
  repeats, and only lines that are too long, or far too short, are rewritten.
- **Emphasis:** in 24/24 lines, the target word chosen to carry the original's stress is present in
  the translation.

### Real clip, end to end (hi_02 → Tamil, gpt-oss-120b)

| Source | IndicTrans2 draft | Step 4 dub |
|---|---|---|
| अरे साक्षात कामदेव चले आ रहे हैं। | ஏய், உண்மையில் காமதேவ் வெளியே வருகிறார். (stiff) | **அடேய், சாட்சாத் மன்மதனே நடந்து வர்றாருப்பா!** *adei, saatchaath manmadhane nadandhu varraarupa!* |
| आजा बेटा आजा आजा। | ஆஜா மகன் ஆஜா ஆஜா (transliterated nonsense) | **வாடா கண்ணா, வா வா!** *vaada kannaa, vaa vaa!* |
| सही-सही बताओ छोकरी कौन छे? | சோகமாக சொல்லுங்கள்… ("say it *sadly*") | **உண்மைய சொல்லு, யாருடா அந்தப் பொண்ணு?** |
| नेम तो कमाल छे, पर सरनेम बवाल छे। | நாம் தோ கமல் சே… (gibberish) | **நேம் என்னவோ அமர்க்களம், ஆனா சர்நேம் பயங்கர கலவரம்!** (keeps the joke's rhyme) |

- **Captioning** (Qwen via Groq, 3.3 s per clip): on ta_02 it correctly saw a woman in a sari with
  arms crossed and a stern, then annoyed face (the *anni* refusing the money) and a smiling young
  man asking, which is exactly the evidence the context step needs.
- **Without video** the pipeline degrades instead of failing: no captions, the context still found
  "younger → elder sister", and the dub was natural spoken Tamil (*akka, please ammakitta sollidaadha*).

### Backup model quality (gpt-oss-20b)

When gpt-oss-120b and Gemini are both out of daily quota, gpt-oss-20b keeps the pipeline running,
but its dubs are not good enough to use: on ta_02 it turned *anni* (brother's wife) into *behan*
(sister) instead of *bhabhi*, copied the draft's wrong meaning for the angry line, produced
gibberish for *attai ponnu* (cousin), and used *yaar* for an elder brother. Each line records the
model that produced it (`_provider`), so the app can flag or redo backup-model dubs.

## Testing

`python -m pytest` runs 73 offline tests in ~2 s; `python -m pytest -m network` runs 2 that call
the real APIs. All pass. They cover JSON parsing, caching, validation, every fallback path, the
token budget, Groq/Gemini error handling, syllable counting, fusion, keyframe selection and tiling
on a real video, the register checker, the length-fit guards, translation batching, CSV helpers,
and step 2's scoring on synthetic voices. Beyond unit tests: the 24-case benchmark, five
end-to-end clip runs, a no-video run, and runs with both main providers out of quota.

### Problems found by running on real clips, all fixed and pinned by tests

| Problem | Fix |
|---|---|
| transformers 5 breaks IndicTrans2; newer 4.x breaks its generation cache | pin `transformers>=4.46,<5`, generate with `use_cache=False` |
| IndicTrans2 can get stuck repeating a word; one stuck line in a 14-line batch ran ~20 min and the process was killed | batches of 4, output length capped relative to input, stuck drafts dropped |
| `\w` does not match Indic vowel signs, so the repetition check missed the Bengali case it was written for | split on whitespace instead (caught by a unit test) |
| Qwen accepts at most 3 images per request | tile frames 2×2 |
| Qwen's free tier: ~800 output tokens and 7k input tokens per minute; 12 frames were cut off | 8 frames, compact prompt; a cut-off reply goes straight to the fallback |
| Groq counts `max_tokens` against the 8k tokens-per-minute limit (413 errors) | lines in batches of 3; `max_tokens` sized to fit beside the input |
| Groq JSON mode sometimes rejects its own output (`json_validate_failed`) | retried; the 20b model runs without JSON mode |
| Gemini returns 503 when overloaded | retried with backoff |
| A malformed context reply was cached and replayed forever | replies validated before caching; bad entries are discarded |
| The first length-fit pass damaged register | register and address-term guards (see above) |
| gpt-oss at medium effort made grammar errors (*nee … theriyaadhu*) | translation uses high effort; the prompt states that politeness applies to every pronoun case |
| `indic-transliteration` mis-romanizes Tamil (பண்ற → "bhaṇṟa") | romanization comes from the LLM |

## Integration with step 3

Step 3 (`docs/reports/step3.md`) changed what step 4 can trust from the audio, and its output
format exposed bugs that the stand-in input could not:

| Change | Why |
|---|---|
| The emotion **label** comes from the scene context; the audio's label is used only without context | emotion2vec's 7-way accuracy on Indic speech is 35-38% (most angry lines come out "happy") |
| **Intensity** comes from `arousal_e2v` (emotion2vec's high-arousal probability), falling back to the prosody arousal model | high vs low arousal: AUC 0.85 hi / 0.81 ta for emotion2vec; the RAVDESS prosody model is mis-calibrated on Indic speech |
| The context model sees the voice's **intensity**, not emotion2vec's label | a wrong "happy" would mislead it |
| "Voice and words disagree" is judged from intensity vs the words' emotion | intensity is the reliable audio signal |
| Words with `stress: null` (unvoiced) are skipped when picking the stressed word | step 3 writes null there; fusion used to crash on it |
| emotion2vec's class names (`fear`, `surprise`) are mapped to ours | |
| The context prompt uses short speaker aliases (S1, S2...) and maps the answer back to step 3's `SPEAKER_00` ids | on a real clip the model answered with S1/S2 for SPEAKER_01/02 speakers, so every addressee was unknown |
| `intent` and `text_emotion` are forced into fixed lists; free-form nuance goes in a new `tone` field (passed to the translator) | the model returned "plea/command", "friendly", "dismissive", which the sarcasm and mismatch checks could not read |
| `dub_clip.py --speech auto` reads `data/cache/speech/<clip>.json`; the source language comes from step 3's `meta` | |

Tested with step-3-format JSON for the four clips (text and timing from the pre-labels, prosody
and arousal measured from the real audio, 25 words with `stress: null`): three clips ran end to
end, the fourth stopped cleanly when all three LLM providers were out of daily quota. Step 3's own
models could not run on the Mac (the disk had 3.9 GB free; they need ~5 GB), so the real check is
step 3's output from the Omen (see open items). Unit tests pin every change above.

## Free-tier limits (design constraint)

- **gpt-oss-120b: 200,000 tokens per day** (rolling). A clip costs roughly 20–40k tokens at high
  effort, so the free tier handles **about 5–8 new clips per day**. Every clip must be processed
  once and cached; the live game loop must not call it.
- Gemini Flash's free daily quota was also used up today, mostly by the fallback during testing.
- The register benchmark costs ~100k tokens per full run (24 cases × several calls); re-runs are
  free thanks to the cache.

## Open items

1. **Re-run the four clips on gpt-oss-120b** once its daily quota recovers. Three of the five end-to-end runs only completed on
   the backup model:
   `for s in "ta_02 hi" "ta_01 hi" "hi_01 ta" "hi_01 bn"; do set -- $(echo $s); python scripts/dub_clip.py data/clips/$1.mp4 --stub --to $2; done`
2. **Native-speaker review** of `docs/reports/step4-register-review.md` and of the four clips'
   labels (`data/labels/`), then set `verified=yes` in `data/testset.csv`.
3. **Run step 4 on step 3's real output.** On the Omen, for each clip:
   `python -m backend.pipeline.speech.run data/clips/ta_02.mp4 --lang ta`, then copy
   `data/cache/speech/*.json` to the Mac and run `python scripts/dub_clip.py data/clips/ta_02.mp4 --speech auto --to hi`.
4. The two remaining register misses (*dei*, *thambi*): add the hostile-stranger and little-brother
   cases to the prompt's examples if they recur on real clips.
5. **Step 3 note:** the Mac stack pins `transformers<5`. If the Omen's speech stack needs
   transformers 5, IndicTrans2 must run in its own environment there.
