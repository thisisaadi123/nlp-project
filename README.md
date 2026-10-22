# Dub-Dub-Goose

Direct Indic-to-Indic dubbing that keeps context and emotion, turned into a game: dub a line
yourself, get scored out of 100 on tone, emotion, emphasis and timing, and earn a **Dub** or a **Goose**.

Languages: Hindi, Bengali, Marathi, Telugu, Tamil (any pair).

## Machines

| Machine | Role |
|---|---|
| Mac | development, prosody + scoring code, API-only stages |
| Omen (RTX 4060, WSL2) | backend + GPU models — see [docs/omen-setup.md](docs/omen-setup.md) |
| Colab | emotion fine-tuning, score calibration |
| Groq (free API) | captioning, scene context, translation refinement, insights |

## Mac setup

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-mac.txt
cp .env.example .env        # fill in GROQ_API_KEY, HF_TOKEN, GEMINI_API_KEY
python scripts/check_setup.py
```

## Layout

```
backend/pipeline/   speech, visual, context, translate, tts stages
backend/scoring/    prosody comparison and the Dub/Goose score
backend/llm/        Groq client, prompts, cache, fallbacks
backend/api/        FastAPI app
frontend/           web app
notebooks/          Colab notebooks
data/testset.csv    labels for the test clips in data/clips/ (clips are not committed)
config.yaml         languages, model names, score weights, thresholds
```
