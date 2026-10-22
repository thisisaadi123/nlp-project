# Test set guide

The test set is the answer key every pipeline stage is checked against: ASR, emotion
detection, scene context, translation and scoring. Each language needs a person who speaks
it to pick the clips and check the labels.

## What we need

**3 clips per language** (Hindi, Bengali, Marathi, Telugu, Tamil), each **3–45 s**, ideally 5–30 s.

Across all 15 clips, at least:

| Requirement | Count |
|---|---|
| angry / shouting | 3 |
| sad / pleading | 3 |
| comic or sarcastic | 2 |
| calm / neutral | 2 |
| background music under the dialogue | 3 |
| two or more speakers talking | 3 |
| register contrast: bhai vs bhaiya, tu/tum/aap, dada, anna vs machan... | 2 |
| self-recorded (you act it out) | 1 per language |

One clip can cover several rows (an angry two-person argument with music counts three times).
`python scripts/testset_status.py` shows what is covered and what is missing.

**Good clips:** clear dialogue, one main emotion, a relationship you can tell from the scene.
**Avoid:** heavy crosstalk, very loud music over the words, songs, clips over 45 s.

**Self-recorded clips are the most valuable**: you know the exact words and the intended
emotion, so they are true ground truth. Act out a short scene with a friend, ideally one with
a register contrast (younger sibling scolding an elder one, two friends joking).

## Setup (once)

```bash
git pull
source .venv/bin/activate
uv pip install -r requirements-mac.txt     # adds yt-dlp
```

`GEMINI_API_KEY` must be set in `.env` (see the README).

## Adding a clip

```bash
# a scene from YouTube: only that part is downloaded
python scripts/add_clip.py "https://www.youtube.com/watch?v=..." --lang hi --start 1:02:15 --end 1:02:31

# a file on your computer
python scripts/add_clip.py ~/Downloads/movie.mp4 --lang ta --start 12:05 --end 12:20

# your own phone recording
python scripts/add_clip.py ~/Desktop/scene.m4a --lang bn --self
```

The clip is saved as `data/clips/<lang>_<nn>.mp4` and a row is added to `data/testset.csv`.

## Draft labels with Gemini

```bash
python scripts/prelabel_clip.py hi_01     # or --all for every clip without labels
```

Gemini watches the clip and writes `data/labels/hi_01.json`: speakers, relationship, each line
with transcript, emotion, intensity and stressed words, register notes, and translation hints.
It also fills the empty fields in `testset.csv`.

## Check the labels (the important part)

The drafts are good at transcripts and register notes, but they **do get emotions and speaker
counts wrong**. In our first test, Gemini heard two speakers where there was one, and called an
angry line "surprised". For each clip:

1. Watch the clip and open `data/labels/<clip_id>.json`.
2. Fix the **transcript** word by word (keep slang and code-mixing as spoken).
3. Fix each line's **speaker**, **addressee**, **emotion** and **stressed words**.
4. Fix **register_note** and **expected_translation_notes**. Would a native speaker of the other
   languages really say it that way?
5. Copy the corrections into the matching row of `data/testset.csv`, then set `verified` to `yes`.

Edit `testset.csv` in VS Code or another plain-text editor. Excel can break the Indic scripts
when it saves; if you use Google Sheets, export as CSV (UTF-8).

## Sharing clips

Clips are **not** in git (copyright and file size). Upload each `data/clips/<clip_id>.mp4` to the
team's shared Google Drive folder. The Omen and everyone else download them into their own
`data/clips/`. `testset.csv` and `data/labels/` **are** committed, so everyone gets the labels
with `git pull`.

To avoid merge conflicts in `testset.csv`: `git pull` before adding clips, and commit and push
soon after. Each person works on their own language.
