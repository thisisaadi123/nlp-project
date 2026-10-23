"""Emotion-Intent Vector per line: how intense the voice is + what the words mean + what the face shows.

Step 3 measured emotion2vec on Indic speech (Rasa): it reliably tells how intense a line is
(high vs low arousal, AUC 0.85 Hindi / 0.81 Tamil) but not which emotion it is (7-way accuracy
35-38%; most angry lines come out "happy"). So:

- the emotion label comes from the scene context (the words, the scene, the faces), and the
  audio's label is used only when there is no context;
- intensity comes from emotion2vec's high-arousal probability (`arousal_e2v`), falling back to
  the prosody arousal model, which ranks Indic speech correctly but scores it too low;
- a line is flagged when the voice's intensity and the words disagree (an intense voice on
  calm words, a flat voice on angry words) or the context marks sarcasm: that is where subtext
  lives and where a literal translation goes wrong.
"""

HIGH = 0.66
LOW = 0.33

# emotion2vec's class names -> ours
ALIASES = {"fear": "fearful", "surprise": "surprised", "disgusted": "disgust", "anger": "angry",
           "happiness": "happy", "sadness": "sad"}
# text_emotion values (fixed list in context/build.py) that imply a calm or an intense delivery
CALM_WORDS = {"calm", "neutral", "sad"}
INTENSE_WORDS = {"angry", "fearful", "surprised"}


def _norm(label: str | None) -> str | None:
    if not label:
        return None
    label = label.strip().lower()
    return ALIASES.get(label, label)


def intensity_level(arousal: float | None) -> str | None:
    if arousal is None:
        return None
    return "high" if arousal > HIGH else "low" if arousal < LOW else "medium"


def fuse(line: dict, ctx_line: dict | None) -> dict:
    ctx_line = ctx_line or {}
    voice = line.get("emotion") or {}
    intent = ctx_line.get("intent")
    text_emotion = _norm(ctx_line.get("text_emotion"))
    voice_label = _norm(voice.get("label"))

    arousal = voice.get("arousal_e2v")
    arousal_source = "emotion2vec"
    if arousal is None:
        arousal, arousal_source = voice.get("arousal"), "prosody"
    if arousal is None:
        arousal_source = None
    intensity = intensity_level(arousal)

    label, label_source = text_emotion, "context"
    if not label:
        label, label_source = voice_label or "neutral", "voice" if voice_label else "default"
    if intent == "sarcasm":
        label = "sarcastic"

    mismatch = intent == "sarcasm" or (
        (intensity == "high" and text_emotion in CALM_WORDS)
        or (intensity == "low" and text_emotion in INTENSE_WORDS))

    stressed = line.get("stressed_words")
    if not stressed and line.get("words"):
        scored = [w for w in line["words"] if w.get("stress") is not None]  # None = unvoiced word
        if scored:
            top = max(scored, key=lambda w: w["stress"])
            stressed = [top["w"]] if top["stress"] > 1.0 else []

    return {
        "label": label,
        "label_source": label_source,
        "intensity": intensity,
        "arousal": arousal,
        "arousal_source": arousal_source,
        "voice_emotion": voice_label,
        "text_emotion": text_emotion,
        "intent": intent,
        "tone": ctx_line.get("tone"),
        "visual_cue": ctx_line.get("visual_cue"),
        "mismatch": bool(mismatch),
        "stressed_words": stressed or [],
    }
