"""One plain-language coaching sentence per line, from its card: no LLM call, always the same.

    "Angry, loud. Hit அண்ணா hard. About 2 seconds."
"""

EMOTION = {
    "angry": "Angry", "sad": "Sad", "happy": "Happy", "fearful": "Scared", "surprised": "Surprised",
    "disgust": "Disgusted", "neutral": "Matter-of-fact", "calm": "Calm", "sarcastic": "Sarcastic",
    "comic": "Playful",
}
INTENSITY = {"low": "soft", "medium": "firm", "high": "loud"}


def coach(card: dict) -> str:
    emotion = card.get("emotion")
    first = EMOTION.get(emotion or "", (emotion or "natural").capitalize())
    if card.get("intensity") in INTENSITY:
        first += f", {INTENSITY[card['intensity']]}"
    parts = [first + "."]
    if card.get("stress"):
        parts.append(f"Hit {card['stress'][0]} hard.")
    d = float(card.get("duration") or 0)
    parts.append("About 1 second." if d < 1.5 else f"About {round(d)} seconds.")
    return " ".join(parts)
