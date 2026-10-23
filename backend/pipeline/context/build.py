"""Scene context: who the speakers are, how they relate, and what each line is doing.

Combines the visual captions with the diarized transcript (step 3's output). The captions alone
cannot tell who said what to whom, and the transcript alone cannot see the room; together they
give the translator what it needs to pick the right register.
"""
import json
import re

from backend.llm.client import LLMError, chat_json
from backend.pipeline.context.fusion import intensity_level
from backend.settings import config

INTENTS = ["question", "accusation", "threat", "command", "plea", "apology", "comfort", "affection",
           "joke", "sarcasm", "complaint", "information", "greeting", "other"]
EMOTIONS = ["angry", "sad", "happy", "fearful", "surprised", "disgust", "neutral", "calm", "sarcastic", "comic"]

SYSTEM = """You analyse short film/TV scenes for a dubbing team. From the visual description and
the transcript, work out who the speakers are, how they are related, and what each line is
doing. Use what is in the material: forms of address (bhaiya, anna, didi, sir, machan...),
pronoun levels (tu/tum/aap, nee/neenga), tone, and what is visible. Say when you are guessing.
"voice_intensity" is measured from the audio (how loud and charged the delivery is); judge the
emotion itself from the words, the scene and the faces. Answer with JSON only."""

OUTPUT_SPEC = """Return JSON:
{"setting": "<where>",
 "summary": "<one sentence: what happens>",
 "mood": "<overall mood>",
 "speakers": {"S1": {"description": "<who they are>", "gender": "<male/female/unknown>",
                     "approx_age": "<e.g. 20s>", "visual_id": "<P-id from the captions, or null>"}},
 "relationships": [{"between": ["S1", "S2"], "relation": "<siblings, friends, boss/employee...>",
                    "senior": "<speaker id or null>", "closeness": "<close/neutral/distant/hostile>"}],
 "lines": [{"line_id": <id>, "addressee": "<speaker id, 'audience', or null>",
            "intent": "<exactly one of %s>",
            "text_emotion": "<exactly one of %s>",
            "tone": "<one or two words of nuance, e.g. dismissive, teasing, pleading>",
            "visual_cue": "<expression/action of the speaker at that moment, or null>"}],
 "confidence": "<high/medium/low, and what is uncertain>"}
Use the speaker ids exactly as they appear in the transcript.""" % ("/".join(INTENTS), "/".join(EMOTIONS))


def _transcript(lines: list[dict], alias: dict[str, str] | None = None) -> list[dict]:
    """The transcript as the context model sees it. From the audio it gets only how intense the
    voice is: emotion2vec's emotion label is wrong too often on Indic speech (step 3: 35-38%) and
    would mislead it. Stand-in transcripts without step 3's measurements pass their label."""
    alias = alias or {}
    out = []
    for ln in lines:
        item = {"line_id": ln["line_id"], "speaker": alias.get(ln.get("speaker"), ln.get("speaker")),
                "start": ln.get("start"), "end": ln.get("end"), "text": ln["text"]}
        emotion = ln.get("emotion") or {}
        arousal = emotion.get("arousal_e2v")
        intensity = intensity_level(arousal if arousal is not None else emotion.get("arousal"))
        if intensity:
            item["voice_intensity"] = intensity
        elif emotion.get("label"):
            item["voice_emotion"] = emotion["label"]
        out.append(item)
    return out


def _one_of(value, allowed: list[str], default: str) -> str:
    """The first allowed word in a free-form answer ("plea/command" -> "plea"), else `default`."""
    for word in re.split(r"[^a-z]+", str(value or "").lower()):
        if word in allowed:
            return word
    return default


def normalize(context: dict, alias: dict[str, str]) -> dict:
    """Map short speaker aliases back to the transcript's ids and force fixed vocabularies."""
    back = {short: real for real, short in alias.items()}

    def real(sid):
        return back.get(sid, sid) if isinstance(sid, str) else sid

    context["speakers"] = {real(k): v for k, v in (context.get("speakers") or {}).items()}
    for rel in context.get("relationships") or []:
        if isinstance(rel, dict):
            rel["between"] = [real(x) for x in rel.get("between") or []]
            rel["senior"] = real(rel.get("senior"))
    for ln in context.get("lines") or []:
        ln["addressee"] = real(ln.get("addressee"))
        ln["intent"] = _one_of(ln.get("intent"), INTENTS, "other")
        ln["text_emotion"] = _one_of(ln.get("text_emotion"), EMOTIONS, "neutral")
    return context


def build_context(lines: list[dict], captions: dict | None, language: str) -> dict:
    # Step 3 names speakers SPEAKER_00, SPEAKER_01...; models answer more reliably (and the prompt
    # is shorter) with S1, S2..., so the prompt uses aliases and the answer is mapped back.
    speakers = list(dict.fromkeys(ln.get("speaker") for ln in lines if ln.get("speaker")))
    alias = {sid: f"S{i}" for i, sid in enumerate(speakers, 1)}
    request = {
        "language": config()["languages"][language]["name"],
        "visual": {k: v for k, v in (captions or {}).items() if not k.startswith("_")} or "no video available",
        "transcript": _transcript(lines, alias),
    }
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=1) + "\n\n" + OUTPUT_SPEC}]
    try:
        result = chat_json(messages, max_tokens=4000, effort="medium", validate=valid_context)
    except LLMError as e:
        # Translation still works without scene context, just less precisely.
        return {"lines": [], "confidence": f"unavailable: {e}"}
    return normalize(dict(result), alias)


def valid_context(result: dict) -> bool:
    lines = result.get("lines")
    return (isinstance(lines, list) and all(isinstance(x, dict) and "line_id" in x for x in lines)
            and isinstance(result.get("speakers", {}), dict)
            and isinstance(result.get("relationships", []), list))
