"""Describe a clip's keyframes with the vision model: setting, people, expressions, actions."""
from backend.llm.client import LLMError, vision_json
from backend.pipeline.visual.keyframes import GRID, keyframe_sheets

PROMPT = """These {n_images} images come from one short film/TV clip. Each image is a {g}x{g} grid
of video frames in reading order (top-left, top-right, bottom-left, bottom-right).
Frame times in seconds, in order: {times}.

Describe what is visible; do not guess dialogue. Keep it SHORT: the whole answer must stay under
500 words. Return JSON:
{{"setting": "<where, max 12 words>",
  "people": [{{"id": "P1", "looks": "<gender, approx age, role cues; max 10 words>"}}],
  "frames": [{{"t": <time>, "who": ["P1", ...], "faces": "<expressions, max 10 words>",
              "action": "<what happens, max 10 words>"}}]}}
Use the same person ids across frames when it is clearly the same person."""


def valid_captions(result: dict) -> bool:
    return isinstance(result.get("frames"), list) and isinstance(result.get("people", []), list)


def caption_clip(path: str) -> dict | None:
    """Captions for the clip, or None if both vision models fail (the rest of step 4 then
    works from the transcript alone)."""
    times, sheets = keyframe_sheets(path)
    prompt = PROMPT.format(n_images=len(sheets), g=GRID, times=", ".join(f"{t:.1f}" for t in times))
    try:
        result = vision_json(prompt, sheets, validate=valid_captions)
    except LLMError:
        return None
    result["frame_times"] = times
    return result
