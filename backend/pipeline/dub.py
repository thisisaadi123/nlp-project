"""Step 4 for one clip: visual captions -> scene context -> emotion-intent fusion -> translation.

Input is step 3's speech JSON (a list of lines). Output adds, per line, the fused emotion-intent
vector and the dubbing translation, plus the clip's captions and scene context.
"""
import json
from pathlib import Path

from backend.pipeline.context.build import build_context
from backend.pipeline.context.fusion import fuse
from backend.pipeline.translate.draft import translator
from backend.pipeline.translate.refine import refine, syllables
from backend.pipeline.visual.caption import caption_clip


def translation_input(line: dict, fused: dict, ctx_line: dict) -> dict:
    item = {"line_id": line["line_id"], "speaker": line.get("speaker"), "addressee": ctx_line.get("addressee"),
            "text": line["text"], "start": line.get("start"), "end": line.get("end"),
            "emotion": {"label": fused["label"], **({"arousal": fused["arousal"]} if fused["arousal"] is not None else {})},
            "intent": fused["intent"], "tone": fused.get("tone"), "stressed_words": fused["stressed_words"],
            "visual_cue": fused["visual_cue"]}
    if fused["mismatch"]:
        item["subtext"] = "the voice and the words disagree: likely sarcasm or a held-back emotion"
    return item


def process_clip(clip: str | None, lines: list[dict], src: str, tgt: str) -> dict:
    captions = caption_clip(clip) if clip else None
    context = build_context(lines, captions, src)
    ctx_lines = {c.get("line_id"): c for c in context.get("lines", []) if isinstance(c, dict)}
    scene = {k: context[k] for k in ("setting", "summary", "mood", "speakers", "relationships") if context.get(k)}

    fused = [fuse(ln, ctx_lines.get(ln["line_id"])) for ln in lines]
    inputs = [translation_input(ln, f, ctx_lines.get(ln["line_id"], {})) for ln, f in zip(lines, fused)]
    drafts = translator().translate([ln["text"] for ln in lines], src, tgt)
    translated = refine(inputs, drafts, src, tgt, scene)

    out_lines = []
    for ln, f, inp, d, t in zip(lines, fused, inputs, drafts, translated):
        out_lines.append({**ln, "addressee": inp["addressee"], "emotion_intent": f, "draft": d, "dub": {
            **{k: v for k, v in t.items() if k != "line_id"},
            "syllables_source": syllables(ln["text"]), "syllables_target": syllables(t["translation"])}})
    return {"source_language": src, "target_language": tgt, "captions": captions, "context": context,
            "lines": out_lines}


def lines_from_label(label_path: Path, language: str) -> list[dict]:
    """Stand-in for step 3 output, built from a Gemini pre-label (data/labels/<clip>.json).

    The addressee is left out on purpose: the context step has to infer it.
    """
    label = json.loads(label_path.read_text(encoding="utf-8"))
    lines = []
    for i, ln in enumerate(label.get("lines", []), 1):
        lines.append({"line_id": i, "speaker": ln.get("speaker"), "start": ln.get("start"), "end": ln.get("end"),
                      "language": language, "text": ln.get("transcript", ""),
                      "emotion": {"label": ln.get("emotion")}, "stressed_words": ln.get("stressed_words", [])})
    return lines
