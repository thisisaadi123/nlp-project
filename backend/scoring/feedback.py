"""Plain-language tips from a score's measurements.

These are the template fallback for when the LLM is unavailable or rate-limited, and the
facts the LLM is given to write its own insights from.
"""
from backend.scoring.compare import Result


def _third(pos: float) -> str:
    return "the start" if pos < 1 / 3 else "the middle" if pos < 2 / 3 else "the end"


def _ending(slope: float) -> str:
    return "rises" if slope > 4 else "falls" if slope < -4 else "stays level"


def tips(r: Result, limit: int = 3) -> list[str]:
    d = r.details
    found = []  # (how far off, tip)

    gap = d["att_arousal"] - d["ref_arousal"]
    if abs(gap) > 0.25:
        if gap < 0:
            found.append((abs(gap), "The original is far more intense. Raise your voice and push more energy into it."))
        else:
            found.append((abs(gap), "You went bigger than the original. Pull it back to a calmer, softer delivery."))

    ratio = d["duration_ratio"]
    if ratio > 1.2:
        found.append((ratio - 1, f"Your line ran {ratio:.1f}x as long as the original. Speed up so it fits the scene."))
    elif ratio < 0.8:
        found.append((1 / ratio - 1, f"Your line was {1 / ratio:.1f}x faster than the original. Slow down and let it breathe."))

    ref_end, att_end = _ending(d["ref_ending_slope"]), _ending(d["att_ending_slope"])
    if ref_end != att_end:
        found.append((0.4, f"The original's pitch {ref_end} at the end; yours {att_end}."))

    ref_third, att_third = _third(d["ref_stress_pos"]), _third(d["att_stress_pos"])
    if ref_third != att_third:
        found.append((0.35, f"The original hits its strongest word near {ref_third}; you stressed {att_third}."))
    elif d["att_stress_strength"] < d["ref_stress_strength"] - 1.0:
        found.append((0.3, f"Punch the stressed word near {ref_third} harder."))

    if "pitch_raise_ref" in d and d["pitch_raise_ref"] - d["pitch_raise_att"] > 2:
        found.append((0.3, "The original speaker pitches well above their normal voice. Go higher."))

    found.sort(key=lambda t: -t[0])
    out = [tip for _, tip in found[:limit]]
    return out or ["Close match on intensity, timing and stress. Nicely done."]
