"""Context- and emotion-aware dubbing translation.

IndicTrans2's draft keeps the content; this step rewrites each line the way the target
language's film dialogue would say it, given who is speaking to whom, the scene, and how the
line is delivered. All lines of a clip go in one call so address terms stay consistent across
the conversation (and to save free-tier quota).
"""
import json

from backend.llm.client import LLMError, chat_json
from backend.settings import config

SYSTEM = """You are an expert film dubbing writer for Indian languages. You translate dialogue
DIRECTLY between Indian languages, as a native screenwriter of the target language would write
it for a dubbed film: natural spoken dialogue, not written or textbook language.

Rules:
1. Choose words by context. Forms of address and pronouns carry the relationship: who is older,
   who is closer, who is angry at whom. Examples: Hindi "bhaiya" to an elder brother is Tamil
   "anna", but Hindi "bhai" between college friends is Tamil "machan" or "da", not "anna" and never
   the formal "sagodharare". Match pronoun levels (Hindi tu/tum/aap, Tamil nee/neenga,
   Bengali tui/tumi/apni, Telugu nuvvu/meeru, Marathi tu/tumhi) to the relationship and emotion.
   The level applies to every case form while the grammar stays correct: informal Tamil is
   nee / unakku / unnai, formal is neenga / ungalukku / ungalai; Hindi tujhe vs aapko; and so on.
   Changing the level never changes the case the verb needs (e.g. Tamil "theriyaadhu" takes the
   dative: "unakku onnum theriyaadhu").
2. Use the spoken register of the target language as heard in its mainstream films, including
   natural colloquial verb forms. Avoid formal written forms unless the speaker is formal.
3. Keep the meaning of the line. The draft is a literal machine translation: use it as a guide
   to content only; fix its register, idioms and word choice.
4. Match the emotion and its intensity. An angry line stays sharp and short; a pleading line soft.
5. Keep the length close to the original (aim for within 20% of its syllable count) so it fits
   the actor's mouth and timing.
6. The words the original speaker stresses must map to the target word carrying the same
   meaning, so the dubber can stress it.
7. Write the translation in the target language's own script. Also give a romanization a
   non-reader can pronounce from, in casual English spelling (e.g. "anna, enna panna?"), not
   scholarly diacritics.
Answer with JSON only."""

OUTPUT_SPEC = """Return JSON:
{"lines": [{"line_id": <same id>,
            "translation": "<target script>",
            "romanized": "<casual romanization>",
            "register": "<intimate | casual | respectful | formal>",
            "address_terms": [{"source": "<source word>", "target": "<target word>", "why": "<short reason>"}],
            "emphasis": [{"source_word": "<stressed source word>", "target_word": "<word in translation>"}],
            "delivery": "<one short instruction for the dubber, in English>",
            "changes_from_draft": "<what you changed from the draft and why, in English, one sentence>"}]}
Return one entry per line in "lines", in the same order. "already_translated" (if present) is
earlier dialogue from the same scene, for consistency only: keep its forms of address, and do
not return it."""


def _line_payload(line: dict, draft: str) -> dict:
    """The facts about one line the translator needs; step 3's JSON fields where they exist."""
    out = {"line_id": line["line_id"], "speaker": line.get("speaker"), "addressee": line.get("addressee"),
           "text": line["text"], "draft": draft or "(no usable draft: translate from the source text)"}
    emotion = line.get("emotion") or {}
    if emotion:
        out["emotion"] = emotion.get("label")
        if "arousal" in emotion:
            out["intensity"] = "high" if emotion["arousal"] > 0.66 else "low" if emotion["arousal"] < 0.33 else "medium"
    for key in ("intent", "tone", "stressed_words", "duration", "visual_cue", "subtext"):
        if line.get(key):
            out[key] = line[key]
    if "start" in line and "end" in line and "duration" not in out:
        out["duration"] = round(line["end"] - line["start"], 2)
    return out


def _valid_lines(n: int):
    def check(result: dict) -> bool:
        lines = result.get("lines")
        return (isinstance(lines, list) and len(lines) == n
                and all(isinstance(x, dict) and str(x.get("translation", "")).strip() for x in lines))
    return check


def refine(lines: list[dict], drafts: list[str], src: str, tgt: str, scene: dict | None = None,
           effort: str | None = None) -> list[dict]:
    """Translate a clip's lines with context. `scene` is the scene context (see context/build.py);
    pass None to translate the lines without any scene information.

    Lines go in batches (Groq's free tier caps input + output per request); each batch sees the
    lines already translated, so forms of address stay consistent across the clip."""
    llm = config()["llm"]
    names = {k: v["name"] for k, v in config()["languages"].items()}
    size = llm["translate_batch"]
    out: list[dict] = []
    for i in range(0, len(lines), size):
        request = {
            "source_language": names[src],
            "target_language": names[tgt],
            "scene": scene or "no scene information available",
            "lines": [_line_payload(line, d) for line, d in zip(lines[i:i + size], drafts[i:i + size])],
        }
        if out:
            request["already_translated"] = [
                {"line_id": ln["line_id"], "speaker": ln.get("speaker"), "text": ln["text"],
                 "translation": t["translation"], "register": t.get("register")}
                for ln, t in list(zip(lines, out))[-6:]]
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=1) + "\n\n" + OUTPUT_SPEC}]
        result = chat_json(messages, max_tokens=6500, effort=effort or llm["translate_effort"],
                           validate=_valid_lines(len(request["lines"])))
        for item in result["lines"]:
            item["_provider"] = result.get("_provider")
        out += result["lines"]
    return fit_lengths(lines, out, tgt)


# A dubbed line has to fit the actor's mouth: aim for this range of the source's syllable count.
FIT_LOW, FIT_HIGH = 0.8, 1.3
# Only lines outside this range are rewritten. A short line can be stretched by the actor; a long
# one has to be rushed, so the upper bound is the one that matters.
REWRITE_BELOW = 0.6

FIT_SPEC = """These dubbed lines do not fit the original actor's timing. Rewrite each so its
syllable count falls inside "target_syllables". Rules:
- Keep the meaning and the exact words listed in "keep" (forms of address, stressed words).
- Keep the SAME politeness level: same pronoun level and the same respectful or casual verb
  endings as "current" (e.g. Tamil -nga/-ungal stays respectful; Hindi aap-forms stay aap-forms).
  "register" must come back unchanged.
- Never pad by repeating words or names. If a line cannot be fitted naturally, return "current"
  unchanged.
Same target language and script as "current". Return JSON:
{"lines": [{"line_id": <id>, "translation": "<line>", "romanized": "<casual romanization>",
            "register": "<intimate | casual | respectful | formal>"}]}"""


def _kept_words(item: dict) -> list[str]:
    words = [a.get("target", "") for a in item.get("address_terms", [])]
    words += [e.get("target_word", "") for e in item.get("emphasis", [])]
    return [w for w in words if w]


def _ratio(item: dict, source: str) -> float:
    return syllables(item["translation"]) / max(syllables(source), 1)


def _acceptable(old: dict, new: dict, source: str) -> bool:
    """A length rewrite must keep every protected word without repeating it, keep the register,
    and actually move the line closer to the target range."""
    def distance(item: dict) -> float:
        r = _ratio(item, source)
        return max(FIT_LOW - r, r - FIT_HIGH, 0.0)

    for w in _kept_words(old):
        if new["translation"].count(w) == 0 or new["translation"].count(w) > old["translation"].count(w):
            return False
    if new.get("register") and old.get("register") and new["register"] != old["register"]:
        return False
    return distance(new) < distance(old)


def fit_lengths(lines: list[dict], out: list[dict], tgt: str) -> list[dict]:
    """One rewrite pass for lines that are too long (or far too short) for the actor's timing."""
    bad = [(ln, item) for ln, item in zip(lines, out)
           if not REWRITE_BELOW <= _ratio(item, ln["text"]) <= FIT_HIGH]
    if bad:
        request = {"target_language": config()["languages"][tgt]["name"], "lines": [
            {"line_id": item["line_id"], "source": ln["text"], "current": item["translation"],
             "register": item.get("register"), "current_syllables": syllables(item["translation"]),
             "target_syllables": [round(syllables(ln["text"]) * FIT_LOW), round(syllables(ln["text"]) * FIT_HIGH)],
             "keep": _kept_words(item)} for ln, item in bad]}
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=1) + "\n\n" + FIT_SPEC}]
        try:
            reply = chat_json(messages, max_tokens=3000, effort="high",
                              validate=lambda r: isinstance(r.get("lines"), list))
        except LLMError:
            reply = {"lines": []}  # fitting is an improvement, not a requirement
        rewrites = {r.get("line_id"): r for r in reply["lines"] if isinstance(r, dict)}
        for ln, item in bad:
            new = rewrites.get(item["line_id"])
            if new and new.get("translation") and _acceptable(item, new, ln["text"]):
                item["fitted_from"] = item["translation"]
                item["translation"] = new["translation"]
                item["romanized"] = new.get("romanized", item.get("romanized"))
    for ln, item in zip(lines, out):
        item["length_ratio"] = round(_ratio(item, ln["text"]), 2)
    return out


# Syllable estimate for Indic scripts: one per vowel letter or consonant that is not followed by a
# virama (a consonant cluster counts once). Good enough to check that a dub fits the original's length.
_VIRAMAS = {"्", "্", "்", "్"}  # Devanagari, Bengali, Tamil, Telugu


def syllables(text: str) -> int:
    count = 0
    for ch, nxt in zip(text, text[1:] + " "):
        # Each Indic script is a 128-codepoint block; offsets 0x05-0x39 are its vowels and consonants.
        if "ऀ" <= ch <= "౿" and 0x05 <= (ord(ch) & 0x7F) <= 0x39 and nxt not in _VIRAMAS:
            count += 1
    return count
