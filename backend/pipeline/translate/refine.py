"""Context- and emotion-aware dubbing translation.

IndicTrans2's draft keeps the content; this step rewrites each line the way the target
language's film dialogue would say it, given who is speaking to whom, the scene, and how the
line is delivered. All lines of a clip go in one call so address terms stay consistent across
the conversation (and to save free-tier quota).
"""
import json

from backend import sarvam_http
from backend.llm.client import LLMError, chat_json
from backend.pipeline.translate.copies import copied_words
from backend.pipeline.translate.kinship import kinship_issues, relation_hints, replace_particles
from backend.pipeline.translate.translit import sarvam_roman
from backend.pipeline.translate.written import written_words
from backend.roman import romanize
from backend.settings import config, secret

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
2. Write how people TALK, not how they write: the spoken register of the target language as
   heard in its mainstream films, with colloquial verb forms, contractions and the everyday
   English words its speakers really use. "target_style" describes it for this language. Formal
   written forms are wrong unless the speaker is being formal.
3. Keep the source's meaning. The draft is a machine translation ("draft_source" says which and
   how it tends to fail). An IndicTrans2 draft comes out bookish; a Sarvam draft is usually
   spoken-style but can miss the meaning, the address terms, or copy source words. For casual, intimate or emotional lines (friends,
   family, teasing, anger), write the line fresh from its meaning the way the character would
   say it out loud, and use the draft only to check nothing is lost. For formal or respectful
   lines, start from the draft: when it already says the meaning naturally with the right forms
   of address, keep it and change only what is wrong. Every change must make the line better
   (more accurate, more natural as speech, the right register), never just different.
   When the draft is wrong, translate from the source: drafts often break on spoken dialogue,
   copying words they don't know into the target script (Tamil "kelambren-da", "I'm leaving",
   came out as Hindi "गुम्बुरेंटा") or getting the meaning wrong. The source always wins.
   "draft_warning" lists words the draft copied instead of translating.
4. Match the emotion and its intensity. An angry line stays sharp and short; a pleading line soft.
5. Keep the length close to the original (aim for within 20% of its syllable count) so it fits
   the actor's mouth and timing.
6. The words the original speaker stresses must map to the target word carrying the same
   meaning, so the dubber can stress it.
7. Never write a source-language word in the target script unless it is a person's or place's
   name, or a word the target language really uses (phone, Aadhaar). Family and address terms
   always become the target language's own word for that relation, whatever the scene notes call
   the person. The same relation in Hindi / Tamil / Bengali / Marathi / Telugu:
     elder brother:        bhaiya / anna / dada / dada / anna, annayya
     elder brother's wife: bhabhi / anni / boudi / vahini / vadina
     elder sister:         didi / akka / didi / tai / akka
     younger brother:      chhotu, the name / thambi / bhai / the name / tammudu
     father's sister:      bua / athai / pishi / atya / atta
     mother's brother:     mama / mama / mama / mama / mavayya
     father's younger brother: chacha / chithappa / kaka / kaka / babai
     close male friend:    yaar, bhai / machan, da / bhai, dost / bhava, mitra / bava, mama, ra
   Tamil "anni" is a sister-in-law, never an elder sister.
8. Verb and adjective forms follow "speaker_gender" and "addressee_gender" wherever the target
   language marks gender (a male speaker in Hindi says "keh raha hoon", not "keh rahi hoon").
9. Write the translation in the target language's own script. Also give a romanization a
   non-reader can pronounce from, in casual English spelling (e.g. "anna, enna panna?"), not
   scholarly diacritics.
Answer with JSON only."""

OUTPUT_SPEC = """Return JSON:
{"lines": [{"line_id": <same id>,
            "meaning": "<what the speaker means here, in plain English, including what each family or address term means (e.g. anni = elder brother's wife)>",
            "draft_verdict": "<kept | edited | rewritten: how much of the draft you kept>",
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


# How each language is actually spoken in its films, against the written forms models default to.
SPOKEN = {
    "ta": ("Spoken Tamil (Chennai/film Tamil), never written or literary Tamil. Verb endings: "
           "போறேன் not போகிறேன், வர்றேன் not வருகிறேன், பண்ற not செய்கிறாய், இருக்கு not இருக்கிறது, "
           "வேணும் not வேண்டும், இல்ல not இல்லை, சொல்லுங்க not சொல்லுங்கள், போலாம் not போகலாம், "
           "அவங்க not அவர்கள். Use pal/brother words by closeness: டா / டி / மச்சான் / அண்ணா / தம்பி. "
           "Everyday English words Tamil speakers really use stay English (நைட், ஃபோன், சாரி). "
           "Examples: 'आजा बेटा आजा' -> 'வா டா, வா வா'; 'रात कैसी कटी?' -> 'நைட் எப்படி போச்சு?'; "
           "'देख लेना' -> 'பாத்துக்கோ' (never 'பார்க்கவும்')."),
    "hi": ("Spoken Bollywood Hindi (Hindustani), not shuddh or textbook Hindi: everyday Urdu and English "
           "words are normal (time, problem, yaar, pakka). 'क्या कर रहे हो' not 'आप क्या कर रहे हैं' between "
           "friends; 'चल', 'अरे', 'ना' tags; 'बोल' not 'कहिए' unless respectful. Never Sanskritised words "
           "like 'प्रसन्न' or 'अनुमति' in casual talk."),
    "bn": ("Spoken Kolkata Bengali (cholito), never sadhu bhasha: করছি not করিতেছি, যাচ্ছি, বলছিস. "
           "Pronoun level by closeness: তুই / তুমি / আপনি. Everyday English words stay English."),
    "te": ("Spoken Telugu as in Telugu films, not bookish Telugu: ఏం చేస్తున్నావ్ not ఏమి చేయుచున్నావు, "
           "వస్తా, చెప్పు రా. Pronoun level: నువ్వు / మీరు; friends use రా / రే / బావ / మామ."),
    "mr": ("Spoken Marathi as in Marathi films (Pune/Mumbai), not textbook: काय करतोयस, चल रे, अरे यार, "
           "बोल ना. Pronoun level: तू / तुम्ही / आपण. Everyday English words stay English."),
}


DRAFT_SOURCES = {
    "indictrans2": "IndicTrans2: literal, written-style; breaks on spoken words (may copy them)",
    "sarvam": "Sarvam Mayura, spoken mode: natural and colloquial, but sometimes wrong in meaning or "
              "address terms (e.g. Hindi 'aaja beta' came out as Tamil 'sari paiyan'); check every line",
}


def translator_model() -> str | None:
    """llm.translator -> the model that understands and writes the dialogue: "sarvam" and "gemini"
    name a provider:model for chat_json(prefer=...); "gpt-oss" -> None (the default chain)."""
    llm = config()["llm"]
    if llm.get("translator") == "sarvam" and sarvam_http.available():
        return f"sarvam:{llm.get('translator_sarvam_model', 'sarvam-105b')}"
    if llm.get("translator") == "gemini":
        return f"gemini:{llm.get('translator_gemini_model')}"
    return None


def _line_payload(line: dict, draft: str, people: dict | None = None, src: str | None = None,
                  tgt: str | None = None) -> dict:
    """The facts about one line the translator needs; step 3's JSON fields where they exist."""
    out = {"line_id": line["line_id"], "speaker": line.get("speaker"), "addressee": line.get("addressee"),
           "text": line["text"], "draft": draft or "(no usable draft: translate from the source text)"}
    for role in ("speaker", "addressee"):
        gender = (people.get(line.get(role)) or {}).get("gender") if isinstance(people, dict) and people else None
        if gender:
            out[f"{role}_gender"] = gender
    hints = relation_hints(line["text"], src, tgt) if src and tgt else []
    if hints:
        out["must_use"] = hints         # family words checked in code after translating
    copied = copied_words(line["text"], draft, src, tgt) if draft and src and tgt else []
    if copied:
        out["draft_warning"] = f"the draft copied these source words instead of translating them: {', '.join(copied)}"
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
    # Gemini and Cerebras take a whole scene per call; Groq's per-minute budget only fits a few lines
    size = llm["translate_batch_cerebras"] if translator_model() or secret("CEREBRAS_API_KEY") else llm["translate_batch"]
    people = (scene or {}).get("speakers") or {}
    primary_only = llm.get("translate_primary_only", True)
    out: list[dict] = []
    for i in range(0, len(lines), size):
        request = {
            "source_language": names[src],
            "target_language": names[tgt],
            "target_style": SPOKEN.get(tgt, "the spoken form of the language, as in its films"),
            "draft_source": DRAFT_SOURCES.get(config()["llm"].get("draft_engine"), DRAFT_SOURCES["indictrans2"])
            if sarvam_http.available() else DRAFT_SOURCES["indictrans2"],
            "scene": scene or "no scene information available",
            "lines": [_line_payload(line, d, people, src, tgt) for line, d in zip(lines[i:i + size], drafts[i:i + size])],
        }
        if out:
            request["already_translated"] = [
                {"line_id": ln["line_id"], "speaker": ln.get("speaker"), "text": ln["text"],
                 "translation": t["translation"], "register": t.get("register")}
                for ln, t in list(zip(lines, out))[-6:]]
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=1) + "\n\n" + OUTPUT_SPEC}]
        result = chat_json(messages, max_tokens=6500, effort=effort or llm["translate_effort"],
                           validate=_valid_lines(len(request["lines"])), primary_only=primary_only, prefer=translator_model())
        for item in result["lines"]:
            item["_provider"] = result.get("_provider")
        out += result["lines"]
    if llm.get("translate_review", True):
        out = review(lines, out, drafts, src, tgt, scene, primary_only)
    out = fix_kinship(lines, out, src, tgt, primary_only)
    out = fix_written(lines, out, tgt, primary_only)
    out = fix_copies(lines, out, src, tgt, primary_only)
    return fit_lengths(lines, out, tgt)


KINSHIP_SPEC = """Each line below uses the wrong word for a family relation, or a particle from another
language ("problems" says exactly what to change). Fix only that: use the word given, in the right
grammatical form, and keep everything else (meaning, register, spoken style, length). Return JSON:
{"lines": [{"line_id": <id>, "translation": "<line>", "romanized": "<casual romanization>"}]}"""


def fix_kinship(lines: list[dict], out: list[dict], src: str, tgt: str, primary_only: bool = True) -> list[dict]:
    """Repair lines whose family words don't match the source's relation (checked again after)."""
    flagged = [(ln, item, kinship_issues(ln["text"], item["translation"], src, tgt)) for ln, item in zip(lines, out)]
    flagged = [(ln, item, p) for ln, item, p in flagged if p]
    if not flagged:
        return out
    request = {"target_language": config()["languages"][tgt]["name"], "lines": [
        {"line_id": item["line_id"], "source": ln["text"], "translation": item["translation"], "problems": p}
        for ln, item, p in flagged]}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=1) + "\n\n" + KINSHIP_SPEC}]
    try:
        reply = chat_json(messages, max_tokens=2000, effort="medium", primary_only=primary_only,
                          prefer=translator_model(), validate=lambda r: isinstance(r.get("lines"), list))
    except LLMError:
        reply = {"lines": []}
    fixes = {r.get("line_id"): r for r in reply["lines"] if isinstance(r, dict)}
    for ln, item, before in flagged:
        new = fixes.get(item["line_id"])
        if new and str(new.get("translation", "")).strip():
            left = kinship_issues(ln["text"], new["translation"], src, tgt)
            if len(left) < len(before):
                item["kinship_fixed_from"] = item["translation"]
                item["translation"] = new["translation"]
                item["romanized"] = new.get("romanized", item.get("romanized"))
                before = left
        if any("particle" in p for p in before):          # the model kept it: swap it in code
            swapped = replace_particles(item["translation"], tgt)
            if swapped != item["translation"]:
                item.setdefault("kinship_fixed_from", item["translation"])
                item["translation"] = swapped
                before = kinship_issues(ln["text"], swapped, src, tgt)
        if before:
            item["kinship_problems"] = before
    return out


WRITTEN_SPEC = """These dubbed lines contain written-only words ("written"): forms people read but never
say out loud. Rewrite each line as the character would actually SAY it in a {tgt} film, following
"target_style": colloquial verb endings and spoken words instead of the written ones. Keep the
meaning, the forms of address, the register (a respectful line stays respectful, in its spoken
form) and roughly the same length. A word may stay only if the speaker is deliberately formal:
then list it in "kept". Return JSON:
{{"lines": [{{"line_id": <id>, "translation": "<line>", "romanized": "<casual romanization>",
              "kept": ["<written words deliberately kept>"]}}]}}"""


def fix_written(lines: list[dict], out: list[dict], tgt: str, primary_only: bool = True) -> list[dict]:
    """One rewrite pass for lines with written-only words (see written.py)."""
    flagged = [(ln, item, written_words(item["translation"], tgt)) for ln, item in zip(lines, out)]
    flagged = [(ln, item, w) for ln, item, w in flagged if w and item.get("register") != "formal"]
    if not flagged:
        return out
    name = config()["languages"][tgt]["name"]
    request = {"target_language": name, "target_style": SPOKEN.get(tgt, "the spoken form of the language"),
               "lines": [{"line_id": item["line_id"], "source": ln["text"], "meaning": item.get("meaning"),
                          "register": item.get("register"), "translation": item["translation"], "written": w}
                         for ln, item, w in flagged]}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=1) + "\n\n"
                 + WRITTEN_SPEC.format(tgt=name)}]
    try:
        reply = chat_json(messages, max_tokens=3000, effort="medium", primary_only=primary_only,
                          prefer=translator_model(), validate=lambda r: isinstance(r.get("lines"), list))
    except LLMError:
        return out
    fixes = {r.get("line_id"): r for r in reply["lines"] if isinstance(r, dict)}
    for ln, item, before in flagged:
        new = fixes.get(item["line_id"])
        if not new or not str(new.get("translation", "")).strip():
            continue
        kept = [w for w in new.get("kept") or [] if isinstance(w, str)]
        left = [w for w in written_words(new["translation"], tgt) if w not in kept]
        if len(left) < len(before):          # only take a rewrite that is actually more spoken
            item["written_fixed_from"] = item["translation"]
            item["translation"] = new["translation"]
            item["romanized"] = new.get("romanized", item.get("romanized"))
            before = left
        if before:
            item["written_words"] = before
    return out


REVIEW_SPEC = """You are now the dialogue editor: a native {tgt} screenwriter checking a dubbed script
before recording. For every line, compare "source" with "translation" and check:
1. Meaning: does it say what the source says in this scene (see "meaning")? Nothing added or lost.
2. Naturalness: would a {tgt} film character say exactly this, out loud, to this person? Fix
   stiff, literal, written or textbook wording; use the spoken forms in "target_style". Formal
   written forms are a mistake unless the speaker is being formal.
3. Address and register: family and address terms are {tgt}'s own words for that relation, and
   pronoun and verb politeness fit who is speaking to whom.
4. Gender: verb forms agree with "speaker_gender" / "addressee_gender".
5. No {src} word written in {tgt} script, except names and real loanwords.
6. Never worse than the draft: "draft" is the machine translation the line started from. If the
   draft says the line more accurately, go back to it (or to it with only the needed fixes).
   Being closer to the draft's wording is not "better" when the draft is written-style.
If a line is already right, return it unchanged with "verdict": "ok". Otherwise return the
corrected line with "verdict": "fixed" and say what was wrong in "problem".
Return JSON:
{{"lines": [{{"line_id": <id>, "verdict": "ok|fixed", "problem": "<short, English, empty if ok>",
              "translation": "<line>", "romanized": "<casual romanization>"}}]}}"""


def review(lines: list[dict], out: list[dict], drafts: list[str], src: str, tgt: str, scene: dict | None = None,
           primary_only: bool = True) -> list[dict]:
    """A second read of the whole scene's translation by the same model, as the dialogue editor."""
    names = {k: v["name"] for k, v in config()["languages"].items()}
    people = (scene or {}).get("speakers") or {}
    items = []
    for ln, item, draft in zip(lines, out, drafts):
        x = {"line_id": item["line_id"], "speaker": ln.get("speaker"), "addressee": ln.get("addressee"),
             "source": ln["text"], "meaning": item.get("meaning"), "draft": draft,
             "translation": item["translation"],
             "register": item.get("register")}
        for role in ("speaker", "addressee"):
            if isinstance(people, dict) and (people.get(ln.get(role)) or {}).get("gender"):
                x[f"{role}_gender"] = people[ln[role]]["gender"]
        items.append(x)
    request = {"scene": scene or "no scene information",
               "target_style": SPOKEN.get(tgt, "the spoken form of the language, as in its films"),
            "draft_source": DRAFT_SOURCES.get(config()["llm"].get("draft_engine"), DRAFT_SOURCES["indictrans2"])
            if sarvam_http.available() else DRAFT_SOURCES["indictrans2"], "lines": items}
    spec = REVIEW_SPEC.format(src=names[src], tgt=names[tgt])
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=1) + "\n\n" + spec}]
    try:
        reply = chat_json(messages, max_tokens=4000, effort=config()["llm"]["translate_effort"],
                          primary_only=primary_only, prefer=translator_model(), validate=_valid_lines(len(items)))
    except LLMError:
        return out              # the review improves a translation; without it the first pass stands
    for item, r in zip(out, reply["lines"]):
        if r.get("verdict") == "fixed" and str(r.get("translation", "")).strip() and r["translation"] != item["translation"]:
            item["reviewed_from"] = item["translation"]
            item["review_problem"] = r.get("problem", "")
            item["translation"] = r["translation"]
            item["romanized"] = r.get("romanized", item.get("romanized"))
    return out


COPY_SPEC = """Each translation below contains words that look like source-language words written in
the target script ("copied"). For each line: a copied word that is a person's or place's name, or a
word the target language really uses (phone, Aadhaar), may stay: list it in "kept". Every other
copied word is a mistake: rewrite the line so its meaning is said in real target-language words
(family and address terms become the target's own word, e.g. Tamil anna -> Hindi bhaiya).
Change nothing else: same meaning, register, pronoun level and roughly the same length.
Return JSON:
{"lines": [{"line_id": <id>, "translation": "<line>", "romanized": "<casual romanization>",
            "kept": ["<copied words that are names or real loanwords>"]}]}"""


def fix_copies(lines: list[dict], out: list[dict], src: str, tgt: str, primary_only: bool = True) -> list[dict]:
    """One repair pass for lines that still contain source words in the target script."""
    flagged = [(ln, item, copied_words(ln["text"], item["translation"], src, tgt)) for ln, item in zip(lines, out)]
    flagged = [(ln, item, c) for ln, item, c in flagged if c]
    if not flagged:
        return out
    names = config()["languages"]
    request = {"source_language": names[src]["name"], "target_language": names[tgt]["name"], "lines": [
        {"line_id": item["line_id"], "source": ln["text"], "meaning": item.get("meaning"),
         "translation": item["translation"], "copied": c} for ln, item, c in flagged]}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=1) + "\n\n" + COPY_SPEC}]
    try:
        reply = chat_json(messages, max_tokens=3000, effort="high", primary_only=primary_only, prefer=translator_model(),
                          validate=lambda r: isinstance(r.get("lines"), list))
    except LLMError:
        reply = {"lines": []}
    fixes = {r.get("line_id"): r for r in reply["lines"] if isinstance(r, dict)}
    for ln, item, before in flagged:
        new = fixes.get(item["line_id"])
        if new and str(new.get("translation", "")).strip():
            kept = [w for w in new.get("kept") or [] if isinstance(w, str)]
            left = copied_words(ln["text"], new["translation"], src, tgt)
            def is_kept(w):
                return any(k in w or w in k for k in kept)
            # accept only if there was a real copy to fix (not just names the model kept) and every
            # copied word that is left was deliberately kept as a name or loanword
            if not all(is_kept(w) for w in before) and all(is_kept(w) for w in left):
                if new["translation"] != item["translation"]:
                    item["copies_fixed_from"] = item["translation"]
                    item["translation"] = new["translation"]
                    item["romanized"] = new.get("romanized", item.get("romanized"))
                before = left
        if before:
            item["copied_words"] = before   # left in: names and loanwords, or a failed repair
    return out


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
            reply = chat_json(messages, max_tokens=3000, effort="high", prefer=translator_model(),
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
        item["romanized"] = checked_romanization(item.get("romanized"), item["translation"], tgt,
                                                 sarvam_roman(item["translation"], tgt))
    return out


def checked_romanization(llm: str | None, translation: str, lang: str, sarvam: str | None = None) -> str:
    """The friendliest romanization that still spells this exact line. Sarvam's and the model's read
    better than ours, but both can change words: the model's can describe an older version of the
    line ("bhaiya" for भाभी), and Sarvam's turned Marathi into Hindi ("आत ठेव" -> "Aat rakh",
    "ठीक आहे" -> "Theek hai"). Each is kept only if it matches ours word by word."""
    ours = romanize(translation, lang)
    from backend.pipeline.translate.copies import _similarity, _sound

    def spells_the_line(candidate: str) -> bool:
        a, b = [[_sound(w) for w in t.split() if _sound(w)] for t in (candidate, ours)]
        # word by word: one wrong word hides inside a long line's overall similarity
        return len(a) == len(b) and all(_similarity(x, y) >= 0.6 for x, y in zip(a, b))

    return next((c for c in (sarvam, llm) if c and spells_the_line(c)), ours)


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
