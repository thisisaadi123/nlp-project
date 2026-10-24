"""Hard check for family and address words: a relation in the source must come out as the target
language's own word for that relation.

Prompt rules weren't enough: with a Sarvam draft that read Tamil அண்ணி (elder brother's wife) as
"elder sister", the writer kept అక్కా / দিদি / ताई in Telugu, Bengali and Marathi even with a
kinship table in the prompt. This finds such lines so they can be repaired and checked again.
Matching is on word stems, so case endings and honorific suffixes still match (அண்ணனுக்கு, भैया जी).
"""
import re

# relation -> "say": target words that are right (the first is the one to suggest); "heard": source
# stems that mark the relation, where they differ from "say". Hindi "भाई" is not a marker: between
# friends it means "bro" (Tamil machan, not anna), which is the register this project is about.
RELATIONS = {
    "elder brother's wife": {"say": {
        "ta": ["அண்ணி"], "hi": ["भाभी"], "bn": ["বৌদি", "বউদি"], "mr": ["वहिनी", "वहिनि"], "te": ["వదిన"]}},
    "elder sister": {"say": {
        "ta": ["அக்கா"], "hi": ["दीदी", "दिदी"], "bn": ["দিদি"], "mr": ["ताई"], "te": ["అక్క"]}},
    "elder brother": {"say": {
        "ta": ["அண்ணா", "அண்ணே", "அண்ணன்"], "hi": ["भैया", "भइया", "भाई"], "bn": ["দাদা"],
        "mr": ["दादा", "भाऊ"], "te": ["అన్న"]},
        "heard": {"hi": ["भैया", "भइया"], "bn": ["দাদা"], "mr": ["दादा"]}},
    "father's sister": {"say": {
        "ta": ["அத்தை"], "hi": ["बुआ"], "bn": ["পিসি"], "mr": ["आत्या"], "te": ["అత్త"]},
        # spoken Tamil says அத்த; only as a whole word (அத்தனை, "that many", is not an aunt)
        "heard": {"ta": ["அத்தை", "=அத்த"]}},
}
# Source stems that must not be read as a relation in that language (அண்ணாமலை is a name).
NOT_A_RELATION = {"ta": ["அண்ணாமலை"]}
# The Tamil "da" between friends, written in another script, and that language's own tag for it.
PARTICLES = {"hi": {"डा": "यार"}, "mr": {"डा": "रे"}, "bn": {"ডা": "রে"}, "te": {"డా": "రా"}}

_WORD = re.compile(r"[\wऀ-෿]+")


def _has(text: str, stems: list[str]) -> bool:
    """A stem matches inside a word; "=word" matches only that whole word."""
    return any(w == s[1:] if s.startswith("=") else s in w for w in _WORD.findall(text) for s in stems)


def replace_particles(translation: str, tgt: str) -> str:
    """Swap a leftover Tamil "da" for the target language's own tag (safe to do without a model)."""
    for p, own in PARTICLES.get(tgt, {}).items():
        translation = re.sub(rf"(?<![\w\u0900-\u0DFF]){re.escape(p)}(?![\w\u0900-\u0DFF])", own, translation)
    return translation


def relation_hints(source: str, src: str, tgt: str) -> list[str]:
    """The relation words a translation of `source` must use, e.g. ["elder brother's wife: వదిన"]."""
    return [p.replace("use ", "") for p in kinship_issues(source, "", src, tgt) if "particle" not in p]


def kinship_issues(source: str, translation: str, src: str, tgt: str) -> list[str]:
    """Plain-English problems, e.g. 'elder brother's wife: use వదిన'; [] when the line is right."""
    problems = []
    if any(n in source for n in NOT_A_RELATION.get(src, [])):
        source = re.sub("|".join(NOT_A_RELATION[src]), " ", source)
    for relation, r in RELATIONS.items():
        heard = r.get("heard", {}).get(src, r["say"].get(src))
        say = r["say"].get(tgt)
        if heard and say and _has(source, heard) and not _has(translation, say):
            problems.append(f"{relation}: use {say[0]}")
    for p, own in PARTICLES.get(tgt, {}).items():
        if p in _WORD.findall(translation):
            problems.append(f"'{p}' is a Tamil particle that means nothing in this language: use '{own}'")
    return problems
