"""Word by word: which words of the line the player said, and which they got wrong or skipped.

Sarvam's speech-to-text writes what it hears, so a word said wrong usually comes out as a different
word. The heard words are lined up with the line's words (an edit-distance alignment over words) and
compared by sound, so spelling variants of the same word still count as right (spoken Tamil
"குடுங்க" for "கொடுங்க"). This judges words, not accent: a subtle mispronunciation that still sounds
like the word passes.
"""
import re

from backend.pipeline.translate.copies import _similarity, _sound
from backend.roman import romanize

SAME_WORD = 0.65            # "है" vs "हैं": the same word, said casually
_TOKEN = re.compile(r"[\wऀ-෿]+")


def _words(text: str) -> list[str]:
    return [w.strip("।॥") for w in _TOKEN.findall(text) if w.strip("।॥")]


def _key(word: str, lang: str) -> str:
    return _sound(romanize(word, lang))


def _same(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a == b or _similarity(a, b) >= SAME_WORD)


def check(heard: str, expected: str, lang: str) -> dict:
    """{"score": share of the line's words said right (0..1),
        "words": [{"word": <line word>, "ok": bool, "heard": <what was heard there, or None>}]}"""
    exp = _words(expected)
    # speech-to-text writes numbers as digits ("₹600"); the line spells them out, so digits are dropped
    heard_words = _words(heard)
    is_number = [any(ch.isdigit() for ch in w) for w in heard_words]
    got = [w for w, num in zip(heard_words, is_number) if not num]
    # where each number sat among the remaining words
    number_at = [sum(not x for x in is_number[:k]) for k, num in enumerate(is_number) if num]
    ek, gk = [_key(w, lang) for w in exp], [_key(w, lang) for w in got]
    n, m = len(exp), len(got)
    if not n:
        return {"score": 1.0, "words": []}
    # cost[i][j]: aligning exp[:i] with got[:j]; a substitution of the same word is free
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1,
                             cost[i - 1][j - 1] + (0 if _same(ek[i - 1], gk[j - 1]) else 1))
    # walk back, preferring matches, then skips, and a wrong word only when nothing else is as cheap,
    # so equally good alignments keep the most words right
    out, i, j = [], n, m
    while i > 0:
        match = j > 0 and _same(ek[i - 1], gk[j - 1])
        if match and cost[i][j] == cost[i - 1][j - 1]:
            out.append({"word": exp[i - 1], "ok": True, "heard": None})
            i, j = i - 1, j - 1
        elif j > 0 and cost[i][j] == cost[i][j - 1] + 1:
            j -= 1                                   # an extra heard word: costs nothing in the score
        elif cost[i][j] == cost[i - 1][j] + 1:
            out.append({"word": exp[i - 1], "ok": False, "heard": None, "at": j})
            i -= 1
        else:
            out.append({"word": exp[i - 1], "ok": False, "heard": got[j - 1]})
            i, j = i - 1, j - 1
    out.reverse()
    # a number heard as digits stands for up to two spelled-out words skipped right where it was said
    for d in number_at:
        near = [w for w in out if not w["ok"] and w["heard"] is None and abs(w["at"] - d) <= 1][:2]
        for w in near:
            w["ok"] = True
    for w in out:
        w.pop("at", None)
    return {"score": sum(w["ok"] for w in out) / n, "words": out}


def tip(result: dict) -> str:
    """One sentence about the first wrong word, or "" when every word was right."""
    for w in result["words"]:
        if not w["ok"]:
            return (f"We heard “{w['heard']}” where the line says “{w['word']}”." if w["heard"]
                    else f"You skipped “{w['word']}”.")
    return ""
