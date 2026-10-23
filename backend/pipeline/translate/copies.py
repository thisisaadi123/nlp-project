"""Find source words copied into a translation in the target script instead of being translated.

The machine draft does this with spoken words it doesn't know (Tamil கெளம்புறேன்டா "I'm leaving"
came out as Hindi "गुम्बुरेंटा"), and a weak refine step keeps them; address terms leak the same
way (அண்ணா -> "अन्ना" instead of "भैया"). Both sides are romanized and compared by sound, loosely:
voicing, aspiration, vowel length and doubled letters are ignored, because the romanizers and the
scripts differ in exactly those. Names and loanwords also match; the caller decides what to keep.
"""
import re

from backend.roman import romanize

_CLASSES = str.maketrans({"g": "k", "q": "k", "j": "c", "d": "t", "b": "p", "f": "p", "w": "v", "z": "s"})
_VOWELS = re.compile(r"[aeiou]")
MIN_SIMILARITY = 0.75


def _sound(word: str) -> str:
    w = re.sub(r"[^a-z]", "", word.lower()).replace("x", "ks").replace("h", "").translate(_CLASSES)
    return re.sub(r"(.)\1+", r"\1", w)


def _skeleton(sound: str) -> str:
    return re.sub(r"(.)\1+", r"\1", _VOWELS.sub("", sound))


def _similarity(a: str, b: str) -> float:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b), 1)


def _words(text: str) -> list[str]:
    words = (w.strip("।॥") for w in re.split(r"[^\wऀ-෿]+", text))  # ।॥ are inside the block
    return [w for w in words if w]


def _same(a: str, b: str) -> bool:
    sa, sb = _sound(a), _sound(b)
    if len(sa) < 2 or len(sb) < 2:
        return False
    ka, kb = _skeleton(sa), _skeleton(sb)
    if min(len(ka), len(kb)) < 3:           # short words: the whole sound has to match
        return sa == sb
    return _similarity(ka, kb) >= MIN_SIMILARITY


_LATIN = re.compile(r"[A-Za-z]{2,}")


def copied_words(source: str, translation: str, src: str, tgt: str) -> list[str]:
    """Words of `translation` that sound like a word of `source`, in the translation's script, and
    any word left in English letters (a dub line is read from the target script: Tamil "da" once
    came through as Latin "da" in a Telugu line)."""
    src_words = [(w, romanize(w, src)) for w in _words(source)]
    found = [w for w in dict.fromkeys(_LATIN.findall(translation))]
    for t in _words(translation):
        if src == tgt or any(t == w for w, _ in src_words):
            continue                         # same script: identical words are usually shared vocabulary
        rt = romanize(t, tgt)
        if any(_same(rt, rs) for _, rs in src_words) and t not in found:
            found.append(t)
    return found
