"""Readable romanization for people who speak a language but don't read its script.

Not a scholarly transliteration: it aims for how people type the language in chats
("anna, enna panra?", "bhaiya, ye kya kar diya?").

- Tamil script doesn't mark voicing (க is k or g), so stops are voiced by position the way
  they're pronounced: after a nasal (ந்த -> ndh) and between vowels (மகன் -> magan), but
  not word-initially or when doubled (பக்கம் -> pakkam).
- Devanagari / Bengali / Telugu go through indic_transliteration (ITRANS) and are simplified.
  Hindi and Marathi drop the word-final inherent vowel (कर -> kar, not kara).
"""
import re
import unicodedata

from indic_transliteration import sanscript

# ---------- Tamil ----------

_TA_VOWELS = {"அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo", "எ": "e", "ஏ": "e",
              "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au"}
_TA_SIGNS = {"ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo", "ெ": "e", "ே": "e", "ை": "ai",
             "ொ": "o", "ோ": "o", "ௌ": "au"}
_TA_CONS = {"க": "k", "ங": "ng", "ச": "s", "ஞ": "nj", "ட": "t", "ண": "n", "த": "th", "ந": "n",
            "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh", "ள": "l",
            "ற": "r", "ன": "n", "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h", "ஶ": "sh"}
_TA_VOICED = {"க": "g", "ச": "j", "ட": "d", "த": "dh", "ப": "b"}       # after a nasal
_TA_MEDIAL = {"க": "g", "ச": "s", "ட": "d", "த": "dh", "ப": "b"}       # between vowels
_TA_DOUBLED = {"க": "kk", "ச": "ch", "ட": "tt", "த": "tth", "ப": "pp", "ற": "tr"}
_TA_NASALS = set("ஙஞணநமன")
_TA_SOFT = set("யரழலள")                                              # voice a following stop too
_PULLI = "்"


def _tamil_word(w: str) -> str:
    # split into units: (consonant, sign) or (vowel, None); sign is a vowel sign, pulli or "" (inherent a)
    units, i = [], 0
    while i < len(w):
        c = w[i]
        if c in _TA_CONS:
            sign = w[i + 1] if i + 1 < len(w) and (w[i + 1] in _TA_SIGNS or w[i + 1] == _PULLI) else ""
            units.append((c, sign)); i += 1 + len(sign)
        else:
            units.append((c, None)); i += 1
    out = []
    for k, (c, sign) in enumerate(units):
        if sign is None:                                    # independent vowel or other character
            out.append(_TA_VOWELS.get(c, c)); continue
        prev = units[k - 1] if k else None
        if c in _TA_DOUBLED and prev and prev[0] == c and prev[1] == _PULLI:
            out[-1] = ""                                    # the pulli consonant is the first half
            base = _TA_DOUBLED[c]
        elif c == "ற" and prev and prev[0] == "ன" and prev[1] == _PULLI:
            base = "dr"                                     # என்று -> endru
        elif c in _TA_VOICED and prev and prev[0] in _TA_NASALS and prev[1] == _PULLI:
            base = _TA_VOICED[c]
            if prev[0] in "ஙஞ":
                out[-1] = "n"                               # ங்க -> ng, not ngg; ஞ்ச -> nj
        elif c in _TA_VOICED and prev and prev[0] in _TA_SOFT and prev[1] == _PULLI:
            base = _TA_VOICED[c]                            # செய்து -> seydhu, சொல்கிறார் -> solgiraar
        elif c in _TA_MEDIAL and prev and (prev[1] is None or prev[1] != _PULLI) and \
                (prev[1] is not None or prev[0] in _TA_VOWELS):
            base = _TA_MEDIAL[c]                            # previous unit ends in a vowel
        elif c == "ச" and prev and prev[1] == _PULLI and prev[0] != "ச":
            base = "ch"
        else:
            base = _TA_CONS[c]
        vowel = "" if sign == _PULLI else _TA_SIGNS.get(sign, "a")
        out.append(base + vowel)
    if units and units[-1][1] == _PULLI and units[-1][0] in "கசதப" and len(units) > 1:
        out[-1] = ""                                        # sandhi doubling before the next word: ஆகிக் -> aagi
    s = "".join(out)
    return s[:-2] + "a" if s.endswith("aa") and len(s) > 3 else s     # அண்ணா -> anna


def _tamil(text: str) -> str:
    return re.sub(r"[஀-௿]+", lambda m: _tamil_word(m.group()), text)


# ---------- Devanagari / Bengali / Telugu ----------

_SCRIPTS = {"hi": sanscript.DEVANAGARI, "mr": sanscript.DEVANAGARI,
            "bn": sanscript.BENGALI, "te": sanscript.TELUGU}
_VIRAMA = {"hi": "्", "mr": "्"}
_ITRANS = [("RRi", "ri"), ("Ch", "chh"), ("Sh", "sh"), ("~N", "n"), ("~n", "n"), (".N", "n"),
           (".Dh", "rh"), (".D", "r"), ("GY", "gy"), ("x", "ksh"), (".h", ""), (".a", ""),
           ("M", "n"), ("H", "h"), ("T", "t"), ("D", "d"), ("N", "n"), ("L", "l"), ("R", "r"),
           ("w", "v"), ("^", "")]


def _is_deva_consonant(c: str) -> bool:
    return "क" <= c <= "ह" or "क़" <= c <= "य़" or "ॸ" <= c <= "ॿ"


def _schwa_deletion(word: str, virama: str) -> str:
    """Hindi/Marathi schwa deletion: the inherent 'a' is silent at the end of a word and in a
    V C _ C V context (करना -> karna, देखता -> dekhta, but समझ -> samajh)."""
    # aksharas: [consonant cluster + modifiers]; track whether each ends in the inherent vowel
    aks, i = [], 0
    while i < len(word):
        j = i + 1
        if _is_deva_consonant(word[i]):
            while j < len(word) and (word[j] == "़" or (word[j] == virama and j + 1 < len(word)
                                                               and _is_deva_consonant(word[j + 1]))):
                j += 2 if word[j] == virama else 1
            schwa = not (j < len(word) and ("ा" <= word[j] <= "्" or word[j] in "ॢॣ"))
            while j < len(word) and not _is_deva_consonant(word[j]) and not ("ऄ" <= word[j] <= "औ"):
                j += 1
            aks.append([word[i:j], schwa])
        else:
            while j < len(word) and not _is_deva_consonant(word[j]) and not ("ऄ" <= word[j] <= "औ"):
                j += 1
            aks.append([word[i:j], False])
        i = j
    if len(aks) < 2:
        return word
    drop = [False] * len(aks)
    if aks[-1][1]:
        drop[-1] = True
    for k in range(1, len(aks) - 1):
        if aks[k][1] and not drop[k - 1] and (aks[k + 1][1] is False or not drop[k + 1]) \
                and not (aks[k + 1][1] and k + 1 == len(aks) - 1):
            drop[k] = True
    return "".join(a + (virama if d else "") for (a, _), d in zip(aks, drop))


def _drop_final_schwa(text: str, lang: str) -> str:
    v = _VIRAMA.get(lang)
    if not v:
        return text
    return re.sub(r"[ऀ-ॿ]+", lambda m: _schwa_deletion(m.group(), v), text)


def _simplify_vowels(word: str) -> str:
    word = re.sub(r"A$", "a", word)                 # kyA -> kya
    word = re.sub(r"I$", "i", word)                 # nahI -> nahi
    return word.replace("A", "aa").replace("I", "ee").replace("U", "oo")


def _brahmic(text: str, lang: str) -> str:
    s = sanscript.transliterate(_drop_final_schwa(text, lang), _SCRIPTS[lang], sanscript.ITRANS)
    s = re.sub(r"[A-Za-z.~^]+", lambda m: _simplify_vowels(m.group()), s)
    for a, b in _ITRANS:
        s = s.replace(a, b)
    s = s.replace("||", ".").replace("|", ".")
    if lang == "te":
        s = re.sub(r"n\b", "m", s)                  # word-final anusvara: ఏం -> em
    return s.lower()


def romanize(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text)
    if lang == "ta":
        return _tamil(text)
    if lang in _SCRIPTS:
        return _brahmic(text, lang)
    return text
