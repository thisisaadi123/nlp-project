"""Find written-style (bookish) words in a dubbed line: forms people read but don't say.

Dubbed dialogue is spoken, so these make a line sound like a news reader or a textbook:
Tamil "பார்க்கவும்" (spoken: பாத்துக்கோ), "போகிறேன்" (போறேன்), "வேண்டும்" (வேணும்);
Hindi "प्रसन्न", "अनुमति"; Bengali sadhu verbs "করিতেছি"; Telugu "చేయుచున్నాను". The lists are
deliberately short and unambiguous: a marker must be wrong in casual speech, not just formal-ish.
"""
import re
import unicodedata

_WORD = re.compile(r"[\wऀ-෿]+")

# whole words that are written-only
WORDS = {
    "ta": {"வேண்டும்", "இல்லை", "அவர்கள்", "இவர்கள்", "உள்ளது", "உள்ளன", "இருக்கிறது", "இருக்கின்றது",
           "மேலும்", "எனவே", "ஆனால்", "மற்றும்", "நலமா", "தயவுசெய்து", "அவ்வாறு", "இவ்வாறு",
           "முதியவன்", "முதியவர்", "இன்றிரவு", "இன்று", "நேற்று", "அவ்விடம்", "யாது"},
    "hi": {"प्रसन्न", "अनुमति", "कृपया", "आवश्यक", "अत्यंत", "परंतु", "किंतु", "यद्यपि", "तथा", "एवं",
           "हेतु", "प्रतीक्षा", "क्षमा", "उपरांत", "अतः", "तत्पश्चात", "सम्मिलित", "वार्तालाप"},
    "mr": {"कृपया", "आवश्यक", "परंतु", "तथापि", "अत्यंत", "प्रतीक्षा", "क्षमस्व", "उपरोक्त"},
    "bn": {"এবং", "অত্যন্ত", "অনুগ্রহপূর্বক", "যাহা", "তাহা", "কিরূপে", "নিমিত্ত"},
    "te": {"మరియు", "కావున", "అత్యంత", "దయచేసి", "యొక్క"},
}

# word endings that are written-only
ENDINGS = {
    # present tense -கிறேன்/-கிறாய்/..., -கின்ற, polite imperative -ங்கள், "-க்கவும்" instructions
    # and the written -ன்ற- past (நின்றான், spoken நின்னான்)
    "ta": re.compile(r"(கிற(ேன்|ாய்|ான்|ாள்|து|ோம்|ார்|ார்கள்|ீர்கள்)|கின்ற\w*|ுங்கள்|(க்க|ச்ச|ட்ட|த்த|ப்ப|ற்ற)வும்"
                     r"|ன்ற(ான்|ாள்|து|ேன்|ோம்|ார்))$"),
    # sadhu bhasha verbs: করিতেছি, যাইতেছে, করিয়া
    "bn": re.compile(r"(িতে(ছি|ছ|ছে|ছেন|ছিল|ছিলাম)|ইয়া|িয়া)$"),
    # bookish progressive -చున్న..., -వలెను
    "te": re.compile(r"(చున్న\w*|వలెను)$"),
}


def written_words(text: str, lang: str) -> list[str]:
    """Words in `text` that only appear in writing, for the target language `lang`."""
    found = []
    for w in _WORD.findall(unicodedata.normalize("NFC", text)):
        w = w.strip("।॥")
        if w and (w in WORDS.get(lang, ()) or (lang in ENDINGS and ENDINGS[lang].search(w))) and w not in found:
            found.append(w)
    return found
