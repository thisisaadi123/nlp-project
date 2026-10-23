"""Literal draft translation with IndicTrans2 (direct Indic -> Indic, no English pivot).

The draft keeps the content accurate; it is sentence-by-sentence and blind to the scene, so
register and emotion are fixed afterwards by refine.py. The distilled 320M model runs on CPU.

The model occasionally gets stuck repeating a word ("কাঁচা কাঁচা কাঁচা ..."). Output length is
capped relative to the input, lines go in small batches so one bad line cannot slow the others,
and a stuck draft is returned as "" so the refinement step translates that line on its own.
"""
from functools import lru_cache

from backend.settings import config

BATCH = 4
MAX_REPEATS = 4   # the same word this many times in a row means the model is stuck
PUNCTUATION = ",.!?;:।॥\"'“”‘’()-"


def degenerate(text: str) -> bool:
    # Split on spaces, not \w: \w does not match Indic vowel signs, which would cut words apart.
    words = [w.strip(PUNCTUATION) for w in text.split()]
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= MAX_REPEATS:
            return True
    return False


class DraftTranslator:
    def __init__(self, device: str = "cpu"):
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = config()["models"]["translation_draft"]
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).to(device).eval()
        self.processor = IndicProcessor(inference=True)

    def _batch(self, sentences: list[str], src_code: str, tgt_code: str, beams: int) -> list[str]:
        batch = self.processor.preprocess_batch(sentences, src_lang=src_code, tgt_lang=tgt_code)
        inputs = self.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt",
                                return_attention_mask=True).to(self.device)
        max_length = min(256, 2 * inputs["input_ids"].shape[1] + 16)
        with self.torch.no_grad():
            # use_cache=False: the model's remote code predates the cache format of transformers >= 4.4x
            out = self.model.generate(**inputs, use_cache=False, min_length=0, max_length=max_length,
                                      num_beams=beams, num_return_sequences=1)
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.processor.postprocess_batch(decoded, lang=tgt_code)

    def translate(self, sentences: list[str], src: str, tgt: str, beams: int = 4) -> list[str]:
        """`src`/`tgt` are our language keys (hi, ta, ...). Stuck outputs come back as ""."""
        langs = config()["languages"]
        src_code, tgt_code = langs[src]["indictrans"], langs[tgt]["indictrans"]
        out = []
        for i in range(0, len(sentences), BATCH):
            out += self._batch(sentences[i:i + BATCH], src_code, tgt_code, beams)
        return ["" if degenerate(t) else t for t in out]


@lru_cache
def translator(device: str = "cpu") -> DraftTranslator:
    return DraftTranslator(device)
