"""Indic Parler-TTS: text + a plain-language voice description -> 16 kHz speech.

The description picks the voice (a named speaker) and the delivery. TTS can mis-speak or read
the description aloud without any error, so callers should check the output with ASR.
"""
import librosa
import numpy as np
import torch

from backend.settings import config

SR = 16000
STYLE = {
    "angry": "speaks in an angry, loud voice with a fast pace and high pitch",
    "sad": "speaks in a sad, soft voice with a slow pace and low pitch",
    "happy": "speaks in a happy, cheerful voice with a lively pace",
    "neutral": "speaks in a calm, neutral voice at a moderate pace",
    "surprise": "speaks in a surprised, excited voice with a high pitch",
    "fear": "speaks in a fearful, trembling voice",
    "disgust": "speaks in a disgusted voice",
}
CLEAN = "The recording is very clear with no background noise."


def describe(speaker: str, emotion: str = "neutral") -> str:
    return f"{speaker} {STYLE.get(emotion, STYLE['neutral'])}. {CLEAN}"


class ParlerTTS:
    def __init__(self, device: str = "cuda"):
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        name = config()["models"]["tts"]
        self.device = device
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sr = self.model.config.sampling_rate

    def say(self, text: str, description: str) -> np.ndarray:
        d = self.desc_tok(description, return_tensors="pt").to(self.device)
        p = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                        prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        y = np.atleast_1d(audio.float().cpu().numpy().squeeze())
        return librosa.resample(y, orig_sr=self.sr, target_sr=SR) if y.size > 1000 else np.zeros(0, np.float32)


def trim(y: np.ndarray, below_peak_db: float = 35) -> np.ndarray:
    """Cut leading/trailing silence: keep from the first to the last 5 ms frame within 35 dB of the peak."""
    if len(y) < 800:
        return y[:0]
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=160, hop_length=80)[0] + 1e-8)
    on = np.flatnonzero(db > db.max() - below_peak_db)
    return y[on[0] * 80: (on[-1] + 2) * 80] if len(on) else y[:0]
