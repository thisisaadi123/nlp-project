"""Word start/end times by CTC forced alignment (torchaudio), two ways:

- CharAligner (preferred where config.yaml lists one): a small character-level wav2vec2 CTC
  model per language (Vakyansh, MIT) with 20 ms frames. IndicConformer supplies the words;
  this model only places them in time. The MMS aligner weights are not used (CC-BY-NC).
- align(): IndicConformer's own CTC log-probs, 80 ms frames. Its output isn't peaky and it
  emits a line's first word before the sound starts, so inner word boundaries are only good to
  ~200 ms (step 3 report). Used as the fallback for languages without a CharAligner.

Both forbid every token in silent frames (same loudness rule as backend/scoring/prosody.py),
so words can't start in leading silence or run across pauses.
"""
import unicodedata
from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torchaudio.functional as F
from scipy.ndimage import median_filter

from backend.pipeline.speech.asr import BLANK, FRAME_S, SR, Transcript
from backend.settings import ROOT, config

WORD_START = "▁"
SUB = int(FRAME_S / 0.01)           # 10 ms loudness frames per encoder frame


@dataclass
class Word:
    w: str
    start: float        # seconds from the start of the audio given to the ASR
    end: float
    score: float        # mean token log-prob along the alignment (closer to 0 = more confident)


def tokenize(text: str, tokens: list[str]) -> list[int]:
    """Split text into the model's subword tokens (fewest tokens; "▁" marks a word start)."""
    index = {t: i for i, t in enumerate(tokens) if i != BLANK and t not in ("<unk>", "|")}
    longest = max(len(t) for t in index)
    ids = []
    for word in text.split():
        s = WORD_START + word
        best = [0] + [None] * len(s)              # best[i] = fewest tokens covering s[:i]
        back = [None] * (len(s) + 1)
        for i in range(len(s)):
            if best[i] is None:
                continue
            for j in range(i + 1, min(len(s), i + longest) + 1):
                if s[i:j] in index and (best[j] is None or best[i] + 1 < best[j]):
                    best[j], back[j] = best[i] + 1, i
        if best[-1] is None:
            ids.append(index.get("<unk>", 0)); continue
        word_ids, j = [], len(s)
        while j > 0:
            word_ids.append(index[s[back[j]:j]]); j = back[j]
        ids.extend(reversed(word_ids))
    return ids


def greedy_ids(tr: Transcript) -> list[int]:
    path = tr.logprobs.argmax(-1)
    keep = np.concatenate([[True], path[1:] != path[:-1]]) & (path != BLANK)
    return path[keep].tolist()


def silent_frames(y: np.ndarray, n_frames: int, sub: int = SUB) -> np.ndarray:
    """True for frames of `sub` x 10 ms with no active speech (same rule as prosody.extract)."""
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0] + 1e-8)
    active = median_filter(db > max(np.percentile(db, 10) + 10, db.max() - 40), size=5)
    active = np.pad(active, (1, n_frames * sub + 1))            # one 10 ms frame of slack each side
    return np.array([not active[f * sub: f * sub + sub + 2].any() for f in range(n_frames)])


def _forced(lp: np.ndarray, ids: list[int]):
    labels, scores = F.forced_align(torch.from_numpy(np.ascontiguousarray(lp, dtype=np.float32))[None],
                                    torch.tensor([ids], dtype=torch.int32), blank=BLANK)
    return F.merge_tokens(labels[0], scores[0].exp(), blank=BLANK)


def align(tr: Transcript, text: str | None = None, y: np.ndarray | None = None, pad_frames: float = 0.0) -> list[Word]:
    """Word timings for `text` (default: the transcript itself) against the CTC log-probs.

    y: the audio the transcript came from; enables the silence constraint (recommended).
    pad_frames: extend each word into neighbouring blank frames by up to this many frames.
    """
    ids = tokenize(text, tr.tokens) if text is not None else greedy_ids(tr)
    lp = tr.logprobs
    if not ids or lp.shape[0] < len(ids):
        return []
    spans = None
    if y is not None:
        silent = silent_frames(y, lp.shape[0])
        if (~silent).sum() >= 2 * len(ids):                    # enough speech frames to fit the text
            masked = lp.copy()
            masked[silent, :BLANK] = -1e4
            try:
                spans = _forced(masked, ids)
            except RuntimeError:                               # infeasible: fall back to no constraint
                spans = None
    if spans is None:
        spans = _forced(lp, ids)

    words, cur = [], None
    for sp in spans:
        tok = tr.tokens[sp.token]
        if tok.startswith(WORD_START) or cur is None:
            if cur:
                words.append(cur)
            cur = {"toks": [], "start": sp.start, "end": sp.end, "scores": []}
        cur["toks"].append(tok); cur["end"] = sp.end; cur["scores"].append(sp.score)
    words.append(cur)

    n_frames = lp.shape[0]
    starts = [float(w["start"]) for w in words]
    ends = [float(w["end"]) for w in words]
    if pad_frames:
        for i in range(len(words)):
            left = starts[i] - (ends[i - 1] if i else 0)
            right = (starts[i + 1] if i + 1 < len(words) else n_frames) - ends[i]
            starts[i] -= min(left / 2 if i else left, pad_frames)
            ends[i] += min(right / 2 if i + 1 < len(words) else right, pad_frames)
    return [Word("".join(w["toks"]).replace(WORD_START, ""), round(s * FRAME_S, 3), round(e * FRAME_S, 3),
                 float(np.log(np.mean(w["scores"]) + 1e-9)))
            for w, s, e in zip(words, starts, ends)]


class CharAligner:
    """Character-level wav2vec2 CTC for one language; places given words in time at 20 ms."""
    FRAME = 0.02

    def __init__(self, lang: str, device: str = "cuda"):
        import json
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        repo = config()["models"]["aligner"][lang]
        path = ROOT / "models" / repo.split("/")[-1]
        if not (path / "config.json").exists():
            snapshot_download(repo, local_dir=path, allow_patterns=["*.json", "pytorch_model.bin"],
                              ignore_patterns=["language_model/*"])
        self.device = torch.device(device)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).to(self.device).eval()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
        self.vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        self.blank = self.vocab.get("<pad>", self.vocab.get("[PAD]"))
        self.sep = self.vocab["|"]

    @staticmethod
    def available(lang: str) -> bool:
        return lang in (config()["models"].get("aligner") or {})

    def logprobs(self, y: np.ndarray) -> np.ndarray:
        x = self.fe(y, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(x).logits[0].float()
        return torch.log_softmax(logits, -1).cpu().numpy()

    def align(self, y: np.ndarray, words: list[str], mask_silence: bool = True, close_gaps: bool = True) -> list[Word]:
        """close_gaps: in continuous speech two words meet at one instant, but CTC leaves the word
        separator's frames between them; put the boundary in the middle unless the gap is silent."""
        lp = self.logprobs(y)
        silent = silent_frames(y, lp.shape[0], sub=int(self.FRAME / 0.01))
        ids, owner = [], []                                   # owner[k] = word index of target k, -1 for "|"
        for i, w in enumerate(words):
            chars = [self.vocab[c] for c in unicodedata.normalize("NFC", w) if c in self.vocab and c != "|"]
            if not chars:
                continue
            if ids:
                ids.append(self.sep); owner.append(-1)
            ids += chars; owner += [i] * len(chars)
        if not ids or lp.shape[0] < 2 * len(ids):
            return []
        if mask_silence:
            if (~silent).sum() >= len(ids):
                lp = lp.copy()
                lp[np.ix_(silent, [k for k in range(lp.shape[1]) if k not in (self.blank, self.sep)])] = -1e4
        labels, scores = F.forced_align(torch.from_numpy(np.ascontiguousarray(lp, dtype=np.float32))[None],
                                        torch.tensor([ids], dtype=torch.int32), blank=self.blank)
        spans = F.merge_tokens(labels[0], scores[0].exp(), blank=self.blank)
        if len(spans) != len(ids):
            return []
        span_of = {}
        for k, sp in enumerate(spans):
            if owner[k] >= 0:
                span_of.setdefault(owner[k], []).append(sp)
        frames = [[sps[0].start, sps[-1].end] if (sps := span_of.get(i)) else None for i in range(len(words))]
        if close_gaps:
            real = [i for i, f in enumerate(frames) if f]
            for a, b in zip(real, real[1:]):
                gap = range(frames[a][1], frames[b][0])
                if len(gap) and not silent[gap.start:gap.stop].any():
                    mid = (frames[a][1] + frames[b][0]) / 2
                    frames[a][1] = frames[b][0] = mid
        out, last_end = [], 0.0
        for i, w in enumerate(words):
            if not frames[i]:                                 # nothing alignable (digits, Latin): zero-length
                out.append(Word(w, last_end, last_end, float("nan"))); continue
            start, end = frames[i][0] * self.FRAME, frames[i][1] * self.FRAME
            out.append(Word(w, round(start, 3), round(end, 3),
                            float(np.log(np.mean([s.score for s in span_of[i]]) + 1e-9))))
            last_end = out[-1].end
        return out
