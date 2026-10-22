"""Word start/end times from IndicConformer's own CTC log-probs (torchaudio forced alignment).

No separate aligner model: the MMS aligner weights are CC-BY-NC. Resolution is one encoder
frame (80 ms). CTC emits each token as a short spike, so a word's raw span covers only its
spikes; `pad` extends each word into the blank frames around it, up to half the gap to its
neighbour, which is closer to where the sound actually starts and stops.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio.functional as F

from backend.pipeline.speech.asr import BLANK, FRAME_S, Transcript

WORD_START = "▁"


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


def align(tr: Transcript, text: str | None = None, pad: bool = True) -> list[Word]:
    """Word timings for `text` (default: the transcript itself) against the CTC log-probs."""
    ids = tokenize(text, tr.tokens) if text is not None else greedy_ids(tr)
    if not ids:
        return []
    lp = torch.from_numpy(np.ascontiguousarray(tr.logprobs, dtype=np.float32))[None]
    if lp.shape[1] < len(ids):                     # audio too short for the text
        return []
    labels, scores = F.forced_align(lp, torch.tensor([ids], dtype=torch.int32), blank=BLANK)
    spans = F.merge_tokens(labels[0], scores[0].exp(), blank=BLANK)

    words, cur = [], None
    for sp in spans:
        tok = tr.tokens[sp.token]
        if tok.startswith(WORD_START) or cur is None:
            if cur:
                words.append(cur)
            cur = {"toks": [], "start": sp.start, "end": sp.end, "scores": []}
        cur["toks"].append(tok); cur["end"] = sp.end; cur["scores"].append(sp.score)
    words.append(cur)

    n_frames = tr.logprobs.shape[0]
    starts = [w["start"] for w in words]
    ends = [w["end"] for w in words]
    if pad:
        for i in range(len(words)):
            left_gap = starts[i] - (ends[i - 1] if i else 0)
            right_gap = (starts[i + 1] if i + 1 < len(words) else n_frames) - ends[i]
            starts[i] -= min(left_gap / 2 if i else left_gap, 1.0)       # at most one frame early
            ends[i] += min(right_gap / 2 if i + 1 < len(words) else right_gap, 2.0)
    return [Word("".join(w["toks"]).replace(WORD_START, ""), round(s * FRAME_S, 3), round(e * FRAME_S, 3),
                 float(np.log(np.mean(w["scores"]) + 1e-9)))
            for w, s, e in zip(words, starts, ends)]
