"""Who spoke when: pyannote speaker-diarization (pyannote.audio 4).

pyannote.audio 4 also loads the PLDA from pyannote/speaker-diarization-community-1, so the HF
account needs access to that repo too (scripts/check_setup.py checks it). Audio is passed as a
waveform so nothing is decoded twice.
"""
from dataclasses import dataclass

import numpy as np
import torch

from backend.settings import config, secret

SR = 16000


@dataclass
class Turn:
    start: float
    end: float
    speaker: str        # "S1", "S2", ... in order of first appearance


class Diarizer:
    def __init__(self, device: str = "cuda"):
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(config()["models"]["diarization"], token=secret("HF_TOKEN"))
        self.pipeline.to(torch.device(device))

    def __call__(self, y: np.ndarray, num_speakers: int | None = None,
                 min_speakers: int | None = None, max_speakers: int | None = None) -> list[Turn]:
        kw = {k: v for k, v in dict(num_speakers=num_speakers, min_speakers=min_speakers,
                                    max_speakers=max_speakers).items() if v}
        out = self.pipeline({"waveform": torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None],
                             "sample_rate": SR}, **kw)
        ann = getattr(out, "speaker_diarization", out)
        names, turns = {}, []
        for seg, _, label in ann.itertracks(yield_label=True):
            names.setdefault(label, f"S{len(names) + 1}")
            turns.append(Turn(round(seg.start, 3), round(seg.end, 3), names[label]))
        return sorted(turns, key=lambda t: t.start)
