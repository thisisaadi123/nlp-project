"""Music removal: Demucs (htdemucs), keeping only the vocals stem.

Demucs works on 44.1 kHz stereo; our audio is 16 kHz mono, so it is resampled in and out.
Whether to run it at all is decided by the step 3 measurement (docs/reports/step3.md).
"""
import numpy as np
import torch
import torchaudio

from backend.settings import config

SR = 16000


class Separator:
    def __init__(self, device: str = "cuda"):
        from demucs.pretrained import get_model
        self.device = torch.device(device)
        self.model = get_model(config()["models"]["separation"]).to(self.device).eval()

    def __call__(self, y: np.ndarray) -> np.ndarray:
        from demucs.apply import apply_model
        msr = self.model.samplerate
        x = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), SR, msr)
        x = x[None].repeat(2, 1)[None].to(self.device)                # (batch, 2 channels, time)
        ref = x.mean(1, keepdim=True)
        mean, std = ref.mean(), ref.std() + 1e-8                     # the normalisation demucs' own CLI uses
        with torch.no_grad():
            out = apply_model(self.model, (x - mean) / std, split=True, overlap=0.25, progress=False)[0]
        vocals = out[self.model.sources.index("vocals")].mean(0) * std + mean
        return torchaudio.functional.resample(vocals.cpu(), msr, SR).numpy()[: len(y)]
