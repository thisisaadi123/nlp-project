"""IndicConformer ASR (CTC branch) on onnxruntime, keeping the CTC log-probs for word timings.

The model ships as ONNX (no NeMo needed). We load only what CTC decoding uses: the TorchScript
feature extractor, the encoder and the CTC head, then mask the shared 5633-token output down to
one language's 256 tokens + blank. The model is copied as plain files into models/ because
onnxruntime refuses external weight files that live outside the model folder (the HF cache
stores them as symlinks into blobs/).
"""
import json
import unicodedata
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download

from backend.settings import ROOT, config

FRAME_S = 0.08                      # one encoder frame = 80 ms (8x subsampling of 10 ms features)
BLANK = 256                         # index of the CTC blank within each language's masked vocab
SR = 16000


@dataclass
class Transcript:
    text: str
    logprobs: np.ndarray            # (frames, 257) CTC log-probs for the language, blank last
    tokens: list[str]               # the language's vocabulary; "▁" marks a word start


def model_dir():
    return ROOT / "models" / config()["models"]["asr"].split("/")[-1]


class ASR:
    def __init__(self, device: str = "cuda", gpu_mem_limit_gb: float | None = None):
        ort.preload_dlls()          # use the CUDA/cuDNN libraries that ship with torch
        path = model_dir()
        if not (path / "assets/encoder.onnx").exists():
            snapshot_download(config()["models"]["asr"], local_dir=path,
                              allow_patterns=["assets/*", "*.json"],
                              ignore_patterns=["assets/joint_*", "assets/rnnt_*"])
        a = path / "assets"
        self.device = torch.device(device)
        self.pre = torch.jit.load(str(a / "preprocessor.ts"), map_location=self.device)

        cuda = {"arena_extend_strategy": "kSameAsRequested"}
        if gpu_mem_limit_gb:
            cuda["gpu_mem_limit"] = int(gpu_mem_limit_gb * 2**30)
        providers = [("CUDAExecutionProvider", cuda), "CPUExecutionProvider"] if device == "cuda" \
            else ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # hide "Memcpy nodes added" notices
        self.encoder = ort.InferenceSession(str(a / "encoder.onnx"), opts, providers=providers)
        self.ctc = ort.InferenceSession(str(a / "ctc_decoder.onnx"), opts, providers=providers)
        self.vocab = json.loads((a / "vocab.json").read_text())
        self._masks = json.loads((a / "language_masks.json").read_text())

    @cached_property
    def _mask_idx(self) -> dict[str, np.ndarray]:
        return {lang: np.flatnonzero(m) for lang, m in self._masks.items()}

    def logprobs(self, y: np.ndarray, lang: str) -> np.ndarray:
        """CTC log-probs (frames, 257) for 16 kHz mono float audio."""
        wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None].to(self.device)
        with torch.no_grad():
            feats, n = self.pre(input_signal=wav, length=torch.tensor([wav.shape[-1]], device=self.device))
        enc, enc_len = self.encoder.run(["outputs", "encoded_lengths"],
                                        {"audio_signal": feats.cpu().numpy(), "length": n.cpu().numpy()})
        lp = self.ctc.run(["logprobs"], {"encoder_output": enc})[0][0, : int(enc_len[0])]
        lp = lp[:, self._mask_idx[lang]]
        return lp - np.logaddexp.reduce(lp, axis=-1, keepdims=True)    # log_softmax

    def transcribe(self, y: np.ndarray, lang: str) -> Transcript:
        lp = self.logprobs(y, lang)
        path = lp.argmax(-1)
        keep = np.concatenate([[True], path[1:] != path[:-1]]) & (path != BLANK)
        tokens = self.vocab[lang]
        text = "".join(tokens[i] for i in path[keep]).replace("▁", " ").strip()
        return Transcript(unicodedata.normalize("NFC", text), lp, tokens)
