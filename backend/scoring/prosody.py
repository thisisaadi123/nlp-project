"""Prosody features for one spoken line.

Everything here is chosen to be comparable across different speakers and microphones:
pitch is in semitones relative to the speaker's own median, loudness is measured as
ranges and ratios in dB rather than absolute levels.
"""
import subprocess
from dataclasses import dataclass, field

import librosa
import numpy as np
import parselmouth
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

SR = 16000
HOP_S = 0.01
HOP = int(SR * HOP_S)
FRAME = 400                 # 25 ms analysis window
MIN_PAUSE_S = 0.15
PITCH_FLOOR, PITCH_CEILING = 70.0, 700.0


@dataclass
class Prosody:
    duration: float                 # seconds from first to last speech frame
    pitch: np.ndarray               # semitones vs speaker median, unvoiced gaps interpolated
    energy: np.ndarray              # dB, per 10 ms frame across the speech span
    prominence: np.ndarray          # z(pitch) + z(energy), smoothed; peaks are stressed syllables
    pauses: list[tuple[float, float]] = field(default_factory=list)  # (start, end) seconds
    stats: dict[str, float] = field(default_factory=dict)


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Decode any audio/video file to mono float32 at `sr` using ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-6 else np.zeros_like(x)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end indices (end exclusive) of consecutive True runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def extract(y: np.ndarray, sr: int = SR) -> Prosody:
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Loudness per frame and speech detection relative to this recording's own noise floor.
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-8)
    floor, peak = np.percentile(db, 10), db.max()
    active = median_filter(db > max(floor + 10, peak - 40), size=5)

    # Pitch on the same 10 ms grid.
    snd = parselmouth.Sound(y.astype(np.float64), SR)
    pitch_obj = snd.to_pitch_ac(time_step=HOP_S, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0_raw = pitch_obj.selected_array["frequency"]
    idx = np.clip(np.round((np.arange(len(db)) * HOP_S - pitch_obj.xs()[0]) / HOP_S).astype(int),
                  0, len(f0_raw) - 1)
    f0_all = f0_raw[idx]

    # The line runs from the first to the last voiced sound, so breaths and clicks
    # before or after it (unvoiced) do not count towards its length.
    voiced_all = (f0_all > 0) & active
    if voiced_all.sum() < 5:
        raise ValueError("too little voiced speech")
    margin = 5
    first = max(np.argmax(voiced_all) - margin, 0)
    last = min(len(voiced_all) - np.argmax(voiced_all[::-1]) + margin, len(db))
    span = slice(first, last)
    active_span = active[span]
    n = last - first
    f0 = f0_all[span]
    voiced = voiced_all[span]
    if voiced.sum() < 5:
        raise ValueError("too little voiced speech")

    st = np.zeros(n)
    st[voiced] = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
    # Remove octave jumps: points far from their local median are replaced by it.
    smooth = median_filter(st[voiced], size=7)
    jumps = np.abs(st[voiced] - smooth) > 6
    st[np.flatnonzero(voiced)[jumps]] = smooth[jumps]
    vi = np.flatnonzero(voiced)
    pitch = np.interp(np.arange(n), vi, st[vi])

    energy = db[span]
    prominence = median_filter(_zscore(pitch) + _zscore(energy), size=9)

    pauses = [(s * HOP_S, e * HOP_S) for s, e in _runs(~active_span) if (e - s) * HOP_S >= MIN_PAUSE_S]
    duration = n * HOP_S
    pause_time = sum(e - s for s, e in pauses)

    # Syllable nuclei: loudness peaks in voiced speech, at least 100 ms apart.
    peaks, _ = find_peaks(np.where(voiced, median_filter(energy, size=5), energy.min()),
                          distance=10, prominence=2.0)
    articulation_time = max(duration - pause_time, 0.1)

    # Spectral tilt: high-band vs low-band energy. Shouting pushes energy upward; it is gain-invariant.
    spec = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP)) ** 2
    spec = spec[:, :len(db)][:, span][:, active_span]
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    low = spec[(freqs >= 50) & (freqs < 1000)].sum()
    high = spec[(freqs >= 1000) & (freqs < 4000)].sum()

    hnr_frames = snd.to_harmonicity_cc(time_step=HOP_S).values[0]
    hnr_frames = hnr_frames[hnr_frames > -100]

    act_energy = energy[active_span]
    stats = {
        # Absolute pitch level (semitones above 100 Hz). Only meaningful against the same
        # speaker's neutral baseline: a raised voice is a strong sign of arousal.
        "pitch_level": float(12 * np.log2(np.median(f0[voiced]) / 100)),
        "pitch_range": float(np.percentile(st[voiced], 90) - np.percentile(st[voiced], 10)),
        "pitch_sd": float(st[voiced].std()),
        "energy_range": float(np.percentile(act_energy, 95) - np.percentile(act_energy, 10)),
        "spectral_tilt": float(10 * np.log10((high + 1e-10) / (low + 1e-10))),
        "hnr": float(hnr_frames.mean()) if len(hnr_frames) else 0.0,
        "rate": float(len(peaks) / articulation_time),
        "pause_ratio": float(pause_time / duration),
        "voiced_ratio": float(voiced.sum() / max(active_span.sum(), 1)),
    }
    return Prosody(duration, pitch, energy, prominence, pauses, stats)


def extract_file(path: str) -> Prosody:
    return extract(load_audio(path))
