"""Waveform <-> log-mel features (pure torch, no torchaudio) and train-only normalisation."""
from __future__ import annotations

import math

import torch


def _hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(audio: dict) -> torch.Tensor:
    """(n_mels, n_fft//2+1) triangular filters on the HTK mel scale."""
    sr, n_fft, n_mels = audio["sample_rate"], audio["n_fft"], audio["n_mels"]
    fmin, fmax = audio["fmin"], audio["fmax"]
    pts = torch.tensor([_mel_to_hz(_hz_to_mel(fmin) + i * (_hz_to_mel(fmax) - _hz_to_mel(fmin))
                                   / (n_mels + 1)) for i in range(n_mels + 2)])
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = torch.zeros(n_mels, len(freqs))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


def _stft(wav: torch.Tensor, audio: dict) -> torch.Tensor:
    win = torch.hann_window(audio["win_length"], device=wav.device)
    return torch.stft(wav, audio["n_fft"], audio["hop_length"], audio["win_length"], win,
                      center=True, return_complex=True)


def wav_to_logmel(wav: torch.Tensor, audio: dict, fb: torch.Tensor | None = None) -> torch.Tensor:
    """wav (B, L) float in [-1, 1] -> natural-log power mel (B, n_mels, frames), edge-padded."""
    fb = (mel_filterbank(audio) if fb is None else fb).to(wav.device)
    power = _stft(wav, audio).abs() ** 2                  # (B, F, T)
    mel = torch.einsum("mf,bft->bmt", fb, power)
    lm = torch.log(mel + audio["log_eps"])
    pad = audio["frames"] - lm.shape[-1]
    if pad < 0:
        raise ValueError(f"frames={audio['frames']} < available {lm.shape[-1]}")
    if pad:
        lm = torch.cat([lm, lm[..., -1:].expand(-1, -1, pad)], dim=-1)
    return lm


class Normalizer:
    """Per-mel-bin mean/std. `fit` must only ever be given TRAIN features."""

    def __init__(self, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.mean, self.std = mean, std

    def fit(self, train_logmel: torch.Tensor) -> "Normalizer":
        self.mean = train_logmel.mean(dim=(0, 2), keepdim=True)[0]       # (n_mels, 1)
        self.std = train_logmel.std(dim=(0, 2), keepdim=True)[0].clamp_min(1e-5)
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, s: dict) -> "Normalizer":
        return cls(s["mean"], s["std"])


def logmel_to_wav(logmel: torch.Tensor, audio: dict, iters: int | None = None,
                  fb: torch.Tensor | None = None, seed: int = 0) -> torch.Tensor:
    """Griffin-Lim vocoding of a log-mel (B, n_mels, frames incl. padding) -> (B, clip_samples)."""
    iters = audio["gl_iters"] if iters is None else iters
    fb = (mel_filterbank(audio) if fb is None else fb).to(logmel.device)
    n_frames = 1 + audio["clip_samples"] // audio["hop_length"]
    lm = logmel[..., :n_frames]
    mel = torch.exp(lm.clamp(-20.0, 20.0)) - audio["log_eps"]
    power = torch.einsum("fm,bmt->bft", torch.linalg.pinv(fb), mel.clamp_min(0.0)).clamp_min(0.0)
    mag = torch.sqrt(power)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    phase = torch.rand(mag.shape, generator=gen).to(mag.device) * 2 * math.pi
    spec = torch.polar(mag, phase)
    win = torch.hann_window(audio["win_length"], device=mag.device)
    kw = dict(n_fft=audio["n_fft"], hop_length=audio["hop_length"],
              win_length=audio["win_length"], window=win, center=True)
    for _ in range(iters):
        wav = torch.istft(spec, length=audio["clip_samples"], **kw)
        est = torch.stft(wav, return_complex=True, **kw)
        spec = torch.polar(mag, torch.angle(est))
    return torch.istft(spec, length=audio["clip_samples"], **kw)
