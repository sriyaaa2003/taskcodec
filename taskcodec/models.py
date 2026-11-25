"""CNN autoencoder on normalised log-mel. Knows nothing about bit budgets."""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


def _up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, n_mels: int, frames: int, groups: int, total_dim: int, channels):
        super().__init__()
        if n_mels % 8 or frames % 8:
            raise ValueError("n_mels and frames must be multiples of 8")
        if total_dim % groups:
            raise ValueError("total_dim must be divisible by groups")
        c0, c1, c2 = channels
        self.convs = nn.Sequential(_block(1, c0, 2), _block(c0, c0, 1), _block(c0, c1, 2),
                                   _block(c1, c1, 1), _block(c1, c2, 2))
        self.proj = nn.Linear(c2 * (n_mels // 8), total_dim)
        self.groups, self.d = groups, total_dim // groups

    def forward(self, x):                      # x: (B, n_mels, frames) normalised
        h = self.convs(x.unsqueeze(1))         # (B, C, F', T')
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)
        return self.proj(h).reshape(b, t, self.groups, self.d)


class Decoder(nn.Module):
    def __init__(self, n_mels: int, frames: int, groups: int, total_dim: int, channels):
        super().__init__()
        c0, c1, c2 = channels
        self.c2, self.f = c2, n_mels // 8
        self.proj = nn.Linear(total_dim, c2 * self.f)
        self.ups = nn.Sequential(nn.ReLU(inplace=True), _up(c2, c1), _block(c1, c1, 1),
                                 _up(c1, c0), _block(c0, c0, 1),
                                 nn.ConvTranspose2d(c0, 1, 4, 2, 1))

    def forward(self, z):                      # z: (B, T', G, d)
        b, t = z.shape[:2]
        h = self.proj(z.reshape(b, t, -1)).reshape(b, t, self.c2, self.f)
        h = h.permute(0, 2, 3, 1)              # (B, C, F', T')
        return self.ups(h).squeeze(1)          # (B, n_mels, frames)


class Autoencoder(nn.Module):
    def __init__(self, n_mels: int, frames: int, groups: int, total_dim: int, channels=(32, 64, 96)):
        super().__init__()
        self.encoder = Encoder(n_mels, frames, groups, total_dim, channels)
        self.decoder = Decoder(n_mels, frames, groups, total_dim, channels)
        self.groups, self.d = groups, total_dim // groups

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))
