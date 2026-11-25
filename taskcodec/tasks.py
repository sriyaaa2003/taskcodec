"""Small CNN classifiers for keyword spotting and speaker ID; trained on clean mel, then frozen."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskCNN(nn.Module):
    def __init__(self, n_classes: int, channels=(32, 64, 96, 128), dropout: float = 0.3):
        super().__init__()
        layers, cin = [], 1
        for i, c in enumerate(channels):
            layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            cin = c
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * cin, n_classes)

    def embed(self, x):                        # (B, n_mels, frames) normalised -> pooled features
        h = self.features(x.unsqueeze(1))
        return torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def train_classifier(model: TaskCNN, x_train, y_train, x_val, y_val, cfg: dict, seed: int,
                     log=print) -> dict:
    """x_*: normalised log-mel on the training device. Keeps the best-validation checkpoint."""
    t = cfg["tasks"]
    dev = x_train.device
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps = t["epochs"] * max(1, len(x_train) // t["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=steps)
    best, best_state = -1.0, None
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(t["epochs"]):
        model.train()
        perm = torch.randperm(len(x_train), generator=g).to(dev)
        for i in range(0, len(perm) - t["batch_size"] + 1, t["batch_size"]):
            idx = perm[i:i + t["batch_size"]]
            xb, yb = x_train[idx], y_train[idx]
            if t["shift_frames"]:
                s = int(torch.randint(-t["shift_frames"], t["shift_frames"] + 1, (1,), generator=g))
                xb = torch.roll(xb, s, dims=-1)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN loss in classifier training (seed {seed})")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc = (predict(model, x_val).argmax(1) == y_val).float().mean().item()
        log(f"  epoch {ep + 1}/{t['epochs']} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"best_val_acc": best}
