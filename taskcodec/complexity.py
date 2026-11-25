"""Parameter and multiply-accumulate counting for conv / linear layers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_macs(model: nn.Module, *inputs) -> int:
    """MACs of one forward pass of conv/linear layers (norms and activations not counted)."""
    total = [0]
    hooks = []

    def conv_hook(m, inp, out):
        k = int(np.prod(m.kernel_size))
        total[0] += int(out.numel() // out.shape[0]) * (m.in_channels // m.groups) * k

    def convt_hook(m, inp, out):
        k = int(np.prod(m.kernel_size))
        x = inp[0]
        total[0] += int(x.numel() // x.shape[0]) * (m.out_channels // m.groups) * k

    def lin_hook(m, inp, out):
        total[0] += int(out.numel() // out.shape[0]) * m.in_features

    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(convt_hook))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(*inputs)
    model.train(was_training)
    for h in hooks:
        h.remove()
    return total[0]
