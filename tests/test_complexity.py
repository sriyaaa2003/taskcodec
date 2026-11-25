import torch
import torch.nn as nn

from taskcodec.complexity import count_macs, count_params


def test_conv_macs_hand_computed():
    m = nn.Conv2d(3, 8, 3, padding=1)
    assert count_macs(m, torch.zeros(1, 3, 10, 10)) == 8 * 10 * 10 * 3 * 9


def test_linear_and_params():
    m = nn.Linear(7, 5)
    assert count_macs(m, torch.zeros(2, 7)) == 35
    assert count_params(m) == 7 * 5 + 5


def test_convtranspose_macs():
    m = nn.ConvTranspose2d(4, 2, 4, 2, 1)
    assert count_macs(m, torch.zeros(1, 4, 5, 5)) == 4 * 5 * 5 * 2 * 16
