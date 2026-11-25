import torch

from taskcodec.config import load_config
from taskcodec.features import Normalizer, logmel_to_wav, mel_filterbank, wav_to_logmel

A = load_config()["audio"]


def test_shape_and_filterbank():
    fb = mel_filterbank(A)
    assert fb.shape == (A["n_mels"], A["n_fft"] // 2 + 1)
    assert (fb.sum(1) > 0).all()
    x = wav_to_logmel(torch.randn(3, A["clip_samples"]) * 0.1, A)
    assert x.shape == (3, A["n_mels"], A["frames"])


def test_normalizer_uses_train_stats_only():
    train = torch.randn(50, 40, 104)
    test = torch.randn(50, 40, 104) + 5.0
    nz = Normalizer().fit(train)
    mean_before = nz.mean.clone()
    out = nz.apply(test)
    assert torch.equal(nz.mean, mean_before)         # applying never refits
    assert out.mean() > 4.0                          # the test shift is NOT removed
    assert torch.allclose(nz.invert(nz.apply(test)), test, atol=1e-4)


def test_griffin_lim_runs():
    w = torch.sin(torch.arange(A["clip_samples"]) * 0.1)[None] * 0.3
    lm = wav_to_logmel(w, A)
    rec = logmel_to_wav(lm, A, iters=5)
    assert rec.shape == w.shape and torch.isfinite(rec).all()
