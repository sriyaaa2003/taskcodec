import torch

from taskcodec.quant import quantize, total_bits

G, d, T = 4, 6, 5


def _z(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor([0.5, 1.0, 2.0, 3.0])[None, None, :, None]
    return torch.randn(n, T, G, d, generator=g) * scale


def _stats(z):
    return z.mean((0, 1)), z.std((0, 1))


def test_levels_at_most_2_pow_b():
    z = _z()
    mu, sg = _stats(z)
    bits = torch.tensor([1, 2, 3, 4])
    q = quantize(z, bits, mu, sg, ste=False)
    for g in range(G):
        for j in range(d):
            assert len(torch.unique(q[:, :, g, j])) <= 2 ** int(bits[g])


def test_error_decreases_with_bits():
    z = _z()
    mu, sg = _stats(z)
    errs = [((quantize(z, torch.full((G,), b), mu, sg, ste=False) - z) ** 2).mean().item()
            for b in range(1, 8)]
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_zero_bits_returns_mean():
    z = _z()
    mu, sg = _stats(z)
    q = quantize(z, torch.zeros(G), mu, sg, ste=False)
    assert torch.allclose(q, mu.expand_as(q), atol=1e-6)


def test_ste_gradient_is_identity():
    z = _z(8).requires_grad_(True)
    mu, sg = _stats(z.detach())
    quantize(z, torch.full((G,), 3), mu, sg).sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z))


def test_per_clip_bits_shape():
    z = _z(10)
    mu, sg = _stats(z)
    bits = torch.randint(0, 5, (10, G))
    assert quantize(z, bits, mu, sg, ste=False).shape == z.shape


def test_total_bits_counts_copies():
    assert total_bits([2, 1], [10, 20], [2, 1]) == 2 * 10 * 2 + 20
