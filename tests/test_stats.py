import numpy as np

from taskcodec.stats import bootstrap_ci, paired_bootstrap


def test_ci_covers_true_mean():
    rng = np.random.default_rng(0)
    hits = 0
    for t in range(100):
        x = rng.normal(0.7, 1.0, 400)
        _, lo, hi = bootstrap_ci(x, n_boot=500, seed=t)
        hits += lo <= 0.7 <= hi
    assert hits >= 88


def test_paired_difference_sign_and_zero():
    rng = np.random.default_rng(1)
    b = rng.normal(0, 1, 2000)
    a = b + 0.5 + rng.normal(0, 0.1, 2000)
    m, lo, hi = paired_bootstrap(a, b, n_boot=1000)
    assert m > 0.4 and lo > 0
    m2, lo2, hi2 = paired_bootstrap(b, b.copy(), n_boot=200)
    assert m2 == 0 and lo2 == hi2 == 0


def test_mean_matches():
    x = np.array([0.0, 1.0, 1.0, 1.0])
    assert bootstrap_ci(x, 100)[0] == 0.75
