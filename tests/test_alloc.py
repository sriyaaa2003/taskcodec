import inspect

import numpy as np
import pytest

from taskcodec import alloc
from taskcodec.alloc import InfeasibleBudget, allocate

SIZES = np.array([100, 100, 100, 100])


def used(b, sizes=SIZES):
    return int((b * sizes).sum())


@pytest.mark.parametrize("policy", ["uniform", "variance", "task_kws", "random"])
def test_budget_respected_and_maximal(policy):
    sens = np.array([4.0, 2.0, 1.0, 0.5])
    for budget in [0, 100, 450, 800, 1600]:
        b = allocate(policy, sens, budget, SIZES, seed=3)
        assert used(b) <= budget
        if (b < 8).any():
            assert budget - used(b) < SIZES.min()    # nothing else fits: budget used exactly


def test_exact_when_divisible():
    b = allocate("task_kws", np.array([4.0, 2.0, 1.0, 0.5]), 1300, SIZES)
    assert used(b) == 1300


def test_monotone_in_budget():
    sens = np.array([4.0, 2.0, 1.0, 0.5])
    prev = allocate("task_kws", sens, 0, SIZES)
    for budget in range(100, 2000, 100):
        cur = allocate("task_kws", sens, budget, SIZES)
        assert (cur >= prev).all()
        prev = cur


def test_uniform_on_equal_sensitivities():
    assert (allocate("task_kws", np.ones(4), 800, SIZES) == 2).all()
    assert (allocate("uniform", None, 800, SIZES) == 2).all()


def test_high_sensitivity_gets_more_bits():
    b = allocate("task_kws", np.array([8.0, 1.0, 1.0, 1.0]), 800, SIZES)
    assert b[0] > b[1]


def test_deterministic_given_seed():
    a = allocate("random", None, 900, SIZES, seed=7)
    b = allocate("random", None, 900, SIZES, seed=7)
    c = allocate("random", None, 900, SIZES, seed=8)
    assert (a == b).all() and not (a == c).all()


def test_raises_below_minimum():
    with pytest.raises(InfeasibleBudget):
        allocate("uniform", None, 399, SIZES, b_min=1)


def test_bmax_cap():
    b = allocate("task_kws", np.array([100.0, 1, 1, 1]), 10000, SIZES, b_max=3)
    assert b.max() <= 3


def test_group_sensitivity_signature_validation_only():
    params = inspect.signature(alloc.group_sensitivity).parameters
    assert "z_val" in params and "y_val" in params
    assert not any("test" in p for p in params)
