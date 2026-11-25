import numpy as np

from taskcodec.config import config_hash, load_config
from taskcodec.data import build_sid_subset, official_split


def test_official_split():
    s = official_split(["a/1", "a/2", "b/3"], {"a/2"}, {"b/3"})
    assert list(s) == [0, 1, 2]


def test_sid_subset_topk_and_disjoint():
    sp = np.array(["x"] * 30 + ["y"] * 20 + ["z"] * 5)
    lab, split = build_sid_subset(list(sp), 2, [0.7, 0.15, 0.15], 0)
    assert set(lab[sp == "z"]) == {-1} and set(split[sp == "z"]) == {-1}
    assert set(lab[sp == "x"]) | set(lab[sp == "y"]) == {0, 1}
    for s in ("x", "y"):
        assert {0, 1, 2} <= set(split[sp == s])
    _, split2 = build_sid_subset(list(sp), 2, [0.7, 0.15, 0.15], 0)
    assert (split == split2).all()


def test_config_hash_changes():
    a = load_config("e1")
    b = load_config("e1", {"seed": 1})
    assert config_hash(a) != config_hash(b)
    assert config_hash(a) == config_hash(load_config("e1"))
