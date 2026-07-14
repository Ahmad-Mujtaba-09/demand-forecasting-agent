"""Phase 2 Increment 1 tests: WMAPE with hand-computed answers.

Every expected value is worked out by hand so a regression is unambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfa import metrics as m


def test_wmape_known():
    # |1-2|+|3-3|+|4-2| = 3 ; sum(actual)=8 -> 0.375
    a = np.array([1, 3, 4.0])
    f = np.array([2, 3, 2.0])
    assert m.wmape(a, f) == pytest.approx(3 / 8)


def test_wmape_perfect_is_zero():
    a = np.array([0, 5, 2.0])
    assert m.wmape(a, a) == 0.0


def test_wmape_zero_denominator_is_nan():
    # all-zero actuals -> denominator 0 -> undefined, not a divide error
    assert np.isnan(m.wmape(np.zeros(4), np.array([1, 0, 2, 0.0])))


def test_wmape_shape_mismatch_raises():
    with pytest.raises(ValueError):
        m.wmape(np.array([1.0, 2]), np.array([1.0]))


def test_wmape_by_group_known_split():
    a = np.array([10, 10, 5, 5.0])
    f = np.array([8, 10, 0, 10.0])
    g = np.array(["x", "x", "y", "y"])
    out = m.wmape_by_group(a, f, g)
    # x: (|10-8|+0)/20 = 0.1 ; y: (5+5)/10 = 1.0
    assert out["x"]["wmape"] == pytest.approx(0.1)
    assert out["y"]["wmape"] == pytest.approx(1.0)
    assert out["x"]["n_obs"] == 2 and out["y"]["denom"] == pytest.approx(10.0)


def test_wmape_by_group_zero_denom_group_is_nan():
    a = np.array([0, 0, 4.0])
    f = np.array([1, 2, 4.0])
    g = np.array(["z", "z", "w"])
    out = m.wmape_by_group(a, f, g)
    assert np.isnan(out["z"]["wmape"])
    assert out["w"]["wmape"] == 0.0


def test_volume_terciles_three_nonempty_buckets():
    ids = [f"s{i}" for i in range(9)]
    vols = list(range(9))  # strictly increasing volume
    t = m.volume_terciles(ids, vols)
    assert t["s0"] == "low" and t["s4"] == "mid" and t["s8"] == "high"
    counts = {b: sum(v == b for v in t.values()) for b in ("low", "mid", "high")}
    assert counts == {"low": 3, "mid": 3, "high": 3}


def test_volume_terciles_ties_still_split():
    # many identical volumes must still fall into non-empty buckets (rank-first)
    ids = [f"s{i}" for i in range(6)]
    t = m.volume_terciles(ids, [5, 5, 5, 5, 5, 5])
    assert set(t.values()) == {"low", "mid", "high"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
