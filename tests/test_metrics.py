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


# --------------------------------------------------------------------------
# Phase 3 metric additions (sub-plan sec 0.2)
# --------------------------------------------------------------------------


def test_wmape_horizon_hand_computed():
    """One series, one fold: totals 10 actual vs 8 forecast -> |10-8|/10 = 0.2.

    Daily WMAPE on the same rows is (|1-4|+|2-2|+|3-1|+|4-1|)/10 = 8/10 = 0.8,
    so this also demonstrates the level-vs-timing decomposition: 0.2 of the
    error is level, the remaining 0.6 is within-window timing.
    """
    actual = [1.0, 2.0, 3.0, 4.0]
    forecast = [4.0, 2.0, 1.0, 1.0]
    ids = ["s1"] * 4
    folds = [0] * 4
    assert m.wmape_horizon(actual, forecast, ids, folds) == pytest.approx(0.2)
    assert m.wmape(actual, forecast) == pytest.approx(0.8)


def test_wmape_horizon_separates_series_and_folds():
    """Errors must not cancel ACROSS series or ACROSS folds, only within a block."""
    # two series, opposite-signed errors that would cancel if pooled together
    actual = [5.0, 5.0]
    forecast = [7.0, 3.0]
    same_block = m.wmape_horizon(actual, forecast, ["s1", "s1"], [0, 0])
    diff_series = m.wmape_horizon(actual, forecast, ["s1", "s2"], [0, 0])
    diff_folds = m.wmape_horizon(actual, forecast, ["s1", "s1"], [0, 1])
    assert same_block == pytest.approx(0.0)      # +2 and -2 cancel within a block
    assert diff_series == pytest.approx(0.4)     # 4/10 -- no cancelling
    assert diff_folds == pytest.approx(0.4)


def test_wmape_horizon_never_exceeds_daily_wmape():
    """The triangle-inequality invariant from sub-plan sec 0.2, as a live check.

    |sum of errors| <= sum of |errors|, and the denominator is shared, so
    WMAPE_28 <= daily WMAPE for EVERY model. This is why a lower number here is
    not the argument for the metric -- the zero line staying at 1.0 is.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 200
        actual = rng.poisson(0.7, n).astype(float)
        forecast = rng.gamma(2.0, 0.4, n)
        ids = rng.integers(0, 7, n).astype(str)
        folds = rng.integers(0, 5, n)
        if actual.sum() == 0:
            continue
        assert m.wmape_horizon(actual, forecast, ids, folds) <= m.wmape(actual, forecast) + 1e-12


def test_zero_forecast_scores_exactly_one_under_both_metrics():
    """The bar is unchanged by the metric switch -- the point of sub-plan sec 0.2."""
    rng = np.random.default_rng(1)
    actual = rng.poisson(0.4, 300).astype(float)
    zero = np.zeros_like(actual)
    ids = rng.integers(0, 9, 300).astype(str)
    folds = rng.integers(0, 5, 300)
    assert m.wmape(actual, zero) == pytest.approx(1.0)
    assert m.wmape_horizon(actual, zero, ids, folds) == pytest.approx(1.0)


def test_oracle_lines_hand_computed():
    """Series of [0,0,0,4]: median 0 -> WMAPE 1.0; mean 1 -> (1+1+1+3)/4 = 1.5."""
    actual = [0.0, 0.0, 0.0, 4.0]
    o = m.oracle_lines(actual, ["s1"] * 4, [0] * 4)
    assert o["oracle_median"] == pytest.approx(1.0)
    assert o["oracle_mean"] == pytest.approx(1.5)


def test_oracle_median_is_one_when_every_series_median_is_zero():
    """The dataset-C situation: no constant point forecast can beat doing nothing."""
    actual = np.array([0.0] * 9 + [5.0] * 1)
    o = m.oracle_lines(actual, ["s1"] * 10, [0] * 10)
    assert o["oracle_median"] == pytest.approx(1.0)


def test_pinball_loss_hand_computed():
    # q=0.9, actual 10, forecast 8 -> under-forecast -> 0.9 * 2 = 1.8
    assert m.pinball_loss([10.0], [8.0], 0.9) == pytest.approx(1.8)
    # q=0.9, actual 8, forecast 10 -> over-forecast -> 0.1 * 2 = 0.2
    assert m.pinball_loss([8.0], [10.0], 0.9) == pytest.approx(0.2)
    # q=0.5 is half the absolute error
    assert m.pinball_loss([10.0, 4.0], [8.0, 8.0], 0.5) == pytest.approx(1.5)


def test_pinball_loss_minimised_at_the_true_quantile():
    """Sanity: the loss a quantile model minimises really is minimised there."""
    rng = np.random.default_rng(3)
    sample = rng.normal(5.0, 2.0, 20000)
    q = 0.9
    truth = np.quantile(sample, q)
    at_truth = m.pinball_loss(sample, np.full_like(sample, truth), q)
    for off in (-0.5, 0.5):
        assert m.pinball_loss(sample, np.full_like(sample, truth + off), q) > at_truth


def test_pinball_rejects_out_of_range_quantile():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            m.pinball_loss([1.0], [1.0], bad)


def test_coverage_counts_ties_as_covered():
    """A q10 forecast of 0 does cover an actual of 0 -- demand is a zero-heavy count."""
    assert m.coverage([0.0, 0.0, 3.0], [0.0, 0.0, 0.0]) == pytest.approx(2 / 3)
    assert m.coverage([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == pytest.approx(1.0)


def test_crossing_rate_and_monotone_repair():
    preds = {
        0.1: np.array([1.0, 5.0, 0.0]),
        0.5: np.array([2.0, 3.0, 0.0]),   # row 1 crosses (5 > 3)
        0.9: np.array([3.0, 4.0, 0.0]),
    }
    assert m.crossing_rate(preds) == pytest.approx(1 / 3)
    fixed = m.monotone_quantiles(preds)
    assert m.crossing_rate(fixed) == pytest.approx(0.0)
    # repair is a per-row sort: the same values, reassigned in order
    assert fixed[0.1][1] == 3.0 and fixed[0.5][1] == 4.0 and fixed[0.9][1] == 5.0
    # non-crossing rows are untouched
    assert fixed[0.1][0] == 1.0 and fixed[0.9][0] == 3.0
