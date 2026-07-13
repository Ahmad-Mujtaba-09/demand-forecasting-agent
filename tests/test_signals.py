"""Increment 2 tests: signal functions on synthetic series with KNOWN answers.

Each test hand-computes the expected value so a regression is unambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfa import signals as sig


# --- intermittency ---

def test_zero_share_known():
    u = np.array([0, 5, 0, 0, 5, 0])  # 4 zeros of 6
    assert sig.zero_share(u) == pytest.approx(4 / 6)


def test_adi_known():
    u = np.array([0, 5, 0, 0, 5, 0])  # 6 periods, 2 non-zero -> 3.0
    assert sig.adi(u) == pytest.approx(3.0)


def test_adi_no_demand_is_nan():
    assert np.isnan(sig.adi(np.zeros(10)))


def test_cv_squared_known():
    # non-zero sizes [2,4,6]: mean 4, sample std 2 -> cv 0.5 -> cv^2 0.25
    u = np.array([0, 2, 4, 6, 0])
    assert sig.cv_squared(u) == pytest.approx(0.25)


def test_cv_squared_undefined_below_two_nonzero():
    # 0 or 1 non-zero observation -> dispersion is undefined, not zero
    assert np.isnan(sig.cv_squared(np.array([0, 7, 0])))
    assert np.isnan(sig.cv_squared(np.zeros(5)))


def test_cv_squared_zero_for_constant_sizes():
    # >=2 identical non-zero sizes -> genuine zero variability (distinct from NaN)
    assert sig.cv_squared(np.array([0, 3, 3, 3])) == 0.0


def test_sb_class_nan_cv2_stays_sparse_quadrant():
    # a near-single-sale series (undefined cv2, high adi) must land intermittent
    assert sig.sb_class(adi_val=50.0, cv2_val=float("nan")) == "intermittent"


def test_sb_class_quadrants():
    # smooth: adi<1.32, cv2<0.49
    assert sig.sb_class(1.1, 0.2) == "smooth"
    assert sig.sb_class(1.1, 0.8) == "erratic"
    assert sig.sb_class(2.0, 0.2) == "intermittent"
    assert sig.sb_class(2.0, 0.8) == "lumpy"
    assert sig.sb_class(float("nan"), 0.0) == "no_demand"


# --- seasonality: day-of-week variance-explained (B3 signal) ---

def test_dow_seasonality_high_for_clean_weekly_signal():
    weekly = np.array([10, 4, 4, 4, 4, 8, 12], dtype=float)  # weekend uplift
    wd = np.array([1, 2, 3, 4, 5, 6, 7])
    eta2, defined = sig.dow_seasonality(np.tile(weekly, 20), np.tile(wd, 20))
    assert defined is True and eta2 > 0.9


def test_dow_seasonality_low_for_flat_random():
    rng = np.random.default_rng(0)
    wd = np.tile(np.array([1, 2, 3, 4, 5, 6, 7]), 20)
    eta2, defined = sig.dow_seasonality(rng.integers(0, 5, 140).astype(float), wd)
    assert defined is True and eta2 < 0.2


def test_dow_seasonality_floor_derived_from_period():
    wd = np.array([1, 2, 3, 4, 5, 6, 7])
    # exactly min_cycles (2) full weekly cycles -> defined
    ok, defined = sig.dow_seasonality(np.tile([1.0, 2, 3, 4, 5, 6, 7], 2), np.tile(wd, 2))
    assert defined is True
    # one short of 2 full cycles (13 < 14) -> undefined, no spurious eta2
    _, defined_short = sig.dow_seasonality(np.arange(13.0), np.tile(wd, 2)[:13])
    assert defined_short is False
    # a non-weekly period generalizes the floor (period=4 -> needs 8)
    _, def4 = sig.dow_seasonality(np.arange(8.0), np.tile([0, 1, 2, 3], 2), period=4)
    assert def4 is True


def test_dow_seasonality_undefined_when_weekday_missing():
    # long enough by count, but not every weekday present -> undefined
    u = np.arange(20.0)
    wd = np.tile([1, 2, 3, 4, 5], 4)  # only 5 of 7 weekday levels
    _, defined = sig.dow_seasonality(u, wd)
    assert defined is False


def test_dow_seasonality_spikes_lower_it():
    # honest: spikes add variance not explained by weekday -> eta2 drops
    weekly = np.tile(np.array([10, 4, 4, 4, 4, 8, 12.0]), 20)
    wd = np.tile(np.array([1, 2, 3, 4, 5, 6, 7]), 20)
    clean = sig.dow_seasonality(weekly, wd)[0]
    spiky = weekly.copy(); spiky[[10, 55, 100]] += 200
    assert sig.dow_seasonality(spiky, wd)[0] < clean


def test_stl_strength_removed():
    # dead code dropped -- the B3 signal is dow_seasonality
    assert not hasattr(sig, "stl_strength")


# --- spikes ---

def test_spike_count_one_injected_spike():
    u = np.full(60, 5.0)  # flat non-zero -> median 5, MAD 0
    u[30] = 100.0         # single outlier above median
    stats = sig.spike_stats(u)
    assert stats["spike_count"] == 1.0
    assert stats["max_med_ratio"] == pytest.approx(20.0)


def test_spike_zero_for_flat_series():
    u = np.full(40, 3.0)
    assert sig.spike_stats(u)["spike_count"] == 0.0


def test_spike_all_zero_series():
    stats = sig.spike_stats(np.zeros(10))
    assert stats["spike_count"] == 0.0
    assert np.isnan(stats["max_med_ratio"])


# --- integration ---

def test_series_signals_keys_and_types():
    rng = np.random.default_rng(1)
    u = np.tile([10, 4, 4, 4, 4, 8, 12], 20) + rng.normal(0, 0.3, 140)
    wd = np.tile([1, 2, 3, 4, 5, 6, 7], 20)
    out = sig.series_signals(u, wday=wd)
    expected = {
        "active_len", "nonzero_count", "mean_nonzero", "zero_share", "adi",
        "cv2", "sb_class", "dow_season", "dow_defined", "spike_count",
        "spike_ratio", "max_med_ratio",
    }
    assert expected <= set(out)
    assert out["sb_class"] == "smooth"  # dense, low-variability weekly signal
    assert out["dow_defined"] is True


def test_series_signals_dow_undefined_without_wday():
    out = sig.series_signals(np.tile([1, 0, 2, 0, 3, 0, 1], 5))
    assert out["dow_defined"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
