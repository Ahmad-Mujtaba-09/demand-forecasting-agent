"""Phase 3 Increment 4 tests: Croston / SBA, the B2 branch's proper estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfa import croston as cr
from dfa.splits import Fold


def test_steady_pattern_recovers_the_true_rate():
    """Demand of 2 every 4 periods is a rate of 0.5 -- Croston must find it."""
    u = np.zeros(40)
    u[3::4] = 2.0
    assert cr.croston_forecast(u, sba=False) == pytest.approx(0.5)


def test_sba_shrinks_croston_by_one_minus_half_alpha():
    """Syntetos-Boylan (2005): the (1 - alpha/2) debiasing factor, exactly."""
    u = np.zeros(40)
    u[3::4] = 2.0
    alpha = 0.1
    plain = cr.croston_forecast(u, alpha=alpha, sba=False)
    assert cr.croston_forecast(u, alpha=alpha, sba=True) == pytest.approx(
        plain * (1 - alpha / 2))


def test_never_seen_a_sale_forecasts_zero():
    """An honest 'no evidence of demand', not an invented positive rate."""
    assert cr.croston_forecast(np.zeros(50)) == 0.0


def test_single_sale_uses_elapsed_periods_as_the_interval():
    """One sale gives no observed interval; fall back to periods up to it."""
    u = np.r_[np.zeros(9), [5.0]]      # one sale of 5 at period 10
    assert cr.croston_forecast(u, alpha=0.1, sba=False) == pytest.approx(5.0 / 10.0)


def test_forecast_is_flat_in_time_since_last_sale():
    """Croston's whole purpose: unlike SES on raw demand, the estimate does not
    decay as zeros accumulate after the last sale."""
    base = np.zeros(40); base[3::4] = 2.0
    padded = np.r_[base, np.zeros(30)]     # 30 more zero periods, same process
    assert cr.croston_forecast(padded, sba=False) == pytest.approx(
        cr.croston_forecast(base, sba=False))


def test_ses_on_raw_demand_would_decay_but_croston_does_not():
    """Contrast that motivates the method (documented in the module docstring)."""
    base = np.zeros(40); base[3::4] = 2.0
    padded = np.r_[base, np.zeros(30)]
    def ses(y, a=0.1):
        lvl = float(y[0])
        for v in y[1:]:
            lvl += a * (float(v) - lvl)
        return lvl
    assert ses(padded) < ses(base) * 0.5          # SES collapses toward zero
    assert cr.croston_forecast(padded) == pytest.approx(cr.croston_forecast(base))


def test_larger_sizes_raise_and_longer_gaps_lower_the_rate():
    small = np.zeros(40); small[3::4] = 1.0
    large = np.zeros(40); large[3::4] = 5.0
    sparse = np.zeros(40); sparse[7::8] = 1.0
    assert cr.croston_forecast(large) > cr.croston_forecast(small)
    assert cr.croston_forecast(sparse) < cr.croston_forecast(small)


def _feat(units_by_id: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for sid, u in units_by_id.items():
        days = np.arange(1, len(u) + 1)
        rows.append(pd.DataFrame({
            "id": sid, "day_idx": days, "units": u.astype(float),
            "active": True, "trainable": True,
        }))
    return pd.concat(rows, ignore_index=True)


def test_predictions_frame_matches_the_baseline_contract():
    """Same columns and same evaluable-row predicate as baseline._floor_predictions,
    so the B2 cohort comparison is like-for-like."""
    u = np.zeros(200); u[3::4] = 2.0
    feat = _feat({"S1": u, "S2": np.zeros(200)})
    folds = [Fold(index=0, origin=100, val_start=101, val_end=128)]
    out = cr.croston_predictions(feat, folds, ids={"S1", "S2"})
    assert list(out.columns) == ["id", "day_idx", "actual", "forecast", "fold", "method"]
    assert out["day_idx"].between(101, 128).all()
    assert (out["method"] == "sba").all()
    # S1 gets a positive rate; the never-sold S2 gets exactly zero
    assert out.loc[out["id"] == "S1", "forecast"].iloc[0] > 0
    assert (out.loc[out["id"] == "S2", "forecast"] == 0).all()


def test_history_is_cut_at_the_origin():
    """A sale AFTER the origin must not influence the forecast (leakage guard)."""
    u = np.zeros(200)
    u[150] = 99.0                     # a big sale well past the origin
    feat = _feat({"S1": u})
    folds = [Fold(index=0, origin=100, val_start=101, val_end=128)]
    out = cr.croston_predictions(feat, folds, ids={"S1"})
    assert (out["forecast"] == 0).all(), "post-origin demand leaked into the forecast"
