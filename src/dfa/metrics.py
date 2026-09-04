"""WMAPE, the Phase 2 yardstick (master plan sec 0.5, Phase 2 sub-plan sec 1).

WMAPE is **volume-weighted, absolute-error based, sum-then-divide**:

    WMAPE = sum_i sum_t |actual - forecast|  /  sum_i sum_t actual

over every (series, day) in the eval window -- one numerator and one denominator,
NOT an average of per-series ratios (which explode to inf/0 on the zero-heavy
intermittent series that dominate here). The volume weighting falls straight out
of the shared denominator: a series counts in proportion to its total units.

Why WMAPE and not M5's WRMSSE: WRMSSE is a scale-free hierarchical *competition*
metric for ranking submissions across the full M5 hierarchy. Our goal is an
interpretable per-dataset demand error that reads as a business number. WMAPE is
robust on zeros, volume-weighted, and coherent with the Tweedie objective chosen
in Phase 1 -- WRMSSE's squared-error core would re-import the exact sensitivity
Tweedie exists to avoid, muddying the Phase 3 bake-off.

`wmape_by_group` powers both the Syntetos-Boylan-class breakdown (all datasets)
and the volume-tercile breakdown (dataset A only): the aggregate is volume-
weighted, so a strong headline can hide a poor intermittent tail -- the grouped
split is the honesty check on the headline. A group whose actuals sum to zero
reports NaN (not a divide-by-zero) with its observation count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wmape(actual, forecast) -> float:
    """sum|actual - forecast| / sum(actual), pooled. NaN if sum(actual) == 0.

    Demand is non-negative, so sum(actual) is the total volume in the window; a
    zero denominator means the window has no demand at all -> WMAPE undefined.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"actual/forecast shape mismatch: {a.shape} vs {f.shape}")
    denom = a.sum()
    if denom <= 0:
        return float("nan")
    return float(np.abs(a - f).sum() / denom)


def wmape_by_group(actual, forecast, group) -> dict[str, dict[str, float]]:
    """Per-group WMAPE with each group's own pooled denominator.

    Returns {group_label: {"wmape": float, "denom": float, "n_obs": int}}. A
    group with zero total actual reports wmape = NaN (honest "undefined") rather
    than raising -- e.g. an SB class whose series never sell in the eval window.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    g = np.asarray(group)
    if not (a.shape == f.shape == g.shape):
        raise ValueError("actual/forecast/group must share shape")
    out: dict[str, dict[str, float]] = {}
    for label in pd.unique(g):
        m = g == label
        out[str(label)] = {
            "wmape": wmape(a[m], f[m]),
            "denom": float(a[m].sum()),
            "n_obs": int(m.sum()),
        }
    return out


def volume_terciles(series_ids, volumes) -> dict[str, str]:
    """Map each series id -> 'low'/'mid'/'high' by total volume (tercile cuts).

    Used for the dataset-A-only volume-tercile WMAPE (Phase 2 sub-plan sec 1.4).
    Ties/duplicate cut edges are handled by ranking then splitting into thirds, so
    the three buckets are always non-empty even when many series share a volume.
    """
    vol = pd.Series(np.asarray(volumes, dtype=float), index=list(series_ids))
    order = vol.rank(method="first")
    labels = pd.qcut(
        order, q=3, labels=["low", "mid", "high"]
    )
    return {str(i): str(lab) for i, lab in labels.items()}


# --------------------------------------------------------------------------
# Phase 3 additions (sub-plan sec 0.2). Daily WMAPE is degenerate on
# intermittent series -- absolute error is minimised by the conditional MEDIAN,
# which is 0 whenever zero-share > 0.5 (Kolassa 2016). Everything below exists
# to make that visible and to give the bake-off a metric it can actually decide
# on, WITHOUT lowering the bar: the zero forecast still scores exactly 1.0.
# --------------------------------------------------------------------------


def wmape_horizon(actual, forecast, series_ids, folds) -> float:
    """WMAPE on per-(series, fold) HORIZON TOTALS -- the decision-relevant cut.

        WMAPE_28 = sum_i sum_k |sum_t y - sum_t f| / sum_i sum_k sum_t y

    A replenishment decision consumes demand over a lead time, not a per-day
    point estimate, so the horizon total is the quantity a planner orders
    against. It is also non-degenerate where the daily metric is not: a 28-day
    total is rarely zero even on sparse series, which restores the conditional
    MEAN as the right estimation target and makes Tweedie/Poisson answerable.

    Two properties worth stating because they are what keep this honest:

    - **Same denominator as `wmape`.** Aggregation happens in the numerator
      only, so the two metrics are directly comparable and their gap decomposes
      the error: this is LEVEL error, `wmape - wmape_horizon` is WITHIN-WINDOW
      TIMING error.
    - **`wmape_horizon <= wmape` always**, by the triangle inequality
      (|sum of errors| <= sum of |errors|). So a lower number here is guaranteed
      for every model and is NOT the argument for the metric. The argument is
      that the zero forecast still scores exactly 1.0 -- the bar is unchanged,
      only made reachable.
    """
    df = pd.DataFrame({
        "actual": np.asarray(actual, dtype=float),
        "forecast": np.asarray(forecast, dtype=float),
        "id": np.asarray(series_ids),
        "fold": np.asarray(folds),
    })
    denom = df["actual"].sum()
    if denom <= 0:
        return float("nan")
    g = df.groupby(["id", "fold"], observed=True)[["actual", "forecast"]].sum()
    return float((g["actual"] - g["forecast"]).abs().sum() / denom)


def oracle_lines(actual, series_ids, folds) -> dict[str, float]:
    """Attainability bounds: the best CONSTANT per-(series, fold) forecast.

    Both are computed FROM the validation window's own actuals -- they cheat, so
    they are reference lines, never results:

    - `oracle_median` -- per-(series, fold) median. The bound relevant to daily
      WMAPE, since absolute error is minimised by the median.
    - `oracle_mean`   -- per-(series, fold) mean. The bound relevant to a
      MEAN-targeting objective, i.e. Tweedie and Poisson. When this sits above
      1.0, a perfect conditional mean would still lose to forecasting nothing,
      and no amount of tuning a mean-estimator can clear the zero line.

    **Scope, stated because it is easy to overclaim:** these are optima over
    *constant-per-series* forecasts, NOT over all daily point forecasts. A model
    conditioning on features (weekday, SNAP, price) can legitimately beat them
    by varying its forecast within the window. So an entrant scoring below its
    oracle is surprising and worth investigating -- it is NOT proof of a leak,
    and this is not a leakage test.
    """
    df = pd.DataFrame({
        "actual": np.asarray(actual, dtype=float),
        "id": np.asarray(series_ids),
        "fold": np.asarray(folds),
    })
    grp = df.groupby(["id", "fold"], observed=True)["actual"]
    return {
        "oracle_median": wmape(df["actual"], grp.transform("median")),
        "oracle_mean": wmape(df["actual"], grp.transform("mean")),
    }


def pinball_loss(actual, forecast, quantile: float) -> float:
    """Mean pinball (quantile) loss at level `quantile` -- the B4 yardstick.

        q * (y - f)      when y >= f   (under-forecast: penalised q)
        (1 - q) * (f - y) when y <  f   (over-forecast:  penalised 1-q)

    The loss a quantile model is actually fit to minimise, so it is what the
    quantile models are scored on. A point metric like WMAPE would penalise a
    q90 forecast for doing its job (deliberately forecasting high).
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"actual/forecast shape mismatch: {a.shape} vs {f.shape}")
    if a.size == 0:
        return float("nan")
    d = a - f
    return float(np.mean(np.maximum(quantile * d, (quantile - 1.0) * d)))


def coverage(actual, forecast) -> float:
    """Fraction of actuals at or below the forecast -- the honest interval check.

    For a well-calibrated q-quantile forecast this should be ~= q. Pinball loss
    alone will not reveal a miscalibrated interval: a band that is nominally 80%
    wide but covers 60% of outcomes is worse than useless for safety stock, and
    only coverage shows it.

    `<=` (not `<`) is deliberate: demand is a non-negative count with heavy mass
    at exactly zero, and a q10 forecast of 0 does cover an actual of 0. Using a
    strict inequality would report ~0% coverage for the low quantiles on
    intermittent series purely as a tie-breaking artifact.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"actual/forecast shape mismatch: {a.shape} vs {f.shape}")
    if a.size == 0:
        return float("nan")
    return float(np.mean(a <= f))


def crossing_rate(quantile_preds: dict[float, np.ndarray]) -> float:
    """Fraction of rows where independently-fit quantiles are out of order.

    Three separate LightGBM fits have nothing tying them together, so q10 > q50
    or q50 > q90 happens -- and emits a negative-width interval straight into a
    safety-stock calculation. We sort the predictions per row before use; this
    reports how often that repair was needed, because a high rate means the
    quantile fits are unstable and the intervals should not be trusted.
    """
    levels = sorted(quantile_preds)
    if len(levels) < 2:
        return 0.0
    stacked = np.column_stack([np.asarray(quantile_preds[q], dtype=float) for q in levels])
    return float(np.mean((np.diff(stacked, axis=1) < 0).any(axis=1)))


def monotone_quantiles(quantile_preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Repair quantile crossing by sorting each row's predictions ascending.

    The minimal, order-preserving fix: the sorted values are still the same set
    of model outputs, just reassigned to the levels in the order the levels
    require. Reported alongside `crossing_rate` so the repair is visible rather
    than silent.
    """
    levels = sorted(quantile_preds)
    stacked = np.column_stack([np.asarray(quantile_preds[q], dtype=float) for q in levels])
    stacked.sort(axis=1)
    return {q: stacked[:, i] for i, q in enumerate(levels)}
