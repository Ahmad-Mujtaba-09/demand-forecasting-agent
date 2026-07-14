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
