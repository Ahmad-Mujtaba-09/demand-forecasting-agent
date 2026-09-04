"""Croston's method and the Syntetos-Boylan Approximation (Phase 3 sub-plan E5).

The literature-standard estimator for intermittent demand, and the B2 branch's
proper comparator. Master plan sec 3 names "Croston-style" as the sparse
fallback; Phase 2 shipped a plain mean floor instead, so the branch that handles
79% of dataset C had never been tested against its own field's standard.

**Croston (1972).** Decompose intermittent demand into two series and smooth each
separately with simple exponential smoothing:

    z -- the SIZE of a non-zero demand
    p -- the INTERVAL (in periods) between consecutive non-zero demands

Both are updated ONLY on periods with non-zero demand; zero periods update
nothing. The per-period forecast is then `z / p`. The point of the split is that
smoothing raw demand lets long zero runs drag the level toward zero, which both
under-forecasts and -- because the level decays with time since the last sale --
makes the forecast depend on WHEN you ask rather than on the demand process.

**SBA (Syntetos & Boylan, 2005).** Croston's estimator is biased high: E[z/p] !=
E[z]/E[p] by Jensen, because 1/p is convex. SBA multiplies by `(1 - alpha/2)`,
which removes the leading bias term. Same paper as the ADI/CV-squared cutoffs
this project already cites and uses as absolute constants.

Both are fit per series on history up to the forecast origin and emit a CONSTANT
per-period forecast over the horizon -- there is no covariate path here, which is
the point: this is the estimator for series with too little signal to justify a
model, so it must not be given one.

`alpha` is the one smoothing parameter. 0.1 is the standard default in the
intermittent-demand literature (Croston's own worked examples and the SBA paper
both operate in the 0.05-0.2 band); it is NOT tuned per dataset here, so E5 stays
a zero-tuning comparator and cannot win by having been tuned harder than the
mean floor it is measured against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ALPHA: float = 0.1


def croston_forecast(units, alpha: float = DEFAULT_ALPHA, sba: bool = True) -> float:
    """Constant per-period forecast from one series' demand history.

    Returns 0.0 for a history with no non-zero demand at all -- an honest
    "we have never seen a sale" rather than an invented positive rate.

    A history with exactly ONE non-zero demand has no observed interval to
    smooth, so the interval estimate falls back to the periods elapsed up to and
    including that sale. That is the only defensible estimate available and it
    keeps the function total, rather than returning NaN into a forecast column.
    """
    y = np.asarray(units, dtype=float)
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return 0.0

    z = float(y[nz[0]])          # size level, initialised at the first sale
    p = float(nz[0] + 1)         # interval level, periods up to that first sale
    for prev, cur in zip(nz[:-1], nz[1:]):
        z += alpha * (float(y[cur]) - z)          # smooth the SIZE
        p += alpha * (float(cur - prev) - p)      # smooth the INTERVAL since last sale
    if p <= 0:
        return 0.0

    rate = z / p
    # SBA debiasing: Croston's z/p is biased high because 1/p is convex (Jensen).
    return rate * (1.0 - alpha / 2.0) if sba else rate


def croston_predictions(
    feat: pd.DataFrame,
    folds,
    ids: set[str],
    alpha: float = DEFAULT_ALPHA,
    sba: bool = True,
    method: str | None = None,
) -> pd.DataFrame:
    """Per-fold Croston/SBA forecasts, shaped like the other prediction frames.

    Mirrors `baseline._floor_predictions` exactly -- same evaluable-row predicate,
    same columns -- so the B2 cohort comparison is like-for-like and the only
    difference is the estimator.
    """
    from . import features as ft  # local import keeps this module import-light

    label = method or ("sba" if sba else "croston")
    out = []
    for fold in folds:
        m = ((feat["day_idx"] >= fold.val_start) & (feat["day_idx"] <= fold.val_end)
             & feat["trainable"] & feat["id"].isin(ids))
        val = feat.loc[m, ["id", "day_idx", "units"]]
        if val.empty:
            continue
        hist = feat[(feat["day_idx"] <= fold.origin) & feat["active"] & feat["id"].isin(ids)]
        rate = (hist.sort_values("day_idx")
                    .groupby("id", observed=True)["units"]
                    .apply(lambda u: croston_forecast(u.to_numpy(), alpha, sba)))
        out.append(val.assign(
            forecast=val["id"].map(rate).fillna(0.0).clip(lower=0.0),
            fold=fold.index, method=label,
        ).rename(columns={"units": "actual"}))
    if not out:
        return pd.DataFrame(columns=["id", "day_idx", "actual", "forecast", "fold", "method"])
    return pd.concat(out, ignore_index=True)[
        ["id", "day_idx", "actual", "forecast", "fold", "method"]]
