"""The honest L2 floor (Phase 2 sub-plan sec 4).

A plain squared-error (L2) linear regression, tuned as well as it honestly can be
so the Phase 3 bake-off is a fair fight -- not a strawman. Design choices, each
with its prior stated in the sub-plan:

- **Pooled Ridge per dataset.** One model over all (series, day) training rows,
  matching how Phase 3's LightGBM trains. Fit on raw units so high-volume series
  dominate the loss -- coherent with WMAPE's volume weighting. Ridge is L2 with an
  L2 penalty (alpha->0 recovers OLS); tuning alpha on the rolling-origin CV gives
  the honest best L2 fit rather than a fragile unregularised one. Numerics are
  standardized per fold on train rows only (no val leakage into the scaler).
- **Target space raw vs log1p**, chosen per dataset by validation WMAPE.
- **Non-negativity clamp.** L2 emits negative demand; we clamp to >=0 (the Phase 4
  critic's min_forecast bound, applied here for consistency).
- **Sparse series -> mean floor, not a pretend L2 fit.** A series the branch logic
  calls B2 (too sparse) is not given to L2; it gets a constant = mean of its active
  demand up to the origin. Reporting L2's ~0 predictions on near-all-zero series
  would understate the honest error and flatter Phase 3. This mirrors the
  architecture's own B2 -> simple baseline branch.
- **Naive comparator (modelable series only).** A constant mean-of-history
  forecast for the series L2 is actually applied to (non-B2). Its whole job is to
  answer "do the modelable series benefit from L2 over a constant?" -- so it is
  scored on exactly those series. Including B2 series would be a built-in tie
  (their baseline forecast *is* the mean), diluting the only comparison that
  matters.
- **Zero comparator (all series).** An all-zero "predict nothing" forecast, scored
  on the same rows as the baseline. It costs nothing to produce and is the true
  trivial baseline for intermittent demand; under sum-then-divide WMAPE it is
  identically 1.0, so it is the line that decides whether a fitted floor is useful
  at all. See `_zero_predictions`.

Hyperparameter selection reports on the same folds it tunes on, mild optimism, 
but Phase 3's challenger is tuned by the identical protocol on the same folds, 
so the optimism is equal on both sides and cancels in the comparison. 
Strongest floor is not the goal; matched protocol is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import features as ft
from .metrics import wmape
from .splits import Fold

ALPHA_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
TRANSFORMS: tuple[str, ...] = ("raw", "log1p")


@dataclass(frozen=True)
class BaselineConfig:
    alpha: float
    transform: str  # "raw" | "log1p"


def _to_target(y: np.ndarray, transform: str) -> np.ndarray:
    return np.log1p(y) if transform == "log1p" else y.astype(float)


def _from_target(pred: np.ndarray, transform: str) -> np.ndarray:
    """Inverse-transform then clamp to >= 0 (demand is non-negative)."""
    out = np.expm1(pred) if transform == "log1p" else pred
    return np.clip(out, 0.0, None)


def _design(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    X = df[list(ft.feature_columns(df))].copy()
    X[list(ft.NUMERIC_COLS)] = scaler.transform(X[list(ft.NUMERIC_COLS)])
    return X.to_numpy()


def _series_mean_to_origin(feat: pd.DataFrame, origin: int) -> pd.Series:
    """Mean of each series' ACTIVE demand on days <= origin (the floor value)."""
    hist = feat[(feat["day_idx"] <= origin) & feat["active"]]
    return hist.groupby("id", observed=True)["units"].mean()


def _val_rows(feat: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Evaluable validation rows for a fold: the SAME predicate as training.

    A row is scored only if `trainable` (series active and >= HORIZON days past
    introduction). This excludes pre-introduction zeros (days the item did not yet
    exist -- scoring a positive forecast there is a spurious penalty) and feature-
    warmup days whose lag/rolling features are 0-filled and unreliable. Applied
    here so L2, the B2 floor, and the naive comparator all share one honest
    denominator.
    """
    m = ((feat["day_idx"] >= fold.val_start) & (feat["day_idx"] <= fold.val_end)
         & feat["trainable"])
    return feat.loc[m, ["id", "day_idx", "units"] + list(ft.feature_columns(feat))].copy()


def _l2_predictions(
    feat: pd.DataFrame, folds: list[Fold], b2_ids: set[str], cfg: BaselineConfig
) -> pd.DataFrame:
    """Per-fold L2 forecasts for the non-B2 (modelable) series only."""
    out = []
    for fold in folds:
        train = feat[(feat["day_idx"] <= fold.origin) & feat["trainable"]
                     & ~feat["id"].isin(b2_ids)]
        val = _val_rows(feat, fold)
        val = val[~val["id"].isin(b2_ids)]
        if train.empty or val.empty:
            continue
        scaler = StandardScaler().fit(train[list(ft.NUMERIC_COLS)])
        model = Ridge(alpha=cfg.alpha).fit(
            _design(train, scaler), _to_target(train["units"].to_numpy(), cfg.transform)
        )
        val = val.assign(
            forecast=_from_target(model.predict(_design(val, scaler)), cfg.transform),
            fold=fold.index, method="l2",
        )
        out.append(val[["id", "day_idx", "units", "forecast", "fold", "method"]])
    if not out:
        return pd.DataFrame(columns=["id", "day_idx", "units", "forecast", "fold", "method"])
    return pd.concat(out, ignore_index=True).rename(columns={"units": "actual"})


def _floor_predictions(
    feat: pd.DataFrame, folds: list[Fold], ids: set[str], method: str
) -> pd.DataFrame:
    """Per-fold constant mean-of-active-history forecast for the given series ids.

    Used for the B2 series (method='floor', the sparse-fallback) and for the
    naive comparator (method='naive'), which is scored on the modelable (non-B2)
    series only. Series with no active history by the origin get 0.
    """
    out = []
    for fold in folds:
        val = _val_rows(feat, fold)
        val = val[val["id"].isin(ids)] if ids is not None else val
        if val.empty:
            continue
        floor = _series_mean_to_origin(feat, fold.origin)
        val = val.assign(
            forecast=val["id"].map(floor).fillna(0.0).clip(lower=0.0),
            fold=fold.index, method=method,
        )
        out.append(val[["id", "day_idx", "units", "forecast", "fold", "method"]])
    if not out:
        return pd.DataFrame(columns=["id", "day_idx", "units", "forecast", "fold", "method"])
    return pd.concat(out, ignore_index=True).rename(columns={"units": "actual"})


def _zero_predictions(
    feat: pd.DataFrame, folds: list[Fold], ids: set[str] | None = None
) -> pd.DataFrame:
    """Per-fold all-zero forecast -- the trivial "predict nothing" comparator.

    Free (no fitting) and the true trivial baseline for intermittent demand. Under
    sum-then-divide WMAPE a zero forecast scores

        sum|y - 0| / sum(y)  ==  1.0

    EXACTLY, on any row set with positive volume (demand is non-negative, so the
    numerator and denominator are the same sum). So this is not an empirical
    finding that happens to land near 1 -- it is an identity, and it is what makes
    "WMAPE > 1" literally mean *worse than forecasting nothing at all*. That makes
    1.0 the real bar on the intermittent datasets, not the fitted floor.

    Computed on the same val rows as the baseline rather than hardcoded, so each
    run re-derives the 1.0; a deviation would signal a bug in the row predicate or
    the WMAPE denominator.
    """
    out = []
    for fold in folds:
        val = _val_rows(feat, fold)
        val = val[val["id"].isin(ids)] if ids is not None else val
        if val.empty:
            continue
        val = val.assign(forecast=0.0, fold=fold.index, method="zero")
        out.append(val[["id", "day_idx", "units", "forecast", "fold", "method"]])
    if not out:
        return pd.DataFrame(columns=["id", "day_idx", "units", "forecast", "fold", "method"])
    return pd.concat(out, ignore_index=True).rename(columns={"units": "actual"})


def select_config(
    feat: pd.DataFrame, folds: list[Fold], b2_ids: set[str]
) -> tuple[BaselineConfig, list[dict]]:
    """Pick (alpha, transform) by pooled WMAPE over the L2-routed val rows.

    Selection is on the L2 series only -- the B2 floor and naive parts don't
    depend on the config, so they can't inform the choice.
    """
    modelable = [i for i in feat["id"].unique() if i not in b2_ids]
    trials: list[dict] = []
    if not modelable:  # dataset is entirely sparse -> no L2 to tune
        return BaselineConfig(alpha=ALPHA_GRID[0], transform="raw"), trials
    for transform in TRANSFORMS:
        for alpha in ALPHA_GRID:
            cfg = BaselineConfig(alpha=alpha, transform=transform)
            pred = _l2_predictions(feat, folds, b2_ids, cfg)
            score = wmape(pred["actual"], pred["forecast"]) if len(pred) else float("nan")
            # sum(forecast)/sum(actual) -- the retransformation-bias diagnostic.
            # log1p+expm1 is biased low by Jensen (expm1(E[log1p y]) <= E[y]); on the
            # intermittent datasets that bias is what makes log1p *win*, by dragging
            # predictions toward zero. Recorded per trial (free -- the fit already
            # happened) so the mechanism is an artifact, not an assertion in prose.
            bias = (float(pred["forecast"].sum() / pred["actual"].sum())
                    if len(pred) and pred["actual"].sum() > 0 else float("nan"))
            trials.append({"alpha": alpha, "transform": transform,
                           "wmape": score, "bias": bias})
    best = min(trials, key=lambda t: (np.isnan(t["wmape"]), t["wmape"]))
    return BaselineConfig(alpha=best["alpha"], transform=best["transform"]), trials


def run_baseline(
    feat: pd.DataFrame, folds: list[Fold], b2_ids: set[str]
) -> dict:
    """Full baseline for one dataset: tuned L2 (+ B2 floor) and both comparators.

    Returns the chosen config, the tuning trials, and three prediction frames, each
    with columns [id, day_idx, actual, forecast, fold, method]:

      - `baseline` -- L2 for modelable + mean floor for B2 (the system under test)
      - `naive`    -- constant mean, MODELABLE series only ("does L2 beat a constant?")
      - `zero`     -- all-zero forecast, ALL series ("is any of this useful at all?")

    The two comparators answer different questions and are deliberately scored on
    different row sets: `naive` on the series L2 is applied to (including B2 would
    be a built-in tie), `zero` on the same rows as `baseline` so it is directly
    comparable to the headline floor. The zero frame is scale-free anyway -- it is
    1.0 on any subset -- so the differing denominators do not make them
    inconsistent.
    """
    all_ids = set(feat["id"].unique())
    b2_ids = b2_ids & all_ids
    modelable_ids = all_ids - b2_ids
    cfg, trials = select_config(feat, folds, b2_ids)

    l2 = _l2_predictions(feat, folds, b2_ids, cfg)
    floor = _floor_predictions(feat, folds, b2_ids, method="floor")
    baseline = pd.concat([l2, floor], ignore_index=True)
    # comparator on the same series L2 is applied to -- not B2 (that would be a tie)
    naive = _floor_predictions(feat, folds, modelable_ids, method="naive")
    # trivial "predict nothing" anchor, on the same rows as `baseline`
    zero = _zero_predictions(feat, folds)

    return {
        "config": cfg,
        "trials": trials,
        "baseline": baseline,
        "naive": naive,
        "zero": zero,
        "n_b2": len(b2_ids),
        "n_l2": len(all_ids - b2_ids),
    }
