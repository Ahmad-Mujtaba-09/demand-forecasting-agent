"""Frozen baseline feature set, origin-aware and leakage-safe (sub-plan sec 3).

This exact feature set is carried UNCHANGED into the Phase 3 bake-off, so the only
thing that changes there is the objective -- the comparison is of objectives, not
of feature engineering. Do not enrich it here.

Leakage discipline (the crux). We forecast a 28-day block *directly* (one model,
no recursive feed-back), so the most recent actual available at prediction time is
the forecast origin. We enforce that mechanically by lagging EVERY dynamic feature
by at least the horizon:

  - lag_28 / lag_35 / lag_42 : units 28/35/42 days before the target day t.
  - roll_28 / roll_56        : mean units over a 28/56-day window ending at t-28.
  - mean_hist                : expanding mean of units up to t-28.

So a feature for target day t reads only units on days <= t-28. For any horizon
day t in (origin, origin+28], t-28 <= origin -> strictly inside the train side.
This is a single feature function shared by train and predict rows, which is why
computing it over the full series (a trailing shift) is identical to computing it
per fold -- there is no future information in a >=28-day-lagged trailing window.

Calendar features are the *target day's* (the calendar is deterministic known-
future, not a leak) and go to every series -- the Phase 1 decision to drop the
per-series seasonal branch and let the model learn what weekly structure exists.

Sell price is the target week's price (known ahead in the M5 setup). It is
forward-filled within a series -- **trailing only** -- so a temporarily-delisted
day inherits the last posted price; leading days before a series' first price have
no trailing price and fall back to 0, flagged by `price_missing`. We never fill
with a per-series median, which would pull a future price back into a past row.

Event types are **extracted from the calendar, not hardcoded** -- any M5-format
dataset carries its own event vocabulary, so we one-hot whatever types the calendar
uses in `event_type_1` and `event_type_2` (both slots, so a second concurrent
event's type is not collapsed to a bare boolean and its demand effect is kept).
Weekday and month vocabularies stay fixed constants -- structural to the format
(7 weekdays, 12 months), not dataset-specific.

The event vocabulary is extracted **once, globally, from the full calendar**
(`event_vocabulary()`, cached), *not* from whichever series/days land in a given
frame. So the one-hot schema is a fixed property of the dataset's calendar,
identical across every frame, subset, and fold: a frame with no instance of a
given event type still carries that column as zeros -- never a shifted design
matrix. A different M5-format calendar yields its own vocabulary (still data-
derived); the extraction (this code) is what is frozen for Phase 3, not a specific
column list.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

# structural (format-fixed) calendar vocabularies -> always-present one-hot columns.
# Event-type vocabularies are NOT here: they are extracted globally from the
# calendar (see event_vocabulary), because any M5-format dataset has its own events.
WDAYS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
MONTHS: tuple[int, ...] = tuple(range(1, 13))
EVENT_COLS: tuple[str, ...] = ("event_type_1", "event_type_2")
EVENT_PREFIX: dict[str, str] = {"event_type_1": "et1", "event_type_2": "et2"}

LAGS: tuple[int, ...] = (config.HORIZON, config.HORIZON + 7, config.HORIZON + 14)  # 28,35,42
ROLL_WINDOWS: tuple[int, ...] = (28, 56)


@lru_cache(maxsize=1)
def event_vocabulary() -> dict[str, tuple[str, ...]]:
    """Event-type vocabulary from the FULL calendar, extracted once (cached).

    Global and frame-independent: the returned per-slot type lists are a property
    of the dataset's calendar, so every frame/subset/fold gets the identical event
    columns. Extracted from the calendar (any M5-format calendar yields its own
    vocabulary) -- data-derived, never hardcoded. Read-only; do not mutate the
    cached result.
    """
    from .data_loader import load_calendar  # lazy: keep module import-light and acyclic
    cal = load_calendar()
    return {col: tuple(sorted(cal[col].dropna().unique())) for col in EVENT_COLS}


def _dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags / rolling means, all lagged >= HORIZON. Assumes df sorted (id, day_idx)."""
    h = config.HORIZON
    grp = df.groupby("id", sort=False)["units"]
    for lag in LAGS:
        df[f"lag_{lag}"] = grp.shift(lag)
    # trailing rolling means whose window ends at t-HORIZON: shift units by HORIZON
    # first, then roll. min_periods=1 -> pandas averages available (non-NaN) points.
    shifted = grp.shift(h)
    sgrp = shifted.groupby(df["id"], sort=False)
    for w in ROLL_WINDOWS:
        df[f"roll_{w}"] = sgrp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    df["mean_hist"] = sgrp.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    return df


def _calendar_price_features(
    df: pd.DataFrame, event_vocab: dict[str, tuple[str, ...]]
) -> pd.DataFrame:
    """One-hot calendar (fixed wday/month + global event vocab) + SNAP + price."""
    for w in WDAYS:
        df[f"wday_{w}"] = (df["wday"] == w).astype("float32")
    for mo in MONTHS:
        df[f"month_{mo}"] = (df["month"] == mo).astype("float32")
    # one-hot BOTH event slots over the GLOBAL calendar vocabulary (not this frame's
    # contents), so the column set is frame-independent and a concurrent second
    # event's type is preserved rather than flattened to a flag.
    for col in EVENT_COLS:
        prefix = EVENT_PREFIX[col]
        for et in event_vocab.get(col, ()):
            df[f"{prefix}_{et}"] = (df[col] == et).astype("float32")

    # SNAP for EACH ROW's own state -- selected per row, not from a single state.
    # A frame may span multiple states; taking one state's column for all series
    # would silently give every out-of-state series the wrong SNAP signal.
    states = df["state_id"].to_numpy()
    missing = sorted({f"snap_{s}" for s in np.unique(states)} - set(df.columns))
    if missing:
        raise KeyError(f"missing SNAP column(s) for states present in the frame: {missing}")
    snap = np.empty(len(df), dtype="float32")
    for state in np.unique(states):
        m = states == state
        snap[m] = df[f"snap_{state}"].to_numpy(dtype="float32")[m]
    df["snap"] = snap

    # price: forward-fill within series -- TRAILING ONLY. A delisted/gap day
    # inherits the last posted price. Leading days before a series' first price
    # have no trailing price to draw on, so they fall back to 0 (flagged by
    # price_missing). We do NOT fill with a per-series median: that would pull in
    # future prices to fill a past row, a leak -- and those rows (pre-first-price)
    # are non-trainable anyway.
    df["price_missing"] = df["sell_price"].isna().astype("float32")
    price = df.groupby("id", sort=False)["sell_price"].ffill()
    df["sell_price_filled"] = price.fillna(0.0).astype("float32")
    return df


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """The model-input column list for a built frame (order is stable).

    Takes the frame because the event one-hot columns come from the (global) event
    vocabulary applied at build time. Fixed structural columns are listed
    explicitly; event columns are discovered by prefix and sorted, so any slice of
    a built frame -- train or val, any fold -- yields the identical ordered list.
    """
    cols = [f"lag_{lag}" for lag in LAGS]
    cols += [f"roll_{w}" for w in ROLL_WINDOWS] + ["mean_hist"]
    cols += [f"wday_{w}" for w in WDAYS]
    cols += [f"month_{mo}" for mo in MONTHS]
    cols += sorted(c for c in feat.columns if c.startswith("et1_"))
    cols += sorted(c for c in feat.columns if c.startswith("et2_"))
    cols += ["snap", "sell_price_filled", "price_missing"]
    return [c for c in cols if c in feat.columns]


# numeric columns that get standardized (per-fold, on train rows) in the model
NUMERIC_COLS: tuple[str, ...] = (
    *[f"lag_{lag}" for lag in LAGS],
    *[f"roll_{w}" for w in ROLL_WINDOWS],
    "mean_hist",
    "sell_price_filled",
)


def build_features(
    long: pd.DataFrame, event_vocab: dict[str, tuple[str, ...]] | None = None
) -> pd.DataFrame:
    """Long M5 frame -> per-(id, day) feature frame with the target `units`.

    Dynamic features are filled with 0 where history is too short (no prior
    demand yet); early rows are excluded from training by `trainable` (active and
    at least HORIZON days past the series' introduction). Keys and `units` are
    kept so the caller can slice train/val by day and join signals by id.

    `event_vocab` is the {slot: sorted_types} mapping that fixes the event one-hot
    columns. It defaults to the GLOBAL calendar vocabulary (`event_vocabulary()`),
    so the schema is frame-independent -- identical across every subset and fold.
    Tests inject a synthetic vocab to stay independent of the calendar file.
    """
    if event_vocab is None:
        event_vocab = event_vocabulary()
    df = long.sort_values(["id", "day_idx"]).reset_index(drop=True)
    df = _dynamic_features(df)
    df = _calendar_price_features(df, event_vocab)

    dyn = [f"lag_{lag}" for lag in LAGS] + [f"roll_{w}" for w in ROLL_WINDOWS] + ["mean_hist"]
    df[dyn] = df[dyn].fillna(0.0)

    # a row is usable for TRAINING once the series is active and >= HORIZON days
    # past introduction (so the target sits on a real active day with history).
    df["trainable"] = df["active"] & (df["day_idx"] >= df["intro_day"] + config.HORIZON)

    keep = (
        ["id", "store_id", "dept_id", "state_id", "day_idx", "units",
         "active", "intro_day", "trainable"]
        + feature_columns(df)
    )
    return df[keep]
