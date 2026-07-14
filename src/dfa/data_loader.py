"""M5 loading + reshape (Phase 1, Increment 1).

Turns the wide Kaggle sales file into a tidy long frame joined to the calendar
and weekly sell prices, restricted to the EDA window (d_1..TRAIN_END_DAY).

Design notes / priors baked in:
- We melt to long because every downstream signal and model works per
  (series, day); the wide `d_*` layout is a storage transpose.
- Absence of a sell_price row for a (store, item, week) means the item was not
  on offer that week. We DO NOT impute here -- price coverage is itself a signal
  (Phase 1). A left join therefore leaves `sell_price` NaN on purpose.
- `day_idx` is the integer N parsed from `d_N`; it is the single time key after
  melting. `active` marks days from a series' first non-zero sale onward, so
  pre-introduction leading zeros are excluded from signal computation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config


def day_to_int(d: str) -> int:
    """`'d_137'` -> `137`."""
    return int(d.split("_", 1)[1])


@lru_cache(maxsize=None)
def last_sales_day(sales_csv: Path = config.SALES_CSV) -> int:
    """Highest `d_N` column present in the sales file (reads only the header).

    Data-driven: derived from whatever M5-format file is supplied, so nothing
    downstream has to hardcode M5's 1941. The last usable *training* day is this
    minus the sealed evaluation horizon (`config.HORIZON`) -- see the runner.
    """
    header = pd.read_csv(sales_csv, nrows=0)
    days = [day_to_int(c) for c in header.columns if c.startswith("d_")]
    if not days:
        raise ValueError(f"no d_* day columns found in sales file {sales_csv}")
    return max(days)


def day_columns(max_day: int = config.TRAIN_END_DAY) -> list[str]:
    """Ordered `['d_1', ..., 'd_max_day']`."""
    return [f"d_{i}" for i in range(1, max_day + 1)]


def load_calendar() -> pd.DataFrame:
    """Calendar with parsed `date` and an integer `day_idx` from `d`."""
    cal = pd.read_csv(config.CALENDAR_CSV)
    cal["date"] = pd.to_datetime(cal["date"])
    cal["day_idx"] = cal["d"].map(day_to_int)
    return cal


def load_sell_prices() -> pd.DataFrame:
    """Weekly sell prices (store_id, item_id, wm_yr_wk, sell_price)."""
    return pd.read_csv(config.SELL_PRICES_CSV)


def load_sales_wide(
    subset_ids: list[str] | None = None,
    max_day: int = config.TRAIN_END_DAY,
) -> pd.DataFrame:
    """Wide sales: id columns + `d_1..d_max_day`.

    `subset_ids` filters to specific series `id`s (used for fast tests and for
    the sampled sub-datasets); None loads all 30,490 series.
    """
    usecols = list(config.ID_COLS) + day_columns(max_day)
    wide = pd.read_csv(config.SALES_CSV, usecols=usecols)
    # keep the declared column order (read_csv does not guarantee it)
    wide = wide[usecols]
    if subset_ids is not None:
        wide = wide[wide["id"].isin(subset_ids)].reset_index(drop=True)
    return wide


def melt_sales(wide: pd.DataFrame, max_day: int = config.TRAIN_END_DAY) -> pd.DataFrame:
    """Wide -> long `(id..keys, d, day_idx, units)`."""
    dcols = day_columns(max_day)
    long = wide.melt(
        id_vars=list(config.ID_COLS),
        value_vars=dcols,
        var_name="d",
        value_name="units",
    )
    long["day_idx"] = long["d"].map(day_to_int)
    long["units"] = long["units"].astype("int32")
    return long


def add_active_flag(long: pd.DataFrame) -> pd.DataFrame:
    """Add `intro_day` (first non-zero day per series) and boolean `active`.

    Leading zeros before a series' first sale are pre-introduction, not
    intermittent demand, so signals are later computed on `active` rows only.
    A series that never sells gets intro_day = NaN and active = False throughout.
    """
    first_sale = (
        long.loc[long["units"] > 0]
        .groupby("id", observed=True)["day_idx"]
        .min()
        .rename("intro_day")
    )
    long = long.merge(first_sale, on="id", how="left")
    long["active"] = long["day_idx"] >= long["intro_day"]
    long["active"] = long["active"].fillna(False)
    return long


def build_long(
    subset_ids: list[str] | None = None,
    max_day: int = config.TRAIN_END_DAY,
    with_calendar: bool = True,
    with_prices: bool = True,
    active_only: bool = False,
) -> pd.DataFrame:
    """End-to-end: load -> melt -> active flag -> join calendar & prices.

    Left joins keep every sales row; a missing price stays NaN by design.
    `active_only=True` drops pre-introduction rows.
    """
    wide = load_sales_wide(subset_ids=subset_ids, max_day=max_day)
    long = melt_sales(wide, max_day=max_day)
    long = add_active_flag(long)

    if with_calendar:
        cal = load_calendar()
        cal_cols = [
            "d", "date", "wm_yr_wk", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ]
        n_before = len(long)
        long = long.merge(cal[cal_cols], on="d", how="left")
        # calendar has one row per d; the join must not multiply rows
        assert len(long) == n_before, "calendar join changed row count"

    if with_prices:
        if "wm_yr_wk" not in long.columns:
            raise ValueError("prices require calendar join (wm_yr_wk) first")
        prices = load_sell_prices()
        n_before = len(long)
        long = long.merge(
            prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
        )
        assert len(long) == n_before, "price join multiplied rows"

    if active_only:
        long = long.loc[long["active"]].reset_index(drop=True)

    return long
