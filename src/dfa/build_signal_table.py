"""Compute branch signals for all series -> parquet.

Runs signals directly off the wide integer matrix (no full melt) for speed; the
whole population builds in ~15s single-threaded (all signals, including the
day-of-week seasonality eta^2, are cheap and vectorizable), so no shortcut is
needed.

Output: artifacts/signal_table.parquet -- one row per series with identity
columns, the three signal families (intermittency, day-of-week seasonality,
spikes), and price_coverage (active weeks that carry a sell price; a proxy for
availability).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from . import data_loader as dl
from . import signals as sig


def _active_vector(row: np.ndarray) -> tuple[np.ndarray, int]:
    """Trim leading (pre-introduction) zeros. Returns (active_units, intro_idx).

    intro_idx is the 0-based position of the first sale; -1 if the series never
    sells.
    """
    nz = np.nonzero(row > 0)[0]
    if nz.size == 0:
        return row[:0], -1
    return row[nz[0]:], int(nz[0])


def _price_coverage(wide: pd.DataFrame, intro_idx: np.ndarray) -> np.ndarray:
    """Fraction of a series' active weeks that carry a sell price.

    Active weeks = distinct calendar weeks from the introduction day to the EDA
    horizon. Priced weeks = weeks the (store, item) appears in sell_prices.
    """
    cal = dl.load_calendar()
    # day_idx (1..TRAIN_END_DAY) -> wm_yr_wk
    week_of_day = (
        cal.loc[cal["day_idx"] <= config.TRAIN_END_DAY]
        .sort_values("day_idx")["wm_yr_wk"]
        .to_numpy()
    )
    prices = dl.load_sell_prices()
    priced_weeks = prices.groupby(["store_id", "item_id"])["wm_yr_wk"].agg(
        lambda s: frozenset(s.to_numpy())
    )

    cov = np.full(len(wide), np.nan)
    keys = list(zip(wide["store_id"].to_numpy(), wide["item_id"].to_numpy()))
    for i, (start, key) in enumerate(zip(intro_idx, keys)):
        if start < 0:
            continue
        active_weeks = np.unique(week_of_day[start:])
        pset = priced_weeks.get(key, frozenset())
        if active_weeks.size == 0:
            continue
        hit = np.fromiter((w in pset for w in active_weeks), dtype=bool)
        cov[i] = hit.mean()
    return cov


def _wday_of_day() -> np.ndarray:
    """Weekday code per day position 0..TRAIN_END_DAY-1 (aligned to d_1..)."""
    cal = dl.load_calendar()
    return (
        cal.loc[cal["day_idx"] <= config.TRAIN_END_DAY]
        .sort_values("day_idx")["wday"]
        .to_numpy()
    )


def compute_signal_table(wide: pd.DataFrame, with_price_coverage: bool = True) -> pd.DataFrame:
    dcols = dl.day_columns(config.TRAIN_END_DAY)
    mat = wide[dcols].to_numpy(dtype=np.int32)
    wday = _wday_of_day()  # day-of-week aligned to the day columns

    records: list[dict] = []
    intro_idx = np.empty(len(wide), dtype=np.int64)
    t0 = time.time()
    for i in range(len(wide)):
        active, start = _active_vector(mat[i])
        intro_idx[i] = start
        # weekday codes over the same active window feed the B3 seasonality signal
        wslice = wday[start:] if start >= 0 else wday[:0]
        s = sig.series_signals(active, wday=wslice)
        s["intro_day"] = start + 1 if start >= 0 else -1  # 1-based day_idx
        records.append(s)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(wide)} series  ({time.time() - t0:.0f}s)")
    sig_df = pd.DataFrame.from_records(records)

    meta = wide[list(config.ID_COLS)].reset_index(drop=True)
    out = pd.concat([meta, sig_df], axis=1)

    if with_price_coverage:
        out["price_coverage"] = _price_coverage(wide, intro_idx)
    return out


def main() -> None:
    t0 = time.time()
    print("loading wide sales ...")
    wide = dl.load_sales_wide(max_day=config.TRAIN_END_DAY)
    print(f"  {wide.shape} in {time.time() - t0:.0f}s")

    print("computing signals for all series ...")
    table = compute_signal_table(wide)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.ARTIFACTS_DIR / "signal_table.parquet"
    table.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  shape={table.shape}  total {time.time() - t0:.0f}s")
    print("\nsb_class distribution:")
    print(table["sb_class"].value_counts())


if __name__ == "__main__":
    main()
