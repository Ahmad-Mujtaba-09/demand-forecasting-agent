"""Increment 1 tests: loader/reshape correctness.

Run: pytest -q  (from repo root, with agno env)
Uses a small subset of real series so tests stay fast but exercise real data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dfa import config
from dfa import data_loader as dl


# a handful of real series ids spanning categories
SUBSET = [
    "HOBBIES_1_001_CA_1_evaluation",
    "HOUSEHOLD_1_001_CA_1_evaluation",
    "FOODS_3_001_CA_1_evaluation",
]
MAX_DAY = 100  # small window for speed


def test_day_to_int_roundtrip():
    assert dl.day_to_int("d_1") == 1
    assert dl.day_to_int("d_1913") == 1913
    assert dl.day_columns(3) == ["d_1", "d_2", "d_3"]


def test_wide_shape_and_columns():
    wide = dl.load_sales_wide(subset_ids=SUBSET, max_day=MAX_DAY)
    assert len(wide) == len(SUBSET)
    assert list(wide.columns) == list(config.ID_COLS) + dl.day_columns(MAX_DAY)


def test_melt_row_count_and_dtype():
    wide = dl.load_sales_wide(subset_ids=SUBSET, max_day=MAX_DAY)
    long = dl.melt_sales(wide, max_day=MAX_DAY)
    # every series x every day
    assert len(long) == len(SUBSET) * MAX_DAY
    assert str(long["units"].dtype) == "int32"
    assert long["day_idx"].min() == 1 and long["day_idx"].max() == MAX_DAY


def test_spot_check_against_raw_wide():
    """A melted (id, day) value must equal the raw wide cell."""
    wide = dl.load_sales_wide(subset_ids=SUBSET, max_day=MAX_DAY)
    long = dl.melt_sales(wide, max_day=MAX_DAY)
    sid = SUBSET[2]  # FOODS_3_001
    raw_row = wide.loc[wide["id"] == sid].iloc[0]
    for d in ["d_1", "d_37", "d_100"]:
        got = long.loc[(long["id"] == sid) & (long["d"] == d), "units"].iloc[0]
        assert got == raw_row[d]


def test_active_flag_excludes_leading_zeros():
    wide = dl.load_sales_wide(subset_ids=SUBSET, max_day=MAX_DAY)
    long = dl.add_active_flag(dl.melt_sales(wide, max_day=MAX_DAY))
    for sid, grp in long.groupby("id"):
        nz = grp.loc[grp["units"] > 0, "day_idx"]
        if nz.empty:
            assert not grp["active"].any()
        else:
            intro = nz.min()
            assert grp["intro_day"].iloc[0] == intro
            # active is exactly day_idx >= intro
            assert (grp["active"] == (grp["day_idx"] >= intro)).all()
            # no active row before the first sale
            assert grp.loc[grp["day_idx"] < intro, "units"].sum() == 0


def test_calendar_join_no_row_multiplication():
    long = dl.build_long(subset_ids=SUBSET, max_day=MAX_DAY, with_prices=False)
    assert len(long) == len(SUBSET) * MAX_DAY
    # calendar features present and dated
    assert long["date"].notna().all()
    assert long["date"].min() == pd.Timestamp(config.DATASET_START_DATE)


def test_price_join_no_row_multiplication_and_coverage_is_partial():
    long = dl.build_long(subset_ids=SUBSET, max_day=MAX_DAY)
    assert len(long) == len(SUBSET) * MAX_DAY
    # prices are intentionally NOT fully covered (missing = not on offer)
    assert long["sell_price"].isna().any()


def test_active_only_filters_rows():
    full = dl.build_long(subset_ids=SUBSET, max_day=MAX_DAY, active_only=False)
    act = dl.build_long(subset_ids=SUBSET, max_day=MAX_DAY, active_only=True)
    assert len(act) <= len(full)
    assert act["active"].all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
