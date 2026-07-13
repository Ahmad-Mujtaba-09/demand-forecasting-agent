"""Increment 3 sanity tests: the persisted signal table.

These validate the artifact produced by `python -m dfa.build_signal_table`.
Skipped (not failed) if the parquet hasn't been built yet, so the suite stays
green on a fresh checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfa import config

PARQUET = config.ARTIFACTS_DIR / "signal_table.parquet"
pytestmark = pytest.mark.skipif(
    not PARQUET.exists(), reason="signal_table.parquet not built yet"
)


@pytest.fixture(scope="module")
def table():
    import pandas as pd
    return pd.read_parquet(PARQUET)


def test_one_row_per_series(table):
    assert len(table) == 30490
    assert table["id"].is_unique


def test_expected_columns_present(table):
    expected = {
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "active_len", "nonzero_count", "mean_nonzero", "zero_share", "adi",
        "cv2", "sb_class", "dow_season", "dow_defined", "spike_count",
        "spike_ratio", "max_med_ratio", "intro_day", "price_coverage",
    }
    assert expected <= set(table.columns)


def test_signal_ranges_sane(table):
    assert table["zero_share"].between(0, 1).all()
    # ADI defined (>=1) for any series that ever sells
    sells = table["nonzero_count"] > 0
    assert (table.loc[sells, "adi"] >= 1.0).all()
    assert table["cv2"].dropna().ge(0).all()
    # day-of-week eta^2 in [0,1] where defined; NaN where not
    defined = table["dow_defined"]
    assert table.loc[defined, "dow_season"].between(0, 1).all()
    assert table.loc[~defined, "dow_season"].isna().all()
    assert table["spike_ratio"].between(0, 1).all()
    assert table.loc[table["price_coverage"].notna(), "price_coverage"].between(0, 1).all()


def test_sb_class_distribution_non_degenerate(table):
    vc = table["sb_class"].value_counts()
    # M5 item level is zero-heavy: intermittent/lumpy should dominate but all
    # four (or at least three) real classes should appear.
    assert (vc.index != "no_demand").sum() >= 3
    assert vc.drop("no_demand", errors="ignore").min() > 0


def test_intro_day_consistency(table):
    # active_len must equal (TRAIN_END_DAY - intro_day + 1) for selling series
    sells = table["intro_day"] > 0
    expected = config.TRAIN_END_DAY - table.loc[sells, "intro_day"] + 1
    assert (table.loc[sells, "active_len"] == expected).all()


def test_no_nan_in_required_signals(table):
    for col in ["zero_share", "sb_class", "spike_count", "active_len"]:
        assert table[col].notna().all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
