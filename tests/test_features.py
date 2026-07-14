"""Phase 2 Increment 3 tests: the frozen feature builder + the leakage invariant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfa import features as ft
from dfa import config


def _synth_long(units, intro_day=1, state="CA", series_id="S1", start_day=1):
    """Minimal M5-shaped long frame for one series with the columns build needs."""
    n = len(units)
    days = np.arange(start_day, start_day + n)
    df = pd.DataFrame({
        "id": series_id,
        "store_id": f"{state}_1",
        "dept_id": "FOODS_1",
        "cat_id": "FOODS",
        "state_id": state,
        "day_idx": days,
        "units": np.asarray(units, dtype="int32"),
        "active": days >= intro_day,
        "intro_day": intro_day,
        "wday": ((days - 1) % 7) + 1,
        "month": ((days // 30) % 12) + 1,
        "event_type_1": None,
        "event_name_2": None,
        "event_type_2": None,
        "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
        "sell_price": 1.5,
    })
    return df


def test_lag_28_equals_units_28_days_earlier():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 10, 200)
    feat = ft.build_features(_synth_long(u))
    feat = feat.sort_values("day_idx").reset_index(drop=True)
    h = config.HORIZON
    # for day_idx d (1-indexed from 1), lag_28 should be units at d-28
    for d in (60, 120, 199):
        assert feat.loc[feat["day_idx"] == d, "lag_28"].iloc[0] == pytest.approx(u[d - 1 - h])


def test_roll_28_is_trailing_window_ending_at_t_minus_horizon():
    u = np.arange(1, 121)  # 1..120, easy to hand-check
    feat = ft.build_features(_synth_long(u)).sort_values("day_idx").reset_index(drop=True)
    h = config.HORIZON
    d = 100
    # window is the 28 units ending at day d-28 -> units[d-28-27 .. d-28]
    expected = u[(d - h - 27) - 1 : (d - h) - 1 + 1].mean()
    assert feat.loc[feat["day_idx"] == d, "roll_28"].iloc[0] == pytest.approx(expected)


def test_no_nan_in_feature_columns():
    u = np.r_[np.zeros(5), np.random.default_rng(1).integers(0, 6, 95)]
    feat = ft.build_features(_synth_long(u, intro_day=6))
    assert not feat[ft.feature_columns(feat)].isna().any().any()


def test_leakage_future_units_do_not_affect_earlier_feature_rows():
    # altering units AFTER day K must leave feature rows with day_idx <= K+HORIZON
    # unchanged, because every dynamic feature reads only days <= t-HORIZON.
    rng = np.random.default_rng(2)
    u = rng.integers(0, 10, 200)
    K = 120
    base = ft.build_features(_synth_long(u)).sort_values("day_idx").reset_index(drop=True)
    u2 = u.copy()
    u2[K:] += 50  # perturb the future
    pert = ft.build_features(_synth_long(u2)).sort_values("day_idx").reset_index(drop=True)

    unaffected = base["day_idx"] <= K + config.HORIZON
    cols = ft.feature_columns(base)
    pd.testing.assert_frame_equal(
        base.loc[unaffected, cols].reset_index(drop=True),
        pert.loc[unaffected, cols].reset_index(drop=True),
    )


def test_event_vocabulary_extracted_globally_from_calendar_not_hardcoded():
    # the vocabulary is read from the calendar file (any M5-format calendar yields
    # its own), not a hardcoded list -- both slots, sorted.
    vocab = ft.event_vocabulary()
    assert vocab["event_type_1"] == ("Cultural", "National", "Religious", "Sporting")
    assert vocab["event_type_2"] == ("Cultural", "Religious")


def test_event_columns_follow_injected_vocab_not_frame_contents():
    # a novel type in the (injected) vocab gets a column; a type present in the
    # FRAME but not the vocab does NOT -- columns follow the global vocab, not the
    # frame's contents.
    df = _synth_long(np.arange(60))
    df.loc[df["day_idx"] == 10, "event_type_1"] = "Festival"   # in vocab below
    df.loc[df["day_idx"] == 11, "event_type_1"] = "Unlisted"   # NOT in vocab
    vocab = {"event_type_1": ("Festival", "National"), "event_type_2": ()}
    feat = ft.build_features(df, event_vocab=vocab)
    assert "et1_Festival" in feat.columns and "et1_National" in feat.columns  # from vocab
    assert "et1_Unlisted" not in feat.columns                                 # frame-only
    assert feat.loc[feat["day_idx"] == 10, "et1_Festival"].iloc[0] == 1.0
    assert (feat["et1_National"] == 0.0).all()  # in vocab, absent from frame -> all-zero col


def test_event_type_2_is_one_hot_not_a_bare_flag():
    # a concurrent second event's TYPE is preserved, not collapsed to a boolean
    df = _synth_long(np.arange(60))
    df.loc[df["day_idx"] == 20, ["event_name_2", "event_type_2"]] = ["Easter", "Cultural"]
    df.loc[df["day_idx"] == 21, ["event_name_2", "event_type_2"]] = ["OrthodoxEaster", "Religious"]
    vocab = {"event_type_1": (), "event_type_2": ("Cultural", "Religious")}
    feat = ft.build_features(df, event_vocab=vocab)
    assert {"et2_Cultural", "et2_Religious"} <= set(feat.columns)
    assert "is_event2" not in feat.columns  # the old bare flag is gone
    assert feat.loc[feat["day_idx"] == 20, "et2_Cultural"].iloc[0] == 1.0
    assert feat.loc[feat["day_idx"] == 21, "et2_Religious"].iloc[0] == 1.0
    assert feat.loc[feat["day_idx"] == 20, "et2_Religious"].iloc[0] == 0.0


def test_event_schema_is_frame_independent_type_absent_from_frame_still_a_column():
    # global vocab -> a type never appearing in this frame STILL gets a (zero)
    # column, so the schema can't shift between folds/subsets.
    df = _synth_long(np.arange(120))
    df.loc[df["day_idx"] == 100, "event_type_1"] = "National"  # only National appears
    vocab = {"event_type_1": ("Cultural", "National"), "event_type_2": ()}
    feat = ft.build_features(df, event_vocab=vocab)
    assert "et1_Cultural" in ft.feature_columns(feat)          # absent from frame...
    assert (feat["et1_Cultural"] == 0.0).all()                 # ...still a zero column
    assert feat.loc[feat["day_idx"] == 100, "et1_National"].iloc[0] == 1.0
    assert (feat.loc[feat["day_idx"] < 100, "et1_National"] == 0.0).all()


def test_no_event_columns_when_vocab_empty():
    # empty vocab -> no et_ columns, model still well-formed
    vocab = {"event_type_1": (), "event_type_2": ()}
    feat = ft.build_features(_synth_long(np.arange(60)), event_vocab=vocab)
    assert not [c for c in feat.columns if c.startswith(("et1_", "et2_"))]
    assert set(ft.feature_columns(feat)) <= set(feat.columns)


def test_price_fill_is_trailing_only_no_future_median():
    # leading NaN prices have no trailing price -> 0 (never a future-inclusive
    # median); a mid-series gap inherits the last TRAILING posted price, not one
    # pulled up by later/higher prices.
    df = _synth_long(np.full(60, 3), intro_day=1)
    df["sell_price"] = np.nan
    df.loc[df["day_idx"].between(6, 29), "sell_price"] = 10.0
    df.loc[df["day_idx"] >= 30, "sell_price"] = 100.0     # future high prices
    df.loc[df["day_idx"] == 20, "sell_price"] = np.nan    # a mid-series gap
    feat = ft.build_features(df).sort_values("day_idx").reset_index(drop=True)

    lead = feat[feat["day_idx"] <= 5]
    assert (lead["sell_price_filled"] == 0.0).all()      # no trailing price -> 0
    assert (lead["price_missing"] == 1.0).all()
    # gap day inherits the trailing 10.0, not a future-inclusive stat
    assert feat.loc[feat["day_idx"] == 20, "sell_price_filled"].iloc[0] == 10.0


def test_price_ffill_and_missing_flag():
    df = _synth_long(np.arange(60))
    df.loc[df["day_idx"] == 30, "sell_price"] = np.nan
    feat = ft.build_features(df).sort_values("day_idx").reset_index(drop=True)
    row = feat[feat["day_idx"] == 30].iloc[0]
    assert row["price_missing"] == 1.0
    assert row["sell_price_filled"] == pytest.approx(1.5)  # inherited via ffill


def test_snap_is_per_row_state_in_multi_state_frame():
    # two series in different states, with distinct SNAP calendars -> each series
    # must get its OWN state's SNAP, not the first row's state applied to all.
    n = 40
    days = np.arange(1, n + 1)
    ca = _synth_long(np.arange(n), state="CA", series_id="CA_series")
    tx = _synth_long(np.arange(n), state="TX", series_id="TX_series")
    # make the state SNAP columns clearly different: CA snaps on even days, TX odd
    for df in (ca, tx):
        df["snap_CA"] = (df["day_idx"] % 2 == 0).astype(int)
        df["snap_TX"] = (df["day_idx"] % 2 == 1).astype(int)
    feat = ft.build_features(pd.concat([ca, tx], ignore_index=True))
    ca_rows = feat[feat["id"] == "CA_series"].sort_values("day_idx")
    tx_rows = feat[feat["id"] == "TX_series"].sort_values("day_idx")
    assert (ca_rows["snap"].to_numpy() == (ca_rows["day_idx"] % 2 == 0)).all()
    assert (tx_rows["snap"].to_numpy() == (tx_rows["day_idx"] % 2 == 1)).all()
    # and the two series genuinely differ (guards against a silent single-state bug)
    assert not np.array_equal(ca_rows["snap"].to_numpy(), tx_rows["snap"].to_numpy())


def test_missing_snap_column_raises():
    df = _synth_long(np.arange(40), state="CA")
    df = df.drop(columns=["snap_CA"])
    with pytest.raises(KeyError):
        ft.build_features(df)


def test_trainable_excludes_rows_before_intro_plus_horizon():
    feat = ft.build_features(_synth_long(np.arange(1, 101), intro_day=10))
    cutoff = 10 + config.HORIZON
    assert not feat.loc[feat["day_idx"] < cutoff, "trainable"].any()
    assert feat.loc[feat["day_idx"] >= cutoff, "trainable"].all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
