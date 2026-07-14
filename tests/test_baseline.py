"""Phase 2 Increment 4 tests: L2 baseline, routing, clamp, and tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfa import baseline as bl
from dfa import features as ft
from dfa import splits
from dfa import config


def _multi_series_long(series: dict[str, np.ndarray], state="CA"):
    """Build a long frame from {series_id: units_array} sharing a calendar."""
    frames = []
    for sid, units in series.items():
        n = len(units)
        days = np.arange(1, n + 1)
        intro = int(np.argmax(units > 0)) + 1 if (units > 0).any() else 1
        frames.append(pd.DataFrame({
            "id": sid, "store_id": f"{state}_1", "dept_id": "FOODS_1",
            "cat_id": "FOODS", "state_id": state, "day_idx": days,
            "units": units.astype("int32"), "active": days >= intro, "intro_day": intro,
            "wday": ((days - 1) % 7) + 1, "month": ((days // 30) % 12) + 1,
            "event_type_1": None, "event_name_2": None, "event_type_2": None,
            "snap_CA": 0, "snap_TX": 0, "snap_WI": 0, "sell_price": 2.0,
        }))
    return pd.concat(frames, ignore_index=True)


def _short_folds():
    # a fold set that fits inside a ~430-day synthetic series: small min-history
    # guard + a cap of 3 so the derivation lands on 3 folds for this short span
    return splits.make_folds(
        train_end=420, horizon=config.HORIZON, min_train_days=200, max_folds=3
    )


def test_forecasts_are_non_negative():
    rng = np.random.default_rng(0)
    # a weekly signal that L2 can partly fit; some noise
    base = np.tile([8, 2, 2, 2, 2, 5, 9], 62)[:430]
    series = {f"s{i}": np.clip(base + rng.integers(-2, 3, 430), 0, None).astype(int)
              for i in range(6)}
    feat = ft.build_features(_multi_series_long(series))
    res = bl.run_baseline(feat, _short_folds(), b2_ids=set())
    assert (res["baseline"]["forecast"] >= 0).all()
    assert (res["naive"]["forecast"] >= 0).all()


def test_b2_series_routed_to_floor_not_l2():
    rng = np.random.default_rng(1)
    dense = {f"d{i}": np.tile([8, 2, 2, 2, 2, 5, 9], 62)[:430].astype(int) for i in range(4)}
    sparse = {"sp1": (rng.random(430) < 0.05).astype(int),
              "sp2": (rng.random(430) < 0.04).astype(int)}
    feat = ft.build_features(_multi_series_long({**dense, **sparse}))
    res = bl.run_baseline(feat, _short_folds(), b2_ids={"sp1", "sp2"})
    methods = res["baseline"].groupby("id")["method"].unique()
    assert set(methods["sp1"]) == {"floor"} and set(methods["sp2"]) == {"floor"}
    assert set(methods["d0"]) == {"l2"}
    assert res["n_b2"] == 2 and res["n_l2"] == 4
    # naive comparator must be scored on the modelable series only, never B2
    naive_ids = set(res["naive"]["id"])
    assert naive_ids == {"d0", "d1", "d2", "d3"}
    assert "sp1" not in naive_ids and "sp2" not in naive_ids


def test_floor_value_is_mean_of_active_history_to_origin():
    # a single series routed to floor -> forecast == mean active units up to origin
    u = np.zeros(430, dtype=int)
    u[::10] = 5  # a sale every 10 days
    feat = ft.build_features(_multi_series_long({"x": u}))
    folds = _short_folds()
    res = bl.run_baseline(feat, folds, b2_ids={"x"})
    f0 = folds[0]
    # intro is day 1 (first sale at idx 0), so active mean == mean of units <= origin
    active_mean = u[:f0.origin].mean()
    got = res["baseline"].query("fold == @f0.index")["forecast"].iloc[0]
    assert got == pytest.approx(active_mean, rel=1e-6)


def test_tuning_selects_a_config_and_scores_all_trials():
    base = np.tile([8, 2, 2, 2, 2, 5, 9], 62)[:430]
    series = {f"s{i}": base.astype(int) for i in range(5)}
    feat = ft.build_features(_multi_series_long(series))
    cfg, trials = bl.select_config(feat, _short_folds(), b2_ids=set())
    assert cfg.transform in bl.TRANSFORMS and cfg.alpha in bl.ALPHA_GRID
    assert len(trials) == len(bl.ALPHA_GRID) * len(bl.TRANSFORMS)
    assert all(np.isfinite(t["wmape"]) for t in trials)


def test_l2_beats_naive_on_a_learnable_weekly_signal():
    # strong, clean weekly pattern -> L2 (with calendar features) should beat a
    # flat per-series mean. This is the sanity that the floor isn't a strawman.
    from dfa.metrics import wmape
    weekly = np.tile([20, 3, 3, 3, 3, 10, 25], 62)[:430].astype(int)
    series = {f"s{i}": weekly for i in range(6)}
    feat = ft.build_features(_multi_series_long(series))
    res = bl.run_baseline(feat, _short_folds(), b2_ids=set())
    l2_wmape = wmape(res["baseline"]["actual"], res["baseline"]["forecast"])
    naive_wmape = wmape(res["naive"]["actual"], res["naive"]["forecast"])
    assert l2_wmape < naive_wmape


def test_validation_excludes_non_trainable_rows():
    # a late-introduced series must not be scored on pre-intro / feature-warmup
    # days: excluded from the early fold, included once it has enough history.
    dense = {f"d{i}": np.tile([8, 2, 2, 2, 2, 5, 9], 62)[:430].astype(int) for i in range(3)}
    late = np.zeros(430, dtype=int)
    late[349:] = np.tile([5, 1, 1, 1, 1, 3, 6], 60)[:430 - 349]  # intro ~ day 350
    feat = ft.build_features(_multi_series_long({**dense, "late": late}))
    folds = _short_folds()  # origins 336/364/392 -> val 337-364, 365-392, 393-420
    res = bl.run_baseline(feat, folds, b2_ids=set())
    pred = res["baseline"]

    # intro=350 -> trainable needs day >= 350 + HORIZON = 378; earliest fold is all < 378
    early = pred[(pred["id"] == "late") & (pred["fold"] == folds[0].index)]
    assert len(early) == 0
    # by the last fold (val 393-420) the series has enough history and is scored
    late_fold = pred[(pred["id"] == "late") & (pred["fold"] == folds[-1].index)]
    assert len(late_fold) > 0


def test_all_sparse_dataset_skips_l2_gracefully():
    rng = np.random.default_rng(3)
    sparse = {f"sp{i}": (rng.random(430) < 0.03).astype(int) for i in range(4)}
    feat = ft.build_features(_multi_series_long(sparse))
    res = bl.run_baseline(feat, _short_folds(), b2_ids=set(sparse))
    assert res["n_l2"] == 0
    assert set(res["baseline"]["method"].unique()) == {"floor"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
