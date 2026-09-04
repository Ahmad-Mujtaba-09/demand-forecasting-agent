"""Phase 3 Increment 3 tests: bake-off entrants, the fit-loop leakage surface,
determinism, and the frozen-feature guarantee.

The Phase-2 leakage check tests FEATURES. These test the FIT LOOP -- the surface
a tree pipeline actually leaks through, which Phase 2 never exercised because
Ridge has no early stopping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfa import config
from dfa import features as ft
from dfa import models as md
from dfa.manifest import hash_columns
from dfa.splits import Fold

# The Phase-2 feature manifest, pinned. Phase 3's whole claim is "only the
# objective changed" -- if features.py drifts, this breaks and the claim is void.
# 34 columns: 3 lags + 2 rolling + mean_hist + 7 wday + 12 month + 4 et1 + 2 et2
# + snap + price + price_missing.
FROZEN_FEATURE_HASH = "447f7447d0e6fb23"
FROZEN_FEATURE_COUNT = 34


@pytest.fixture(scope="module")
def synth_feat():
    """Two series x 400 days of M5-shaped data -- enough for folds and lags."""
    rng = np.random.default_rng(7)
    rows = []
    for sid, rate in (("S1", 3.0), ("S2", 0.4)):
        n = 400
        days = np.arange(1, n + 1)
        rows.append(pd.DataFrame({
            "id": sid, "store_id": "CA_1", "dept_id": "FOODS_1", "cat_id": "FOODS",
            "state_id": "CA", "day_idx": days,
            "units": rng.poisson(rate, n).astype("int32"),
            "active": True, "intro_day": 1,
            "wday": ((days - 1) % 7) + 1, "month": ((days // 30) % 12) + 1,
            "event_type_1": None, "event_type_2": None,
            "snap_CA": 0, "snap_TX": 0, "snap_WI": 0, "sell_price": 2.0,
        }))
    long = pd.concat(rows, ignore_index=True)
    return ft.build_features(long, event_vocab={"event_type_1": ("National",),
                                                "event_type_2": ()})


@pytest.fixture
def fold():
    return Fold(index=0, origin=300, val_start=301, val_end=300 + config.HORIZON)


# --- the frozen-feature guarantee (sub-plan sec 4.4) ------------------------

def test_feature_manifest_is_frozen_at_phase2_values():
    """Licenses the claim that the bake-off varies only the objective."""
    import json
    from dfa.data_loader import build_long
    sel = json.loads((config.ARTIFACTS_DIR / "datasets" / "selection.json").read_text())
    ids = sel["datasets"]["C_sparse"]["ids"][:3]
    cols = ft.feature_columns(ft.build_features(build_long(subset_ids=ids, max_day=400)))
    assert len(cols) == FROZEN_FEATURE_COUNT
    assert hash_columns(cols) == FROZEN_FEATURE_HASH, (
        "feature set changed -- Phase 3's 'only the objective varies' claim is void. "
        "If the change is intended, it invalidates comparability with Phase 2."
    )


# --- fit-loop leakage (sub-plan sec 4.2) -----------------------------------

def test_early_stopping_set_never_touches_validation(synth_feat, fold):
    """The likeliest tree-pipeline leak: stopping on the fold's own val block."""
    fit_rows, stop_rows, val = md._train_val_rows(synth_feat, fold, exclude=set())
    assert not fit_rows.empty and not stop_rows.empty and not val.empty
    # everything used for fitting OR stopping is training-side
    assert fit_rows["day_idx"].max() <= fold.origin
    assert stop_rows["day_idx"].max() <= fold.origin
    # validation is strictly after the origin
    assert val["day_idx"].min() > fold.origin
    # fit and stop are disjoint, and stop is the horizon-length tail of train
    assert fit_rows["day_idx"].max() < stop_rows["day_idx"].min()
    assert stop_rows["day_idx"].min() == fold.origin - config.HORIZON + 1


def test_excluded_series_reach_neither_training_nor_prediction(synth_feat, fold):
    """B2 routing must hold on both sides -- a sparse series is not silently scored."""
    fit_rows, stop_rows, val = md._train_val_rows(synth_feat, fold, exclude={"S2"})
    for frame in (fit_rows, stop_rows, val):
        assert "S2" not in set(frame["id"])
        assert "S1" in set(frame["id"])


# --- determinism (sub-plan sec 2) ------------------------------------------

def test_refit_is_bit_identical(synth_feat, fold):
    """Same seed, same data -> same numbers. Required for a re-checkable Phase 5."""
    e = md.bakeoff_entrants()["lgb_poisson"][0]
    a = md.fit_predict_fold(synth_feat, fold, e, exclude=set())
    b = md.fit_predict_fold(synth_feat, fold, e, exclude=set())
    np.testing.assert_array_equal(a["forecast"].to_numpy(), b["forecast"].to_numpy())


def test_determinism_params_are_actually_set():
    for e in md.bakeoff_entrants()["lgb_l2"]:
        p = e.params
        assert p["deterministic"] is True and p["force_row_wise"] is True
        assert p["seed"] == md.SEED and p["num_threads"] == md.NUM_THREADS


# --- entrant contract ------------------------------------------------------

def test_forecasts_are_non_negative_for_every_entrant(synth_feat, fold):
    """L2 and quantile CAN emit negatives; the clamp is load-bearing, not cosmetic."""
    for name, entrants in md.bakeoff_entrants().items():
        pred = md.fit_predict_fold(synth_feat, fold, entrants[0], exclude=set())
        assert (pred["forecast"] >= 0).all(), name


def test_grid_is_pre_registered_and_small():
    """Sub-plan sec 2: the grid is a module constant, written before any run."""
    assert md.CAPACITY_GRID == (
        {"num_leaves": 15, "min_data_in_leaf": 20},
        {"num_leaves": 15, "min_data_in_leaf": 100},
        {"num_leaves": 31, "min_data_in_leaf": 20},
        {"num_leaves": 31, "min_data_in_leaf": 100},
    )
    counts = {k: len(v) for k, v in md.bakeoff_entrants().items()}
    assert counts == {"lgb_l2": 4, "lgb_poisson": 4, "lgb_quantile50": 4, "lgb_tweedie": 12}


def test_config_id_is_stable_and_distinguishing():
    es = md.bakeoff_entrants()["lgb_tweedie"]
    ids = [e.config_id for e in es]
    assert len(set(ids)) == len(ids)
    assert md.Entrant("x", {"objective": "poisson"}, {"num_leaves": 15}).config_id == \
           md.Entrant("x", {"objective": "poisson"}, {"num_leaves": 15}).config_id


def test_prediction_frame_shape_and_alignment(synth_feat, fold):
    e = md.bakeoff_entrants()["lgb_l2"][0]
    pred = md.fit_predict_fold(synth_feat, fold, e, exclude=set())
    assert list(pred.columns) == ["id", "day_idx", "actual", "forecast", "fold"]
    assert (pred["fold"] == fold.index).all()
    assert pred["day_idx"].between(fold.val_start, fold.val_end).all()
    assert not pred["forecast"].isna().any()
