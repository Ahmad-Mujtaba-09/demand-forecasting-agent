"""Phase 2 Increment 2 tests: rolling-origin folds, data-driven count, guard."""

from __future__ import annotations

import pytest

from dfa import splits
from dfa import config


def test_m5_yields_five_folds():
    # data-driven formula lands on the reviewed choice for M5's history length
    folds = splits.make_folds()
    assert len(folds) == 5
    assert splits.n_folds_for() == 5


def test_horizon_is_28_and_val_starts_after_origin():
    for f in splits.make_folds():
        assert f.val_end - f.val_start + 1 == config.HORIZON
        assert f.val_start == f.origin + 1  # the leakage boundary invariant


def test_latest_fold_ends_at_train_end_never_past():
    folds = splits.make_folds()
    assert max(f.val_end for f in folds) == config.TRAIN_END_DAY
    assert all(f.val_end <= config.TRAIN_END_DAY for f in folds)


def test_folds_are_contiguous_and_non_overlapping():
    folds = splits.make_folds()  # earliest first
    for a, b in zip(folds, folds[1:]):
        assert a.val_end < b.val_start        # non-overlapping
        assert b.val_start == a.val_end + 1   # contiguous 28-day blocks


def test_earliest_origin_matches_plan_and_respects_guard():
    folds = splits.make_folds()
    assert folds[0].origin == 1773                       # 1913 - 5*28
    assert folds[0].origin >= splits.MIN_TRAIN_DAYS       # min-history guard holds


def test_indices_are_ordered_earliest_first():
    folds = splits.make_folds()
    assert [f.index for f in folds] == [0, 1, 2, 3, 4]
    assert folds[0].origin < folds[-1].origin


def test_count_is_data_driven_shorter_history_fewer_folds():
    # fewer days -> fewer folds, and the guard is still respected
    n = splits.n_folds_for(train_end=500, min_train_days=365)  # (500-365)//28 = 4
    assert n == 4
    folds = splits.make_folds(train_end=500, min_train_days=365)
    assert len(folds) == 4
    assert folds[0].origin >= 365
    assert max(f.val_end for f in folds) == 500


def test_count_is_capped_by_max_folds():
    # long history would allow many folds but the cap holds
    assert splits.n_folds_for(train_end=5000, min_train_days=365) == splits.MAX_FOLDS
    assert splits.n_folds_for(train_end=5000, min_train_days=365, max_folds=3) == 3


def test_insufficient_history_raises():
    # not even one guarded fold fits -> raise, don't train on a thin window
    with pytest.raises(ValueError):
        splits.make_folds(train_end=380, min_train_days=365)  # only 15 usable days


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
