"""Increment 5 tests: threshold calibration + branch classifier.

Covers held-out integrity (D excluded from calibration), the precedence of the
deterministic classifier, and non-degenerate branch coverage.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dfa import config
from dfa import calibrate_thresholds as ct

PARQUET = config.ARTIFACTS_DIR / "signal_table.parquet"
SELECTION = config.ARTIFACTS_DIR / "datasets" / "selection.json"
pytestmark = pytest.mark.skipif(
    not (PARQUET.exists() and SELECTION.exists()),
    reason="signal table / selection not built yet",
)


@pytest.fixture(scope="module")
def thr():
    return ct.calibrate(ct.load_nond_pool())


# --- held-out integrity ---

def test_calibration_pool_excludes_heldout():
    import pandas as pd
    pool = ct.load_nond_pool()
    d_ids = set(json.loads(SELECTION.read_text())["datasets"]["D_heldout"]["ids"])
    assert len(pool) == 30490 - len(d_ids)
    assert not (set(pool["id"]) & d_ids), "held-out D leaked into calibration pool"


# --- classifier precedence & determinism ---

def test_sparsity_precedes_tweedie(thr):
    # a very zero-heavy AND very sparse series must be B2, not B1
    branch, _ = ct.classify_branch(zero_share=0.95, adi=20.0, dow_season=0.3, thr=thr)
    assert branch == "B2_baseline"


def test_modelable_zeroheavy_is_tweedie(thr):
    branch, _ = ct.classify_branch(zero_share=0.75, adi=3.0, dow_season=0.3, thr=thr)
    assert branch == "B1_tweedie"


def test_dense_is_standard(thr):
    branch, _ = ct.classify_branch(zero_share=0.30, adi=1.5, dow_season=0.3, thr=thr)
    assert branch == "standard"


def test_b1_cut_means_high_not_median(thr):
    """Issue 1 guard: 'high zero-share' must be genuinely high, not the median.

    A modelable series at the non-sparse median zero-share must route to standard,
    and only a clear minority of modelable series should clear the Tweedie cut.
    """
    import pandas as pd
    pool = ct.load_nond_pool()
    non_sparse = pool[pool["adi"] < thr.adi_sparse_cut]
    median_zero = float(non_sparse["zero_share"].median())
    assert median_zero < thr.zero_share_tweedie_cut  # median is below the cut
    assert ct.classify_branch(median_zero, adi=3.0, dow_season=0.0, thr=thr)[0] == "standard"
    frac_tweedie = float((non_sparse["zero_share"] >= thr.zero_share_tweedie_cut).mean())
    assert frac_tweedie <= 0.45  # a minority ("high"), not ~half (which P50 gives)


def test_b1_cut_calibrated_on_nonsparse_not_full_pool():
    """Issue 2 guard: the Tweedie cut is a percentile of the non-sparse subset.

    A sparse, very-zero-heavy tail must not inflate the cut for the (non-sparse)
    population it actually governs.
    """
    import numpy as np
    import pandas as pd
    non_sparse = pd.DataFrame({
        "adi": [2.0] * 100, "zero_share": np.linspace(0.30, 0.80, 100),
        "dow_season": [0.02] * 100, "max_med_ratio": [5.0] * 100,
    })
    sparse = pd.DataFrame({  # huge ADI + near-1 zero-share; caught by B2 first
        "adi": [50.0] * 100, "zero_share": [0.98] * 100,
        "dow_season": [0.02] * 100, "max_med_ratio": [5.0] * 100,
    })
    pool = pd.concat([non_sparse, sparse], ignore_index=True)
    thr = ct.calibrate(pool)
    expected = round(float(np.percentile(non_sparse["zero_share"], ct.ZERO_TWEEDIE_PCT)), 3)
    assert thr.zero_share_tweedie_cut == expected
    # the full-pool percentile would be dragged up by the 0.98 tail
    assert thr.zero_share_tweedie_cut < float(np.percentile(pool["zero_share"], ct.ZERO_TWEEDIE_PCT))


def test_nan_adi_is_sparse(thr):
    branch, _ = ct.classify_branch(zero_share=0.5, adi=float("nan"), dow_season=0.3, thr=thr)
    assert branch == "B2_baseline"


def test_seasonal_flag_orthogonal(thr):
    _, seasonal_hi = ct.classify_branch(0.75, 3.0, thr.dow_season_cut + 0.05, thr)
    _, seasonal_lo = ct.classify_branch(0.75, 3.0, thr.dow_season_cut - 0.05, thr)
    assert seasonal_hi is True and seasonal_lo is False


def test_thresholds_ordered_and_bounded(thr):
    assert 0.0 < thr.zero_share_tweedie_cut < 1.0
    assert thr.adi_sparse_cut > 1.0
    assert 0.0 < thr.dow_season_cut < 1.0
    assert thr.min_forecast == 0.0


def test_branch_coverage_non_degenerate(thr):
    pool = ct.load_nond_pool()
    branches = pool.apply(
        lambda r: ct.classify_branch(r["zero_share"], r["adi"], r["dow_season"], thr)[0],
        axis=1,
    )
    counts = branches.value_counts()
    for b in ["standard", "B1_tweedie", "B2_baseline"]:
        assert counts.get(b, 0) > 0, f"branch {b} is empty"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
