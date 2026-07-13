"""Group profiling + dataset-selection tests.

Two kinds: end-to-end tests on the real signal table (skipped if it isn't built),
and synthetic-`prof` unit tests for the exclusion/fallback/error logic, which run
without the parquet.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dfa import config
from dfa import select_datasets as sd

PARQUET = config.ARTIFACTS_DIR / "signal_table.parquet"


@pytest.fixture(scope="module")
def table():
    if not PARQUET.exists():
        pytest.skip("signal_table.parquet not built yet")
    return pd.read_parquet(PARQUET)


def _prof(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal group-profile frame for select_groups().

    Each row needs: store_id, dept_id, n, med_zero, med_nonzero,
    pct_intermittent, dist_to_median. Missing keys get benign defaults.
    """
    defaults = dict(med_zero=0.5, med_nonzero=2.0, pct_intermittent=0.5, dist_to_median=1.0)
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_four_datasets_selected(table):
    sel = sd.build_selection(table)
    assert set(sel["datasets"]) == {"A_dense", "B_intermittent", "C_sparse", "D_heldout"}


def test_selection_is_deterministic(table):
    a = sd.build_selection(table)["datasets"]
    b = sd.build_selection(table)["datasets"]
    for k in a:
        assert a[k]["ids"] == b[k]["ids"]


def test_datasets_disjoint_heldout_integrity(table):
    ds = sd.build_selection(table)["datasets"]
    id_sets = {k: set(v["ids"]) for k, v in ds.items()}
    labels = list(id_sets)
    for i, x in enumerate(labels):
        for y in labels[i + 1:]:
            assert not (id_sets[x] & id_sets[y]), f"{x} and {y} overlap"


def test_sample_profile_fidelity(table):
    ds = sd.build_selection(table)["datasets"]
    for k, v in ds.items():
        assert v["fidelity_ok"], f"{k} sample profile drifted from full group"
        assert v["n_sampled"] <= sd.SAMPLE_N


def test_extremes_are_ordered(table):
    """A must be the densest and C the sparsest of the four picks."""
    ds = sd.build_selection(table)["datasets"]
    z = {k: v["full_profile"]["med_zero"] for k, v in ds.items()}
    assert z["A_dense"] == min(z.values())
    assert z["C_sparse"] == max(z.values())


def test_b_is_modelable_and_intermittent(table):
    """B (Tweedie) must clear the modelable non-zero floor and be zero-heavy."""
    ds = sd.build_selection(table)["datasets"]
    b = ds["B_intermittent"]["full_profile"]
    assert b["med_nonzero"] >= sd.MODELABLE_NONZERO
    assert b["med_zero"] > 0.5


def test_heldout_distinct_department(table):
    ds = sd.build_selection(table)["datasets"]
    d_dept = ds["D_heldout"]["dept_id"]
    others = {ds[k]["dept_id"] for k in ["A_dense", "B_intermittent", "C_sparse"]}
    assert d_dept not in others


# --- synthetic-prof unit tests: exclusions, fallbacks, errors (no parquet) ---

def _cells(groups: dict) -> set:
    return {(g["store_id"], g["dept_id"]) for g in groups.values()}


def test_no_evaluable_group_raises():
    prof = _prof([{"store_id": "S1", "dept_id": "D1", "n": 50}])  # < MIN_EVAL_SERIES
    with pytest.raises(sd.DatasetSelectionError, match="MIN_EVAL_SERIES"):
        sd.select_groups(prof)


def test_a_raises_when_no_full_population_group():
    # evaluable but none reaches SAMPLE_N -> A cannot form a full training population
    prof = _prof([
        {"store_id": "S1", "dept_id": "D1", "n": 120},
        {"store_id": "S2", "dept_id": "D2", "n": 130},
    ])
    with pytest.raises(sd.DatasetSelectionError, match="A_dense"):
        sd.select_groups(prof)


def test_exclusions_symmetric_four_distinct_cells():
    # four clean depts -> four distinct cells, none shared
    prof = _prof([
        {"store_id": "S1", "dept_id": "FOODS", "n": 300, "med_zero": 0.20, "med_nonzero": 3.0, "pct_intermittent": 0.3, "dist_to_median": 0.9},
        {"store_id": "S1", "dept_id": "HOBBY", "n": 300, "med_zero": 0.95, "med_nonzero": 1.2, "pct_intermittent": 0.9, "dist_to_median": 0.9},
        {"store_id": "S1", "dept_id": "HOUSE", "n": 300, "med_zero": 0.70, "med_nonzero": 1.8, "pct_intermittent": 0.85, "dist_to_median": 0.9},
        {"store_id": "S1", "dept_id": "OTHER", "n": 300, "med_zero": 0.55, "med_nonzero": 1.7, "pct_intermittent": 0.5, "dist_to_median": 0.1},
    ])
    groups = sd.select_groups(prof)
    assert len(_cells(groups)) == 4  # all disjoint cells
    assert groups["A_dense"]["dept_id"] == "FOODS"   # densest
    assert groups["C_sparse"]["dept_id"] == "HOBBY"  # sparsest
    assert groups["D_heldout"]["dept_id"] == "OTHER"  # closest to median, unused dept


def test_b_relaxes_distinct_dept_when_needed():
    # the only modelable+big intermittent group shares a dept with A; the sole
    # distinct-dept group is too small to be "big" -> B must relax distinct-dept
    prof = _prof([
        {"store_id": "S1", "dept_id": "FOODS", "n": 300, "med_zero": 0.20, "med_nonzero": 3.0, "pct_intermittent": 0.2},
        {"store_id": "S2", "dept_id": "FOODS", "n": 300, "med_zero": 0.68, "med_nonzero": 1.6, "pct_intermittent": 0.9},  # shares dept with A
        {"store_id": "S1", "dept_id": "HOBBY", "n": 300, "med_zero": 0.95, "med_nonzero": 1.2, "pct_intermittent": 0.9},
        {"store_id": "S3", "dept_id": "HOUSE", "n": 150, "med_zero": 0.55, "med_nonzero": 1.7, "pct_intermittent": 0.4, "dist_to_median": 0.1},  # evaluable but < SAMPLE_N
    ])
    groups = sd.select_groups(prof)
    assert groups["B_intermittent"]["dept_id"] == "FOODS"       # reused dept, distinct cell
    assert groups["B_intermittent"]["store_id"] == "S2"
    assert "RELAXED" in groups["B_intermittent"]["rule"]
    assert groups["D_heldout"]["dept_id"] == "HOUSE"            # D still gets a novel dept
    assert len(_cells(groups)) == 4                              # still disjoint


def test_b_raises_when_no_room():
    # exactly two evaluable groups -> A and C consume both, B has no cell left
    prof = _prof([
        {"store_id": "S1", "dept_id": "FOODS", "n": 300, "med_zero": 0.20, "med_nonzero": 3.0},
        {"store_id": "S1", "dept_id": "HOBBY", "n": 300, "med_zero": 0.95, "med_nonzero": 1.2},
    ])
    with pytest.raises(sd.DatasetSelectionError, match="B_intermittent"):
        sd.select_groups(prof)


def test_d_degrades_to_reused_dept_when_depts_exhausted():
    # A/B/C use FOODS/HOBBY/HOUSE; only remaining group reuses HOUSE (distinct cell)
    prof = _prof([
        {"store_id": "S1", "dept_id": "FOODS", "n": 300, "med_zero": 0.20, "med_nonzero": 3.0, "pct_intermittent": 0.2},
        {"store_id": "S2", "dept_id": "HOBBY", "n": 300, "med_zero": 0.95, "med_nonzero": 1.2, "pct_intermittent": 0.9},
        {"store_id": "S3", "dept_id": "HOUSE", "n": 300, "med_zero": 0.70, "med_nonzero": 1.8, "pct_intermittent": 0.9},
        {"store_id": "S4", "dept_id": "HOUSE", "n": 300, "med_zero": 0.55, "med_nonzero": 1.7, "pct_intermittent": 0.5, "dist_to_median": 0.1},  # reused dept
    ])
    groups = sd.select_groups(prof)
    assert groups["D_heldout"]["dept_id"] == "HOUSE"          # reused dept
    assert groups["D_heldout"]["store_id"] == "S4"            # distinct cell
    assert "DEGRADED" in groups["D_heldout"]["rule"]
    assert len(_cells(groups)) == 4                            # still series-disjoint


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
