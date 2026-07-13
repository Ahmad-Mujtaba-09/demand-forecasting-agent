"""Per-dataset branch-threshold calibration.

The thresholds here are **data-relative**: `calibrate(pool)` fits every cut to
the distribution of whatever dataset is passed in, on that dataset's own
non-held-out pool. There are no dataset-specific magic numbers -- the same code
recalibrates for any M5-format dataset. (This is the "format-general, threshold-
adaptive" property from the master plan; the literature-standard Syntetos-Boylan
cutoffs, which are NOT tuned, live in signals.py as absolute constants.)

Held-out integrity: whatever pool is passed must exclude the held-out split, so
no boundary the agent uses is informed by held-out data. `load_nond_pool` does
this for the M5 validation run.

`classify_branch` is the deterministic executor rule Phases 3-4 reuse -- defined
once so the boundaries are defensible.

Findings that shaped the *method* (validated on M5, but the logic is general):
- Absolute non-zero count is not the scarce resource: even M5's 10th-percentile
  series has ~133 non-zero days. What defeats a boosted model is the demand
  *interval* (ADI), so the B2 floor is a high percentile of the dataset's own ADI
  (rationale: sells far less often than typical -> too few events to forecast the
  horizon reliably), not an absolute count.
- Weekly seasonality (day-of-week variance-explained) is weak and clustered at
  the item level, so B3 rarely fires -- for most series calendar features carry
  seasonality. Surfaced, not hidden.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config


# percentiles that define each data-relative cut (documented, not magic)
ADI_SPARSE_PCT: float = 90     # B2 floor: upper tail of the dataset's ADI
# B1 cut: "high zero-share" must actually mean HIGH, not "above median" -- so an
# upper-portion percentile (P60), and computed on the NON-SPARSE subset (the pop
# that actually reaches the B1-vs-standard decision; see calibrate()), not the
# full pool whose sparse tail would inflate it. P60 lands ~0.66 (zeros ~2/3 of
# days) on M5 -- genuinely zero-heavy, while the median modelable series routes
# to `standard`. Chosen at the low end of a defensible P60-P75 band to keep the
# most-intermittent sub-dataset Tweedie-dominant; raise it for a stricter "high".
ZERO_TWEEDIE_PCT: float = 60
DOW_SEASONAL_PCT: float = 90   # B3 cut: upper tail of dataset's DoW variance-explained
MAX_RATIO_PCT: float = 99      # critic: dataset's demand-tail ratio


@dataclass(frozen=True)
class Thresholds:
    # --- data-relative: fit to the passed dataset's own distribution ---
    adi_sparse_cut: float          # ADI >= this -> B2 (P{ADI_SPARSE_PCT} of full-pool ADI)
    zero_share_tweedie_cut: float  # zero_share >= this (and not B2) -> B1 (P60 of NON-SPARSE pool)
    dow_season_cut: float          # dow_season >= this -> add seasonal features (P90)
    max_median_ratio: float        # forecast > this * series median non-zero is implausible
    # --- absolute: domain facts / literature, NOT tuned per dataset ---
    min_forecast: float            # 0.0 -- demand is non-negative (domain fact)
    basis: str                     # provenance note


def calibrate(pool: pd.DataFrame) -> Thresholds:
    """Fit thresholds to *this dataset's* distribution (its non-held-out pool).

    Each cut is a percentile of the passed pool -- pass a different dataset, get
    different cuts; the percentiles are the fixed method.

    Two cuts govern the same objective decision but at different stages, so they
    are calibrated on the populations they actually apply to:
    - adi_sparse_cut (B2) is applied to *every* series first, so it is a
      percentile of the full pool.
    - zero_share_tweedie_cut (B1) is only ever tested on series that survived B2
      (the non-sparse subset), so it is calibrated on exactly that subset -- not
      the full pool, whose sparse, very-zero-heavy tail would inflate the cut for
      a population it never governs. Calibrated-on == applied-on.
    """
    p = lambda col, q, df=pool: float(np.percentile(df[col].dropna(), q))
    adi_sparse = round(p("adi", ADI_SPARSE_PCT), 2)
    non_sparse = pool[pool["adi"] < adi_sparse]  # the series B1 actually governs
    return Thresholds(
        adi_sparse_cut=adi_sparse,
        zero_share_tweedie_cut=round(p("zero_share", ZERO_TWEEDIE_PCT, non_sparse), 3),
        dow_season_cut=round(p("dow_season", DOW_SEASONAL_PCT), 3),
        max_median_ratio=round(p("max_med_ratio", MAX_RATIO_PCT), 1),
        min_forecast=0.0,
        basis=(f"data-relative: full pool n={len(pool)}, non-sparse n={len(non_sparse)}; "
               f"ADI P{ADI_SPARSE_PCT} (full), zero-share P{ZERO_TWEEDIE_PCT} (non-sparse), "
               f"DoW P{DOW_SEASONAL_PCT} (full), ratio P{MAX_RATIO_PCT} (full). "
               f"S-B cutoffs absolute (see signals.py)."),
    )


def classify_branch(
    zero_share: float, adi: float, dow_season: float, thr: Thresholds
) -> tuple[str, bool]:
    """Deterministic executor rule. Returns (objective_branch, add_seasonal).

    Objective branches (documented in master plan sec 3):
    - B2_baseline: too sparse (ADI >= cut) -> simple baseline (Croston/MA).
    - B1_tweedie:  modelable AND high zero-share -> Tweedie objective.
    - standard:    modelable but NOT high zero-share -> standard objective
                   (L2/Poisson, decided by the Phase 3 bake-off). Lower-
                   intermittency modelable series genuinely don't need Tweedie.

    Precedence: sparsity (B2) is checked first -- a too-sparse series falls back
    to a baseline regardless of zero_share; only modelable series reach the
    Tweedie-vs-standard decision, which is why zero_share_tweedie_cut is
    calibrated on the non-sparse subset. Seasonality is orthogonal (features
    on/off).
    """
    if not np.isfinite(adi) or adi >= thr.adi_sparse_cut:
        branch = "B2_baseline"
    elif zero_share >= thr.zero_share_tweedie_cut:
        branch = "B1_tweedie"
    else:
        branch = "standard"
    add_seasonal = bool(np.isfinite(dow_season) and dow_season >= thr.dow_season_cut)
    return branch, add_seasonal


def load_nond_pool() -> pd.DataFrame:
    table = pd.read_parquet(config.ARTIFACTS_DIR / "signal_table.parquet")
    sel = json.loads((config.ARTIFACTS_DIR / "datasets" / "selection.json").read_text())
    d_ids = set(sel["datasets"]["D_heldout"]["ids"])
    return table[~table["id"].isin(d_ids)].copy()


def _distribution_summary(pool: pd.DataFrame) -> dict:
    cols = ["zero_share", "adi", "nonzero_count", "dow_season",
            "spike_ratio", "max_med_ratio", "mean_nonzero"]
    qs = [10, 25, 50, 75, 90, 95, 99]
    return {
        c: {f"p{q}": round(float(np.percentile(pool[c].dropna(), q)), 3) for q in qs}
        for c in cols
    }


def main() -> None:
    pool = load_nond_pool()
    thr = calibrate(pool)

    # apply to the pool to report how many series each branch would claim
    branches = pool.apply(
        lambda r: classify_branch(r["zero_share"], r["adi"], r["dow_season"], thr)[0],
        axis=1,
    )
    seasonal = pool.apply(
        lambda r: classify_branch(r["zero_share"], r["adi"], r["dow_season"], thr)[1],
        axis=1,
    )

    out = {
        "thresholds": asdict(thr),
        "branch_counts_non_d": branches.value_counts().to_dict(),
        "branch_fracs_non_d": branches.value_counts(normalize=True).round(3).to_dict(),
        "seasonal_feature_frac": round(float(seasonal.mean()), 3),
        "distribution_summary_non_d": _distribution_summary(pool),
    }
    out_path = config.ARTIFACTS_DIR / "thresholds.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"wrote {out_path}\n")
    print("Thresholds (calibrated on non-D pool):")
    for k, v in asdict(thr).items():
        print(f"  {k:24} {v}")
    print("\nBranch assignment over non-D pool:")
    for b, n in out["branch_counts_non_d"].items():
        print(f"  {b:14} {n:6}  ({out['branch_fracs_non_d'][b]:.1%})")
    print(f"  add_seasonal (B3):  {out['seasonal_feature_frac']:.1%} of series")


if __name__ == "__main__":
    main()
