"""Phase 2 predictions re-scored under the Phase 3 metrics (sub-plan increment 2).

No new models. This runs the existing Ridge / B2-floor / naive / zero forecasts
through `wmape_horizon` and `oracle_lines` so the bake-off has reference lines in
BOTH metrics, and so the sub-plan sec 0.1 finding is reproduced by the pipeline
rather than by an ad-hoc script.

The numbers this emits are the bar Phase 3 entrants are compared against:

  - `zero`          -- identically 1.0 under both metrics; "worse than doing nothing"
  - `oracle_median` -- best constant-per-series forecast; the daily-WMAPE reference
  - `oracle_mean`   -- same for a MEAN-targeting objective (Tweedie/Poisson). Where
                       this exceeds 1.0, no mean-estimator can clear the zero line,
                       which is the whole reason Phase 3 reports WMAPE_28.
  - `ridge`/`naive` -- the Phase 2 floor and its constant comparator
"""

from __future__ import annotations

import json

import pandas as pd

from . import baseline as bl
from . import config
from . import features as ft
from .calibrate_thresholds import Thresholds, classify_branch
from .data_loader import build_long, last_sales_day
from .manifest import hash_ids, run_manifest
from .metrics import oracle_lines, wmape, wmape_horizon
from .splits import make_folds

WORKING: dict[str, str] = {"A_dense": "A", "B_intermittent": "B", "C_sparse": "C"}


def load_thresholds() -> Thresholds:
    t = json.loads((config.ARTIFACTS_DIR / "thresholds.json").read_text())["thresholds"]
    return Thresholds(**t)


def b2_ids(signal_rows: pd.DataFrame, thr: Thresholds) -> set[str]:
    """Series the deterministic router sends to the sparse baseline (B2)."""
    is_b2 = signal_rows.apply(
        lambda r: classify_branch(r["zero_share"], r["adi"], thr) == "B2_baseline", axis=1
    )
    return set(signal_rows.loc[is_b2, "id"])


def score_frame(pred: pd.DataFrame) -> dict[str, float]:
    """Both metrics for one prediction frame [id, day_idx, actual, forecast, fold]."""
    if pred.empty:
        return {"wmape": float("nan"), "wmape_h": float("nan")}
    return {
        "wmape": round(wmape(pred["actual"], pred["forecast"]), 4),
        "wmape_h": round(wmape_horizon(pred["actual"], pred["forecast"],
                                       pred["id"], pred["fold"]), 4),
    }


def evaluate(label: str, meta: dict, table: pd.DataFrame, thr: Thresholds) -> dict:
    ids = meta["ids"]
    signal_rows = table[table["id"].isin(ids)].copy()
    train_end = last_sales_day() - config.HORIZON
    long = build_long(subset_ids=ids, max_day=train_end)
    feat = ft.build_features(long)
    folds = make_folds(train_end=train_end)
    res = bl.run_baseline(feat, folds, b2_ids(signal_rows, thr))

    base = res["baseline"]
    oracles = oracle_lines(base["actual"], base["id"], base["fold"])
    return {
        "cell": f'{meta["store_id"]} x {meta["dept_id"]}',
        "n_series": len(ids),
        "ids_hash": hash_ids(ids),
        "lines": {
            "ridge_floor": score_frame(base),
            "ridge_l2_subset": score_frame(base[base["method"] == "l2"]),
            "naive_modelable": score_frame(res["naive"]),
            "zero": score_frame(res["zero"]),
        },
        "oracles": {k: round(v, 4) for k, v in oracles.items()},
        "mean_oracle_above_zero_line": bool(oracles["oracle_mean"] > 1.0),
        "routing": {"l2": res["n_l2"], "floor": res["n_b2"]},
    }


def main() -> None:
    table = pd.read_parquet(config.ARTIFACTS_DIR / "signal_table.parquet")
    sel = json.loads((config.ARTIFACTS_DIR / "datasets" / "selection.json").read_text())
    thr = load_thresholds()

    out = {"manifest": run_manifest(step="phase3_reference_lines"), "datasets": {}}
    for key, short in WORKING.items():
        out["datasets"][short] = evaluate(key, sel["datasets"][key], table, thr)

    path = config.ARTIFACTS_DIR / "phase3_reference_lines.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}\n")

    hdr = f"{'ds':<3}{'line':<20}{'WMAPE':>9}{'WMAPE_28':>10}"
    print(hdr); print("-" * len(hdr))
    for short, r in out["datasets"].items():
        for name, sc in r["lines"].items():
            print(f"{short:<3}{name:<20}{sc['wmape']:>9.4f}{sc['wmape_h']:>10.4f}")
        print(f"{short:<3}{'oracle_median':<20}{r['oracles']['oracle_median']:>9.4f}{'-':>10}")
        print(f"{short:<3}{'oracle_mean':<20}{r['oracles']['oracle_mean']:>9.4f}{'-':>10}"
              f"   {'<-- ABOVE the zero line: no mean-estimator can win on daily WMAPE'
                    if r['mean_oracle_above_zero_line'] else ''}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
