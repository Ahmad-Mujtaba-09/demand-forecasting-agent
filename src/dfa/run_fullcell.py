"""Full-cell confirmation for the bake-off winner (Phase 3 sub-plan sec 6).

Phase 2 deferred the A/B full-cell run; it is closed here. Tuning stays on the
250-series samples -- Phase 1 verified sample fidelity and re-tuning on the full
cells would multiply compute for no decision value -- but the headline of a phase
should not rest on a sample when the full cell costs one extra fit per fold.

The winning objective is refit with its FROZEN config (the same params the
contract ships to Phase 4) on every series in the cell. If the full-cell number
diverges materially from the sample, that is itself a finding about sample
fidelity and gets reported rather than buried.
"""

from __future__ import annotations

import json
import time

import pandas as pd

from . import config
from . import features as ft
from . import models as md
from .build_contract import _params_for
from .data_loader import build_long, last_sales_day
from .manifest import hash_ids, run_manifest
from .reference_lines import load_thresholds
from .run_bakeoff import branch_of, score
from .splits import make_folds

# Only A and B were sampled; C already ran at full-cell size (149 = the whole cell).
FULL_CELL_TARGETS: dict[str, str] = {"A_dense": "A", "B_intermittent": "B"}


def _entrant_from_config_id(config_id: str) -> md.Entrant:
    params = _params_for(config_id)
    name = config_id.split("[")[0]
    objective = {"objective": params["objective"]}
    for k in ("alpha", "tweedie_variance_power"):
        if k in params:
            objective[k] = params[k]
    capacity = {k: params[k] for k in ("num_leaves", "min_data_in_leaf")}
    return md.Entrant(name=name, objective=objective, capacity=capacity)


def full_cell_ids(meta: dict, table: pd.DataFrame) -> list[str]:
    """Every series in the cell -- the sampled ids are a subset of these."""
    cell = table[(table["store_id"] == meta["store_id"])
                 & (table["dept_id"] == meta["dept_id"])]
    return sorted(cell["id"])


def main() -> None:
    table = pd.read_parquet(config.ARTIFACTS_DIR / "signal_table.parquet")
    sel = json.loads((config.ARTIFACTS_DIR / "datasets" / "selection.json").read_text())
    results = json.loads((config.ARTIFACTS_DIR / "phase3_bakeoff_results.json").read_text())
    thr = load_thresholds()

    out = {"manifest": run_manifest(step="phase3_full_cell"), "datasets": {}}
    for key, short in FULL_CELL_TARGETS.items():
        meta = sel["datasets"][key]
        r = results["datasets"][short]
        entrant = _entrant_from_config_id(r["winner_config"])

        ids = full_cell_ids(meta, table)
        signal_rows = table[table["id"].isin(ids)].copy()
        branches = branch_of(signal_rows, thr)
        b2 = set(branches[branches == "B2_baseline"].index)

        train_end = last_sales_day() - config.HORIZON
        t0 = time.time()
        feat = ft.build_features(build_long(subset_ids=ids, max_day=train_end))
        folds = make_folds(train_end=train_end)
        pred = md.predict_all_folds(feat, folds, entrant, b2)
        secs = round(time.time() - t0, 1)

        full = score(pred)
        sample = r["entrants"][r["winner"]]["pooled"]
        out["datasets"][short] = {
            "cell": meta["cell"] if "cell" in meta else r["cell"],
            "winner": r["winner"], "winner_config": r["winner_config"],
            "n_sample": meta["n_sampled"], "n_full": len(ids),
            "ids_hash_full": hash_ids(ids),
            "n_modelable_full": len(ids) - len(b2), "n_b2_full": len(b2),
            "sample_score": sample, "full_score": full,
            "delta_wmape_h": round(full["wmape_h"] - sample["wmape_h"], 4),
            "delta_wmape": round(full["wmape"] - sample["wmape"], 4),
            "fit_seconds": secs,
        }
        print(f"  {short}: n {meta['n_sampled']} -> {len(ids)}   "
              f"WMAPE_28 {sample['wmape_h']:.4f} -> {full['wmape_h']:.4f}   ({secs}s)",
              flush=True)

    path = config.ARTIFACTS_DIR / "phase3_fullcell_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
