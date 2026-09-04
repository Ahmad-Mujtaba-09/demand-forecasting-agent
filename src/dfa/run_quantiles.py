"""B4 quantile models: inventory intervals (Phase 3 sub-plan sec 5).

Requirement-driven, NOT a bake-off entrant. Quantiles answer a different question
than a point forecast -- a range, for safety-stock sizing -- so they are judged on
their own terms and never scored against Tweedie on WMAPE, which would penalise a
q90 forecast for correctly forecasting high. (The q=0.5 fit is the same model as
bake-off entrant E4; because every fit is seeded and deterministic, refitting it
here reproduces E4 exactly.)

Three diagnostics, each answering a question pinball loss alone cannot:

- **pinball loss** -- the loss the model is actually fit to minimise, per level.
- **coverage** -- fraction of actuals at or below the q-forecast; should be ~= q.
  A band that is nominally 80% wide but covers 60% of outcomes is worse than
  useless for safety stock, and only coverage reveals it.

  **Read low-quantile coverage against the zero share, not against the target.**
  Demand is a zero-inflated COUNT, so a forecast of 0 is simultaneously the 10th,
  the 30th and (on a 91%-zero series) the 90th percentile -- the quantile
  function is flat across that whole range and calibration there is not
  identified. A q10 model correctly predicting 0 therefore "covers" every zero
  day, and its coverage converges to P(y = 0) rather than to 0.10. That is a
  property of the distribution, not a fault in the model, which is why
  `zero_share_rows` is reported beside every coverage figure and why
  `coverage_strict` (a strict `<`, ties excluded) is reported as the other
  bound. The honest reading is that a low-quantile coverage sitting at the zero
  share means the interval carries no information at that level.
- **crossing rate** -- the three levels are three INDEPENDENT fits, so nothing
  stops q10 > q90, which would emit a negative-width interval into a
  replenishment calculation. We sort each row's predictions before use and report
  how often that repair was needed; a high rate means the fits are unstable and
  the intervals should not be trusted.

Deferred and NOT approximated (sub-plan sec 5): intervals on the 28-day TOTAL,
which is what a lead-time safety stock actually needs. The quantile of a sum is
not the sum of quantiles, and summing daily quantiles would be wrong in the
direction that UNDER-sizes safety stock. Named as future work instead.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from . import features as ft
from . import models as md
from .data_loader import build_long, last_sales_day
from .manifest import run_manifest
from .metrics import coverage, crossing_rate, monotone_quantiles, pinball_loss
from .reference_lines import WORKING, b2_ids, load_thresholds
from .run_bakeoff import branch_of
from .splits import make_folds

LEVELS: tuple[float, ...] = (0.1, 0.5, 0.9)


def evaluate_dataset(meta: dict, table: pd.DataFrame, thr) -> dict:
    ids = meta["ids"]
    signal_rows = table[table["id"].isin(ids)].copy()
    branches = branch_of(signal_rows, thr)
    b2 = set(branches[branches == "B2_baseline"].index)

    train_end = last_sales_day() - config.HORIZON
    feat = ft.build_features(build_long(subset_ids=ids, max_day=train_end))
    folds = make_folds(train_end=train_end)
    sel_folds = {f.index for f in folds[:-1]}

    chosen, frames = {}, {}
    for q in LEVELS:
        trials = []
        for e in md.quantile_entrants(q):
            pred = md.predict_all_folds(feat, folds, e, b2)
            sel = pred[pred["fold"].isin(sel_folds)]
            # selected on the loss the model is fit to, on the selection folds only
            loss = (pinball_loss(sel["actual"], sel["forecast"], q)
                    if len(sel) else float("inf"))
            trials.append((loss, e.config_id, pred))
        loss, cfg, pred = min(trials, key=lambda t: t[0])
        chosen[q] = {"config_id": cfg, "pinball_selection": round(loss, 5)}
        frames[q] = pred.sort_values(["id", "day_idx"]).reset_index(drop=True)

    # all three frames share row order, so they can be stacked per row
    base = frames[LEVELS[0]]
    raw = {q: frames[q]["forecast"].to_numpy() for q in LEVELS}
    rate = crossing_rate(raw)
    fixed = monotone_quantiles(raw)

    actual = base["actual"].to_numpy()
    zero_share = float(np.mean(actual == 0))
    per_level = {}
    for q in LEVELS:
        cov = coverage(actual, fixed[q])
        per_level[str(q)] = {
            "config_id": chosen[q]["config_id"],
            "pinball": round(pinball_loss(actual, fixed[q], q), 5),
            "coverage": round(cov, 4),
            # strict `<`: the other bound on coverage when ties at zero dominate
            "coverage_strict": round(float(np.mean(actual < fixed[q])), 4),
            "coverage_target": q,
            # a coverage sitting at the zero share means the level is not
            # identified on this data, not that the model is miscalibrated
            "confounded_by_zero_mass": bool(abs(cov - zero_share) < 0.05 and q < 0.5),
            "mean_forecast": round(float(np.mean(fixed[q])), 4),
        }
    return {
        "cell": f'{meta["store_id"]} x {meta["dept_id"]}',
        "n_series_modelable": len(ids) - len(b2),
        "n_rows": len(base),
        "zero_share_rows": round(zero_share, 4),
        "levels": per_level,
        "crossing_rate": round(rate, 5),
        "mean_interval_width_q10_q90": round(float(np.mean(fixed[0.9] - fixed[0.1])), 4),
        "interval_coverage_q10_q90": round(
            float(np.mean((actual >= fixed[0.1]) & (actual <= fixed[0.9]))), 4),
        "interval_coverage_target": 0.8,
    }


def main() -> None:
    table = pd.read_parquet(config.ARTIFACTS_DIR / "signal_table.parquet")
    sel = json.loads((config.ARTIFACTS_DIR / "datasets" / "selection.json").read_text())
    thr = load_thresholds()

    out = {"manifest": run_manifest(step="phase3_quantiles", levels=list(LEVELS)),
           "datasets": {}}
    for key, short in WORKING.items():
        out["datasets"][short] = evaluate_dataset(sel["datasets"][key], table, thr)
        print(f"  {short} done", flush=True)

    path = config.ARTIFACTS_DIR / "phase3_quantile_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}\n")
    for short, r in out["datasets"].items():
        print(f"=== {short}: {r['cell']}   crossing={r['crossing_rate']:.2%}   "
              f"zero-share of eval rows={r['zero_share_rows']:.4f}")
        print(f"  {'level':<8}{'pinball':>10}{'coverage':>10}{'strict':>9}{'target':>9}")
        for q, v in r["levels"].items():
            flag = "  <-- at the zero share: level not identified" if v[
                "confounded_by_zero_mass"] else ""
            print(f"  q{float(q):<7.2f}{v['pinball']:>10.5f}{v['coverage']:>10.4f}"
                  f"{v['coverage_strict']:>9.4f}{v['coverage_target']:>9.2f}{flag}")
        print(f"  q10-q90 interval: coverage={r['interval_coverage_q10_q90']:.4f} "
              f"(target 0.80), mean width={r['mean_interval_width_q10_q90']:.3f}\n")


if __name__ == "__main__":
    main()
