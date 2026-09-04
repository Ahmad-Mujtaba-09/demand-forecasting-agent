"""Phase 3 runner: the LightGBM bake-off across datasets A, B, C.

Protocol (Phase 3 sub-plan sec 0.3, sec 2, sec 3):

- Every entrant sees the FROZEN Phase-2 feature set, the same folds, the same B2
  routing and the same evaluable-row predicate. Only the objective varies.
- Each config is fitted on ALL folds once and its predictions cached, so both the
  selection view and the reporting views come from the same fits -- no wasted
  compute and no chance of the two diverging.
- **Selection uses folds 0..n-2 only; the last fold is never seen by selection.**
  Phase 2 tuned and reported on the same folds, justified by Ridge having no
  capacity to exploit the choice (its alpha-grid WMAPE spread was 1e-4). Trees
  have real capacity, so that argument is retired here: we report the pooled
  number (Phase-2-comparable, mildly optimistic, equal protocol) AND the
  selection-free last-fold number, and the gap between them IS the measured
  optimism rather than an assumption about it.
- The winner is chosen on **WMAPE_28**, with daily WMAPE as the tiebreaker inside
  fold-to-fold noise -- the daily metric is degenerate on 2 of 3 datasets and a
  rule that silently changes per dataset is not a rule.
- Per-BRANCH cohort scores are emitted alongside per-dataset ones, because the
  branch is the unit Phase 4's executor actually decides on.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import baseline as bl
from . import config
from . import croston as cr
from . import features as ft
from . import models as md
from .calibrate_thresholds import Thresholds, classify_branch
from .data_loader import build_long, last_sales_day
from .manifest import hash_columns, hash_ids, run_manifest
from .metrics import oracle_lines, wmape, wmape_by_group, wmape_horizon
from .reference_lines import WORKING, load_thresholds
from .run_baseline import leakage_check
from .splits import make_folds

# Below this many series a pooled model is not a model; the cohort routes to the
# B2 fallback instead (sub-plan sec 3). Set from the observed cohort sizes on
# A/B/C -- the smallest cohort that produced a stable score here was ~28 series,
# so 20 is a floor just below that, not a number tuned to make a cohort pass.
MIN_COHORT_SERIES: int = 20


def branch_of(signal_rows: pd.DataFrame, thr: Thresholds) -> pd.Series:
    """Per-series branch from the deterministic router -- the Phase 4 decision unit."""
    return signal_rows.set_index("id").apply(
        lambda r: classify_branch(r["zero_share"], r["adi"], thr), axis=1
    )


def score(pred: pd.DataFrame) -> dict:
    """Both metrics on one prediction frame, plus the row count behind them."""
    if pred.empty or pred["actual"].sum() <= 0:
        return {"wmape": float("nan"), "wmape_h": float("nan"), "n_rows": len(pred)}
    return {
        "wmape": round(wmape(pred["actual"], pred["forecast"]), 4),
        "wmape_h": round(wmape_horizon(pred["actual"], pred["forecast"],
                                       pred["id"], pred["fold"]), 4),
        "n_rows": len(pred),
    }


def _critic_violations(pred: pd.DataFrame, feat: pd.DataFrame, thr: Thresholds) -> dict:
    """Exercise the Phase 4 critic's bounds NOW (sub-plan sec 4.3).

    Phase 4 wires these into a gate that can REJECT the bake-off winner.
    Discovering there that the winner trips it on 3% of rows -- or that the bound
    never fires and is decorative -- is discovering it a phase too late.
    """
    nonzero = feat[feat["units"] > 0]
    med_nz = nonzero.groupby("id", observed=True)["units"].median()
    cap = pred["id"].map(med_nz) * thr.max_median_ratio
    over = (pred["forecast"] > cap) & cap.notna()
    return {
        "below_min_forecast": int((pred["forecast"] < thr.min_forecast).sum()),
        "above_max_median_ratio": int(over.sum()),
        "above_max_median_ratio_rate": round(float(over.mean()), 6) if len(pred) else 0.0,
    }


def run_entrant(feat, folds, exclude, entrant) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    pred = md.predict_all_folds(feat, folds, entrant, exclude)
    return pred, round(time.time() - t0, 1)


def evaluate_dataset(label: str, meta: dict, table: pd.DataFrame, thr: Thresholds) -> dict:
    ids = meta["ids"]
    signal_rows = table[table["id"].isin(ids)].copy()
    branches = branch_of(signal_rows, thr)
    b2 = set(branches[branches == "B2_baseline"].index)

    train_end = last_sales_day() - config.HORIZON
    long = build_long(subset_ids=ids, max_day=train_end)
    feat = ft.build_features(long)
    folds = make_folds(train_end=train_end)
    sel_folds = {f.index for f in folds[:-1]}      # selection sees all but the last
    last_fold = folds[-1].index

    def views(pred: pd.DataFrame) -> dict:
        """Pooled (all folds) + selection-free last fold. Same fits, two views."""
        return {
            "pooled": score(pred),
            "last_fold": score(pred[pred["fold"] == last_fold]),
            "selection": score(pred[pred["fold"].isin(sel_folds)]),
        }

    results, timings, preds = {}, {}, {}
    for name, configs in md.bakeoff_entrants().items():
        trials = []
        for e in configs:
            pred, secs = run_entrant(feat, folds, b2, e)
            trials.append({"config_id": e.config_id, **views(pred),
                           "fit_seconds": secs})
            preds[e.config_id] = pred
            timings[e.config_id] = secs
        # pick this entrant's config on the SELECTION folds only, by WMAPE_28
        best = min(trials, key=lambda t: (np.isnan(t["selection"]["wmape_h"]),
                                          t["selection"]["wmape_h"]))
        results[name] = {"chosen_config": best["config_id"], "trials": trials,
                         **{k: best[k] for k in ("pooled", "last_fold", "selection")}}

    # --- per-branch cohort scores for the chosen config of each entrant -------
    cohort_ids = {b: set(branches[branches == b].index) for b in branches.unique()}
    cohorts = {}
    for b, members in cohort_ids.items():
        if b == "B2_baseline":
            continue                      # B2 never reaches a LightGBM entrant
        cohorts[b] = {"n_series": len(members),
                      "thin": len(members) < MIN_COHORT_SERIES, "entrants": {}}
        for name, res in results.items():
            p = preds[res["chosen_config"]]
            cohorts[b]["entrants"][name] = score(p[p["id"].isin(members)])

    # --- B2 cohort: Croston/SBA vs the Phase 2 mean floor --------------------
    b2_block = {}
    if b2:
        mean_floor = bl._floor_predictions(feat, folds, b2, method="floor")
        sba = cr.croston_predictions(feat, folds, b2, sba=True)
        croston = cr.croston_predictions(feat, folds, b2, sba=False, method="croston")
        zero_b2 = bl._zero_predictions(feat, folds, b2)
        b2_block = {"n_series": len(b2),
                    "mean_floor": score(mean_floor), "sba": score(sba),
                    "croston": score(croston), "zero": score(zero_b2)}

    # --- reference lines on the modelable rows the entrants actually saw -----
    any_pred = preds[next(iter(preds))]
    oracles = oracle_lines(any_pred["actual"], any_pred["id"], any_pred["fold"])
    zero_ref = any_pred.assign(forecast=0.0)

    # --- winner by WMAPE_28 on the selection folds ---------------------------
    winner = min(results, key=lambda n: (np.isnan(results[n]["selection"]["wmape_h"]),
                                         results[n]["selection"]["wmape_h"]))
    win_pred = preds[results[winner]["chosen_config"]]
    critic = _critic_violations(win_pred, feat, thr)

    # Per-SB-class breakdown for the winner -- the Phase 2 honesty check carried
    # forward. The aggregate is volume-weighted, so a strong headline can hide a
    # poor intermittent tail; the split is what prevents reading the headline as
    # if it applied uniformly.
    sb = signal_rows.set_index("id")["sb_class"]
    klass = win_pred["id"].map(sb).to_numpy()
    by_class = {
        k: {"wmape": round(v["wmape"], 4), "n_obs": v["n_obs"],
            "wmape_h": round(wmape_horizon(
                win_pred.loc[klass == k, "actual"], win_pred.loc[klass == k, "forecast"],
                win_pred.loc[klass == k, "id"], win_pred.loc[klass == k, "fold"]), 4)}
        for k, v in wmape_by_group(win_pred["actual"], win_pred["forecast"], klass).items()
    }

    return {
        "cell": f'{meta["store_id"]} x {meta["dept_id"]}',
        "n_series": len(ids), "ids_hash": hash_ids(ids),
        "n_modelable": len(ids) - len(b2), "n_b2": len(b2),
        "folds": {"selection": sorted(sel_folds), "held_out": last_fold},
        "entrants": results,
        "winner": winner,
        "winner_config": results[winner]["chosen_config"],
        "cohorts": cohorts,
        "b2_cohort": b2_block,
        "reference": {
            "zero": score(zero_ref),
            "oracle_median": round(oracles["oracle_median"], 4),
            "oracle_mean": round(oracles["oracle_mean"], 4),
            "mean_oracle_above_zero_line": bool(oracles["oracle_mean"] > 1.0),
        },
        "winner_wmape_by_sb_class": by_class,
        # per-fold spread for the winner: is the headline a one-window artifact?
        "winner_wmape_by_fold": {
            f"fold_{int(k)}": score(g) for k, g in win_pred.groupby("fold")
        },
        # the Phase 2 empirical check, carried forward unchanged: perturbing
        # future units must change features only at/after origin + horizon
        "leakage_check_pass": leakage_check(long),
        "critic_violations": critic,
        "fit_seconds_total": round(sum(timings.values()), 1),
    }


def main() -> None:
    table = pd.read_parquet(config.ARTIFACTS_DIR / "signal_table.parquet")
    sel = json.loads((config.ARTIFACTS_DIR / "datasets" / "selection.json").read_text())
    thr = load_thresholds()

    probe = ft.build_features(build_long(
        subset_ids=sel["datasets"]["C_sparse"]["ids"][:3], max_day=400))
    out = {
        "manifest": run_manifest(
            step="phase3_bakeoff",
            feature_manifest_hash=hash_columns(ft.feature_columns(probe)),
            min_cohort_series=MIN_COHORT_SERIES,
        ),
        "datasets": {},
    }
    for key, short in WORKING.items():
        t0 = time.time()
        out["datasets"][short] = evaluate_dataset(key, sel["datasets"][key], table, thr)
        print(f"  {short} done in {time.time() - t0:.0f}s", flush=True)

    path = config.ARTIFACTS_DIR / "phase3_bakeoff_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}\n")

    for short, r in out["datasets"].items():
        print(f"=== {short}: {r['cell']}  (n={r['n_series']}, "
              f"modelable={r['n_modelable']}, B2={r['n_b2']})")
        hdr = f"  {'entrant':<16}{'WMAPE':>8}{'WMAPE_28':>10}{'last-fold_28':>14}"
        print(hdr)
        for name, res in r["entrants"].items():
            mark = "  <-- WINNER" if name == r["winner"] else ""
            print(f"  {name:<16}{res['pooled']['wmape']:>8.4f}"
                  f"{res['pooled']['wmape_h']:>10.4f}"
                  f"{res['last_fold']['wmape_h']:>14.4f}{mark}")
        ref = r["reference"]
        print(f"  {'zero':<16}{ref['zero']['wmape']:>8.4f}{ref['zero']['wmape_h']:>10.4f}")
        print(f"  {'oracle_median':<16}{ref['oracle_median']:>8.4f}")
        print(f"  {'oracle_mean':<16}{ref['oracle_mean']:>8.4f}"
              + ("   (above the zero line)" if ref["mean_oracle_above_zero_line"] else ""))
        print()


if __name__ == "__main__":
    main()
