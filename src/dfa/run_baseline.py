"""Phase 2 runner: the L2 baseline across datasets A, B, C (sub-plan sec 5-6).

Loads each selected sub-dataset (sampled ids from Phase 1), builds the frozen
features, routes sparse (B2) series to the mean floor, tunes and fits the pooled
Ridge on the rolling-origin folds, and reports:

  - overall WMAPE (baseline) + the naive-mean and zero comparators
  - the Phase 3 bar: min(baseline, zero). A fitted floor that loses to "predict
    nothing" is not the bar -- zero is.
  - WMAPE by Syntetos-Boylan class (the intermittent-tail honesty check)
  - WMAPE by volume tercile -- DATASET A ONLY (sub-plan sec 1.4)
  - per-fold WMAPE spread (stability)
  - routing composition (L2 vs floor); C is the sparse-fallback showcase and its
    L2 number is tagged indicative when the L2 subset is small
  - chosen config (alpha, target space)

A/B run on the 250-series sample; C is the full cell (149). Full-cell confirmation
for A/B is deferred (stated, not run here).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from . import features as ft
from . import baseline as bl
from .calibrate_thresholds import Thresholds, classify_branch
from .data_loader import build_long, last_sales_day
from .metrics import wmape, wmape_by_group, volume_terciles
from .splits import make_folds

WORKING = {"A_dense": "A", "B_intermittent": "B", "C_sparse": "C"}
INDICATIVE_N_L2 = 50  # below this, an L2 WMAPE is high-variance -> tag indicative


def leakage_check(long: pd.DataFrame, perturb_from_day: int | None = None) -> bool:
    """Empirical leakage check with teeth (sub-plan sec 2.3, sec 6 item 8).

    Perturb units from `perturb_from_day` onward, rebuild features, and confirm:
      (1) feature rows on days <= perturb_from_day + HORIZON are UNCHANGED
          (no future actual leaks into a feature), AND
      (2) feature rows beyond that boundary DO change (the check has teeth -- it
          would fail if features simply ignored their inputs).

    `perturb_from_day` defaults to the MIDPOINT of the frame's own day range (data-
    driven, not a hardcoded 1000), guaranteeing rows on both sides of the boundary
    for any dataset's history length.
    """
    if perturb_from_day is None:
        lo, hi = int(long["day_idx"].min()), int(long["day_idx"].max())
        perturb_from_day = lo + (hi - lo) // 2
    base = ft.build_features(long).sort_values(["id", "day_idx"]).reset_index(drop=True)
    cols = ft.feature_columns(base)
    pert = long.copy()
    pert.loc[pert["day_idx"] >= perturb_from_day, "units"] += 100
    pf = ft.build_features(pert).sort_values(["id", "day_idx"]).reset_index(drop=True)

    # a feature at day t reads units no more recent than t - HORIZON (lag_28),
    # so perturbing days >= P changes features exactly at t >= P + HORIZON.
    boundary = perturb_from_day + config.HORIZON
    safe = (base["day_idx"] < boundary).values
    unchanged = base.loc[safe, cols].reset_index(drop=True).equals(
        pf.loc[safe, cols].reset_index(drop=True))
    later = (base["day_idx"] >= boundary).values
    reacted = not base.loc[later, cols].reset_index(drop=True).equals(
        pf.loc[later, cols].reset_index(drop=True))
    return bool(unchanged and reacted)


def _load_thresholds() -> Thresholds:
    t = json.loads((config.ARTIFACTS_DIR / "thresholds.json").read_text())["thresholds"]
    return Thresholds(**t)


def _b2_ids(signal_rows: pd.DataFrame, thr: Thresholds) -> set[str]:
    """Series the deterministic branch logic routes to the sparse baseline (B2)."""
    b2 = signal_rows.apply(
        lambda r: classify_branch(r["zero_share"], r["adi"], r["dow_season"], thr)[0]
        == "B2_baseline",
        axis=1,
    )
    return set(signal_rows.loc[b2, "id"])


def _transform_diagnostics(trials: list[dict]) -> dict[str, dict[str, float]]:
    """Best trial per target space, with its forecast bias -- why a transform wins.

    Reports each transform's best achievable WMAPE alongside `bias` =
    sum(forecast)/sum(actual). The pair is the point: on the intermittent datasets
    log1p wins *while* biased far below 1.0, i.e. it wins by underpredicting toward
    zero rather than by modelling better -- which is only legible when the two
    numbers sit side by side. Also exposes the alpha spread within the winning
    transform, the evidence that alpha selection is immaterial here.
    """
    out: dict[str, dict[str, float]] = {}
    for tr in {t["transform"] for t in trials}:
        rows = [t for t in trials if t["transform"] == tr and not np.isnan(t["wmape"])]
        if not rows:
            continue
        best = min(rows, key=lambda t: t["wmape"])
        out[tr] = {
            "best_alpha": best["alpha"],
            "wmape": round(best["wmape"], 4),
            "bias": round(best["bias"], 4),
            "alpha_spread": round(max(r["wmape"] for r in rows)
                                  - min(r["wmape"] for r in rows), 6),
        }
    return out


def _wmape_by_fold(pred: pd.DataFrame) -> dict[str, float]:
    return {
        f"fold_{int(k)}": round(wmape(g["actual"], g["forecast"]), 4)
        for k, g in pred.groupby("fold")
    }


def evaluate_dataset(label: str, meta: dict, table: pd.DataFrame, thr: Thresholds) -> dict:
    ids = meta["ids"]
    signal_rows = table[table["id"].isin(ids)].copy()
    b2_ids = _b2_ids(signal_rows, thr)

    # True last usable TRAINING day, derived from the data -- NOT hardcoded to M5's
    # 1913: the last day column present in the sales file minus the sealed
    # evaluation horizon (the file's final `HORIZON` days are the Phase-5 held-out
    # window). Both the loaded history and the folds then adapt to whatever
    # M5-format dataset is supplied.
    train_end = last_sales_day() - config.HORIZON
    long = build_long(subset_ids=ids, max_day=train_end)
    feat = ft.build_features(long)
    folds = make_folds(train_end=train_end)
    res = bl.run_baseline(feat, folds, b2_ids)

    base, naive, zero = res["baseline"], res["naive"], res["zero"]
    sb = signal_rows.set_index("id")["sb_class"]

    overall = wmape(base["actual"], base["forecast"])

    # per-SB-class (all datasets)
    by_class = wmape_by_group(
        base["actual"], base["forecast"], base["id"].map(sb).to_numpy()
    )
    # L2 vs naive on the SAME modelable series -> does L2 beat a constant?
    l2_only = base[base["method"] == "l2"]
    l2_wmape = wmape(l2_only["actual"], l2_only["forecast"]) if len(l2_only) else float("nan")
    naive_l2 = wmape(naive["actual"], naive["forecast"]) if len(naive) else float("nan")
    # trivial "predict nothing" anchor, same rows as `base` -> identically 1.0
    zero_wmape = wmape(zero["actual"], zero["forecast"]) if len(zero) else float("nan")

    # The bar Phase 3 must clear is whichever is HARDER: the fitted floor, or doing
    # nothing. On the intermittent datasets the fitted floor scores worse than zero,
    # so beating the baseline there is not evidence of a useful model -- the zero
    # forecast is. Taking the min keeps the bar honest in both regimes.
    bar = overall if np.isnan(zero_wmape) else min(overall, zero_wmape)

    # "full" only when we can prove it: n_full present AND equal to len(ids).
    # A missing n_full labels the dataset "sample" -- never silently overclaim
    # full coverage.
    n_full = meta.get("n_full")
    sample_or_full = "full" if (n_full is not None and len(ids) == n_full) else "sample"

    out = {
        "cell": f'{meta["store_id"]} x {meta["dept_id"]}',
        "n_series": len(ids),
        "sample_or_full": sample_or_full,
        "wmape_overall": round(overall, 4),
        "wmape_l2_subset": round(l2_wmape, 4),
        "wmape_naive_modelable": round(naive_l2, 4),
        "wmape_zero": round(zero_wmape, 4),
        "phase3_bar": round(bar, 4),
        "baseline_beats_zero": bool(overall < zero_wmape),
        "l2_indicative": bool(res["n_l2"] < INDICATIVE_N_L2),
        "routing": {"l2": res["n_l2"], "floor": res["n_b2"]},
        "config": {"alpha": res["config"].alpha, "transform": res["config"].transform},
        "wmape_by_sb_class": {k: round(v["wmape"], 4) for k, v in by_class.items()},
        "wmape_by_fold": _wmape_by_fold(base),
        "transform_diagnostics": _transform_diagnostics(res["trials"]),
        "leakage_check_pass": leakage_check(long),
        "trials": res["trials"],
    }

    # volume-tercile breakdown -- DATASET A ONLY
    if label == "A_dense":
        vol = feat.groupby("id", observed=True)["units"].sum()
        terc = volume_terciles(vol.index, vol.to_numpy())
        by_terc = wmape_by_group(
            base["actual"], base["forecast"],
            base["id"].map(terc).to_numpy(),
        )
        out["wmape_by_volume_tercile"] = {k: round(v["wmape"], 4) for k, v in by_terc.items()}
    return out


def main() -> None:
    table = pd.read_parquet(config.ARTIFACTS_DIR / "signal_table.parquet")
    sel = json.loads((config.ARTIFACTS_DIR / "datasets" / "selection.json").read_text())
    thr = _load_thresholds()

    results = {}
    for key, short in WORKING.items():
        results[short] = evaluate_dataset(key, sel["datasets"][key], table, thr)

    out_path = config.ARTIFACTS_DIR / "phase2_baseline_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}\n")

    for short, r in results.items():
        tag = "  [SPARSE-FALLBACK SHOWCASE]" if short == "C" else ""
        print(f"=== {short}: {r['cell']}  (n={r['n_series']}, {r['sample_or_full']}){tag}")
        print(f"  WMAPE baseline : {r['wmape_overall']:.4f}   (floor: L2 + B2 mean-fallback, all series)")
        ind = "  (indicative, low n)" if r["l2_indicative"] else ""
        print(f"  L2 vs naive    : L2={r['wmape_l2_subset']:.4f}  naive={r['wmape_naive_modelable']:.4f}"
              f"  (modelable series only){ind}")
        verdict = "baseline" if r["baseline_beats_zero"] else "ZERO WINS"
        print(f"  vs zero        : zero={r['wmape_zero']:.4f}  -> {verdict}")
        print(f"  PHASE 3 BAR    : {r['phase3_bar']:.4f}   (min of baseline and zero)")
        print(f"  routing        : L2={r['routing']['l2']}  floor={r['routing']['floor']}")
        print(f"  config         : alpha={r['config']['alpha']}  target={r['config']['transform']}")
        print(f"  by SB class    : "
              + "  ".join(f"{k}={v:.3f}" for k, v in r["wmape_by_sb_class"].items()))
        if "wmape_by_volume_tercile" in r:
            print(f"  by vol tercile : "
                  + "  ".join(f"{k}={v:.3f}" for k, v in r["wmape_by_volume_tercile"].items()))
        print(f"  fold spread    : "
              + "  ".join(f"{k}={v:.3f}" for k, v in r["wmape_by_fold"].items()))
        print(f"  leakage check  : {'PASS' if r['leakage_check_pass'] else 'FAIL'}")
        print()


if __name__ == "__main__":
    main()
