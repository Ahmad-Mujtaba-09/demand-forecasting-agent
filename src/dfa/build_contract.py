"""Emit the agent-handoff contract (Phase 3 sub-plan sec 11).

Phases 1-3 exist to produce a decision function plus a configuration; Phase 4
wraps an agent around them. Master plan sec 0.4 -- *the agent selects which branch
applies per documented rules; it does not invent modelling decisions* -- only
holds if that handoff is a DATA ARTIFACT rather than prose an implementer has to
interpret. This module writes that artifact.

Two properties the emitter enforces rather than assumes:

- **Completeness.** Every branch `classify_branch` can return has an entry, so
  the executor can never meet a branch it has no model for. Asserted here and
  re-asserted by a test.
- **Frozen hyperparameters.** The params written here are the ones Phase 3
  selected; Phase 4 does NOT re-tune. An agent that re-runs a grid search on
  incoming data has unbounded runtime and a non-deterministic decision path,
  which is precisely the "inventing modelling decisions" master plan sec 0.4
  forbids. Freezing also makes Phase 5 a stronger test: the whole configuration
  transfers to sealed D unchanged, rather than being refitted there.

The winner per branch is decided on the (cohort x dataset) MATRIX, not on a
cross-dataset pooled number. Pooling would be invalid: total validation volume is
A 123k / B 18k / C 2.4k units, so a volume-weighted pooled cohort score is ~86%
dataset A. Cells below `MIN_COHORT_SERIES` are marked thin and excluded from the
vote instead of being silently averaged in.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import config
from .calibrate_thresholds import Thresholds
from .manifest import run_manifest
from .models import CAPACITY_GRID, bakeoff_entrants, quantile_entrants
from .reference_lines import load_thresholds
from .run_bakeoff import MIN_COHORT_SERIES

# Every branch the deterministic router can emit. The contract must cover all of
# them -- this list is the completeness contract, asserted below.
ROUTER_BRANCHES: tuple[str, ...] = ("B1_tweedie", "standard", "B2_baseline")


def _params_for(config_id: str) -> dict:
    """Recover an entrant's full LightGBM params from its config_id."""
    for entrants in bakeoff_entrants().values():
        for e in entrants:
            if e.config_id == config_id:
                return e.params
    for level in (0.1, 0.5, 0.9):
        for e in quantile_entrants(level):
            if e.config_id == config_id:
                return e.params
    raise KeyError(f"no entrant matches config_id {config_id!r}")


def decide_branch(cohort: str, results: dict) -> dict:
    """Vote over the (cohort x dataset) matrix; thin cells are excluded.

    Returns the winning entrant plus the evidence: every non-thin cell's score
    and its margin over that cell's own zero line (WMAPE_28 of a zero forecast is
    exactly 1.0, so `1 - wmape_h` is the skill against doing nothing).
    """
    wins: dict[str, int] = defaultdict(int)
    cells, thin = {}, {}
    for ds, r in results["datasets"].items():
        block = r["cohorts"].get(cohort)
        if block is None:
            continue
        scores = {n: v["wmape_h"] for n, v in block["entrants"].items()
                  if not np.isnan(v["wmape_h"])}
        if not scores:
            continue
        best = min(scores, key=scores.get)
        record = {"n_series": block["n_series"], "scores": scores,
                  "best": best, "skill_vs_zero": round(1.0 - scores[best], 4)}
        if block["thin"]:
            thin[ds] = record          # reported, but does not vote
        else:
            cells[ds] = record
            wins[best] += 1
    if not wins:
        raise ValueError(f"cohort {cohort!r} has no non-thin cell to decide on")

    top = max(wins.values())
    tied = [n for n, w in wins.items() if w == top]
    # tie-break on mean skill across the deciding cells -- stated, not arbitrary
    winner = min(tied, key=lambda n: np.mean([c["scores"][n] for c in cells.values()]))
    chosen_cfg, freeze_cost = freeze_config(results, winner, sorted(cells))
    return {
        "objective": winner,
        "config_id": chosen_cfg,
        "params": _params_for(chosen_cfg),
        "evidence": {
            "decided_on": sorted(cells),
            "votes": dict(wins),
            "cells": cells,
            "thin_cells_excluded": thin,
            "tie_broken_on_mean_skill": len(tied) > 1,
            "frozen_config": freeze_cost,
        },
    }


def freeze_config(results: dict, entrant: str, datasets: list[str]) -> tuple[str, dict]:
    """Pick ONE hyperparameter config for a branch, across the deciding datasets.

    Phase 4 does not re-tune (sub-plan sec 2), so the contract must carry a single
    config per branch -- but each dataset selects its own optimum, and on this run
    they genuinely differ (Tweedie's variance_power lands at 1.5 / 1.3 / 1.1 on
    A / B / C). Taking whichever dataset happened to be iterated first would make
    the shipped model an artifact of dict order, so the rule is stated instead:

        the config with the best MEAN selection-fold WMAPE_28 across the
        deciding datasets.

    The compromise has a price and we report it rather than hiding it:
    `cost_vs_per_dataset_best` is how much worse the one frozen config is, on each
    dataset, than that dataset's own best config. A large cost would mean a single
    frozen config is not defensible and the branch needs per-dataset calibration --
    a finding, not something to paper over.
    """
    by_config: dict[str, dict[str, float]] = defaultdict(dict)
    for ds in datasets:
        for t in results["datasets"][ds]["entrants"][entrant]["trials"]:
            v = t["selection"]["wmape_h"]
            if not np.isnan(v):
                by_config[t["config_id"]][ds] = v
    # only configs scored on every deciding dataset are comparable
    complete = {c: v for c, v in by_config.items() if len(v) == len(datasets)}
    if not complete:
        raise ValueError(f"no config of {entrant!r} scored on all of {datasets}")
    frozen = min(complete, key=lambda c: float(np.mean(list(complete[c].values()))))

    per_ds_best = {ds: min(v[ds] for v in complete.values()) for ds in datasets}
    return frozen, {
        "rule": "min mean selection-fold WMAPE_28 across the deciding datasets",
        "mean_score": round(float(np.mean(list(complete[frozen].values()))), 4),
        "per_dataset_best_config": {
            ds: min(complete, key=lambda c: complete[c][ds]) for ds in datasets
        },
        "cost_vs_per_dataset_best": {
            ds: round(complete[frozen][ds] - per_ds_best[ds], 4) for ds in datasets
        },
    }


def decide_b2(results: dict) -> dict:
    """B2 fallback: Croston/SBA vs the Phase 2 mean floor, on the B2 cohort."""
    wins: dict[str, int] = defaultdict(int)
    cells = {}
    for ds, r in results["datasets"].items():
        b2 = r.get("b2_cohort") or {}
        scores = {k: b2[k]["wmape_h"] for k in ("mean_floor", "sba", "croston")
                  if k in b2 and not np.isnan(b2[k]["wmape_h"])}
        if not scores or b2.get("n_series", 0) < MIN_COHORT_SERIES:
            continue
        best = min(scores, key=scores.get)
        cells[ds] = {"n_series": b2["n_series"], "scores": scores, "best": best,
                     "skill_vs_zero": round(1.0 - scores[best], 4)}
        wins[best] += 1
    if not wins:
        raise ValueError("no B2 cohort large enough to decide the fallback estimator")
    top = max(wins.values())
    tied = [n for n, w in wins.items() if w == top]
    # Same documented tie-break as decide_branch: mean WMAPE_28 across the
    # deciding cells. A 1-1 split on votes is genuinely possible here (two
    # eligible cohorts), and picking whichever dict key happened to come first
    # would make the fallback estimator an artifact of iteration order.
    winner = min(tied, key=lambda n: np.mean(
        [c["scores"][n] for c in cells.values() if n in c["scores"]]))
    from .croston import DEFAULT_ALPHA
    params = ({"alpha": DEFAULT_ALPHA, "sba": winner == "sba"}
              if winner in ("sba", "croston")
              else {"note": "constant mean of active history up to the origin"})
    return {"estimator": winner, "params": params,
            "evidence": {"votes": dict(wins), "cells": cells}}


def build(results: dict, quantiles: dict | None, thr: Thresholds) -> dict:
    branches = {b: decide_branch(b, results) for b in ("B1_tweedie", "standard")}
    branches["B2_baseline"] = decide_b2(results)

    missing = set(ROUTER_BRANCHES) - set(branches)
    if missing:
        raise AssertionError(
            f"contract incomplete: router can emit {sorted(missing)} with no model. "
            "Phase 4's executor would meet a branch it cannot serve.")

    viol = {ds: r["critic_violations"] for ds, r in results["datasets"].items()}
    q_block = None
    if quantiles:
        first = next(iter(quantiles["datasets"].values()))
        q_block = {
            "levels": [float(q) for q in first["levels"]],
            "params": {q: _params_for(v["config_id"])
                       for q, v in first["levels"].items()},
            "monotone_sort": True,
            "observed_crossing_rate": {ds: r["crossing_rate"]
                                       for ds, r in quantiles["datasets"].items()},
            "horizon_total_intervals": "deferred -- quantile of a sum != sum of quantiles",
        }

    return {
        "run_manifest": run_manifest(step="phase3_contract"),
        "contract_version": 1,
        "consumed_by": "Phase 4 Executor -- reads this, does NOT re-tune or re-threshold",
        "thresholds_ref": "artifacts/thresholds.json",
        "feature_manifest_hash": results["manifest"]["feature_manifest_hash"],
        "capacity_grid_pre_registered": [dict(c) for c in CAPACITY_GRID],
        "min_cohort_series": MIN_COHORT_SERIES,
        "min_cohort_fallback_branch": "B2_baseline",
        "router": {
            "function": "dfa.calibrate_thresholds.classify_branch",
            "inputs": ["zero_share", "adi"],
            "branches": list(ROUTER_BRANCHES),
        },
        "branches": branches,
        "quantiles": q_block,
        "critic_bounds": {
            "min_forecast": thr.min_forecast,
            "max_median_ratio": thr.max_median_ratio,
            "observed_violations": viol,
        },
    }


def main() -> None:
    results = json.loads((config.ARTIFACTS_DIR / "phase3_bakeoff_results.json").read_text())
    qpath = config.ARTIFACTS_DIR / "phase3_quantile_results.json"
    quantiles = json.loads(qpath.read_text()) if qpath.exists() else None
    contract = build(results, quantiles, load_thresholds())

    path = config.ARTIFACTS_DIR / "phase3_model_contract.json"
    path.write_text(json.dumps(contract, indent=2))
    print(f"wrote {path}\n")
    for name, b in contract["branches"].items():
        pick = b.get("objective") or b.get("estimator")
        ev = b["evidence"]
        detail = (f"votes={ev['votes']}" if "votes" in ev else "")
        print(f"  {name:<14} -> {pick:<16} {detail}")
    print(f"\n  min_cohort_series = {contract['min_cohort_series']} "
          f"(-> {contract['min_cohort_fallback_branch']})")


if __name__ == "__main__":
    main()
