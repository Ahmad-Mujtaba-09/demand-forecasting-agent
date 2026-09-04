"""Phase 3 Increment 10 tests: the agent-handoff contract (sub-plan sec 11).

The contract is the only thing standing between a reviewed decision and an agent
improvising a model choice at runtime, so its invariants are tested rather than
trusted: completeness over the router's branches, thin-cell exclusion, and no
cross-dataset pooling.
"""

from __future__ import annotations

import json

import pytest

from dfa import build_contract as bc
from dfa.calibrate_thresholds import Thresholds, classify_branch


@pytest.fixture
def thr():
    return Thresholds(adi_sparse_cut=8.77, zero_share_tweedie_cut=0.663,
                      max_median_ratio=24.0, min_forecast=0.0, basis="test")


from dfa.models import bakeoff_entrants

REAL_CFG = {name: entrants[0].config_id for name, entrants in bakeoff_entrants().items()}


def _cohort(n, thin, scores):
    return {"n_series": n, "thin": thin,
            "entrants": {k: {"wmape": v, "wmape_h": v, "n_rows": 100}
                         for k, v in scores.items()}}


def _results(datasets):
    return {"manifest": {"feature_manifest_hash": "abc123"}, "datasets": datasets}


def _trials(name, per_config):
    """Per-config selection scores, the input freeze_config votes on."""
    return [{"config_id": c, "selection": {"wmape_h": v, "wmape": v, "n_rows": 10},
             "pooled": {"wmape_h": v, "wmape": v, "n_rows": 10},
             "last_fold": {"wmape_h": v, "wmape": v, "n_rows": 10}}
            for c, v in per_config.items()]


def _ds(cohorts, b2_n=100, b2_scores=None, chosen=None, trials=None):
    names = ("lgb_l2", "lgb_tweedie", "lgb_poisson", "lgb_quantile50")
    return {
        "cohorts": cohorts,
        "b2_cohort": ({"n_series": b2_n,
                       **{k: {"wmape": v, "wmape_h": v, "n_rows": 50}
                          for k, v in (b2_scores or
                                       {"mean_floor": 0.7, "sba": 0.6, "croston": 0.65}).items()}}),
        # real config_ids: _params_for resolves them back to actual LightGBM params,
        # so a fabricated id would (correctly) raise
        "entrants": {n: {"chosen_config": (chosen or {}).get(n, REAL_CFG[n]),
                         "trials": (trials or {}).get(n, _trials(n, {REAL_CFG[n]: 0.5}))}
                     for n in names},
        "critic_violations": {"below_min_forecast": 0, "above_max_median_ratio": 0},
    }


def test_contract_covers_every_branch_the_router_can_emit(thr):
    """Completeness: the executor must never meet a branch it has no model for."""
    coh = {"B1_tweedie": _cohort(120, False, {"lgb_tweedie": 0.4, "lgb_l2": 0.5}),
           "standard": _cohort(200, False, {"lgb_l2": 0.3, "lgb_tweedie": 0.35})}
    contract = bc.build(_results({"A": _ds(coh), "B": _ds(coh)}), None, thr)
    assert set(contract["branches"]) == set(bc.ROUTER_BRANCHES)
    # and the router really can only emit those three
    emitted = {classify_branch(z, a, thr)
               for z in (0.1, 0.7, 0.99) for a in (1.5, 5.0, 20.0)}
    assert emitted <= set(contract["branches"])


def test_missing_branch_raises_rather_than_shipping_a_gap(thr, monkeypatch):
    monkeypatch.setattr(bc, "ROUTER_BRANCHES", ("B1_tweedie", "standard", "B2_baseline", "B9_new"))
    coh = {"B1_tweedie": _cohort(120, False, {"lgb_tweedie": 0.4}),
           "standard": _cohort(200, False, {"lgb_l2": 0.3})}
    with pytest.raises(AssertionError, match="contract incomplete"):
        bc.build(_results({"A": _ds(coh)}), None, thr)


def test_thin_cells_are_reported_but_do_not_vote():
    """A 5-series cohort must not decide a branch (sub-plan sec 3)."""
    results = _results({
        # thin cell prefers tweedie; the deciding cell prefers l2
        "A": _ds({"standard": _cohort(5, True, {"lgb_tweedie": 0.1, "lgb_l2": 0.9})}),
        "B": _ds({"standard": _cohort(200, False, {"lgb_tweedie": 0.8, "lgb_l2": 0.2})}),
    })
    d = bc.decide_branch("standard", results)
    assert d["objective"] == "lgb_l2"
    assert d["evidence"]["decided_on"] == ["B"]
    assert "A" in d["evidence"]["thin_cells_excluded"]


def test_decision_is_a_vote_over_cells_not_a_pooled_number():
    """Two cells for l2, one (better-scoring) for tweedie -> l2 wins on votes.

    Pooling would hand the decision to whichever cell has the most volume; the
    matrix vote deliberately does not.
    """
    results = _results({
        "A": _ds({"standard": _cohort(100, False, {"lgb_l2": 0.40, "lgb_tweedie": 0.41})}),
        "B": _ds({"standard": _cohort(100, False, {"lgb_l2": 0.40, "lgb_tweedie": 0.41})}),
        "C": _ds({"standard": _cohort(100, False, {"lgb_l2": 0.90, "lgb_tweedie": 0.05})}),
    })
    d = bc.decide_branch("standard", results)
    assert d["objective"] == "lgb_l2"
    assert d["evidence"]["votes"] == {"lgb_l2": 2, "lgb_tweedie": 1}


def test_skill_vs_zero_is_reported_per_cell():
    """WMAPE_28 of a zero forecast is exactly 1.0, so skill = 1 - score."""
    results = _results({"A": _ds({"standard": _cohort(100, False, {"lgb_l2": 0.3})})})
    d = bc.decide_branch("standard", results)
    assert d["evidence"]["cells"]["A"]["skill_vs_zero"] == pytest.approx(0.7)


def test_b2_fallback_is_decided_against_the_phase2_mean_floor(thr):
    coh = {"standard": _cohort(200, False, {"lgb_l2": 0.3})}
    r = _results({"A": _ds(coh, b2_scores={"mean_floor": 0.80, "sba": 0.55, "croston": 0.60})})
    d = bc.decide_b2(r)
    assert d["estimator"] == "sba"
    assert d["params"]["sba"] is True
    assert d["evidence"]["cells"]["A"]["scores"]["mean_floor"] == 0.80


def test_frozen_params_are_recoverable_and_deterministic(thr):
    """Phase 4 reads params from the contract; they must carry the seed settings."""
    coh = {"B1_tweedie": _cohort(120, False, {"lgb_tweedie": 0.4}),
           "standard": _cohort(200, False, {"lgb_l2": 0.3})}
    contract = bc.build(_results({"A": _ds(coh)}), None, thr)
    p = contract["branches"]["standard"]["params"]
    assert p["deterministic"] is True and p["force_row_wise"] is True
    assert "seed" in p and "num_threads" in p


def test_contract_is_json_serialisable(thr):
    coh = {"B1_tweedie": _cohort(120, False, {"lgb_tweedie": 0.4}),
           "standard": _cohort(200, False, {"lgb_l2": 0.3})}
    contract = bc.build(_results({"A": _ds(coh)}), None, thr)
    assert json.loads(json.dumps(contract))["contract_version"] == 1


def test_b2_tie_on_votes_is_broken_on_mean_score_not_dict_order():
    """A 1-1 split is real here (two eligible cohorts); iteration order must not decide."""
    coh = {"standard": _cohort(200, False, {"lgb_l2": 0.3})}
    r = _results({
        # sba wins A by a wide margin; mean_floor wins B by a narrow one -> 1-1 votes
        "A": _ds(coh, b2_scores={"mean_floor": 0.90, "sba": 0.60, "croston": 0.95}),
        "B": _ds(coh, b2_scores={"mean_floor": 0.70, "sba": 0.71, "croston": 0.95}),
    })
    d = bc.decide_b2(r)
    assert d["evidence"]["votes"] == {"sba": 1, "mean_floor": 1}
    assert d["estimator"] == "sba"          # mean 0.655 beats mean_floor's 0.80


def test_frozen_config_is_chosen_by_mean_score_not_iteration_order():
    """Datasets genuinely disagree on the best config; the rule must be stated."""
    from dfa.models import bakeoff_entrants
    cfgs = [e.config_id for e in bakeoff_entrants()["lgb_tweedie"][:3]]
    coh = {"standard": _cohort(100, False, {"lgb_tweedie": 0.4, "lgb_l2": 0.9})}
    # cfg0 wins A outright but is terrible on B; cfg1 is second-best on both
    r = _results({
        "A": _ds(coh, trials={"lgb_tweedie": _trials("t", {cfgs[0]: 0.10, cfgs[1]: 0.30, cfgs[2]: 0.90})}),
        "B": _ds(coh, trials={"lgb_tweedie": _trials("t", {cfgs[0]: 0.90, cfgs[1]: 0.32, cfgs[2]: 0.95})}),
    })
    frozen, cost = bc.freeze_config(r, "lgb_tweedie", ["A", "B"])
    assert frozen == cfgs[1]                       # mean 0.31 beats cfg0's 0.50
    assert cost["per_dataset_best_config"]["A"] == cfgs[0]
    # and the price of freezing is reported, not hidden
    assert cost["cost_vs_per_dataset_best"]["A"] == pytest.approx(0.20)
    assert cost["cost_vs_per_dataset_best"]["B"] == pytest.approx(0.00)


def test_freeze_config_ignores_configs_not_scored_everywhere():
    from dfa.models import bakeoff_entrants
    cfgs = [e.config_id for e in bakeoff_entrants()["lgb_tweedie"][:2]]
    coh = {"standard": _cohort(100, False, {"lgb_tweedie": 0.4})}
    r = _results({
        "A": _ds(coh, trials={"lgb_tweedie": _trials("t", {cfgs[0]: 0.10, cfgs[1]: 0.50})}),
        "B": _ds(coh, trials={"lgb_tweedie": _trials("t", {cfgs[1]: 0.50})}),
    })
    frozen, _ = bc.freeze_config(r, "lgb_tweedie", ["A", "B"])
    assert frozen == cfgs[1]        # cfgs[0] is incomparable -- only scored on A
