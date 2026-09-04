# Phase 3 Results — LightGBM bake-off

**Status:** For review. Plan: [../plans/03_phase3_bakeoff_plan.md](../plans/03_phase3_bakeoff_plan.md). Artifacts: `artifacts/phase3_{reference_lines,bakeoff_results,quantile_results,fullcell_results,model_contract}.json`.
**Reproduce:** `PYTHONPATH=src python -m dfa.reference_lines && python -m dfa.run_bakeoff && python -m dfa.run_quantiles && python -m dfa.run_fullcell && python -m dfa.build_contract` (~20 min).

**Headline:** **Tweedie wins every dataset and every branch cohort.** The B1-vs-Standard split does not survive its own test and is recommended for removal. Croston/SBA takes the B2 fallback. All 141 tests pass; the run reproduced bit-identically across three independent executions.

---

## 0. The metric amendment, settled by the data

Phase 3's plan §0 argued daily WMAPE is degenerate on intermittent series and proposed adding horizon-aggregated WMAPE_28 as co-primary. The run confirms both halves.

| Dataset | zero | oracle_median | oracle_mean | Phase 2 Ridge (daily) | best entrant (daily) |
|---|---|---|---|---|---|
| A | 1.000 | 0.463 | 0.490 | 0.621 | 0.582 |
| B | 1.000 | 0.880 | **1.078** | 1.106 | 0.939 |
| C | 1.000 | 0.998 | **1.517** | 1.631 | 1.000 |

- **The prediction held exactly.** `oracle_mean` sits above 1.0 on B and C, so no mean-targeting objective can clear the zero line on daily WMAPE there — and indeed Tweedie and Poisson score 1.13–1.64 on those datasets. That is not a modelling failure; it is the metric.
- **The median-targeting entrant behaves precisely as theory says.** `lgb_quantile50` is the daily-WMAPE winner on all three datasets, and on **C it scores exactly 1.0000 — it predicts all zeros**, because C's conditional median *is* zero (100% of series have a zero-median validation window). On B it reaches 0.9388, genuinely beating the zero line, which is the ~12% of headroom `oracle_median = 0.880` said was available.
- So daily WMAPE on this data does not rank forecasts by usefulness; it ranks them by how closely they approximate zero. **WMAPE_28 is the reported decision metric**, exactly as the plan pre-registered, and the same q50 model that "wins" daily is the **worst** entrant under it (0.71 on B against Tweedie's 0.35). That contrast is the finding.

**The bar was not lowered.** The zero forecast still scores exactly 1.000 under WMAPE_28 (an identity, re-derived each run), and `wmape_horizon <= wmape` is enforced as a property test. What changed is that the bar became reachable: **the Phase 2 Ridge floor already clears it on all three** (A 0.326, B 0.473, C 0.639), so Phase 2's "the baseline loses to doing nothing on B and C" was a metric artifact, not a broken baseline.

---

## 1. Bake-off table

WMAPE_28 pooled over 5 folds; `last-fold` is the selection-free fold 4. Winner chosen on the **selection folds (0–3) only**, by WMAPE_28 — the rule was fixed before the run.

| Dataset | entrant | daily WMAPE | **WMAPE_28** | last-fold 28 | selection 28 |
|---|---|---|---|---|---|
| **A** | **lgb_tweedie** | 0.592 | **0.3015** | 0.2766 | **0.3083** |
| | lgb_poisson | 0.609 | 0.3171 | 0.2857 | 0.3258 |
| | lgb_quantile50 | *0.582* | 0.3467 | 0.3117 | 0.3562 |
| | lgb_l2 | 0.657 | 0.3780 | 0.3319 | 0.3907 |
| | *Phase 2 Ridge* | *0.621* | *0.3255* | — | — |
| **B** | **lgb_tweedie** | 1.135 | **0.3475** | 0.3330 | **0.3518** |
| | lgb_poisson | 1.144 | 0.3527 | 0.3354 | 0.3577 |
| | lgb_l2 | 1.166 | 0.3696 | 0.3397 | 0.3783 |
| | lgb_quantile50 | *0.939* | 0.7114 | 0.7121 | 0.7112 |
| | *Phase 2 Ridge* | *1.106* | *0.4731* | — | — |
| **C** | **lgb_tweedie** | 1.635 | **0.4882** | 0.5990 | **0.4658** |
| | lgb_poisson | 1.640 | 0.4967 | 0.6250 | 0.4707 |
| | lgb_l2 | 1.649 | 0.5005 | 0.6236 | 0.4756 |
| | lgb_quantile50 | *1.000* | 1.0000 | 1.0000 | 1.0000 |
| | *Phase 2 Ridge* | *1.631* | *0.6388* | — | — |

*Italic daily-WMAPE figures mark the daily winner, retained to show how completely the two metrics disagree.*

**Reading the objective effect cleanly.** This is why E1 (LightGBM-L2) was in the field: it isolates model class from objective.

| | Ridge → LGBM-L2 (model class) | LGBM-L2 → Tweedie (objective) |
|---|---|---|
| A | 0.3255 → 0.3780 (**worse**, +0.053) | 0.3780 → 0.3015 (better, −0.077) |
| B | 0.4731 → 0.3696 (better, −0.104) | 0.3696 → 0.3475 (better, −0.022) |
| C | 0.6388 → 0.5005 (better, −0.138) | 0.5005 → 0.4882 (better, −0.012) |

**The objective, not the model class, is what wins on A** — a gradient-boosted tree fitted with squared error is *worse* than the linear Ridge there (0.378 vs 0.326), and only switching the objective recovers and beats it. Without E1 in the field, "Tweedie beat the Phase 2 baseline" on A would have been credited to trees, which the data does not support. On B and C the ordering is reversed: most of the gain is model class, and the objective adds a smaller increment.

---

## 2. Branch-rule validation — B1 does not survive its own test

Scored per **(cohort × dataset) cell**. No cross-dataset pooling: validation volume is A 123.1k / B 18.1k / C 2.4k units, so a pooled cohort score would be 86% dataset A.

| cohort | ds | n | lgb_tweedie | lgb_poisson | lgb_l2 | lgb_quantile50 |
|---|---|---|---|---|---|---|
| **standard** | A | 220 | **0.2988** | 0.3136 | 0.3620 | 0.3392 |
| | B | 88 | **0.3084** | 0.3146 | 0.3190 | 0.6144 |
| | C | 0 | — | — | — | — |
| **B1_tweedie** | A | 28 | **0.4716** | 0.5358 | 1.3852 | 0.8155 |
| | B | 135 | **0.4238** | 0.4268 | 0.4683 | 0.9005 |
| | C | 31 | **0.4882** | 0.4967 | 0.5005 | 1.0000 |

### Verdicts

- **B1 (Tweedie for high zero-share) — the prior is vindicated, the branch is not.** On A's B1 cohort, L2 collapses to **1.385** against Tweedie's 0.472: on genuinely zero-heavy series, squared error is catastrophic, exactly as master plan §3 predicted. But Tweedie *also* wins the Standard cohort (0.2988 vs L2's 0.3620), so **the cut does not separate anything**. The branch buys nothing that giving every modelable series Tweedie would not.
- **Standard (L2/Poisson) — refuted.** The bake-off was supposed to decide L2 vs Poisson for this branch. Neither wins it. Poisson is consistently second and L2 consistently last.
- **Recommendation: collapse `B1_tweedie` and `standard` into one modelable branch with the Tweedie objective.** This is the B3 precedent repeating — a branch that does not earn its complexity gets cut. **Not applied here:** per plan §9 this is deferred to the Phase 4 sub-plan, because it changes the router's shape and that is a reviewed decision, not a results-doc decision. The contract as shipped assigns Tweedie to *both* branches, so the executor behaves correctly either way.
- **Caveat, stated:** C contributes **no evidence about the Standard branch** — it has zero Standard series. The Standard verdict rests on A and B only.

### WMAPE by Syntetos–Boylan class (winner)

The Phase 2 honesty check carried forward: the aggregate is volume-weighted, so it can hide the intermittent tail.

| ds | smooth | erratic | intermittent | lumpy |
|---|---|---|---|---|
| A | 0.255 | 0.232 | 0.478 | 0.406 |
| B | 0.187 | 0.447 | 0.357 | 0.318 |
| C | — | — | 0.491 | 0.484 |

Under WMAPE_28 the class spread is **far narrower than Phase 2's daily figures** (which ran 0.48→1.68). The intermittent tail is ~1.9× the smooth class on A rather than 2–2.5×, and **every class on every dataset beats the 1.000 zero line** — including C's, which under daily WMAPE could not.

---

## 3. B2 fallback — Croston/SBA vs the Phase 2 mean floor

The branch master plan §3 named "Croston-style" and Phase 2 shipped as a plain mean. WMAPE_28 on the B2 cohort:

| ds | n | mean_floor | **SBA** | Croston | zero |
|---|---|---|---|---|---|
| A | 2 | 0.9719 | 0.9621 | 0.9670 | 1.000 |
| B | 27 | 0.7668 | **0.7109** | 0.7160 | 1.000 |
| C | 118 | **0.7141** | 0.7206 | 0.7397 | 1.000 |

- **Split decision, resolved by a stated rule.** A is thin (n=2, below `MIN_COHORT_SERIES=20`) and excluded. B and C then split 1–1, broken on mean score: SBA 0.7158 vs mean_floor 0.7405. **SBA ships.** Its win on B (−0.056) is ~8× its loss on C (+0.007).
- **The honest size of the win is small.** SBA improves the B2 branch by ~3% relative on average. It is the right estimator on principle — it is the literature standard for this exact regime, it is untuned, and it beats the mean where the two differ most — but nobody should read this as a large gain.
- Croston without the SBA debiasing is worse than SBA everywhere, which is the expected direction (Syntetos & Boylan 2005) and a small independent check that the implementation is behaving.

---

## 4. Quantile models (B4) — and a real limit on what they can say

Requirement-driven; scored on pinball loss and coverage, never against Tweedie on WMAPE.

| ds | zero-share | level | pinball | coverage | coverage (strict) | target |
|---|---|---|---|---|---|---|
| A | 0.380 | q10 | 0.350 | 0.380 | 0.009 | 0.10 |
| | | q50 | 1.031 | 0.556 | 0.530 | 0.50 |
| | | q90 | 0.655 | **0.906** | 0.906 | 0.90 |
| B | 0.677 | q10 | 0.057 | 0.677 | 0.000 | 0.10 |
| | | q50 | 0.265 | 0.681 | 0.060 | 0.50 |
| | | q90 | 0.207 | **0.925** | 0.916 | 0.90 |
| C | 0.828 | q10 | 0.025 | 0.828 | 0.000 | 0.10 |
| | | q50 | 0.123 | 0.828 | 0.000 | 0.50 |
| | | q90 | 0.148 | **0.951** | 0.844 | 0.90 |

- **Crossing rate: 0.33% (A), 0.00% (B), 0.00% (C).** The monotone sort is applied regardless; it was essentially never needed, so the quantile fits are stable.
- **q90 is well calibrated** — 0.906 against a 0.90 target on A is close to ideal, and B/C over-cover moderately. The q10–q90 interval covers 0.896 / 0.925 / 0.951 against an 0.80 target: usable but conservative, so it would over-size safety stock rather than under-size it.
- **Low-quantile coverage is not identified on this data, and the number should not be read as miscalibration.** Demand is a zero-inflated *count*, so a forecast of 0 is simultaneously the 10th, 30th and (on C) the 83rd percentile — the quantile function is flat across that entire range. Note that q10 coverage equals the zero-share to three decimals on every dataset (0.380/0.380, 0.677/0.677, 0.828/0.828). The model is correctly predicting 0; "coverage" is measuring P(y=0). Both bounds (`<=` and strict `<`) are reported so the effect is visible rather than being resolved by a convention choice.
- **Deferred, not approximated:** intervals on the 28-day total — what a lead-time safety stock actually needs. The quantile of a sum is not the sum of quantiles, and summing daily quantiles would err in the direction that *under*-sizes safety stock. Named as future work.

---

## 5. Honesty checks

### Selection optimism — measured, not assumed
Phase 2's "the optimism cancels" argument was retired for trees (§2 of the plan). Gap = last-fold minus pooled WMAPE_28:

| ds | lgb_tweedie | lgb_poisson | lgb_l2 | lgb_quantile50 |
|---|---|---|---|---|
| A | −0.025 | −0.031 | −0.046 | −0.035 |
| B | −0.015 | −0.017 | −0.030 | +0.001 |
| C | +0.111 | +0.128 | +0.123 | 0.000 |

**No detectable selection overfitting.** A and B score *better* on the held-out fold than pooled; C scores worse. The gap tracks **fold difficulty, not entrant**, and it moves all four entrants together — which is what a genuinely uninformative selection looks like. With a 4-config grid the concern the protocol was built to catch did not materialise, and the protocol is what lets us say that rather than assume it.

### Per-fold spread (winner, WMAPE_28)

| ds | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|
| A | 0.304 | 0.323 | 0.289 | 0.316 | 0.277 |
| B | 0.350 | 0.313 | 0.371 | 0.374 | 0.333 |
| C | 0.412 | 0.455 | 0.455 | 0.577 | 0.599 |

A and B are tight (spread ≤0.06). **C drifts upward monotonically across folds** (0.41→0.60) — the later windows are genuinely harder on the sparse cell, so C's headline is the average of a deteriorating series, not a stable level. Worth watching in Phase 5, since D's test window is later still.

### Leakage — all checks clean
- **Empirical (carried forward from Phase 2):** PASS on all three. Perturbing future units changes features only at/after `origin + horizon`, and *does* change them after — the check has teeth.
- **Fit-loop (new for trees):** early stopping runs on the `HORIZON`-day tail of the fold's own *training* window; a test asserts the stopping set never touches the validation block. This surface did not exist in Phase 2 (Ridge has no early stopping) and is the likeliest place a tree pipeline leaks.
- **Frozen features:** the column manifest hashes to `447f7447d0e6fb23` (34 columns), asserted equal to Phase 2's by test. This is what licenses "only the objective changed."
- **Oracle crossing:** no entrant scored below its own oracle line. Treated as an investigate-flag, not an assert — the oracle bounds constant-per-series forecasts, so crossing it is surprising but legitimate.

### Critic bounds — exercised, and currently inert
`min_forecast = 0.0` and `max_median_ratio = 24.0` from `thresholds.json`, evaluated on each winner:

| ds | below min_forecast | above max_median_ratio | rate |
|---|---|---|---|
| A | 0 | 0 | 0.0 |
| B | 0 | 0 | 0.0 |
| C | 0 | 0 | 0.0 |

**Zero violations everywhere.** The bounds will not block the Phase 4 critic — but they also never fire, so on this evidence they are **not yet a working gate**. Phase 4 must include a deliberately-failing case (which its plan already requires) or the critic will pass everything by construction. Flagged rather than quietly accepted.

---

## 6. Full-cell confirmation (Phase 2's deferred item, closed)

Winner refit with its frozen config on every series in the cell:

| ds | n sample → full | WMAPE_28 sample → full | Δ | daily Δ |
|---|---|---|---|---|
| A | 250 → 823 | 0.3015 → 0.3008 | **−0.0007** | +0.013 |
| B | 250 → 515 | 0.3475 → 0.3454 | **−0.0021** | −0.048 |

**Sample fidelity confirmed.** The 250-series stratified samples reproduce the full cell to within 0.2% WMAPE_28. Phase 1's sampling decision and every Phase 2/3 number reported on the sample stand.

---

## 7. The agent contract (`artifacts/phase3_model_contract.json`)

The terminal deliverable — what Phase 4's Executor reads. It does **not** re-tune.

| branch | assignment | votes | evidence |
|---|---|---|---|
| `B1_tweedie` | `lgb_tweedie` | 3–0 | decided on A, B, C |
| `standard` | `lgb_tweedie` | 2–0 | decided on A, B (C has no Standard series) |
| `B2_baseline` | `sba` (α=0.1) | 1–1, tie broken on mean score | decided on B, C; A thin (n=2) excluded |

- **Completeness is asserted, not assumed:** every branch `classify_branch()` can emit has an entry, tested. The executor can never meet a branch it has no model for.
- **One frozen config per branch:** `lgb_tweedie[min_data_in_leaf=20, num_leaves=31, tweedie_variance_power=1.5]`. The datasets genuinely disagreed on the optimum (variance_power 1.5/1.3/1.1 on A/B/C), so the choice follows a stated rule — best mean selection-fold WMAPE_28 — and **the price of freezing is reported: 0.0029 (A), 0.0042 (B), 0.0051 (C)**. Under 0.6% relative, so a single frozen config is defensible; had it been large, that would have been a finding against freezing.
- **`min_cohort_series = 20`**, below which a cohort routes to `B2_baseline`. Set just under the smallest cohort that produced a stable score here (28), not tuned to make a cohort pass.
- Ships the feature-manifest hash, the pre-registered grid, the critic bounds with their observed violation rates, and a full run manifest.

---

## 8. Reproducibility

- **Determinism verified empirically, not just configured.** The bake-off was executed three times independently; every reported figure was **bit-identical**. Seeds, `deterministic=True`, `force_row_wise=True` and a fixed `num_threads=4` are set on every fit. (For contrast, the Phase 2 Ridge drifts at ~1e-14 across runs from BLAS reduction order — small, but it is why this is configured rather than hoped for.)
- **Dependencies pinned** to `==` (pandas 3.0.3, numpy 2.5.1, pyarrow 25.0.0, lightgbm 4.6.0, scikit-learn 1.9.0) and stamped into every artifact by `dfa.manifest` alongside the git SHA.
- **Compute:** 838s of fitting for the bake-off (~360 fits), 235s for the full-cell confirmation, ~5 min for quantiles. Comfortably inside the plan's 1-hour budget, so the grid did not need shrinking.
- **Tests:** 141 passing (45 new in Phase 3: metrics, models, croston, contract, manifest).

---

## 9. Open items carried to Phase 4

1. **Collapse `B1_tweedie` and `standard`?** The evidence says the split earns nothing (§2). This changes the router's shape, so it is a Phase 4 sub-plan decision, not one taken here.
2. **The critic bounds never fire** (§5). Phase 4 needs the deliberately-failing case its plan already specifies, or the gate is decorative.
3. **C's per-fold drift** (0.41→0.60 across folds, §5) — D's Phase 5 window is later still, so expect the sparse cohort to score worse there than these numbers suggest.
4. **Do quantiles ship as a third branch output**, or stay a reported side-deliverable? Deferred from the plan; unchanged by these results.
