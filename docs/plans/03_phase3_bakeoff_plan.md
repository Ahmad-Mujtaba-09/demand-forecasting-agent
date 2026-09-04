# Phase 3 Sub-Plan — LightGBM bake-off

**Status:** Draft for review. No code is written until this is approved. Parent: [00_master_plan.md](00_master_plan.md). Predecessors: [01_phase1_eda_plan.md](01_phase1_eda_plan.md), [02_phase2_baseline_plan.md](02_phase2_baseline_plan.md) (both complete; Phase 2 results in [../results/02_phase2_baseline_results.md](../results/02_phase2_baseline_results.md)).

**Goal:** pick the point-forecast model per dataset, honestly, on the frozen Phase-2 feature set — so the only thing that varies is the **objective**. Add the B4 quantile models (requirement-driven). Validate that the deterministic branch rules Phase 4's executor will apply are actually the right rules. **D** (`TX_1 × FOODS_2`) stays sealed.

**Discipline reminder (unchanged):** state the prior before the choice; every artifact feeds a decision; small tested increments each ending on a working run; a suspiciously good score is a prompt to investigate, not to celebrate.

---

## 0. Amendment raised for approval: the metric is degenerate on B and C

**This section must be resolved before §1 is executed.** It changes what the bake-off can conclude, and it is a correction to master plan §0.5, not a Phase-3 detail.

### 0.1 The finding
Measured on the actual Phase-2 validation rows (5 folds, same evaluable-row predicate). The **oracle** columns are computed *on the validation window itself* — each is the best **constant per (series, fold)** forecast, using that window's own actuals. They cheat, so they are attainability bounds, not results.

**Scope of the bound (stated precisely, because it is easy to overclaim):** these are optima over *constant-per-series* forecasts, **not** over all daily point forecasts. A model conditioning on features (day-of-week, SNAP, price) can legitimately beat them by varying its forecast within the window. So they are strong reference lines — beating one requires genuine within-window signal — but not proofs of impossibility, and **not** a leakage test (see §4.2).

| Dataset | zero forecast | oracle **median** per series (bound for daily WMAPE) | oracle **mean** per series (bound for a mean-targeting objective) | Phase 2 baseline |
|---|---|---|---|---|
| A | 1.000 | 0.463 | 0.490 | 0.621 |
| B | 1.000 | **0.882** | **1.091** | 1.106 |
| C | 1.000 | **0.999** | **1.650** | 1.631 |

Two consequences, both structural:

- **Tweedie and Poisson estimate the conditional mean.** On B the *perfect* conditional mean scores 1.091 and on C 1.650 — both worse than forecasting nothing. So the Phase-3 bar of 1.000 on B and C is not a hard target for those objectives; it is **unreachable by construction**. Running the bake-off as written would produce a guaranteed null result and we would have learned nothing about the objectives.
- **On C there is effectively no headroom at all.** 100% of C's series have a validation-window median of 0, so the best constant-per-series forecast *is* all-zeros, at 0.999 ≈ 1. Clearing 1.000 on C would require a feature-conditioned model to extract within-window timing signal from series that are 91% zeros — not impossible in principle, but there is no reason to expect it and the plan does not bet on it.

**Why:** WMAPE is an absolute-error metric, and absolute error is minimized by the conditional **median**. On a series with zero-share > 0.5 the conditional median is 0. This is the standard result for intermittent demand (Kolassa 2016, *Evaluating predictive count data distributions in retail sales forecasting*; consistent with Syntetos & Boylan). Master plan §0.5 committed to WMAPE before this consequence was visible in our own numbers. Phase 2's results are correct and honestly read — the problem is the yardstick, surfaced by Phase 2 doing its job.

### 0.2 Proposal — keep WMAPE, add two things
Neither replaces the headline; master plan §0.5 stays intact and Phase-2 comparability is preserved.

**(a) Horizon-aggregated WMAPE — co-primary decision metric.**
Same rows, same folds, same predicate; aggregate to per-`(series, fold)` **28-day totals** before the sum-then-divide:

```
WMAPE_28 = Σ_i Σ_k | Σ_t∈fold_k y_{i,t} − Σ_t∈fold_k f_{i,t} |  /  Σ_i Σ_k Σ_t y_{i,t}
```

- **Prior:** a replenishment decision consumes *demand over a lead time*, not a per-day point estimate — the 28-day total is the quantity a planner orders against, so this is the more decision-relevant number, not a metric chosen to look better. It is non-degenerate precisely because a 28-day total is rarely zero (zero on 4.0% / 8.6% / 20.9% of `(series, fold)` cells for A/B/C), which restores the conditional **mean** as the right estimation target and makes Tweedie/Poisson answerable questions again. Verified on the honest trailing-mean forecast (no peeking): **A 0.481, B 0.429, C 0.650** — all meaningfully below the 1.000 zero line, including on C where the daily view has no headroom at all.
- **Known property, stated so the metric switch cannot be mistaken for metric-shopping:** the two metrics share the *same denominator*, and by the triangle inequality `|Σ(y−f)| ≤ Σ|y−f|`, so **WMAPE_28 ≤ daily WMAPE always, for every model**. A lower number is therefore guaranteed and is *not* the argument for it. The argument is that (i) the **zero line stays exactly 1.000** under WMAPE_28 — `|Σy − 0|/Σy = 1` — so the *bar is not lowered, only made reachable*, and (ii) the gap between the two metrics is a clean error decomposition: **WMAPE_28 is level error, daily − WMAPE_28 is within-window timing error.** We report both, so we are reporting the decomposition, not choosing the flattering half.

**(b) The two oracle lines, reported per dataset.**
`oracle_median` and `oracle_mean` (as in §0.1) shipped alongside every result table. They cost one groupby.
- **Prior:** they convert "the model lost to zero" into "the best constant-per-series forecast for this objective class is X, and we landed at Y." That is the difference between a null result and a measured one, and it is what makes an honest negative finding reportable rather than embarrassing.
- **They are a reference line, not a leakage test.** An entrant scoring below its oracle is *surprising* and worth investigating (it claims real within-window conditional signal), but it is legitimately possible — see the scope note in §0.1. The leakage tests are the three in §4, which check mechanism rather than score.

### 0.3 What the bake-off then decides

**The decision unit is the BRANCH, not the dataset.** This is a correction to how the phase was framed in master plan §4: Phase 4's executor never receives "dataset A" — it calls `classify_branch()` per series and needs to know *which objective each branch gets*. A per-dataset winner is not a thing the agent can consume. So:

- **Primary output:** an objective + frozen hyperparameters **per branch** (`B1_tweedie`, `standard`, `B2_baseline`), emitted as the machine-readable contract in §11. This is what Phase 4 executes and Phase 5 tests.
- **Per-dataset winners are a reporting cut** — a legibility and consistency check ("does the branch-level answer hold on each cell?"), not the deliverable.
- **Selection metric:** **WMAPE_28**, because the daily metric is known-degenerate on 2 of 3 datasets and a rule that silently changes per dataset is not a rule. Daily WMAPE is reported for every entrant and is the **tiebreaker** when WMAPE_28 differences sit inside the fold-to-fold spread.
- **If every entrant loses the daily-WMAPE zero line on B/C, that is an expected and explained outcome**, not a failure — §0.1 predicts it. The claim we get to make is about WMAPE_28.

**If this amendment is rejected**, the honest fallback is to run the bake-off on daily WMAPE only and report that B and C are structurally unwinnable, with §0.1 as the evidence. That is a defensible phase outcome, just a much thinner one. Flagging the fork rather than choosing it unilaterally.

---

## 1. The entrants

All trained on the **frozen Phase-2 feature set, unchanged** (`dfa.features`), on the **same folds**, with the **same B2 routing** and the **same evaluable-row predicate**. The feature manifest is asserted equal to Phase 2's in a test (§4.4), so "only the objective changed" is mechanically true, not a claim.

| # | Entrant | Objective | Why it is in |
|---|---|---|---|
| E1 | LightGBM **L2** | `regression` | The tree-model control. Isolates *objective* from *model class*: the gap E1 − Ridge is what trees buy, the gap E2/E3 − E1 is what the objective buys. Without it, "Tweedie beat Ridge" confounds the two. |
| E2 | LightGBM **Tweedie** | `tweedie`, `variance_power` ∈ {1.1, 1.3, 1.5} | The B1 branch's model. Compound Poisson-Gamma matches zero-inflated non-negative demand. |
| E3 | LightGBM **Poisson** | `poisson` | Count-native, one fewer tuned parameter than Tweedie. If it ties Tweedie, the simpler branch rule wins. |
| E4 | LightGBM **quantile α=0.5** | `quantile` | **Promoted into the bake-off** (change from master plan §3, where quantiles were B4-only). It is the only entrant whose loss is aligned with what daily WMAPE actually rewards — the conditional median. Excluding it would mean testing three mean-estimators against a median metric and calling the result a bake-off. |
| E5 | **Croston / SBA** | — | The literature-standard intermittent-demand estimator (Croston 1972; Syntetos–Boylan approximation, 2005). Master plan §3 names "Croston-style" as the B2 branch's fallback and it was never built — Phase 2 used a mean floor. **B2 is 79% of dataset C**, so C's headline is mostly a floor number and the floor has never been tested against its own field's standard. Cheap, no tuning, closes a real gap. |
| — | Reference lines (not entrants) | — | Phase-2 Ridge floor, naive-mean, zero, both oracles — carried forward unchanged for continuity. |

**Prior on E5's placement:** Croston/SBA competes **on the B2 cohort only** (against the mean floor), not against LightGBM on modelable series. It is a fallback estimator; testing it where the heavy model belongs would be a category error in the other direction.

---

## 2. Hyperparameter protocol — and why Phase 2's argument does not carry over

Phase 2 tuned and reported on the same folds, justified by: *the optimism is equal on both sides and cancels.* That argument was sound there because **Ridge had no capacity to exploit the selection** — the α-grid WMAPE spread was 0.00014 (A), 0.0017 (B), 0.018 (C), i.e. selection was a no-op. **LightGBM has real capacity, so the optimism no longer cancels** and reusing the argument would be quietly dishonest.

**Decision:**
- **Pre-registered grid, identical for every objective** — written into the plan *before* any run, so no per-objective tuning effort asymmetry can creep in. Small and defensible: `num_leaves ∈ {15, 31}`, `learning_rate ∈ {0.05}`, `n_estimators` by early stopping on the fold's own train tail, `min_data_in_leaf ∈ {20, 100}`. Plus Tweedie's `variance_power` (3 values), which is a distributional parameter, not a capacity knob. Total ≤ 4 configs per objective (12 for Tweedie).
  - **Prior:** a large grid would make the selection-optimism problem worse and would drift the phase into hyperparameter search, which is not what it is for. The bake-off tests objectives; the grid exists only so no objective is handicapped by an obviously-wrong capacity setting.
- **Selection on folds 0–3; confirmation reported on fold 4 (never used for selection).** Report **both** the pooled 0–4 number (Phase-2-comparable, mildly optimistic, equal-protocol) and the **selection-free fold-4 number** (honest, higher variance). The gap between them *is* the measured optimism — reported, not argued about.
  - **Prior:** a nested/rolling inner CV would be cleaner but multiplies compute by the fold count for a phase whose question is "which objective," not "what is the exact generalization error." Holding out the latest fold is the cheapest construct that makes the optimism a measured quantity instead of an assumption.
- **Identical protocol for every entrant, including the Phase-2 Ridge when re-reported.**
- **The selected hyperparameters are FROZEN into the §11 contract — the Phase 4 agent does not re-tune at runtime.**
  - **Prior:** an agent that re-runs a grid search on incoming data has unbounded runtime and a non-deterministic decision path, which breaks master plan §0.4 (*deterministic, defensible branch logic*) — the agent would be inventing modelling decisions, which is exactly what §0.4 forbids. Freezing also makes Phase 5 a **stronger** test, not a weaker one: hyperparameters tuned on A/B/C are applied *unchanged* to sealed D, so D tests the whole frozen configuration rather than a fresh fit. The honest limit to state alongside: these params were tuned on three M5 cells, and their transfer is evidenced only by D.
- **Determinism is a hard requirement, not a nicety.** Every LightGBM fit sets `seed`/`bagging_seed`/`feature_fraction_seed`, `deterministic=True`, `force_row_wise=True`, and a **fixed `num_threads`**. LightGBM's default histogram construction is thread-count-dependent, so without this the *same code on the same data* returns different numbers on a different machine — fatal for a project whose deliverable is a reproducible, defensible pipeline, and fatal for Phase 5's sealed run being re-checkable. A test asserts two fits with the same seed produce identical predictions.

**Increment deliverable:** a `tuning` helper + tests: the grid is a module constant (pre-registered), selection sees only folds 0–3, a test asserts fold 4 never enters any selection path, and a **bit-identical-refit test** for determinism.

---

## 3. Branch-rule validation — the deliverable Phase 4 actually needs

Master plan §3 has the executor apply deterministic rules: **B1 → Tweedie** if zero-share ≥ 0.663, **Standard → L2/Poisson** otherwise, **B2 → simple baseline** if ADI ≥ 8.77. Phase 3 is the only chance to test whether those rules are *right* before Phase 4 hardcodes them.

**Method:** every entrant runs on every dataset's full modelable set; then score each objective **per (branch cohort × dataset) cell**.

**Do NOT pool cohorts across datasets.** Measured on the val rows, total volume is **A 123.1k / B 18.1k / C 2.4k units** — so under a volume-weighted metric a "pooled B1 cohort" would be **85.7% dataset A and 1.7% dataset C**, i.e. A's 28 B1 series would decide a verdict nominally about 194 series. The cohorts also do not span the datasets:

| | B1_tweedie | standard | B2_baseline |
|---|---|---|---|
| A | 28 | 220 | 2 |
| B | 135 | 88 | 27 |
| C | 31 | **0** | 118 |

So the cohort verdict is read off the **matrix**, by counting which objective wins each cell and reporting the per-cell margin against that cell's own zero line. Cells that are thin (A's 2 B2 series, A's 28 B1) are **labelled thin and excluded from the verdict count**, not silently averaged in. C contributes no evidence at all about the Standard branch, and the write-up must say so rather than implying three-dataset support.

- **Prior:** a single pooled number would have been more quotable and would have been wrong. The whole point of the phase is a branch rule that survives scrutiny; a verdict dominated 50:1 by one cell's volume is not that.
- If Tweedie wins on the B1 cohort and loses on Standard → **B1 is confirmed**, with a number behind it.
- If one objective wins everywhere → **say so and simplify the branch**, exactly as B3 was dropped in Phase 1 when the evidence didn't support it. A branch that doesn't earn its complexity gets cut.
- If the cut point looks wrong (e.g. Tweedie's advantage starts at a different zero-share), report the **empirical crossover** against the P60-derived 0.663 and flag the discrepancy for a threshold revision — do not silently retune it.

- **Prior:** the branch rules are currently justified by distributional theory and a percentile, not by measured forecast accuracy. Phase 4 wraps an agent around them and Phase 5 tests that agent on sealed data. If a rule is wrong, this is the last phase where it is cheap to find out.

**Minimum cohort size (a Phase 4 production rule this phase must supply).** An incoming dataset can produce a cohort of 3 series, and a pooled model fit on 3 series is not a model. Phase 3 records the observed cohort sizes and their scores, and **states a `MIN_COHORT_SERIES` floor with a documented fallback** (below the floor, the cohort routes to `B2_baseline`, the branch that exists precisely for "not enough signal"). Without this, the Phase 4 agent has an unhandled edge case on any dataset less M5-shaped than ours.

- **Prior:** the floor is a judgement call, not a measurable optimum, so it is set from the observed evidence and **recorded as an explicit constant with its rationale** rather than discovered as a crash in Phase 5.

**Deliverable:** objective × (cohort × dataset) WMAPE matrix + a stated verdict per rule (confirmed / simplified / flagged) + the `MIN_COHORT_SERIES` constant.

---

## 4. Leakage discipline

### 4.1 Carried forward
`run_baseline.leakage_check` (perturb-and-diff, with teeth: asserts features are unchanged before `origin + horizon` **and** do change after) runs unchanged for every dataset. It passed on all three in Phase 2.

### 4.2 New for trees
- **Oracle-crossing flag (investigate, do not halt):** an entrant scoring **below its own oracle line** (§0.2b) is claiming within-window conditional signal that beats the best constant-per-series forecast. That is possible but unlikely on this data, so it is logged as a **flag requiring investigation before the number is reported** — not an assertion failure, because the oracle is not a hard bound (§0.1 scope note). Treated per master plan §Phase 3: a suspiciously strong score is a prompt to inspect, not accept.
- **Early stopping must not see validation.** Early-stopping rounds are selected on a **tail slice of the fold's own training days**, never on the fold's validation block. This is the most likely place for a tree pipeline to leak and it is not covered by the Phase-2 check (which tests features, not the fit loop). Explicit test.

### 4.3 Exercise the Phase 4 critic's bounds now
The critic's gates already exist as calibrated numbers in `artifacts/thresholds.json`: **`min_forecast = 0.0`** and **`max_median_ratio = 24.0`** (a forecast above 24× a series' median non-zero demand is implausible). Phase 3 counts, per entrant, how many predictions violate each.

- **Prior:** Phase 4 wires these into a gate that can **reject the chosen model**. Discovering in Phase 4 that the bake-off winner trips the critic on 3% of rows — or that the bound never fires and is therefore decorative — is discovering it one phase too late. Measuring it here costs two comparisons and either validates the bound or flags it for recalibration *before* it becomes a blocker.
- Non-negativity is enforced by the same clamp Phase 2 used, applied uniformly to every entrant. Tweedie and Poisson are non-negative by their log link; **L2 and quantile are not**, so the clamp is load-bearing for E1/E4 and its effect (rows clamped, per entrant) is reported rather than hidden.

### 4.4 Frozen-feature assertion
A test asserts the Phase-3 feature-column manifest **equals** the Phase-2 manifest exactly. This is what licenses the claim "only the objective changed."

- **Prior (master plan §Phase 3):** do not introduce errors on purpose. The checks run as standard; a clean result is a valid outcome and is reported as such.

---

## 5. Quantile models (B4) — requirement-driven, judged on their own terms

Fit LightGBM `quantile` at **0.1 / 0.5 / 0.9** per dataset (0.5 also enters the bake-off as E4; the fit is shared).

- **Evaluated with pinball loss** (per quantile) + **empirical coverage** (fraction of actuals below q10 / below q90; target 10% / 90%) + **mean interval width**.
- **Not compared to Tweedie on WMAPE.** Quantiles answer a different question — a range, for inventory sizing — and scoring a q90 forecast on a point metric would penalize it for doing its job. Master plan §3 B4 already states this; it is restated so no results table blurs it.
- **Prior:** coverage is the honest check on an interval. An interval that is 90%-wide but only covers 60% of outcomes is worse than useless for a safety-stock decision, and pinball loss alone will not reveal that.
- **Quantile crossing must be handled, not assumed away.** The three quantiles are fit as three independent models, so nothing stops `q10 > q50` or `q50 > q90` on a given row — a standard failure of independent quantile regression, and one that emits a negative-width interval straight into a safety-stock calculation. We apply a **post-hoc monotone sort** across the three predictions per row and **report the crossing rate** as a diagnostic. A high rate means the quantile fits are unstable and the intervals should not be trusted, which is a finding.
- **Deferred, stated:** intervals on the **28-day total** (the quantity a lead-time safety stock actually needs) are *not* obtainable by summing daily quantiles — the quantile of a sum is not the sum of quantiles. Daily quantiles are what we ship; horizon-total intervals are named as future work and **not** approximated by summation, because that approximation would be silently wrong in the direction that under-sizes safety stock.

**Increment deliverable:** `pinball_loss` + `coverage` in `dfa.metrics` with hand-computed unit tests; a known-quantile synthetic fixture.

---

## 6. Scale — the deferred full-cell confirmation, closed here

Phase 2 deferred the A/B full-cell run ("end-of-phase or rolled into Phase 3"). **It is rolled in here.**
- Bake-off and tuning run on the **250-series samples** (A, B) and the **full cell** (C = 149) — the Phase-2 iteration surface, so the comparison to the floor is like-for-like.
- The **winning objective only** is then re-run on **full A (823)** and **full B (515)** as a confirmation, reported beside the sample number.
- **Prior:** tuning on the full cells would multiply compute for no decision value — Phase 1 verified sample fidelity. But the headline of a phase should not rest on a sample when the full cell is available for one extra fit. If the full-cell number diverges materially from the sample, that is itself a finding about sample fidelity and gets reported.

---

## 6b. Reproducibility and observability (the production bar)

The core has to be runnable, re-checkable, and diagnosable by someone who is not me — and in Phase 4 by an agent that must report what it did.

- **Run manifest, emitted into every results JSON:** git SHA, `lightgbm`/`sklearn`/`pandas`/`numpy` versions, the RNG seed, `num_threads`, the resolved fold origins, the feature-manifest hash (§4.4), and the dataset id-list hash. **Prior:** a results table with no provenance cannot be audited, and Phase 5's sealed run is only meaningful if the configuration that produced it is pinned to a commit. This is also the payload Phase 4's Report node has to surface.
- **Wall-clock budget, recorded per entrant per dataset** (fit + predict, per fold). **Prior:** Phase 4 runs this pipeline end-to-end inside an agent loop and Phase 5 runs it on sealed data; if a pooled fit on full A (823 series ≈ 1.6M rows) takes minutes per config, the agent's runtime is a design constraint that must be known *before* the agent is built, not discovered during it. Rough expected budget for the phase: ~540 fits (5 entrants × grid × 5 folds × 3 datasets, incl. quantiles) — if that exceeds ~1 hour, the grid is what shrinks, and the plan says so up front rather than quietly widening scope.
- **Dependency pinning is currently `>=`, not pinned.** `requirements.txt` allows `pandas>=3.0`, `numpy>=2.5`, `scikit-learn>=1.5`, `lightgbm>=4.6`; the environment resolves to pandas 3.0.3 / numpy 2.5.1 / sklearn 1.9.0 / lightgbm 4.6.0. **A `>=` range does not support a reproducibility claim** — a fresh install a month from now can silently change a tree-split tie-break or a pandas groupby default. Pin the four numeric dependencies to `==` (or add a lock file) as part of this phase's first increment, and record the versions in the run manifest regardless.
- **Out of scope, named:** trained-model serialization and a model registry. Phase 4's agent refits per run, so nothing needs persisting yet; a production deployment would need it and it is deferred deliberately, not overlooked.

---

## 7. Results to present (for review)

Per dataset A / B / C:

1. **Bake-off table** — entrant × {daily WMAPE, WMAPE_28}, pooled 0–4 **and** selection-free fold 4, against zero / oracle_median / oracle_mean / Phase-2 Ridge.
2. **Chosen winner + why**, per the §0.3 rule stated in advance.
3. **WMAPE by SB class** for the winner (the Phase-2 honesty check, carried forward).
4. **Branch-cohort matrix** (§3) + verdict per branch rule.
5. **B2 cohort:** Croston/SBA vs mean floor (§1 E5) — does the fallback branch improve?
6. **Quantile results** — pinball, coverage, width (§5).
7. **Full-cell confirmation** for A/B winner (§6).
8. **Leakage-check output** — all four checks (§4), reported either way.
9. **Measured selection optimism** — pooled vs fold-4 gap, per entrant.
10. **Per-fold spread** for the winner (stability).
11. **Critic-bound violations** per entrant (§4.3) — counts for `min_forecast` and `max_median_ratio`, plus rows clamped.
12. **Quantile crossing rate** (§5).
13. **Wall-clock per entrant** and the **run manifest** (§6b).
14. **The §11 branch contract** — the machine-readable artifact Phase 4 consumes, shown as part of the review.

Table in `docs/results/03_phase3_bakeoff_results.md` + `artifacts/phase3_bakeoff_results.json`. An HTML review artifact only if a visual earns its place (the objective × cohort matrix probably does).

---

## 8. Increment order (each ends on a working, tested run)

0. **Pin dependencies** (§6b) + a `run_manifest()` helper with a test. *(Trivial, but everything downstream is stamped by it, so it goes first.)*
1. **`metrics` extension** — `wmape_horizon_aggregated`, `oracle_lines`, `pinball_loss`, `coverage` + unit tests on synthetics with hand-computed answers. Includes a test asserting **WMAPE_28 ≤ daily WMAPE** on random inputs (the §0.2 triangle-inequality property, as an executable invariant).
2. **Re-score Phase 2 under the new metrics** — no new models, just the existing Ridge/naive/zero/floor predictions through §0.2. Gives the corrected reference lines the bake-off is measured against, and confirms the §0.1 table reproduces from the pipeline rather than from an ad-hoc script.
3. **`models` module** — LightGBM entrant wrappers (E1–E4) sharing one fit/predict interface with the Phase-2 baseline, + the pre-registered grid, + fold-4 holdout selection (§2) + tests (§4.2 early-stopping leakage, §4.4 frozen manifest, §2 determinism refit).
4. **Croston/SBA** (E5) + unit tests against hand-worked examples from the literature.
5. **Run the bake-off** across A/B/C → results JSON.
6. **Branch-cohort analysis** (§3) → cohort × dataset matrix + verdicts + `MIN_COHORT_SERIES`.
7. **Quantile models** (§5) → pinball/coverage/crossing table.
8. **Critic-bound exercise** (§4.3) → violation counts.
9. **Full-cell confirmation** for the winning objective (§6).
10. **Emit the §11 branch contract** + a test that it round-trips and that every branch `classify_branch` can return has an entry.
11. **Results write-up** → review.

Nothing proceeds to Phase 4 until the bake-off and the branch verdicts are reviewed and approved.

---

## 9. Decisions I'm making vs deferring

**Making now (priors above):**
- **Winner selected on WMAPE_28**, daily WMAPE reported for all and used as tiebreaker (§0.3) — *conditional on the §0 amendment being approved.*
- **Entrants E1–E5**, with LightGBM-L2 as the control that separates model class from objective, and **quantile-0.5 promoted into the bake-off**.
- **Croston/SBA competes on the B2 cohort only**, against the mean floor.
- **Pre-registered small grid; selection on folds 0–3; fold 4 selection-free** — Phase 2's "optimism cancels" argument explicitly retired for trees, with the reason stated.
- **Feature set frozen and asserted equal to Phase 2's** — the objective is the only variable.
- **Branch-rule validation is a phase deliverable**, with the right to simplify or flag a rule (precedent: B3 dropped in Phase 1).
- **A/B full-cell confirmation runs here** for the winner.
- **The deliverable is a per-BRANCH contract (§11), not a per-dataset winner** — per-dataset results are a reporting cut.
- **Hyperparameters are frozen into the contract; the Phase 4 agent does not re-tune** (§2).
- **No cross-dataset cohort pooling** — the verdict is read off the cohort × dataset matrix (§3), because pooling is 86% dataset A by volume.
- **Determinism enforced and tested** (seeds, `deterministic=True`, fixed `num_threads`); **deps pinned**; **run manifest** on every result (§2, §6b).
- **Quantile crossing sorted and its rate reported**; horizon-total intervals deferred, not approximated (§5).

**Deferring (flagged, not decided):**
- Whether a **confirmed branch-rule change** (e.g. a revised B1 zero-share cut) is applied in Phase 3 or carried into the Phase 4 sub-plan — depends on what §3 finds; the finding is reported either way and the threshold is **not** silently retuned.
- Whether the quantile models ship as a **third branch output** in Phase 4 or stay a reported side-deliverable.
- **Categorical handling:** LightGBM natively handles categoricals and our frozen set is one-hot, which mildly handicaps every tree entrant *equally*. Keeping one-hot for the frozen-feature guarantee; a native-categorical run is a possible **post-bake-off sensitivity check**, not part of the comparison.

---

## 10. Phase 2 items carried in (raised at the Phase 3 gate)

1. **Metric degeneracy** — §0. The blocking one.
2. **Selection optimism does not transfer to trees** — §2.
3. **A/B full-cell confirmation still outstanding** — §6. (The commit titled *"phase 2 — first run (full dataset)"* refers to the signal-table build; `artifacts/phase2_baseline_results.json` correctly records A and B as `"sample"`, n=250 of 823 / 515.)
4. **Croston/SBA named in master plan §3 but never built** — §1 E5. C's headline is 79% mean-floor, so the B2 branch's estimator has never been tested against its field's standard.
5. **Dead B3 surface — RESOLVED (done at this gate).** `classify_branch()` returned an unused `add_seasonal` flag and `thresholds.json` carried `dow_season_cut = 0.042`, contradicting master plan §3/§7 which already declared B3 dropped. Verified dead on both axes before removal: (a) *no consumer* — across all six commits, `[1]` was only ever read to report `seasonal_feature_frac`, and `features.py` has no conditional on the calendar block, so the flag was a no-op for both of its values; (b) *dead in general, not just on M5* — since every series gets calendar features unconditionally, a seasonal gate has no "off" state on **any** M5-format dataset. Removed the cut, the flag, `DOW_SEASONAL_PCT`, `seasonal_feature_frac`, and the gate test; `classify_branch(zero_share, adi, thr) -> str`. A replacement test pins the deletion. The η² signal, `med_dow` profiling, the sample-fidelity gate, and the η² percentiles in `distribution_summary_non_d` all stay. Regenerating `thresholds.json` left all three live cuts and all branch counts byte-identical; re-running Phase 2 reproduced every reported figure exactly (max float delta 1.1e-14, BLAS noise — itself a datapoint for §2's determinism requirement).
6. **Doc wording drift (no code change needed):** `docs/results/02_phase2_baseline_results.md` describes the B2 cut as "ADI ≥ 8.77, data-relative, **recomputed per dataset**", but `run_baseline._load_thresholds` reads one global `artifacts/thresholds.json` calibrated once on the non-D pool. The behavior matches the Phase-1 design (data-relative *to the incoming dataset*, which is M5-minus-D; A/B/C are sub-datasets of it, not separate datasets). **The sentence is wrong, not the code** — fix the wording.

---

## 11. The agent-handoff contract (what Phase 4 actually consumes)

Phases 1–3 exist to produce a **decision function plus a configuration**, and Phase 4 wraps an agent around them. Master plan §0.1 (*core first, agent last*) and §0.4 (*the agent selects which branch applies per documented rules; it does not invent modelling decisions*) only hold if the handoff is a **data artifact**, not prose in a results document that an implementer has to interpret.

Phase 3's terminal deliverable is therefore `artifacts/phase3_model_contract.json`:

```
{
  "run_manifest":  { git_sha, lib_versions, seed, num_threads, ... },   # §6b
  "thresholds_ref": "artifacts/thresholds.json",                        # Phase 1, unchanged
  "feature_manifest_hash": "...",                                       # §4.4 -- what "frozen" means
  "min_cohort_series": <int>,                                           # §3, with fallback branch
  "branches": {
    "B1_tweedie":   { "objective": ..., "params": {...}, "evidence": {...} },
    "standard":     { "objective": ..., "params": {...}, "evidence": {...} },
    "B2_baseline":  { "estimator": ..., "params": {...}, "evidence": {...} }
  },
  "quantiles":     { "levels": [0.1,0.5,0.9], "params": {...}, "monotone_sort": true },
  "critic_bounds": { "min_forecast": 0.0, "max_median_ratio": 24.0, "observed_violation_rate": {...} }
}
```

- **Every branch `classify_branch()` can return must have an entry** — asserted by a test, so the executor cannot meet a branch it has no model for. This is the completeness property that makes the deterministic router safe.
- **`evidence` carries the number that justified the choice** (the cohort × dataset cells and their margin over that cell's zero line), so Phase 4's Report node and Phase 5's write-up can cite *why* a branch has the objective it has without re-deriving it.
- **Consumed, not re-decided.** Phase 4's Executor reads this file; it does not re-tune, re-threshold, or re-rank objectives. The agent's freedom is *routing* (which branch a series is in) and *reporting* — never modelling choices.
- **Prior:** every phase so far has emitted its state as a machine-readable artifact (`signal_table.parquet`, `selection.json`, `thresholds.json`) rather than as a claim in a document, which is why Phase 2 could be rebuilt on Phase 1's numbers without re-litigating them. Phase 3 is the phase where that habit matters most, because its output is the only thing standing between a reviewed decision and an agent improvising a model choice at runtime.
