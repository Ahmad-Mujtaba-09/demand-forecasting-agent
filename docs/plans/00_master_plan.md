# Demand-Forecasting Agent — Master Plan

**Format:** Retail sales data in the **M5 canonical format** — a sales table (series × time), a calendar, and weekly sell prices. The system is written to that *format*, not to M5 specifically: M5 is the **validation dataset**, not the scope.

**Scope:** A demand-forecasting system. The inputs carry unit sales, a calendar, and sell prices — **no cost or margin data** — so this forecasts demand and does **not** optimize profit. Forecasts are shaped so a downstream price/inventory optimizer *could* consume them later; that layer is out of scope.

**Threshold stance:** Branch thresholds are set from **each dataset's own distribution** where the cut would otherwise be tuned to a particular dataset (e.g. what counts as "high" zero-share, or the too-sparse floor). Literature-standard constants are used as published and cited, not re-tuned — the Syntetos–Boylan cutoffs (ADI 1.32, CV² 0.49; Syntetos & Boylan 2005) stay absolute. See §7.

**Honesty boundary (holds throughout):** The claim is **format-general and threshold-adaptive** — it ingests any M5-format retail dataset and calibrates its cuts to that dataset. The **evidence is M5 only**: we have not run it on another retailer's data because we have none. State both; never let "designed to generalize" read as "shown to generalize." Generality here is a **design property we argue for**, not a set of features we build — there are deliberately **no** format converters, column auto-detection, or per-dataset quirk handling. The validator stays strict to the M5 format and flags mismatches rather than adapting to them.

**Status:** Draft for review. This is the overarching plan for all five phases. Each phase gets its own detailed sub-plan, reviewed and approved *before* execution. No phase starts until the prior phase's results are reviewed.

---

## 0. Governing principles

These hold across every phase. They exist so the reasoning stays visible and defensible.

1. **Core first, agent last.** Build and validate the ML core as plain code before any Agno orchestration. If time runs short, the core must stand alone as a working, honest forecasting pipeline.
2. **State the prior at every fork.** At each decision point (a threshold, a model choice, a feature toggle) write the driving assumption in plain language *before* the choice. Reasoning is explicit, never implied.
3. **Every EDA output feeds a decision.** If a computed statistic or plot does not change a later branch, threshold, or dataset pick, it is cut. No decorative analysis.
4. **Deterministic, defensible branch logic.** The modeling branches are functions *I* write and can defend. The agent selects which branch applies per documented rules; it does not invent modeling decisions.
5. **WMAPE is the yardstick.** Weighted Mean Absolute Percentage Error is the headline metric. Every model after the baseline must beat the baseline's WMAPE on a proper time-series holdout to be taken seriously.
6. **Time-series discipline.** Holdouts are expanding/rolling windows on the time axis. **Never shuffle.** Features (lags, rolling stats) are computed *within* the train/test split boundary, never across the full series.
7. **Test as you go.** Small verifiable increments. Write tests and show them passing. Every executable step ends on a working run. No large untested dumps.
8. **Flag, don't fake.** Validation and the critic flag problems rather than guessing or silently patching. A model that fails the critic is flagged, not accepted.

---

## 1. Dataset facts (verified from the files on disk)

Grounding the plan in the actual schema so later steps don't rediscover it.

| File | Shape / key columns | Notes |
|---|---|---|
| `calendar.csv` | 1,969 rows; `date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI` | One row per day from **2011-01-29**. `d` (`d_1`…) is the join key to sales. `wm_yr_wk` joins to prices. |
| `sales_train_validation.csv` | 30,490 series × `d_1`…`d_1913` | Wide format. Keys: `id, item_id, dept_id, cat_id, store_id, state_id`. |
| `sales_train_evaluation.csv` | 30,490 series × `d_1`…`d_1941` | Same series, **28 extra days** (`d_1914`…`d_1941`) — the labels for the validation horizon. Use as ground truth for final scoring. |
| `sell_prices.csv` | 6,841,121 rows; `store_id, item_id, wm_yr_wk, sell_price` | **Weekly** price per store/item. Sparse in time: a series only has price rows once it's on sale. Absence of a price row ≈ item not sold that week. |

Hierarchy: 3 states (CA, TX, WI) × 10 stores × 3 categories (HOBBIES, HOUSEHOLD, FOODS) × 7 departments × 3,049 items = 30,490 store-item series.

**Prior stated:** Demand at the individual store-item level is zero-heavy and intermittent (many days with zero sales). This single fact drives most modeling choices downstream — it is why squared-error regression is expected to underperform and why Tweedie/Poisson objectives are on the table.

---

## 2. Target architecture (built in Phase 4, described here for orientation)

The agent is the *last* thing built. The pipeline it will orchestrate:

```
Dataset in
   │
   ▼
[Validator]  — light, timeboxed. Confirms M5 schema: required columns present,
              dates continuous, no missing crucial fields. FLAGS problems,
              does not guess or repair. NOT a general CSV parser — schema is fixed.
   │
   ▼
[Planner]    — runs EDA, interprets the three signals, does preprocessing.
   │
   ▼
[Executor]   — fits models per the deterministic branch rules (§3).
   │
   ▼
[Critic]     — challenges the executor's choices AND its reported performance.
              Gates acceptance: WMAPE vs threshold + sanity checks
              (no negative forecasts, magnitudes in plausible range).
              Only passes if all checks hold. Otherwise flags.
   │
   ▼
[Report]     — short structured summary to the user.
```

The critic is a real gate, not a rubber stamp. If a model fails, the report says so.

---

## 3. The branch logic (deterministic rules the executor applies)

These are the documented, defensible rules. Thresholds are set in Phase 1/3 from real EDA distributions, not guessed here — the values below are placeholders to be calibrated and recorded.

| # | Condition (signal) | Action | Prior driving it |
|---|---|---|---|
| B1 | **High intermittency** — modelable *and* zero-share genuinely high (an upper percentile of the non-sparse population, not merely above median) | LightGBM with **Tweedie** objective | Zero-inflated non-negative demand; Tweedie's compound Poisson-Gamma form matches this shape, unlike squared error. Tweedie isn't free, so it's reserved for series that are *actually* zero-heavy. |
| **Standard** | **Modelable but not high-intermittency** (survives the sparsity gate, zero-share below the B1 cut) | LightGBM with a **standard objective** (L2 / Poisson, decided by the Phase 3 bake-off) | Lower-intermittency modelable series genuinely don't need Tweedie — the wrong objective for a series that isn't zero-heavy. This is the explicit complement of B1, not an undocumented path. |
| B2 | **Too sparse to model reliably** (ADI in the dataset's upper tail / signal too thin) | Fall back to **simple baseline** (moving average or Croston-style) | Not enough signal to justify a heavy model; a defensible simple estimate beats an overfit complex one. Knowing when *not* to use the heavy model is deliberate. |
| B3 | **Seasonality above threshold** (day-of-week variance-explained, η²) | Add seasonal features | When a stable weekly cycle exists, giving the model explicit seasonal structure helps more than making it rediscover it. |
| B3′ | **Seasonality below threshold** | Skip seasonal features; rely on calendar features (day-of-week, month, events, SNAP) | Weak/no seasonal signal → seasonal features add noise and cost; let LightGBM learn from calendar features directly. |
| B4 | **Use case needs inventory intervals** | Add LightGBM **quantile** models at 0.1 / 0.5 / 0.9 | **Requirement-driven, not a bake-off winner.** Quantiles answer a different question (a range) than Tweedie (a point). Added because the use case asks for intervals, never tested *against* Tweedie. |

Signals computed in EDA: **intermittency ratio** (zero-share), **seasonality** (day-of-week variance-explained, η² — chosen over STL, which is config-fragile on spiky retail data), **outlier/spike count**.

---

## 4. Phase-by-phase plan

Each phase: **produce a sub-plan → review → revise if flagged → execute in small tested increments → show results across datasets → gate before proceeding.**

### Phase 1 — EDA & dataset selection
**Goal:** Characterize M5 and pick the working datasets.
- Compute the three branch signals per series (and aggregated): intermittency ratio, seasonality (day-of-week η²), outlier/spike count.
- Every output must map to a later decision (branch threshold or dataset pick). Cut anything decorative.
- Select **3 sub-datasets at the pattern extremes** — fast-moving/dense, intermittent/zero-heavy, slow/sparse — plus **1 held-out sub-dataset** reserved for final agent testing (Phase 5), never touched before then.
- **Deliverable:** an artifact (visual review) presenting the signal distributions and the proposed dataset picks, for review *before* anything is locked.
- **Prior to state at the fork:** what defines each "extreme," and why those three cover the branch space (they map 1:1 to B1/B2/B3).

### Phase 2 — Baseline (the number to beat)
**Goal:** An honest floor.
- Plain **L2 / linear regression** baseline, trained to the best it can honestly do on each of the 3 datasets.
- This WMAPE is the bar every later model must beat.
- **Deliverable:** baseline WMAPE across all 3 datasets + code, for review.

### Phase 3 — LightGBM bake-off
**Goal:** Pick the point-forecast model per dataset, honestly.
- On each dataset, bake-off: **baseline L2 vs Tweedie vs Poisson**, compared on **WMAPE** on a proper time-series holdout (expanding/rolling, never shuffled). WMAPE picks the winner per dataset.
- **Prior to state up front:** demand is zero-heavy, so squared error is dragged around by the zeros — that's why Tweedie and Poisson are worth testing and vanilla regression likely isn't. Then let the metric decide.
- Add the **quantile models (0.1/0.5/0.9)** here per **B4** — framed as requirement-driven (intervals), *not* part of the bake-off.
- **Leakage discipline (explicit, standard practice):** time-series feature engineering is leakage-prone. Treat any suspiciously strong validation score as a signal to investigate, not accept. Build an explicit leakage check confirming lag/rolling features are computed within the split boundary, never across the full series. If it surfaces a problem: diagnose, fix, rerun. If it comes back clean, that's a valid outcome too. **Do not introduce errors on purpose** — the point is the check runs as standard and the reasoning is visible either way.
- **Deliverable:** per-dataset WMAPE table (baseline vs Tweedie vs Poisson), chosen winner + why, quantile-model results, leakage-check output. For review.

### Phase 4 — Agent wrapper (Agno)
**Goal:** Wrap the *working* core in the agent.
- Only after EDA and training are solid. Planner → Executor → Critic → Report (§2).
- The **critic must actually gate acceptance** (WMAPE vs threshold + sanity checks). A failing model is flagged, not set.
- Branch selection uses the deterministic functions from §3.
- **Deliverable:** end-to-end agent run on the 3 working datasets + tests showing the critic gating (including a deliberately-failing case that gets flagged). For review.

### Phase 5 — Final test
**Goal:** Prove it on unseen data.
- Run the finished agent end-to-end on the **held-out sub-dataset** from Phase 1.
- **Deliverable:** the agent's structured report + WMAPE on held-out data, for final review.

---

## 5. Repo layout (proposed, filled in as phases land)

```
demand_forecasting_agent/
├── data/m5-forecasting-accuracy/     # raw M5 (present)
├── docs/plans/
│   ├── 00_master_plan.md             # this file
│   ├── 01_phase1_eda_plan.md         # created at Phase 1 start
│   └── ...                           # one sub-plan per phase
├── src/                              # ML core (Phases 1–3), plain defensible code
├── tests/                            # tests per increment
├── artifacts/                        # EDA artifact, result tables
└── agent/                            # Agno wrapper (Phase 4)
```

Structure is provisional and firmed up per phase; not scaffolded ahead of need.

---

## 6. Definition of done (per phase)

A phase is done when: its sub-plan was approved; code runs end-to-end on a clean invocation; tests pass and were shown passing; results are presented across the relevant datasets; and the results are reviewed and approved before proceeding to the next phase.

---

## 7. Open decisions to resolve within each sub-plan

- **Branch thresholds — data-relative vs absolute, kept distinct:**
  - **Data-relative (computed per-dataset, at processing time, on that dataset's non-held-out pool):** the B1 "high zero-share" cut (an **upper percentile** so "high" means high, computed on the **non-sparse subset** — the population that actually reaches the B1-vs-Standard decision, since B2 catches sparse series first; calibrated-on == applied-on), the B2 too-sparse floor (an upper percentile of the dataset's own ADI/demand-interval distribution — not a fixed number), the B3 seasonality cut (from the dataset's own day-of-week variance-explained distribution), and the critic's plausible-magnitude bound (from the dataset's own demand tail). These are **recomputed for whatever dataset comes in**, never a single global fit, and never calibrated across the held-out split.
  - **Absolute, with citation:** the Syntetos–Boylan classification cutoffs, **ADI = 1.32, CV² = 0.49** (Syntetos & Boylan, 2005) — published constants, not tuned by us. Kept as-is and cited. We do not make these relative for the sake of consistency.
  - Rule: *literature-standard constants stay absolute and cited; anything we would otherwise fit to a dataset becomes data-relative, computed per-dataset.*
- WMAPE weighting scheme (M5 uses a specific hierarchical weighting; decide whether to adopt M5's WRMSSE-style weights or a simpler per-series WMAPE, and state the prior).
- Holdout window sizes and number of rolling folds (Phase 3 sub-plan).
- ~~STL period(s) to test~~ — resolved in Phase 1: seasonality is day-of-week variance-explained (η²), weekly cadence; STL was evaluated and dropped as config-fragile.

---

*Next step: on approval of this master plan, produce the **Phase 1 sub-plan** (EDA & dataset selection) for review before any code is written.*
