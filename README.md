# Demand-Forecasting Agent

A demand-forecasting system for retail sales data in the **M5 canonical format** — a
sales table (series × time), a calendar, and weekly sell prices. The ML core is built
and validated as plain, defensible code first; an agent layer (planner → executor →
critic → report) wraps it last.

**Scope.** This forecasts demand. The M5 inputs carry unit sales, a calendar, and sell
prices — **no cost or margin data** — so profit is out of scope. Forecasts are shaped so
a downstream price/inventory optimizer *could* consume them later; that layer is not
built here.

**Design claim vs evidence.** The system is **format-general and threshold-adaptive** — it
ingests any M5-format retail dataset and calibrates its decision thresholds to that
dataset's own distribution. The **evidence is M5 only**: it has not been run on another
retailer's data. Generality here is a design property argued for, not a set of features
built — there are deliberately no format converters or column auto-detection.

> **Status:** Phase 1 (EDA & dataset selection) and Phase 2 (L2 baseline) complete. ML core in progress.
> **94 tests passing.** · Validated on M5 (30,490 store-item series).

---

## 📊 Phase 1 review

An interactive, self-contained review of the three branch signals and the four selected
sub-datasets — distribution charts, the intermittency map, all 70 candidate groups with
the picks highlighted, and the calibrated thresholds.

**Open it locally — no account or internet needed:**

➡️ **[`artifacts/phase1_review_standalone.html`](artifacts/phase1_review_standalone.html)**
— download and **double-click to open in any browser**. It's a single self-contained file
(all CSS/JS/data inlined, no external requests) and follows your system light/dark theme.

> Hosted copy: [claude.ai/code/artifact/500b6d04…](https://claude.ai/code/artifact/500b6d04-b4fb-4556-99b2-732edbdbcf2c)
> (private to the author — the standalone file above is the shareable version).

**Prefer plain text?** The same results are written up in
[`docs/artifacts/phase1_review.md`](docs/artifacts/phase1_review.md) — readable and
diffable on GitHub, no browser needed.

There are two HTML files in `artifacts/`:
[`phase1_review_standalone.html`](artifacts/phase1_review_standalone.html) is the complete,
double-clickable document; `phase1_review.html` is the same content without the
`<!doctype>`/`<head>`/`<body>` wrapper (the form used to publish the hosted copy).

---

## Target architecture

The agent (built last, Phase 4) orchestrates a core that already stands on its own:

```
dataset ─▶ Validator ─▶ Planner ─▶ Executor ─▶ Critic ─▶ Report
           (M5-schema   (EDA +      (fits per   (gates       (structured
            strict,      signals)    branch)     acceptance)   summary)
            flags)
```

The **Executor** applies deterministic, documented branch rules — the agent *selects*
which branch applies, it does not invent modeling decisions:

| Branch | Condition | Action | Prior |
|---|---|---|---|
| **B1** | High intermittency — modelable *and* zero-share genuinely high | LightGBM **Tweedie** | Zero-inflated demand; Tweedie's compound Poisson-Gamma form fits, unlike squared error. |
| **Standard** | Modelable but below the B1 cut | LightGBM **L2 / Poisson** (per bake-off) | Lower-intermittency series don't need Tweedie. Explicit complement of B1. |
| **B2** | Too sparse (ADI in the upper tail) | **Simple baseline** (MA / Croston) | Too little signal for a heavy model; knowing when *not* to use it is deliberate. |
| **B4** | Use case needs inventory intervals | LightGBM **quantiles** (0.1/0.5/0.9) | Requirement-driven, not a bake-off winner — a range answers a different question than a point. |

**Seasonality is not a branch.** An earlier per-series seasonal gate (B3/B3′) was
**dropped after Phase 1**: day-of-week explains only ~1% of the median item's demand
variance (median η² ≈ 0.012) and the gate would have fired for only ~10% of series. Every
series instead gets the calendar features (weekday, month, events, SNAP) and the model
learns whatever weekly structure exists.

The **Critic** is a real gate: WMAPE vs threshold plus sanity checks (non-negative
forecasts, magnitudes within a plausible range). A model that fails is flagged, not set.

Full rationale: [docs/plans/00_master_plan.md](docs/plans/00_master_plan.md).

---

## What's implemented (Phase 1)

The ML core is plain Python under [`src/dfa/`](src/dfa/), one concern per module:

| Module | Role |
|---|---|
| [`config.py`](src/dfa/config.py) | Paths and day-window boundaries (train `d_1..d_1913`, sealed `d_1914..d_1941`). Single source of truth. |
| [`data_loader.py`](src/dfa/data_loader.py) | M5 wide→long reshape, calendar + weekly-price joins, active-window flag. Missing price = "not on offer" (kept as signal, not imputed). |
| [`signals.py`](src/dfa/signals.py) | Per-series branch signals: intermittency (zero-share, ADI, CV², Syntetos–Boylan class), day-of-week seasonality (η²), MAD-based spikes. Pure functions. |
| [`build_signal_table.py`](src/dfa/build_signal_table.py) | Runs signals over all 30,490 series → `signal_table.parquet` (~15 s). |
| [`select_datasets.py`](src/dfa/select_datasets.py) | Profiles `(store, dept)` groups; selects four sub-datasets with explicit, symmetric exclusions and graceful fallbacks. |
| [`calibrate_thresholds.py`](src/dfa/calibrate_thresholds.py) | Fits branch thresholds **per-dataset** on the non-held-out pool; `classify_branch` is the deterministic executor rule reused by later phases. |

### Outputs (in [`artifacts/`](artifacts/))

- **`signal_table.parquet`** — one row per series with all signals + `price_coverage`.
- **`datasets/selection.json`** — the four sub-datasets, with frozen series-id lists.
- **`thresholds.json`** — calibrated cuts + branch coverage over the non-D pool.
- **`phase1_review_standalone.html`** — the review artifact, double-clickable in any browser.

### Selected sub-datasets

Each pick sits at a pattern extreme and stresses one branch; **D is held out**, frozen and
untouched until Phase 5.

| Pick | Cell | Median zero-share | Stresses |
|---|---|---|---|
| **A** — dense / fast | `CA_3 × FOODS_3` | 0.35 | Standard objective + baseline |
| **B** — intermittent | `CA_2 × HOUSEHOLD_2` | 0.72 | B1 (Tweedie) |
| **C** — slow / sparse | `CA_4 × HOBBIES_2` | 0.93 | B2 (baseline fallback) |
| **D** — held-out (mixed) | `TX_1 × FOODS_2` | 0.64 | Phase 5 general test |

### Calibrated thresholds (M5 non-held-out pool)

| Threshold | Value | Basis |
|---|---|---|
| B2 sparse cut | `ADI ≥ 8.77` | P90 of full-pool ADI (data-relative) |
| B1 Tweedie cut | `zero-share ≥ 0.663` | **P60 of the non-sparse subset** — genuinely high, computed on the population it governs (data-relative) |
| Critic bounds | `≥ 0`, `≤ 24×` median non-zero | Non-negativity (absolute) + P99 tail (data-relative) |
| SB class cutoffs | `ADI 1.32`, `CV² 0.49` | **Absolute** — Syntetos & Boylan (2005), cited, never tuned |

Resulting branch coverage: **standard 54% · Tweedie 36% · sparse-baseline 10%**. (The
day-of-week η² cut was calibrated too, but the seasonal branch was dropped — see above —
so it no longer gates anything.)

---

## What's implemented (Phase 2 — the baseline)

An honest **L2 (squared-error) linear-regression floor** — the number every Phase 3 model
must beat. Tuned as well as it can honestly be, so the bake-off is a fair fight. New modules
under [`src/dfa/`](src/dfa/):

| Module | Role |
|---|---|
| [`metrics.py`](src/dfa/metrics.py) | **WMAPE** — volume-weighted, sum-then-divide (`Σ\|a−f\| / Σa`), robust on zero-heavy series; per-group and volume-tercile breakdowns. WRMSSE rejected (re-imports the squared-error sensitivity Tweedie avoids). |
| [`splits.py`](src/dfa/splits.py) | **Rolling-origin** (expanding-window) folds, horizon 28, never shuffled. Fold count is **data-driven** — `min(5, (train_end − 365) // horizon)` — with a min-history guard that raises rather than train on a thin window. |
| [`features.py`](src/dfa/features.py) | The **frozen feature set** (carried unchanged into Phase 3): calendar one-hots (fixed weekday/month; event types **extracted from the calendar** for both slots; SNAP resolved per-row by state), trailing-only price fill, lags 28/35/42, rolling means. Every dynamic feature is lagged **≥ horizon** → direct multi-horizon, leakage-safe by construction. |
| [`baseline.py`](src/dfa/baseline.py) | Pooled **Ridge** per dataset (α + raw/log1p tuned on the CV), non-negativity clamp, **B2 → mean-of-history floor** (sparse series aren't given to L2), and a naive comparator scored on the modelable series only. |
| [`run_baseline.py`](src/dfa/run_baseline.py) | Runs the baseline across A/B/C; the true last usable day is derived from the data (`last_sales_day() − horizon`). Emits WMAPE overall + by SB class + (A only) by volume tercile, per-fold spread, routing, and an empirical leakage check. |

### Baseline results (5-fold rolling-origin CV, WMAPE)

| Dataset | Cell | n | Sample/Full | **WMAPE (floor)** | L2 vs naive (modelable) | Config |
|---|---|---|---|---|---|---|
| **A** — dense | `CA_3 × FOODS_3` | 250 | sample | **0.621** | 0.621 vs 0.732 | Ridge α=1000, raw |
| **B** — intermittent | `CA_2 × HOUSEHOLD_2` | 250 | sample | **1.106** | 1.092 vs 1.207 | Ridge α=0.1, log1p |
| **C** — slow / sparse | `CA_4 × HOBBIES_2` | 149 | **full** | **1.631** | 1.401 vs 1.669 *(indicative, low n)* | Ridge α=10, log1p |

- **L2 beats a constant mean on every dataset's modelable series** — a real floor, not a
  strawman. WMAPE climbs A→B→C, tracking intermittency; > 1 on B/C is expected for
  zero-heavy L2 and is exactly the gap Tweedie should close in Phase 3.
- **C is the sparse-fallback showcase:** 118 of 149 series route to the mean floor; its
  31-series L2 number is tagged *indicative* (high-variance, not a stable floor).
- Per-class and (A-only) volume-tercile breakdowns confirm the aggregate is carried by the
  dense / high-volume series — the intermittent tail and low-volume third are 2–2.5× worse.
- **Leakage check passes** on all three: perturbing future units changes features only
  at/after `origin + horizon`, never before. Full write-up:
  [`docs/results/02_phase2_baseline_results.md`](docs/results/02_phase2_baseline_results.md).

---

## Key design decisions

Each is documented in the plans and enforced by a test.

- **Active-window signals.** Signals are computed from a series' first sale onward;
  pre-introduction zeros are not counted as intermittency (M5 items launch at different
  times).
- **Day-of-week η², not STL, for seasonality.** STL strength swings ~3× on its `robust`
  flag on spiky retail data. η² (variance explained by weekday) is deterministic,
  config-free, and spike-honest. *Finding:* weekly seasonality is weak at the item level
  (median η² ≈ 0.012) — which is exactly what **retired the per-series seasonal branch**;
  calendar features now go to every series instead.
- **Data-relative vs absolute thresholds.** Anything otherwise tuned to a dataset is a
  percentile of that dataset's own distribution; literature constants (Syntetos–Boylan)
  stay absolute and cited.
- **Calibrated-on == applied-on.** The B1 cut is fit on the *non-sparse* subset — the
  series that actually reach the Tweedie-vs-Standard decision — so the sparse tail can't
  inflate it. And "high" is an upper percentile, not the median.
- **Held-out integrity.** D informs no distribution and no threshold; the four
  sub-datasets are disjoint by construction (explicit, symmetric cell exclusions), so the
  Phase 5 test is genuinely unseen — thresholds included.
- **Graceful degradation, not silent forcing.** Selection relaxes constraints in a
  documented order and raises a located error rather than forcing an unsound pick.

---

## Repository layout

```
demand_forecasting_agent/
├── README.md
├── requirements.txt
├── pyproject.toml                  # pytest config (adds src/ to path)
├── data/m5-forecasting-accuracy/   # M5 CSVs (not tracked; see Setup)
├── docs/
│   ├── plans/
│   │   ├── 00_master_plan.md       # overarching plan, all phases
│   │   ├── 01_phase1_eda_plan.md   # Phase 1 sub-plan
│   │   └── 02_phase2_baseline_plan.md
│   └── results/
│       └── 02_phase2_baseline_results.md
├── src/dfa/                        # ML core (Phases 1–3)
├── tests/                          # one suite per module (94 tests)
└── artifacts/                      # signal table, selection, thresholds, review, baseline results
```

---

## Getting started

**Requirements:** Python 3.12, the M5 dataset.

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate     # or: conda create -n dfa python=3.12
pip install -r requirements.txt

# 2. data — place the Kaggle M5 files here:
#    data/m5-forecasting-accuracy/{calendar,sell_prices,sales_train_evaluation}.csv
#    (https://www.kaggle.com/competitions/m5-forecasting-accuracy)

# 3. run the Phase 1 pipeline (src/ must be importable)
export PYTHONPATH=src
python -m dfa.build_signal_table      # -> artifacts/signal_table.parquet   (~15 s)
python -m dfa.select_datasets         # -> artifacts/datasets/selection.json
python -m dfa.calibrate_thresholds    # -> artifacts/thresholds.json

# 4. run the Phase 2 baseline across A/B/C
python -m dfa.run_baseline            # -> artifacts/phase2_baseline_results.json  (~1 min)

# 5. tests
pytest                                # 94 passing
```

`pyproject.toml` puts `src/` on the path for pytest automatically; the `PYTHONPATH=src`
export is only needed for the `python -m dfa.*` module runs.

---

## Roadmap

Phase 1 is done and reviewed. Later phases follow the same discipline — a reviewed
sub-plan, small tested increments, results shown across all datasets before proceeding.

| Phase | Scope | Status |
|---|---|---|
| **1 — EDA & dataset selection** | Three branch signals over M5; four sub-datasets; calibrated thresholds. | ✅ Complete |
| **2 — Baseline** | Plain L2 (linear regression) baseline per dataset — the number every later model must beat. WMAPE on a rolling-origin holdout, frozen feature set, sparse-series mean fallback. | ✅ Complete |
| **3 — LightGBM bake-off** | Per dataset: L2 vs Tweedie vs Poisson on WMAPE, proper time-series holdout (expanding/rolling, never shuffled), with an explicit leakage check on lag/rolling features. Quantile models added per B4. | Planned |
| **4 — Agent wrapper (Agno)** | Wrap the working core: planner → executor → critic → report. The critic must actually gate acceptance. | Planned |
| **5 — Final test** | Run the finished agent end-to-end on the held-out sub-dataset D. | Planned |

---

## References

- **Master plan:** [docs/plans/00_master_plan.md](docs/plans/00_master_plan.md)
- **Phase 1 sub-plan:** [docs/plans/01_phase1_eda_plan.md](docs/plans/01_phase1_eda_plan.md)
- **Phase 2 sub-plan:** [docs/plans/02_phase2_baseline_plan.md](docs/plans/02_phase2_baseline_plan.md) · **results:** [docs/results/02_phase2_baseline_results.md](docs/results/02_phase2_baseline_results.md)
- **Dataset:** M5 Forecasting – Accuracy (Kaggle)
- **Intermittency classification:** Syntetos, A. A., & Boylan, J. E. (2005). *The accuracy
  of intermittent demand estimates.* International Journal of Forecasting, 21(2), 303–314.
