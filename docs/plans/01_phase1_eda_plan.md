# Phase 1 Sub-Plan — EDA & Dataset Selection

**Status:** Draft for review. No code is written until this is approved. Parent: [00_master_plan.md](00_master_plan.md).

**Goal:** Characterize M5 by computing the three branch signals per series, then select 3 working sub-datasets at the pattern extremes + 1 held-out sub-dataset for Phase 5. Deliver a review artifact before anything is locked.

**Discipline reminder:** every output below maps to a specific downstream decision (a branch threshold in §B1–B4 of the master plan, or a dataset pick). If a statistic or plot doesn't, it's cut.

---

## 1. Data scope and loading

- **Source file:** `sales_train_evaluation.csv`, but EDA uses **only `d_1`…`d_1913`**. The 28 columns `d_1914`…`d_1941` are the validation-horizon labels — treated as sacred future and excluded from all Phase 1 computation.
  - **Prior:** letting any signal be computed over days we will later forecast is a peeking risk. Excluding the horizon now keeps Phase 5's final test honest.
- **Reshape:** wide → long (`melt`) to `(id, d, units)`, join `calendar.csv` on `d` for `date, wday, month, year, event_*, snap_*`, and `sell_prices.csv` on `(store_id, item_id, wm_yr_wk)`.
  - Absence of a price row ≈ item not offered that week. **We do not impute prices in Phase 1** — price coverage is itself an EDA output (proxy for item availability / introduction date).
- **Active window:** for each series define `[first_nonzero_day, 1913]`. Leading zeros before the first sale are **pre-introduction**, not intermittent demand.
  - **Prior:** M5 items enter the assortment at different times; counting pre-launch zeros as intermittency would inflate zero-share and misclassify new-but-dense items as sparse. Signals are computed over the active window, and `intro_day`/active-length is recorded per series.

**Increment 1 deliverable:** a loader/reshape utility + tests (correct shapes; row count = series × days over the window; a hand-checked spot series matches the raw wide row; calendar/price joins produce no unexpected row multiplication).

---

## 2. The three signals — exact definitions

Computed per series over its active window. Each has a stated prior and a named downstream consumer.

### 2.1 Intermittency — zero-share + ADI + CV²
- **zero_share** = (# zero-demand days) / (active-window length).
- **ADI** (Average Demand Interval) = mean gap in days between consecutive non-zero demands.
- **CV²** = squared coefficient of variation of the *non-zero* demand sizes.
- **Syntetos–Boylan classification** using standard cutoffs **ADI = 1.32**, **CV² = 0.49** → {smooth, intermittent, erratic, lumpy}.
- **Prior:** zero_share alone conflates "steady low volume" with "bursty." ADI/CV² separate *how often* demand occurs from *how variable its size is* — the distinction between a Tweedie-friendly series (B1) and a too-sparse-to-model one (B2).
- **Consumes into:** B1 (Tweedie) vs B2 (sparse baseline) threshold calibration.

### 2.2 Seasonality — day-of-week variance-explained (η²)
- **Method:** η² = `SS_between / SS_total` over the 7 weekday groups — the fraction of a series' demand variance explained by which day of the week it is. Computed on the active window.
- **Why not STL strength (revised after testing):** STL weekly-strength swings ~3× on its `robust` flag (spikes either smear into the seasonal component or dump into the residual), so it is not a defensible per-series gate on spiky retail data. η² is **deterministic, config-free, fast, and interpretable**, and it is honestly spike-aware — a spiky series reads as *less* weekly-driven because the spikes are variance not explained by weekday. STL was evaluated and **removed** (dead code); the B3 signal is η². Its floor is derived from the cadence — `period × min_cycles` (weekly ⇒ 14) and all weekday levels present — not a fixed number, so it generalizes and correctly returns undefined on windows too short to hold a stable weekly pattern.
  - **Prior (unchanged):** the dominant, reliably estimable cycle in daily retail is **weekly**; annual/holiday effects are carried by explicit calendar/event/SNAP features (Phase 3), not a period-365 decomposition.
- **Short-window caveat:** η² needs the window long enough for each weekday to appear; below `min_len` it is recorded **undefined** rather than misleading — itself a signal toward B2/B3′.
- **Consumes into:** B3 vs B3′ (add seasonal features or skip them).

### 2.3 Outlier / spike count
- **Method:** robust, on non-zero days. Flag day as spike if `units > median_nonzero + 3 × 1.4826 × MAD_nonzero` (MAD-based, ~3σ-robust). Report **spike_count** and **spike_ratio** (spike days / active days) and **max/median ratio** (tail heaviness).
- **Prior:** promotions, holidays, and SNAP windows cause legitimate demand spikes; these are real, not errors. Counting them tells us how heavy-tailed each series is, which informs whether later models need a robust objective and/or winsorization — and confirms the zero-heavy + spiky shape that motivates Tweedie over L2.
- **Consumes into:** model-objective sanity (Phase 3) and the critic's "plausible magnitude range" check (Phase 4).

**Increment 2 deliverable:** a `signals` module with one pure function per statistic + **unit tests on synthetic series with known answers** (e.g., a series with a known zero pattern → known zero_share/ADI; a clean weekly signal → high day-of-week η²; a flat series + one injected spike → spike_count = 1).

---

## 3. Compute strategy & runtime

- Scale: 30,490 series × 1,913 days ≈ 58M observations.
- zero_share / ADI / CV² / spike: fully vectorizable (groupby over the long frame) — cheap.
- All signals — including η² — are cheap and fully vectorizable per series; the whole table over 30,490 series builds in ~15s. *(This replaced an STL-based seasonality signal that alone cost ~185s; the swap made the build ~13× faster and removed the robust-flag fragility.)*
- Persist a **per-series signal table** (parquet): `id, store_id, dept_id, cat_id, state_id, intro_day, active_len, mean_nonzero, zero_share, adi, cv2, sb_class, dow_season, dow_defined, spike_count, spike_ratio, price_coverage`.

**Increment 3 deliverable:** the signal table for all 30,490 series + sanity tests (no NaNs where a value is required; ranges in bounds; `sb_class` distribution is non-degenerate).

---

## 4. Sub-dataset selection

### 4.1 Grouping level
- A "sub-dataset" = a coherent group of series sharing a pattern, large enough to train/evaluate but small enough to iterate fast. **Unit: `(store_id, dept_id)` cells** (e.g., `CA_1 × FOODS_3` ≈ hundreds of items). *(Approved.)*
  - **Prior:** grouping by store×dept keeps series economically related (same store, same product family), gives a natural train/eval population of a few hundred to ~800 series, and matches how a real planner would batch forecasts. Category level is too coarse (mixes patterns); single-item is too thin to evaluate a model class.
- **Size cap (approved):** sample **N = 250 series per selected group** (deterministic seed, stratified to preserve the group's signal profile) for fast iteration. The full-cell run is deferred until the pipeline is solid — the sampled set must reproduce the group's median zero_share / DoW η² within a stated tolerance, else N is raised.
  - **Prior:** the goal in Phases 1–3 is fast, honest iteration on representative data; a stratified 250-series sample preserves the pattern while keeping training seconds-not-minutes. The frozen full cell remains available for a final confirmation run.
- I'll compute each candidate group's **profile** = distribution summary of its members' signals (median zero_share, %{sb_class}, median DoW η², median spike_ratio, total volume).

### 4.2 The four picks — mapped 1:1 to the branches
Selected at the extremes so each stresses a different branch:

| Pick | Target profile | Branch it stresses | Why this extreme |
|---|---|---|---|
| **A — dense / fast-moving** | low zero_share, high volume, mostly `smooth`/`erratic` | **B3** (seasonal features) + fairest test of the L2 baseline | Where a point model and seasonal structure should shine; the baseline is most competitive here, so beating it is a real test. Likely a `FOODS_3` cell. |
| **B — intermittent / zero-heavy** | high zero_share but non-zero mass still modelable, mostly `intermittent`/`lumpy` | **B1** (Tweedie) | The core zero-inflated case Tweedie exists for. Likely a `HOBBIES` cell. |
| **C — slow / sparse** | extreme zero_share, very low non-zero counts, high ADI | **B2** (baseline fallback) | Where the heavy model *shouldn't* win and knowing not to use it is the point. Likely low-velocity `HOUSEHOLD`/`HOBBIES` items. |
| **D — held-out (Phase 5)** | mixed/representative, non-overlapping with A/B/C | full agent test | A fair general test that exercises branch *selection*, not one hand-picked pattern. Locked and untouched until Phase 5. |

- Final group identities are chosen **from the observed profiles**, not asserted now — the table above states the target and the likely M5 region, and the artifact will show the actual ranked candidates with the picks highlighted.
- **Held-out integrity (two-stage, closes the threshold-leakage hole):**
  - **Stage 1 — selection.** Per-series signals are computed over *all* 30,490 series and used **only to partition and pick the four groups**, including confirming D is mixed/representative. Measuring a group to decide it should be held out is not leakage — you cannot certify "representative" without looking, and nothing is trained here.
  - **Stage 2 — calibration.** Once D's `id` list is frozen, **every data-relative threshold that feeds the agent's decision function is (re)computed on the non-D pool only** (A/B/C + all other non-D series). D informs *no* decision boundary. These cuts are **fit to this dataset's own distribution** (see the data-relative vs absolute split below), not hardcoded — a different M5-format dataset re-fits them.
  - **Data-relative vs absolute (master-plan §7):** the B1 zero-share cut (**P60 of the non-sparse subset** — "high" means an upper percentile, computed on the population that actually faces the B1-vs-Standard decision, not the full pool whose sparse tail would inflate it), the B2 sparse floor (P90 of the dataset's ADI — sells far less often than typical), the B3 seasonality cut (P90 of the dataset's η²), and the critic magnitude bound (P99 of the demand tail) are **fit per-dataset**. The **Syntetos–Boylan cutoffs (ADI 1.32, CV² 0.49; S&B 2005)** are published constants, used as-is and cited — never tuned. A modelable series below the B1 cut takes the **Standard** objective branch (L2/Poisson per the Phase 3 bake-off) — the documented complement of B1.
  - D is then excluded from Phases 2–4 as well.
  - **Prior:** the project's whole value is having no leakage holes. Calibrating thresholds over a pool that includes D would let the held-out data softly inform the agent's decision boundaries — defensible but a real seam. Recomputing thresholds ex-D lets us state the stronger claim: *the held-out set informed nothing, not even the thresholds.*

**Increment 4 deliverable:** candidate-group profile table, ranked; the 4 selected groups with their profiles and a one-line justification each tied to the branch.

---

## 5. Review artifact

An HTML artifact (built via the dataviz skill) for review, containing **only decision-driving visuals**:
1. **Zero-share distribution** → shows where B1/B2 thresholds should fall.
2. **ADI vs CV² scatter** with Syntetos–Boylan quadrant lines → the intermittency landscape.
3. **Day-of-week η² distribution** (defined series) → B3/B3′ threshold. (Weak/clustered near zero — the seasonal branch rarely fires.)
4. **Spike-ratio distribution** → tail-heaviness / critic magnitude bounds.
5. **Candidate-group profile table** with the **4 picks highlighted** and their branch mapping.

**Leakage note for the visuals:** the threshold-calibration distributions (1–4) are plotted over the **non-D pool** — the numbers we read thresholds off never include held-out data. The selection view (5, and any plot marking where the four groups sit) shows all groups *including* D, since justifying D as representative is Stage-1 selection, not calibration.

Each visual captioned with the exact decision it feeds. No decorative charts.

**Increment 6 deliverable:** the artifact, presented for review **before** the picks are locked.

---

## 6. Increment order (each ends on a working, tested run)

1. Loader + reshape (+ tests) → prints shapes/spot-check.
2. Signal functions (+ unit tests on synthetic series) → all green.
3. Run signals over all series → persist signal-table parquet (+ sanity tests). *(Stage 1 — selection inputs.)*
4. Group profiling + selection logic → ranked table + 4 picks (+ **frozen held-out id list**).
5. **Recompute distributions + calibrate B1–B4 thresholds on the non-D pool** → record threshold values. *(Stage 2 — the numbers baked into the agent, D excluded.)*
6. Review artifact → hand off for review.

Nothing proceeds to Phase 2 until the picks are reviewed and approved.

---

## 7. Decisions I'm making vs deferring

**Making now (with priors above):** exclude horizon days from EDA; active-window signal computation; ADI/CV² + Syntetos–Boylan cutoffs (1.32 / 0.49, cited, absolute); **day-of-week η² for seasonality** (not STL); MAD-based spike rule; store×dept grouping; **sample N = 250 series/group (stratified)**; **mixed/representative held-out D**; four-pick branch mapping; **data-relative thresholds fit per-dataset**. *(Grouping, held-out type, and size cap approved.)*

**Deferring (flagged, not decided):**
- Exact numeric thresholds for B1–B4 — **data-relative, fit per-dataset** from the distributions **over the non-D pool** (see §4.2 Stage 2); S–B cutoffs stay absolute-and-cited. Recorded at the end of Phase 1, not guessed now.
- WMAPE weighting scheme (M5 WRMSSE-style vs simple per-series) — carried to the Phase 2 sub-plan.
- ~~STL full-population vs above-floor-only~~ — resolved: STL replaced by day-of-week η² (deterministic, cheap, config-free), computed for all series.

---

## 8. Approved decisions

1. **Grouping unit** → `(store_id, dept_id)` cells.
2. **Sub-dataset size** → sample **N = 250 series/group**, stratified to preserve the group's signal profile; full-cell run deferred to a later confirmation.
3. **Held-out D** → **mixed/representative** group, so Phase 5 tests the agent's branch *selection*, not one hand-picked pattern.
