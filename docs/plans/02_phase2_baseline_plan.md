# Phase 2 Sub-Plan — Baseline (the number to beat)

**Status:** Draft for review. No code is written until this is approved. Parent: [00_master_plan.md](00_master_plan.md). Predecessor: [01_phase1_eda_plan.md](01_phase1_eda_plan.md) (complete, locked).

**Goal:** Establish the **honest floor** — a plain **L2 (squared-error) linear regression**, trained as well as it can honestly be trained, on datasets **A** (`CA_3 × FOODS_3`), **B** (`CA_2 × HOUSEHOLD_2`), **C** (`CA_4 × HOBBIES_2`). This WMAPE is the bar every Phase 3 model must beat. It has to be a real floor, not a strawman: tuned properly so the bake-off is a fair fight. **D** (`TX_1 × FOODS_2`) stays frozen — untouched here.

**Discipline reminder (unchanged from Phase 1):** state the prior at every fork *before* the choice; every artifact feeds a decision; small tested increments, each ending on a working run; a suspiciously good score is a signal to investigate, not accept.

---

## 1. The metric — WMAPE (decided, not open)

### 1.1 Definition (exact)
Over an evaluation window, pooled across the series in a dataset:

```
WMAPE = Σ_i Σ_t |actual_{i,t} − forecast_{i,t}|  /  Σ_i Σ_t actual_{i,t}
```

**Sum-then-divide** — one numerator and one denominator accumulated over every (series, day) in the window, *not* an average of per-series ratios.

- **Prior — why sum-then-divide, not a mean of ratios:** a per-series `|a−f|/a` blows up (÷0 or ÷tiny) on the zero-heavy intermittent series that dominate here — exactly the tail we care about. Pooling the absolute errors over a single shared denominator is finite and stable whenever the dataset sells anything at all.
- **Weighting is the natural volume weighting** that falls out of the formula: a series contributes to the denominator in proportion to its total units, so high-selling series count more. We state this explicitly and add *no* extra weight scheme. This is the resolution of the master-plan §7 open item (adopt simple volume-weighted WMAPE, **not** M5's hierarchical WRMSSE weights).

### 1.2 Why WMAPE and not M5's WRMSSE
- WRMSSE is a **scale-free hierarchical competition metric** built to rank submissions across the *full* M5 aggregation hierarchy (state → store → dept → item …). Our goal is different: an **interpretable per-dataset demand-forecast error** that reads as a business number and generalizes to any M5-format retailer.
- WMAPE is **absolute-error based** → robust on the zero-heavy series where squared error is dragged around by a handful of spikes.
- WMAPE is **volume-weighted** → high-selling series dominate, which is the right business emphasis.
- WMAPE is **coherent with the Tweedie objective** we chose in Phase 1 for the same zero-inflation reason. WRMSSE's squared-error core would partly **re-import the exact squared-error sensitivity Tweedie exists to avoid**, muddying the Phase 3 comparison.

### 1.3 Report per Syntetos–Boylan class, not just the aggregate
Alongside the overall dataset WMAPE, report WMAPE **broken down by SB class** (smooth / intermittent / erratic / lumpy — headlining the intermittent + lumpy tail the user named).

- **Prior:** aggregate WMAPE is volume-weighted, so a strong number on a few dense series can **hide poor performance on the intermittent tail**. The per-class split pre-empts that — it's the honesty check on the headline. Each class uses its own pooled denominator (`Σ actual` within the class); a class whose eval-window actuals sum to zero reports **N/A** with its series count rather than a divide-by-zero.

### 1.4 Volume-tercile breakdown — dataset A only
On **dataset A only** (`CA_3 × FOODS_3`, the dense/fast cell), also report WMAPE split into **volume terciles** — series ranked by total training-window units, cut into low / mid / high thirds, each with its own pooled denominator.

- **Prior:** A is where the L2 baseline is *most* competitive, so it's where "the aggregate is carried by the biggest sellers" is the real risk. The tercile split shows whether the floor holds across the volume range or just on the head. **One breakdown, one dataset** — B and C are already dominated by the intermittent tail, so the SB-class split (§1.3) is the more informative cut there and the tercile view would be redundant.

**Increment 1 deliverable:** a `metrics` module — pure `wmape(actual, forecast)` + `wmape_by_group(...)` (used for both the SB-class and the A-only volume-tercile splits) — with unit tests on synthetic arrays with hand-computed answers, including the zero-denominator guard and a known grouped split.

---

## 2. Validation design — time-series holdout, never shuffled

### 2.1 Rolling-origin (expanding-window) CV
- **Split on the day axis only.** Train = days strictly before a *forecast origin*; validate = the 28 days after it. **Never shuffle**, never random k-fold.
- **Horizon = 28 days**, to match M5. **Prior:** M5's evaluation horizon is 28 days; matching it makes this floor directly comparable to the M5 context and to Phase 5's sealed 28-day held-out test. No reason found to change it, so we don't.
- **Folds:** expanding-window rolling origin over the tail of the training range. The **count is data-driven, not hardcoded** — it's the number of horizon blocks that fit after reserving a minimum training window, capped so long histories don't over-fold:

  ```
  n_folds = min(MAX_FOLDS, (train_end − MIN_TRAIN_DAYS) // horizon)
  ```

  - **Min-history guard (`MIN_TRAIN_DAYS = 365`):** the earliest fold must still train on ≥ one full year, so every fit has annual calendar coverage and ample lag/rolling history (56-day max lookback). Because the count is derived from `train_end − MIN_TRAIN_DAYS`, the earliest origin is **≥ `MIN_TRAIN_DAYS` by construction**. Too little history to place even one guarded fold → **raise**, rather than silently training on a thin window.
  - **Cap (`MAX_FOLDS = 5`):** more folds tighten the variance estimate but cost compute and eat history; 5 is the point past which the marginal gain isn't worth shrinking the earliest train window.
  - **On M5** (`train_end = 1913`) the formula yields exactly **5 folds** at origins `d_1773/1801/1829/1857/1885` → validation blocks `d_1774–1801` … `d_1886–1913`, earliest train range `d_1..d_1772` (~4.85 yr) — the reviewed choice. A shorter M5-format dataset gets proportionally fewer folds automatically; this is the format-general property (master plan §honesty boundary) applied to validation, not just thresholds.
  - **Prior:** hardcoding a fold count silently misbehaves on any dataset whose history isn't M5's length (too-thin train windows, or a fold reaching past the data). Deriving the count from length with an explicit guard makes the failure mode loud and the behavior defensible on any history.
  - The **sealed horizon `d_1914–1941` is never touched** — Phase 5 only; all folds live inside `d_1..d_1913`.
- **Reported number:** WMAPE **pooled across all folds** (one accumulated numerator/denominator over every fold), with the per-fold spread shown so we can see stability. Pooling keeps the volume weighting coherent across folds.
- **Evaluation set = the training predicate.** A validation day is scored only if it is *evaluable* on the same rule training uses: the series is active and ≥ horizon days past introduction. This excludes **pre-introduction zeros** (days the item didn't exist — a positive forecast there is a spurious penalty) and **feature-warmup days** whose lag/rolling inputs are 0-filled and unreliable. L2, the B2 floor, and the naive comparator therefore share **one honest denominator**. *(No-op on the A/B/C cells — all series introduced well before the earliest origin — but a real safeguard for late-introduced series and any shorter/newer M5-format dataset.)*

### 2.2 Direct multi-horizon forecast (the leakage-critical fork)
Forecasting a 28-day block forces a choice on what lag information is *legitimately available at the origin*:

- **Recursive** (predict day-by-day, feed predictions back) — lets you use short lags (lag-1) but compounds error and is fiddly to keep honest.
- **Direct multi-horizon** (one model predicts all 28 days from features frozen at the origin) — the most recent usable actual is at the origin, so **no lag shorter than the horizon can leak** a within-horizon actual.

**Decision: direct multi-horizon.** **Prior:** it is the leakage-safe default and matches how the Phase 3 LightGBM models will be trained, so the bake-off compares objectives, not forecasting protocols. It also means the honest floor never "cheats" by peeking at actuals inside the horizon it's supposed to predict.

- **Consequence for lags (stated so it isn't rediscovered):** usable lags are **≥ 28** (`lag_28`, `lag_35`, …). `lag_7`/`lag_1` are *not* available for a 28-day-ahead direct forecast and are excluded. Rolling means are **trailing windows ending at the origin**, identical across the 28 target days. Calendar features belong to the *target* day (the calendar is deterministic known-future, not a leak). Sell price is the target week's price (M5 provides horizon-week prices; known at forecast time).

### 2.3 Leakage discipline
- **Every lag/rolling feature is computed within the fold's train boundary only** (≤ origin), never over the full series. This is the master-plan §6 rule, made mechanical here.
- A **leakage assertion** ships as a test: for each fold, assert no feature row for a validation target day draws on any day `> origin`. A suspiciously strong fold score is treated as a **prompt to inspect this**, not to celebrate.

**Increment 2 deliverable:** a `splits` module (rolling-origin fold generator) + tests: folds are contiguous, non-overlapping, horizon = 28, no fold reaches `d_1914+`, the origin/feature-boundary invariant holds, the **fold count is data-driven** (shorter history → fewer folds, capped by `MAX_FOLDS`), and **insufficient history raises** (min-history guard).

---

## 3. Features — the frozen set (constant into Phase 3)

The **exact same feature set** carries unchanged into the Phase 3 bake-off, so the only thing that changes there is the **objective**. Locking it here is what makes Phase 3 a test of objectives, not of feature engineering.

| Group | Features | Handling / prior |
|---|---|---|
| **Calendar** | `wday`, `month`, `event_type_1`, `event_type_2`, `snap_{state}` | One-hot for categoricals. **wday/month** are fixed structural vocabularies; **event types are extracted from the calendar, not hardcoded** (any M5-format dataset carries its own), and **both event slots are one-hot** so a concurrent second event's type isn't collapsed to a bare flag. SNAP taken **per row from the series' own state** column. Target-day calendar (known future). Every series gets these — the Phase 1 decision to drop B3 and let the model learn weekly structure. |
| **Price** | `sell_price` (target week) | Weekly, sparse in time. **Forward-fill within series (trailing only)** — a delisted/gap day inherits the last posted price. Leading days before a series' first price have no trailing price → **0, flagged by `price_missing`** (never a median, which would pull future prices into a past row; those rows are non-trainable anyway). |
| **Lags** | `lag_28`, `lag_35`, `lag_42` | ≥ horizon only (§2.2). Missing early-history lags → 0 (pre-introduction / no prior demand). |
| **Rolling means** | trailing mean of units over the 28 / 56 days **ending at the origin** | Carries the series' recent level; identical across the 28 target days. Same info LightGBM will get in Phase 3. |
| **Series level** | trailing-mean feature above *is* the level carrier | **Prior:** pooled model (§4) has no per-series intercept, so the trailing mean gives L2 an honest handle on each series' scale — the same handle the tree models get. |

- **Prior on the feature altitude:** this is the master-plan baseline feature list (calendar, price, lags, rolling means) and nothing richer. Keeping it deliberately plain is the point — a fancy baseline feature set would make Phase 3's objective comparison dishonest.

**Increment 3 deliverable:** a `features` builder (origin-aware) + tests: correct shapes, no NaN in required columns, the **leakage test** (features for a target day use only ≤ origin days), **event types extracted from data** (a novel type gets a column; both slots one-hot), **SNAP resolved per-row by state** (multi-state frame gets each series its own signal; missing column raises), and **event one-hot schema stable across folds** (a type appearing only in the val window still carries a zero column in train — vocab derived once from the full frame).

---

## 4. The baseline model — L2, tuned honestly, with a sparse-series floor

### 4.1 What "L2, trained as well as it honestly can" means
- **One pooled linear model per dataset**, fit over all `(series, day)` training rows in that dataset (not per-series). **Prior:** Phase 3's LightGBM will also be pooled-per-dataset; the floor must share that training regime. Fitting on **raw units** means high-volume series dominate the squared-error loss — **coherent with WMAPE's volume weighting** and with how the eventual model is judged.
- **Ridge (L2-penalised) linear regression**, α selected on the rolling-origin CV. **Prior:** pure OLS is Ridge at α→0; with one-hot calendar + lags + price the design is collinear and unregularised OLS overfits noise, which would give a *fragile, artificially low-variance* floor — a strawman in the other direction. Tuning α (grid, incl. α→0) on the time-series CV yields the honest best L2 fit; if α→0 wins, so be it. Features standardised so the penalty is fair.
- **Target space:** evaluate raw-units vs `log1p(units)` (back-transformed for scoring) and pick per dataset by validation WMAPE; **report which was chosen.** **Prior:** `log1p` often helps a linear model on skewed counts, but it's a tuning knob, not a new model class — choosing by WMAPE keeps it honest and keeps the comparison to Phase 3 clean.
- **Non-negativity clamp:** forecasts **clamped to ≥ 0**. **Prior:** L2 will emit negative demand; negatives are nonsensical and would be caught by the Phase 4 critic's `min_forecast = 0` sanity bound — so we clamp here and stay consistent with the architecture.

### 4.2 Sparse series — an honest floor, not a pretend fit
Some series (especially in **C**, `CA_4 × HOBBIES_2`) are too sparse for L2 to model meaningfully.

- **Routing:** a series flagged **B2** by the Phase 1 sparse cut (**ADI ≥ 8.77**, data-relative, recomputed per dataset) is **not** given to L2. It gets a **naive mean floor**: a constant forecast = mean daily units over the series' active history up to the origin.
- **Prior:** on a near-all-zero series L2 just predicts ≈ 0, and reporting that as "the L2 baseline" would both understate the honest error and make Phase 3 look artificially good against a degenerate comparator. The mean floor is the honest ML-free estimate for series with too little signal — and it mirrors the architecture's own **B2 → simple baseline** branch, so the floor is internally consistent.
- **Reported explicitly:** how many series in each dataset route to L2 vs the mean floor, so the composition of the floor is visible.
- **C is the sparse-fallback showcase.** `CA_4 × HOBBIES_2` is expected to route a large share to the mean floor, so its results are **framed as the B2-fallback demonstration**, not primarily an L2 result. Where C's L2 subset is small, that **L2 WMAPE is tagged "indicative (low n)"** — reported for completeness but not to be read as a stable floor. **Prior:** a WMAPE over a handful of L2 series is high-variance; labelling it prevents over-reading a number that C isn't really there to produce. C's job is to show the fallback firing correctly.

### 4.3 A naive comparator, on the modelable series only
For **all three** datasets, report a **naive-mean WMAPE** (constant per-series mean forecast) beside the L2 WMAPE — scored on the **modelable (non-B2) series only**, the set L2 is actually applied to.

- **Prior:** the floor only means something if L2 actually beats a constant. If L2 ≈ naive on a dataset, then the honest baseline for that dataset *is* the mean, and Phase 3 must beat that — reporting both prevents us from overselling L2 as the floor where it adds nothing. It must be scored on the modelable series only: a B2 series' baseline forecast *is* its mean, so including B2 in the comparator would be a built-in tie that dilutes the very question it answers ("do the modelable series benefit from L2 over a constant?"). Cheap, and it strengthens the honesty claim.

**Increment 4 deliverable:** the `baseline` model (Ridge + α/target tuning on CV, non-neg clamp, B2→mean routing, naive comparator) + tests: fits and predicts non-negative; B2 series are routed to the mean; a known tiny fixture reproduces a hand-checked forecast.

---

## 5. Sample vs full cell (state it, per dataset)

- Iterate and **report on the sampled sets**: **A** and **B** at 250 series, **C** at **149 = the full cell** (so C's number is already full-cell).
- **Prior:** Phase 1 verified the 250-stratified sample reproduces each cell's median zero-share / DoW η² within tolerance, so it's a faithful iteration surface. For **A and B the full-cell confirmation run (823 / full-B series) is deferred** — noted, not run now — and the reported headline is labelled **(sample)** vs **(full)** per dataset so nothing is ambiguous.

---

## 6. Results to present (for review)

A compact table + a short write-up, per dataset **A / B / C**:

1. **Overall WMAPE** (L2 baseline), labelled sample/full. C's L2 number **tagged "indicative (low n)"** where its L2 subset is small (§4.2).
2. **WMAPE by SB class** (smooth / intermittent / erratic / lumpy; N/A + count where a class denominator is 0).
3. **WMAPE by volume tercile — dataset A only** (§1.4).
4. **Naive-mean WMAPE** beside it (§4.3).
5. **Per-fold WMAPE spread** (the rolling origins; 5 on M5) — stability.
6. **Routing composition** — # series to L2 vs mean floor (§4.2); **C highlighted as the sparse-fallback showcase**.
7. **Chosen config per dataset** — α, target space (raw / log1p).
8. **Leakage-check output** — the assertion result, shown either way.

- **Prior:** each row maps to a Phase 3 decision or an honesty check; no decorative output (master-plan §0.3). A small HTML artifact is optional — a clean table in the results doc is sufficient for this phase and preferred unless a visual earns its place.

---

## 7. Increment order (each ends on a working, tested run)

1. `metrics` — WMAPE + per-class + tests (synthetic, known answers, ÷0 guard).
2. `splits` — rolling-origin folds + leakage-boundary tests.
3. `features` — frozen origin-aware feature builder + leakage/shape tests.
4. `baseline` — Ridge + tuning + non-neg clamp + B2→mean routing + comparator + tests.
5. **Run across A/B/C** → the §6 results table (sample for A/B, full for C).
6. Results write-up handed off for review.

Nothing proceeds to Phase 3 until the baseline numbers are reviewed and approved.

---

## 8. Decisions I'm making vs deferring

**Making now (priors above):**
- Metric = **volume-weighted WMAPE**, sum-then-divide, pooled; **+ per-SB-class breakdown** (all datasets) **+ a volume-tercile breakdown on A only**. WRMSSE rejected (§1.2). *(Resolves master-plan §7 WMAPE-weighting item.)*
- **Rolling-origin expanding-window CV**, **data-driven fold count** (`min(MAX_FOLDS=5, (train_end−MIN_TRAIN_DAYS=365)//horizon)`; M5 → 5), **min-history guard** (raise if <1 guarded fold), **horizon 28**, pooled scoring; sealed `d_1914+` untouched.
- **Direct multi-horizon** forecast → usable lags **≥ 28**; trailing rolling means at origin; calendar = target-day known-future.
- **Frozen feature set** (calendar, price, lags≥28, rolling means) carried unchanged into Phase 3.
- **Pooled Ridge** per dataset, α + target-space tuned on CV, **non-neg clamp**; **B2 (ADI ≥ 8.77) → naive mean floor** (unchanged routing); **naive-mean comparator** reported everywhere. **C = sparse-fallback showcase**, its low-n L2 number tagged **indicative**.
- Iterate/report on **sample** (A, B = 250) and **full** (C = 149); **A/B full-cell confirmation deferred**.

**Deferring (flagged, not decided):**
- A/B **full-cell confirmation run** — deferred to end-of-phase or rolled into Phase 3.
- Whether to add a **seasonal-naive** comparator (same-day-last-year) — likely noise on this intermittent data; only added if a dataset's naive-mean looks suspiciously strong.
- **α grid and target-transform** final values — reported as outcomes after the CV run, not guessed now.
