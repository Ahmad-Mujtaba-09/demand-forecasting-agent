# Phase 2 Results — Baseline (the number to beat)

**Status:** For review. Plan: [../plans/02_phase2_baseline_plan.md](../plans/02_phase2_baseline_plan.md). Artifact: `artifacts/phase2_baseline_results.json`. Reproduce: `PYTHONPATH=src python -m dfa.run_baseline`.

**What this is:** the honest L2 floor, and — where that floor is not good enough to be worth beating — the trivial forecast that replaces it as the bar. Metric is volume-weighted **WMAPE** (sum-then-divide), on a 5-fold rolling-origin holdout (horizon 28), features frozen for the bake-off, sparse (B2) series routed to a mean floor. All numbers are CV WMAPE pooled across the 5 folds.

---

## Headline

| Dataset | Cell | n | Sample/Full | WMAPE floor (all series) | L2 (modelable) | Naive (modelable) | Zero (all series) | **PHASE 3 BAR** | Config |
|---|---|---|---|---|---|---|---|---|---|
| **A** dense | `CA_3 × FOODS_3` | 250 | sample | 0.621 | 0.620 | 0.732 | 1.000 | **0.621** | Ridge α=1e4, raw |
| **B** intermittent | `CA_2 × HOUSEHOLD_2` | 250 | sample | 1.106 | 1.092 | 1.207 | 1.000 | **1.000** | Ridge α=0.01, log1p |
| **C** slow/sparse | `CA_4 × HOBBIES_2` | 149 | **full** | 1.631 | 1.401 *(indicative, low n)* | 1.669 | 1.000 | **1.000** | Ridge α=10, log1p |

- **The Phase 3 bar is `min(baseline, zero)` per dataset.** For **A** the fitted floor genuinely is the bar (0.621) — a real, hard target. For **B and C the bar is 1.000**, because the baseline system scores *worse than forecasting nothing*. Beating 1.106 on B would not demonstrate a useful model; it would only demonstrate beating a floor that a zero-cost constant already beats. Phase 3 is scored against the bar column, not the floor column.
- **The zero comparator is exactly 1.000 by construction, not by measurement.** Under sum-then-divide WMAPE a zero forecast scores `sum|y − 0| / sum(y) ≡ 1` on any row set with positive volume. It is reported because it makes the threshold legible: **"WMAPE > 1" means "worse than predicting nothing at all"**, which is precisely the situation on B and C. It costs no fitting, so there is no excuse for a model not to clear it. (It is computed each run rather than hardcoded — a deviation from 1.0 would flag a bug in the row predicate or the WMAPE denominator; see `baseline._zero_predictions`.)
- **L2 vs Naive is scored on the modelable series only** (the set L2 is actually applied to): a constant mean-of-history forecast would be a built-in tie on B2 series, so including them would dilute that comparison. **L2 beats the constant-mean comparator on every dataset** (0.620<0.732, 1.092<1.207, 1.401<1.669) — **though on B and C neither beats a zero forecast, which is the real bar.** So the correct reading is narrow: L2 is not a strawman *for the mean*, and the Ridge is doing genuine work relative to a constant — but on the intermittent datasets that work is not yet enough to be worth anything, because both sit above 1.0. Only on A does L2 clear both comparators.
- **WMAPE rises monotonically A→B→C**, exactly tracking the intermittency the datasets were picked to span. B and C landing above 1.0 is expected and honest: on zero-heavy series the mean absolute error exceeds mean demand. **This is the gap Tweedie exists to close in Phase 3** — and the zero comparator is what turns that from a qualitative hope into a pass/fail line.
- **Leakage check: PASS on all three** (teeth-having — perturbing future units changes features only at/after `origin + horizon`, never before; see `run_baseline.leakage_check`).

---

## WMAPE by Syntetos–Boylan class

The honesty check on the volume-weighted headline. Denominator is per-class volume.

| Dataset | smooth | erratic | intermittent | lumpy |
|---|---|---|---|---|
| A | 0.478 | 0.548 | 1.029 | 0.869 |
| B | 0.583 | 1.113 | 1.143 | 1.069 |
| C | — | — | 1.678 | 1.485 |

- Every dataset's aggregate is carried by its **smooth/dense** series; the **intermittent + lumpy tail is 2–2.5× worse** than the headline. On A the aggregate (0.62) hides an intermittent-class WMAPE of 1.03 — precisely why the per-class split is reported. C has only intermittent/lumpy series present (no smooth/erratic), consistent with it being the sparse cell.
- **Against the 1.0 line, the per-class split is sharper than the headline suggests:** every cell at or above 1.0 in this table is a class where the baseline loses to a zero forecast. That includes **A's own intermittent class (1.029)** — so even the one dataset whose headline clears the bar has an intermittent segment that does not. A's aggregate passes on the strength of its smooth and erratic series alone.

## WMAPE by volume tercile — dataset A only

| low | mid | high |
|---|---|---|
| 1.112 | 0.884 | 0.512 |

- The floor is **carried by the high-volume third** (0.51); the low-volume third is more than 2× worse (1.11). Even in the dense cell where L2 is most competitive, the aggregate is a volume-weighted average masking weak performance on the smaller sellers — the tercile split makes that explicit. Note the low tercile is **above 1.0**, i.e. beaten by a zero forecast: A's headline pass is a high-volume phenomenon.

## Per-fold spread (stability)

| Dataset | fold 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| A | 0.744 | 0.629 | 0.567 | 0.616 | 0.578 |
| B | 1.122 | 1.072 | 1.142 | 1.155 | 1.048 |
| C | 1.510 | 1.684 | 1.617 | 1.683 | 1.700 |

- Tight across folds (spread ≤ ~0.18), so the headline isn't a one-window artifact. A's fold 0 (earliest origin, `d_1774–1801`) is the worst, consistent with the shortest training history.
- **B is above 1.0 in all five folds and C in all five** — losing to the zero forecast is not a one-window artifact either, it is the stable state of the baseline on the intermittent datasets. A is below 1.0 in all five.

## Routing composition (L2 vs mean floor)

| Dataset | → L2 | → mean floor | Notes |
|---|---|---|---|
| A | 248 | 2 | dense; essentially all modelable |
| B | 223 | 27 | intermittent but mostly still modelable |
| **C** | **31** | **118** | **sparse-fallback showcase** — 79% of series too sparse for L2 |

- **C is the sparse-fallback showcase:** 118 of 149 series route to the mean floor, exactly the B2 branch the architecture reserves for series with too little signal to justify a heavy model. Its L2-subset WMAPE (1.401 over 31 series) is **tagged indicative** — a WMAPE over 31 series is high-variance and should not be read as a stable floor. C's job is to demonstrate the fallback firing correctly, not to produce a headline L2 number.

---

## Config chosen (tuned honestly on the CV)

Selected per dataset by pooled L2-series CV WMAPE over α ∈ {0.01, 0.1, 1, 10, 100, 1e3, 1e4} × {raw, log1p}.

- **The target transform is the only choice that matters** — A takes **raw** (0.620 vs 1.108), B and C take **log1p** (1.092 vs 1.146; 1.401 vs 1.645). Large, real gaps. The mechanism is *not* "log helps on skewed counts"; it is two different effects, measured below as forecast bias `sum(forecast)/sum(actual)` on the L2-routed rows:

  | | raw: WMAPE / bias | log1p: WMAPE / bias |
  |---|---|---|
  | A | **0.620** / 1.00 | 1.108 / **1.10** |
  | B | 1.146 / 0.98 | **1.092** / **0.69** |
  | C | 1.645 / 1.04 | **1.401** / **0.65** |

  **On B and C, log1p wins by systematically underpredicting.** Fitting on `log1p` and inverting with `expm1` is biased low by Jensen's inequality (`expm1(E[log1p y]) ≤ E[y]`), and here that bias is severe — forecasts total **31% and 36% below actuals**. That drags predictions toward zero, and *zero is near-optimal on these datasets*, so the transform is rewarded. **This is not a better model — it is an approximation of the trivial forecast**, which is exactly why both still land above 1.0. Read the log1p win on B/C as evidence for the zero bar, not against it.

  **On A the mechanism is different, and the opposite sign.** log1p **over**predicts (bias 1.10), so retransformation bias is not what costs it. The damage is concentrated entirely in the **high-volume tercile — WMAPE 1.154 at bias 1.16, against raw's 0.512 at bias 0.96** — while the mid tercile is actually *better* under log1p (0.855 vs 0.884). Fitting in log space equalizes series in the loss, so the high-volume series stop dominating the fit; but they still dominate WMAPE's volume-weighted denominator, and `expm1` amplifies their residuals on the way back. Raw wins on A because raw's loss is aligned with the metric's weighting, not because "the dense signal is learnable directly."
- **α is immaterial here — do not read a story into it.** Across the full six-order-of-magnitude grid the WMAPE spread within the winning transform is **0.00014 (A), 0.0017 (B), 0.018 (C)**. The selected values (A: 1e4, B: 0.01, C: 10) are decided in the 4th–5th decimal place, i.e. at noise level; A and B each land on a *grid boundary*, and widening the grid simply moves them to the new boundary without changing WMAPE. An earlier version of this doc explained A's choice as "strong shrinkage stabilizes the many one-hot calendar columns" — **that explanation is not supported by the data** and has been removed: α=0.01 and α=1e4 score the same on A to four decimals. The honest statement is that this design is insensitive to L2 penalty strength, so no tuning effort was hidden and none was needed.
- **Selection reports on the same folds it tunes on.** This is mild optimism, and the reason it is acceptable is *not* that it makes the floor harder to beat — it does the opposite, and a floor already beaten by a zero forecast is not made conservative by being slightly optimistic. The reason it is acceptable is that **Phase 3's challenger is tuned by the identical protocol on the same folds, so the optimism is equal on both sides and cancels in the comparison.** Matched protocol is the goal, not the strongest possible floor. Stated, not hidden.

---

## Method (as built)

- **Metric** — volume-weighted WMAPE, `sum|a−f| / sum(a)` pooled; per-SB-class everywhere, per-volume-tercile on A only. WRMSSE rejected (see plan §1.2). `dfa.metrics`.
- **Holdout** — rolling-origin, expanding window, horizon 28; fold count data-driven (`min(MAX_FOLDS=5, (train_end−MIN_TRAIN_DAYS=365)//horizon)`), which is **5 on M5** at origins `d_1773/1801/1829/1857/1885`; min-history guard; sealed `d_1914–1941` untouched. `dfa.splits`.
- **Features (frozen for Phase 3)** — calendar one-hots (fixed wday/month; event types for **both** `event_type_1/2` slots **extracted from the calendar, not hardcoded**; SNAP resolved per-row by state), forward-filled price, lags 28/35/42, trailing rolling means (28/56) and expanding mean, all lagged ≥ horizon → direct multi-horizon, leakage-safe by construction. `dfa.features`.
- **Model** — pooled Ridge per dataset (per-fold standardized numerics, non-negativity clamp); B2 (ADI ≥ 8.77) → constant mean-of-active-history floor; naive-mean comparator scored on the modelable (non-B2) series only; all-zero comparator scored on all series (same rows as the baseline). `dfa.baseline`.
- **Evaluation set** — a validation day is scored only if evaluable on the same predicate as training (series active and ≥ horizon past introduction), so pre-introduction zeros and feature-warmup days don't inflate the denominator. No rows were excluded on A/B/C (max intro day d_1658, earliest origin d_1773), so these numbers are unchanged by it; it is a safeguard for late-introduced series.
- **Tests** — 40 Phase-2 unit tests (metrics, splits, features incl. leakage invariant, baseline routing/clamp/tuning, zero-comparator identity); full suite green (96).

## Deferred (flagged, not run)

- **A/B full-cell confirmation** (823 / full-B) — headline is on the 250-sample; Phase 1 verified sample fidelity, so this is a confirmation, deferred to end-of-phase or folded into Phase 3.
- Seasonal-naive comparator — only if a dataset's naive-mean looks suspiciously strong; it does not here.
