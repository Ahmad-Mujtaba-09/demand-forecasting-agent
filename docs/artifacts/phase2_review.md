# Phase 2 Review — Baseline & the Bar

*Markdown record of the Phase 2 review for convenient reading/diffing. The
interactive version (comparator charts against the 1.000 rule, per-class and
per-fold views, transform diagnostics, table view) is
[`artifacts/phase2_review_standalone.html`](../../artifacts/phase2_review_standalone.html)
— double-click to open in any browser.*

*Source of record for the full write-up is
[`docs/results/02_phase2_baseline_results.md`](../results/02_phase2_baseline_results.md);
every number in both is generated from `artifacts/phase2_baseline_results.json`.*

An L2 baseline was tuned as honestly as it can be on each of the three datasets,
then measured against **two** comparators: a constant mean-of-history, and a
zero forecast. The second one changed the conclusion.

| | |
|---|---|
| Metric | WMAPE, volume-weighted, sum-then-divide |
| Holdout | 5-fold rolling origin, expanding window, horizon 28 |
| Origins | `d_1773 / 1801 / 1829 / 1857 / 1885` · sealed `d_1914–1941` untouched |
| Datasets | A, B, C (A/B are 250-series samples; C is the full 149 cell) |
| Leakage check | **PASS** on all three |

---

## Headline — and the bar

| Dataset | Cell | n | Baseline | L2 *(modelable)* | Naive *(modelable)* | Zero | **Phase 3 bar** | Config |
|---|---|--:|--:|--:|--:|--:|--:|---|
| **A** — dense | `CA_3 × FOODS_3` | 250 | 0.621 | 0.620 | 0.732 | 1.000 | **0.621** | Ridge α=1e4, raw |
| **B** — intermittent | `CA_2 × HOUSEHOLD_2` | 250 | 1.106 | 1.092 | 1.207 | 1.000 | **1.000** | Ridge α=0.01, log1p |
| **C** — slow/sparse | `CA_4 × HOBBIES_2` | 149 | 1.631 | 1.401 *(indicative)* | 1.669 | 1.000 | **1.000** | Ridge α=10, log1p |

**The bar for Phase 3 is `min(baseline, 1.0)` per dataset.** Only **A** sets a bar
of its own.

---

## What the run says

- **The zero forecast is an identity, not a measurement.** Under sum-then-divide
  WMAPE, an all-zero forecast scores `sum|y−0| / sum(y)` **≡ 1.000** on any row set
  with positive volume. It is reported because it makes the threshold legible:
  **"WMAPE > 1" means "worse than predicting nothing at all."** It costs no fitting,
  so there is no excuse for a model not to clear it. Computed each run rather than
  hardcoded — a deviation from 1.0 would flag a bug in the row predicate or the
  WMAPE denominator.
- **L2 beats the constant mean on every dataset — and it is still not enough.**
  0.620<0.732, 1.092<1.207, 1.401<1.669, so the Ridge does real work and is not a
  strawman for the mean. **But on B and C neither beats a zero forecast**, so that
  work does not yet buy anything. The narrow claim is the only defensible one.
- **B and C are above 1.0 in all five folds.** Losing to zero is the stable state
  of the baseline on the intermittent datasets, not a one-window artifact.
- **α is immaterial; the target transform is everything.** See below.

---

## WMAPE by Syntetos–Boylan class

Denominator is per-class volume. **Every cell ≥ 1.000 is a class the baseline loses
to zero on.**

| Dataset | smooth | erratic | intermittent | lumpy |
|---|--:|--:|--:|--:|
| A | 0.478 | 0.548 | **1.029** | 0.869 |
| B | 0.583 | **1.113** | **1.143** | **1.069** |
| C | — | — | **1.678** | **1.485** |

**A's own intermittent class is above the rule.** A's aggregate pass is carried by
its smooth and erratic series, not earned across the board — which is exactly what
the volume-weighted headline hides and why this split is reported.

Dataset A by volume tercile: **low 1.112 · mid 0.884 · high 0.512.** Same story —
A's pass is a high-volume phenomenon; its low-volume third is itself above 1.0.

## Per-fold spread

| Dataset | f0 | f1 | f2 | f3 | f4 |
|---|--:|--:|--:|--:|--:|
| A | 0.744 | 0.629 | 0.567 | 0.616 | 0.578 |
| B | 1.122 | 1.072 | 1.142 | 1.155 | 1.048 |
| C | 1.510 | 1.684 | 1.617 | 1.683 | 1.700 |

Tight (spread ≤ ~0.18). A's fold 0 is worst, consistent with the shortest training
history.

---

## Why the chosen transform wins

Grid: α ∈ {0.01, 0.1, 1, 10, 100, 1e3, 1e4} × {raw, log1p}. Bias is
`sum(forecast)/sum(actual)` on the L2-routed rows.

| Dataset | raw: WMAPE / bias | log1p: WMAPE / bias | α spread (winning transform) |
|---|--:|--:|--:|
| A | **0.620** / 1.00 | 1.108 / **1.10** | 0.00014 |
| B | 1.146 / 0.98 | **1.092** / **0.69** | 0.00171 |
| C | 1.645 / 1.04 | **1.401** / **0.65** | 0.01787 |

- **On B and C, log1p wins by systematically underpredicting.** Fitting on `log1p`
  and inverting with `expm1` is biased low by Jensen's inequality
  (`expm1(E[log1p y]) ≤ E[y]`), and here severely: forecasts total **31% and 36%
  below actuals**. That drags predictions toward zero, and *zero is near-optimal on
  these datasets*, so the transform is rewarded. **This is an approximation of the
  trivial forecast, not a better model** — which is precisely why both still land
  above 1.0.
- **On A the mechanism is different and the sign flips.** log1p **over**predicts
  (bias 1.10), so retransformation bias is not what costs it. The damage is
  concentrated in the **high-volume tercile — WMAPE 1.154 at bias 1.16 vs raw's
  0.512 at bias 0.96** — while the mid tercile is actually better under log1p.
  Fitting in log space stops the high-volume series dominating the loss while they
  still dominate WMAPE's volume-weighted denominator.
- **α selection is noise.** The spread across the full six-order grid is 1e-4 to
  1e-2; A and B both land on grid *boundaries*, and widening the grid just moves
  them to the new edge without changing WMAPE. Read no story into the chosen α.

---

## Routing — how many series L2 ever sees

| Dataset | → L2 | → sparse mean-fallback | |
|---|--:|--:|---|
| A | 248 | 2 | dense; essentially all modelable |
| B | 223 | 27 | intermittent but mostly still modelable |
| **C** | **31** | **118** | **sparse-fallback showcase** — 79% too sparse for L2 |

C's L2 number is **tagged indicative**: a WMAPE over 31 series is high-variance and
should not be read as a stable floor. C's job is to demonstrate the fallback firing
correctly.

---

## Honest notes

- **Selection reports on the same folds it tunes on.** Mild optimism. The reason
  this is acceptable is *not* that it makes the floor harder to beat — it does the
  opposite, and a floor already beaten by a zero forecast is not made conservative
  by being slightly optimistic. It is acceptable because **Phase 3's challenger is
  tuned by the identical protocol on the same folds, so the optimism is equal on
  both sides and cancels in the comparison.** Matched protocol is the goal.
- **A and B are 250-series samples.** C is the full cell. Full-cell confirmation for
  A/B is deferred, not run.

---

## Decisions for your review

1. **Adopt `min(baseline, 1.0)` as the Phase 3 bar.** Already computed per dataset
   (`phase3_bar`). Consequence: on B and C a Phase 3 model must reach **WMAPE <
   1.000** to count as useful at all — materially harder than beating 1.106 / 1.631.
2. **Sampling parity for the bake-off.** Either hold the same 250-series sample into
   Phase 3 or re-run Phase 2 at full cell size first; mixing the two makes the
   comparison unfalsifiable.
3. **C stays tagged indicative.** Read it as a fallback demonstration, not a floor.

---

## Provenance

- Code: [`src/dfa/baseline.py`](../../src/dfa/baseline.py) ·
  [`run_baseline.py`](../../src/dfa/run_baseline.py) ·
  [`build_phase2_review.py`](../../src/dfa/build_phase2_review.py) — 96 tests passing.
- Recorded outputs: [`artifacts/phase2_baseline_results.json`](../../artifacts/phase2_baseline_results.json),
  [`artifacts/phase2_chartdata.json`](../../artifacts/phase2_chartdata.json).
- Reproduce: `PYTHONPATH=src python -m dfa.run_baseline && PYTHONPATH=src python -m dfa.build_phase2_review`.
  The HTML is generated *from* the results JSON, so the charts cannot drift from the run.
- Plans: [master plan](../plans/00_master_plan.md) · [Phase 2 sub-plan](../plans/02_phase2_baseline_plan.md).
