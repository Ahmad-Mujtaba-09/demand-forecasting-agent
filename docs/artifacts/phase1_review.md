# Phase 1 Review — EDA & Dataset Selection

*Markdown record of the Phase 1 review for convenient reading/diffing. The
interactive version (distribution charts, intermittency map, sortable group
table) is [`artifacts/phase1_review_standalone.html`](../../artifacts/phase1_review_standalone.html)
— double-click to open in any browser.*

Three branch signals were computed over every store-item series, and four
sub-datasets were selected from the pattern extremes. Every figure below is
generated from the shipped artifacts (`signal_table.parquet`, `selection.json`,
`thresholds.json`); thresholds are **fit to the dataset's own distribution**
(format-general, threshold-adaptive) and **validated on M5**, the reference dataset.

| | |
|---|---|
| Series total | 30,490 |
| Non-held-out pool | 30,240 |
| Held-out (D) | 250 |
| Store×dept groups | 70 |
| History | `d_1`–`d_1913` · 2011-01-29 → 2016-04-24 |

---

## What the signals say

- **Zero-heavy, as expected.** The median series is **63% zeros**; 73% classify
  as intermittent, 18% lumpy. Squared-error regression is the wrong default — the
  **Tweedie branch claims 36%** of series.
- **Sparsity is about timing, not count.** Even the 10th-percentile series has ~130
  non-zero days. What defeats a boosted model is the demand **interval** (ADI), so
  the B2 fallback keys on ADI — it claims **10%**.
- **Weekly seasonality is weak at the item level.** Day-of-week explains only ~1% of
  the median item's demand variance (median η² **0.012**). The seasonal gate would
  fire for just **~10%** of series → **B3 was dropped as a branch** (see below).

---

## Signal distributions (non-held-out pool)

Percentiles of each per-series signal, used to place the data-relative cuts.

| Signal | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---|--:|--:|--:|--:|--:|--:|--:|
| zero-share | 0.258 | 0.431 | 0.634 | 0.797 | 0.886 | 0.919 | 0.956 |
| ADI (days) | 1.35 | 1.76 | 2.73 | 4.92 | **8.77** | 12.39 | 22.69 |
| day-of-week η² | 0.003 | 0.006 | 0.012 | 0.023 | **0.042** | 0.059 | 0.107 |
| spike ratio | 0.000 | 0.002 | 0.009 | 0.022 | 0.038 | 0.047 | 0.069 |
| max/median ratio | 3.0 | 4.0 | 6.0 | 8.0 | 11.0 | 13.6 | **24.0** |

Syntetos–Boylan class mix (non-D): intermittent 73% · lumpy 18% · smooth 6% · erratic 3%.

---

## The four sub-datasets

Each pick sits at a pattern extreme and stresses one branch. **D is held out** —
frozen and untouched until Phase 5. Groups are `store × dept` cells, stratified-
sampled to ≤250 series (C is 149 = full cell); the samples reproduce the group's
signal profile.

| Pick | Cell | n (full / sampled) | median zero-share | median ADI-class mix | Stresses |
|---|---|--:|--:|---|---|
| **A** — dense / fast | `CA_3 × FOODS_3` | 823 / 250 | 0.345 | 40% interm · 29% lumpy · 20% smooth | Standard objective + baseline |
| **B** — intermittent | `CA_2 × HOUSEHOLD_2` | 515 / 250 | 0.718 | **93% intermittent** | B1 (Tweedie) — cleanest showcase |
| **C** — slow / sparse | `CA_4 × HOBBIES_2` | 149 / 149 | 0.929 | 88% interm · 12% lumpy | B2 (baseline fallback) |
| **D** — held-out (mixed) | `TX_1 × FOODS_2` | 398 / 250 | 0.638 | 75% interm · 18% lumpy · 6% smooth | Phase 5 general test |

The four cells are disjoint by construction (no shared series). D informed no
distribution and no threshold — it is genuinely unseen going into Phase 5,
thresholds included.

---

## Calibrated thresholds & branch coverage

Data-relative, recomputed per dataset on the non-held-out pool.

| Threshold | Value | Basis |
|---|---|---|
| B2 · sparse cut | `ADI ≥ 8.77` | P90 of full-pool ADI |
| B1 · Tweedie cut | `zero-share ≥ 0.663` | **P60 of the non-sparse subset** — genuinely high, computed on the population it governs (not the median, not the full pool) |
| Critic · magnitude | `≥ 0` and `≤ 24× median non-zero` | non-negativity (absolute) + P99 tail (data-relative) |
| SB class cutoffs | `ADI 1.32`, `CV² 0.49` | **absolute** — Syntetos & Boylan (2005), cited, never tuned |

Objective-branch coverage over the non-D pool: **Standard 54% · Tweedie (B1) 36%
· sparse-baseline (B2) 10%**.

---

## Post-review decision: B3 dropped as a branch

Day-of-week η² is weak and clustered (median 0.012, P90 0.042), so a per-series
seasonal toggle would fire for only ~10% of series and isn't worth its complexity.
**Decision:** drop B3/B3′ as a branch; give **every** series the calendar features
(`wday`, `month`, events, SNAP) and let the model learn what weekly structure
exists. Assessing seasonality on *aggregated* series (where the weekly signal is
stronger) is noted as **future work — not built now**. The η² signal is retained as
descriptive EDA — it earned its place by driving this decision.

---

## Provenance

- Signals & selection: [`src/dfa/`](../../src/dfa/) — 56 tests passing.
- Recorded outputs: [`artifacts/datasets/selection.json`](../../artifacts/datasets/selection.json),
  [`artifacts/thresholds.json`](../../artifacts/thresholds.json).
- Plans: [master plan](../plans/00_master_plan.md) · [Phase 1 sub-plan](../plans/01_phase1_eda_plan.md).
- Reference: Syntetos, A. A., & Boylan, J. E. (2005). *The accuracy of intermittent
  demand estimates.* Int. J. Forecasting, 21(2), 303–314.
