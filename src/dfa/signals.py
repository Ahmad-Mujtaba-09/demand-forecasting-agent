"""Per-series branch signals (Phase 1, Increment 2).

Each function takes a 1-D array of a single series' demand over its ACTIVE
window (first non-zero day onward), ordered by day. The three signal families
map 1:1 to the branch rules in the master plan:

- intermittency (zero_share + ADI + CV^2 -> Syntetos-Boylan class) -> B1 vs B2
- seasonality (day-of-week variance-explained, eta^2)              -> B3 vs B3'
- spike stats (MAD-based)                                           -> critic bounds

Priors:
- ADI and CV^2 separate *how often* demand occurs from *how variable its size*
  is -- the distinction between a Tweedie-friendly series and a too-sparse one.
- Weekly seasonality is measured as day-of-week variance-explained (eta^2), not
  STL strength. STL strength swings ~3x on its robust flag (spikes either smear
  into the seasonal or dump into the residual), so it is not a defensible per-
  series gate on spiky retail data. eta^2 is deterministic, fast, interpretable,
  and config-free: "what fraction of demand variance is explained by day-of-week".
- Spikes (promos/events/SNAP) are real demand, not errors. We count them to
  quantify tail-heaviness, which feeds the critic's plausible-magnitude check.

Constants: ADI_CUT / CV2_CUT are the published Syntetos-Boylan classification
cutoffs (Syntetos & Boylan, 2005, "The accuracy of intermittent demand
estimates", Int. J. Forecasting 21(2):303-314). They are literature standards,
used as-is and NOT tuned to any dataset -- unlike the branch thresholds in
calibrate_thresholds.py, which are fit per-dataset.
"""

from __future__ import annotations

import numpy as np

# Syntetos-Boylan classification cutoffs -- published constants (S&B 2005),
# used as-is, never tuned per-dataset. See module docstring for citation.
ADI_CUT: float = 1.32
CV2_CUT: float = 0.49
# robust-scale constant: MAD * 1.4826 ~ sigma for a normal
MAD_K: float = 1.4826


def zero_share(units: np.ndarray) -> float:
    """Fraction of active-window days with zero demand."""
    u = np.asarray(units)
    if u.size == 0:
        return float("nan")
    return float((u == 0).mean())


def adi(units: np.ndarray) -> float:
    """Average Demand Interval = periods / non-zero periods.

    NaN if the series never sells (no demand to have an interval between).
    """
    u = np.asarray(units)
    nnz = int((u > 0).sum())
    if nnz == 0:
        return float("nan")
    return float(u.size / nnz)


def cv_squared(units: np.ndarray) -> float:
    """Squared coefficient of variation of the *non-zero* demand sizes.

    NaN (undefined) when fewer than 2 non-zero observations -- with 0 or 1
    demands there is no dispersion to measure, which is distinct from a genuine
    0.0 (>=2 demands that are all the same size). Reporting NaN keeps that
    distinction honest and mirrors dow_seasonality's "undefined" outcome. On M5
    this branch never fires (sparsest series has 7 non-zero days), but it makes the
    function safe on sparser inputs / single series inside the agent.
    sb_class treats a NaN cv2 as non-lumpy, so a near-single-sale series still
    lands in the sparse (intermittent) quadrant, driven by its ADI.
    """
    u = np.asarray(units, dtype=float)
    nz = u[u > 0]
    if nz.size < 2:
        return float("nan")
    mean = nz.mean()
    if mean == 0:
        return 0.0
    std = nz.std(ddof=1)
    return float((std / mean) ** 2)


def sb_class(adi_val: float, cv2_val: float) -> str:
    """Syntetos-Boylan quadrant.

    - ADI undefined (series never sells) -> 'no_demand'.
    - CV^2 undefined (<2 non-zero demands, so size variability is unmeasurable)
      -> treated explicitly as non-lumpy: the class is then driven by ADI, which
      for such a near-single-sale series is large, placing it in the sparse
      (intermittent) quadrant. Made explicit rather than relying on NaN>=cut
      evaluating to False.
    """
    if not np.isfinite(adi_val):
        return "no_demand"
    lumpy_cv = bool(np.isfinite(cv2_val) and cv2_val >= CV2_CUT)
    intermittent_adi = adi_val >= ADI_CUT
    if not intermittent_adi and not lumpy_cv:
        return "smooth"
    if not intermittent_adi and lumpy_cv:
        return "erratic"
    if intermittent_adi and not lumpy_cv:
        return "intermittent"
    return "lumpy"


def dow_seasonality(
    units: np.ndarray,
    wday: np.ndarray,
    period: int = 7,
    min_cycles: int = 2,
) -> tuple[float, bool]:
    """Weekly seasonality as day-of-week variance-explained (eta^2), in [0,1].

    eta^2 = SS_between / SS_total over the weekday groups: the fraction of total
    demand variance explained by which day of the week it is. This is the B3
    seasonality signal -- deterministic and config-free.

    Requirement floor is **derived from the cadence, not a fixed number**: the
    window must cover at least `min_cycles` full cycles (`period` days each) AND
    contain every one of the `period` weekday levels, so each weekday recurs
    enough to estimate a group mean. That generalizes to any M5-format dataset
    (weekly cadence -> period=7) and correctly returns undefined on windows too
    short to hold a stable weekly pattern -- rather than reporting a spurious
    eta^2 off one or two observations per weekday.

    `wday` is the day-of-week code (any `period`-level encoding) aligned to
    `units`. Returns (eta2, defined); `defined` is False (eta2 NaN) when the
    window is shorter than `period * min_cycles` or is missing weekday levels.
    Spikes still enter the variance, so a spiky series honestly reads as less
    weekly-driven -- that is the true "fraction explained by day-of-week".
    """
    u = np.asarray(units, dtype=float)
    w = np.asarray(wday)
    if w.size != u.size:
        return float("nan"), False
    min_len = period * min_cycles
    if u.size < min_len or np.unique(w).size < period:
        return float("nan"), False
    grand = u.mean()
    ss_total = float(((u - grand) ** 2).sum())
    if ss_total == 0.0:
        return 0.0, True
    ss_between = 0.0
    for k in np.unique(w):
        uk = u[w == k]
        ss_between += uk.size * (uk.mean() - grand) ** 2
    return float(min(max(ss_between / ss_total, 0.0), 1.0)), True


def spike_stats(units: np.ndarray) -> dict[str, float]:
    """MAD-based spike detection on non-zero days.

    A non-zero day is a spike if units > median + 3*1.4826*MAD (over non-zero
    days). Returns spike_count, spike_ratio (spikes / active days), and
    max_med_ratio (max / median non-zero -- tail heaviness).

    Degenerate-MAD guard: low-count integer demand very often has median 1 and
    MAD 0 (most non-zero days sell the same small amount), which collapses the
    robust scale to zero and would flag every above-typical day. When MAD is 0
    we fall back to a multiplicative tail rule -- a spike is demand more than 3x
    the typical non-zero day -- which keeps the "unusually large" meaning intact.
    """
    u = np.asarray(units, dtype=float)
    nz = u[u > 0]
    if nz.size == 0:
        return {"spike_count": 0.0, "spike_ratio": 0.0, "max_med_ratio": float("nan")}
    med = np.median(nz)
    mad = np.median(np.abs(nz - med))
    scale = MAD_K * mad
    threshold = med + 3.0 * scale if scale > 0 else 3.0 * med
    spike_count = float((nz > threshold).sum())
    return {
        "spike_count": spike_count,
        "spike_ratio": float(spike_count / u.size),
        "max_med_ratio": float(nz.max() / med) if med > 0 else float("nan"),
    }


def series_signals(
    units: np.ndarray, wday: np.ndarray | None = None
) -> dict[str, float | str | bool]:
    """All branch signals for one series' active-window demand vector.

    `wday` (day-of-week codes aligned to `units`) enables the B3 seasonality
    signal; omit it and dow_season is reported undefined.
    """
    u = np.asarray(units, dtype=float)
    adi_val = adi(u)
    cv2_val = cv_squared(u)
    if wday is not None:
        dow, dow_def = dow_seasonality(u, wday)
    else:
        dow, dow_def = float("nan"), False
    out: dict[str, float | str | bool] = {
        "active_len": int(u.size),
        "nonzero_count": int((u > 0).sum()),
        "mean_nonzero": float(u[u > 0].mean()) if (u > 0).any() else 0.0,
        "zero_share": zero_share(u),
        "adi": adi_val,
        "cv2": cv2_val,
        "sb_class": sb_class(adi_val, cv2_val),
        "dow_season": dow,
        "dow_defined": dow_def,
    }
    out.update(spike_stats(u))
    return out
