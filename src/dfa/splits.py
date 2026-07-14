"""Rolling-origin (expanding-window) time-series folds (Phase 2 sub-plan sec 2).

Splits are on the DAY axis only -- never shuffled, never random k-fold. Each fold
trains on all days up to a forecast *origin* and validates on the 28 days after
it (the M5 horizon). We walk the origin back from the end of the training range
in non-overlapping 28-day blocks:

    origin 1885 -> val d_1886..1913
    origin 1857 -> val d_1858..1885
    ...

**The fold count is data-driven, not hardcoded.** It is the number of horizon
blocks that fit into the history *after* reserving a minimum training window,
capped so very long datasets don't over-fold:

    n_folds = min(MAX_FOLDS, (train_end - MIN_TRAIN_DAYS) // horizon)

- **Min-history guard (MIN_TRAIN_DAYS).** The earliest fold must still train on at
  least one full year, so every fit has annual calendar coverage and ample lag/
  rolling history (max lookback is 56 days). Because n_folds is derived from
  `train_end - MIN_TRAIN_DAYS`, the earliest origin is always >= MIN_TRAIN_DAYS by
  construction. Too little history to place even one guarded fold -> raise, don't
  silently train on a thin window.
- **Cap (MAX_FOLDS).** More folds tighten the score's variance estimate but cost
  compute and eat history; 5 is the point past which the marginal variance gain
  isn't worth shrinking the earliest train window. On M5 (train_end=1913) the
  formula yields exactly 5 (earliest origin d_1773, ~4.85 yr of history) -- the
  reviewed choice. A shorter M5-format dataset gets proportionally fewer folds.

The sealed horizon d_1914..1941 is Phase 5 only: `make_folds` never emits a fold
whose validation reaches past `train_end`.

Leakage discipline lives in the feature builder (features lagged >= horizon); the
fold only carves days. `val_start == origin + 1` is the boundary invariant tests
assert.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config

# one full year of training history before the earliest origin (annual coverage +
# room for the 56-day max feature lookback). The min-history guard.
MIN_TRAIN_DAYS: int = 365
# cap on fold count: beyond this the variance gain isn't worth the lost history.
MAX_FOLDS: int = 5


@dataclass(frozen=True)
class Fold:
    index: int      # 0 = earliest origin ... n-1 = latest
    origin: int     # last training day (inclusive)
    val_start: int  # first validation day == origin + 1
    val_end: int    # last validation day == origin + horizon


def n_folds_for(
    train_end: int = config.TRAIN_END_DAY,
    horizon: int = config.HORIZON,
    min_train_days: int = MIN_TRAIN_DAYS,
    max_folds: int = MAX_FOLDS,
) -> int:
    """Data-driven fold count: horizon blocks fitting after the min-history guard.

    Raises if the history can't hold even one guarded fold.
    """
    usable = train_end - min_train_days
    if usable < horizon:
        raise ValueError(
            f"insufficient history: train_end={train_end} leaves {usable} days after "
            f"the {min_train_days}-day min-history guard, but one fold needs "
            f"horizon={horizon}. Provide more history or lower min_train_days."
        )
    return min(max_folds, usable // horizon)


def make_folds(
    train_end: int = config.TRAIN_END_DAY,
    horizon: int = config.HORIZON,
    min_train_days: int = MIN_TRAIN_DAYS,
    max_folds: int = MAX_FOLDS,
) -> list[Fold]:
    """Non-overlapping rolling-origin folds, earliest origin first.

    The latest fold's validation ends exactly at `train_end`; earlier folds step
    back one horizon at a time. The count is derived from history length (see
    `n_folds_for`); the earliest origin is >= `min_train_days` by construction.

    `train_end` is the dataset's **true last usable training day** and should be
    passed by the caller (e.g. `last_sales_day() - HORIZON` in the runner). The
    default is only M5's convenience value; a different M5-format dataset with a
    different history length must pass its own, or the folds would be misplaced.
    """
    n_folds = n_folds_for(train_end, horizon, min_train_days, max_folds)
    folds: list[Fold] = []
    for k in range(n_folds):
        val_end = train_end - (n_folds - 1 - k) * horizon
        val_start = val_end - horizon + 1
        origin = val_start - 1
        folds.append(Fold(index=k, origin=origin, val_start=val_start, val_end=val_end))
    return folds

