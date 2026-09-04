"""LightGBM bake-off entrants (Phase 3 sub-plan sec 1-2).

Every entrant trains on the FROZEN Phase-2 feature set, the same folds, the same
B2 routing and the same evaluable-row predicate, so the only thing that varies is
the objective. That is what makes this a test of objectives rather than of
feature engineering, and `dfa.features.feature_columns` is asserted unchanged
against the Phase-2 manifest by a test.

Entrants (sub-plan sec 1):
  E1 lgb_l2       -- the tree-model CONTROL. Isolates model class from objective:
                     (E1 - Ridge) is what trees buy, (E2/E3 - E1) is what the
                     objective buys. Without it, "Tweedie beat Ridge" confounds
                     the two.
  E2 lgb_tweedie  -- the B1 branch's model; compound Poisson-Gamma matches
                     zero-inflated non-negative demand.
  E3 lgb_poisson  -- count-native, one fewer tuned parameter. If it ties Tweedie,
                     the simpler branch rule wins.
  E4 lgb_quantile50 -- promoted INTO the bake-off: the only entrant whose loss is
                     aligned with what daily WMAPE rewards (the conditional
                     median). Excluding it would mean testing three
                     mean-estimators against a median metric.

Two protocol rules that are load-bearing:

**Determinism.** LightGBM's histogram construction is thread-count dependent, so
without a pinned seed, `deterministic=True`, `force_row_wise=True` and a fixed
`num_threads` the same code on the same data returns different numbers on a
different host. For a project whose deliverable is a reproducible pipeline -- and
whose Phase 5 sealed run must be re-checkable -- that is not acceptable. Even the
Phase-2 Ridge drifts at ~1e-14 across runs from BLAS reduction order; trees drift
far more.

**Early stopping never sees the fold's validation block.** The stopping set is a
tail slice of the fold's own TRAINING days (the `HORIZON` days ending at the
origin), with the model fit on everything before it. This is the likeliest place
for a tree pipeline to leak and it is not covered by the Phase-2 feature-level
leakage check, which tests features rather than the fit loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from . import features as ft
from .manifest import NUM_THREADS, SEED
from .splits import Fold

# --- pre-registered capacity grid (sub-plan sec 2) --------------------------
# Written down BEFORE any run so no objective can be handicapped by an
# obviously-wrong capacity setting and no per-objective tuning asymmetry can
# creep in. Deliberately small: the phase tests objectives, not hyperparameters,
# and a wide grid would worsen the selection-optimism problem it is measured
# against.
CAPACITY_GRID: tuple[dict, ...] = (
    {"num_leaves": 15, "min_data_in_leaf": 20},
    {"num_leaves": 15, "min_data_in_leaf": 100},
    {"num_leaves": 31, "min_data_in_leaf": 20},
    {"num_leaves": 31, "min_data_in_leaf": 100},
)
LEARNING_RATE: float = 0.05
MAX_ROUNDS: int = 600            # ceiling; the stopping set picks the real count
EARLY_STOPPING_ROUNDS: int = 30

# Tweedie's variance_power is a DISTRIBUTIONAL parameter, not a capacity knob:
# 1.1 is near-Poisson, 1.9 near-Gamma. Swept because the right value depends on
# how the zero mass and the size distribution trade off, which is exactly the
# question the B1 branch exists to answer.
TWEEDIE_POWERS: tuple[float, ...] = (1.1, 1.3, 1.5)

# Reproducibility settings -- see the module docstring.
DETERMINISM: dict = {
    "seed": SEED,
    "bagging_seed": SEED,
    "feature_fraction_seed": SEED,
    "data_random_seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": NUM_THREADS,
    "verbosity": -1,
}


@dataclass(frozen=True)
class Entrant:
    """One bake-off contestant: a name plus the objective params that define it."""
    name: str
    objective: dict                       # the objective-defining params
    capacity: dict = field(default_factory=dict)   # from CAPACITY_GRID

    @property
    def params(self) -> dict:
        return {
            "objective": self.objective["objective"],
            **{k: v for k, v in self.objective.items() if k != "objective"},
            **self.capacity,
            "learning_rate": LEARNING_RATE,
            **DETERMINISM,
        }

    @property
    def config_id(self) -> str:
        """Stable label for one (entrant, capacity, objective-param) combination."""
        bits = [f"{k}={v}" for k, v in sorted(self.capacity.items())]
        bits += [f"{k}={v}" for k, v in sorted(self.objective.items()) if k != "objective"]
        return f"{self.name}[{','.join(bits)}]"


def entrant_grid(name: str, objective: dict) -> list[Entrant]:
    """All pre-registered configs for one entrant."""
    return [Entrant(name=name, objective=objective, capacity=cap) for cap in CAPACITY_GRID]


def bakeoff_entrants() -> dict[str, list[Entrant]]:
    """E1-E4. Tweedie additionally sweeps its distributional parameter."""
    out: dict[str, list[Entrant]] = {
        "lgb_l2": entrant_grid("lgb_l2", {"objective": "regression"}),
        "lgb_poisson": entrant_grid("lgb_poisson", {"objective": "poisson"}),
        "lgb_quantile50": entrant_grid(
            "lgb_quantile50", {"objective": "quantile", "alpha": 0.5}
        ),
    }
    out["lgb_tweedie"] = [
        e
        for vp in TWEEDIE_POWERS
        for e in entrant_grid("lgb_tweedie", {"objective": "tweedie", "tweedie_variance_power": vp})
    ]
    return out


def quantile_entrants(level: float) -> list[Entrant]:
    """B4 interval models (sub-plan sec 5). q=0.5 is shared with bake-off E4."""
    return entrant_grid(f"lgb_quantile{int(level * 100)}",
                        {"objective": "quantile", "alpha": level})


def _train_val_rows(feat: pd.DataFrame, fold: Fold, exclude: set[str]):
    """Fold slices, sharing Phase 2's `trainable` predicate exactly.

    The stopping split is the crux: `stop` is the HORIZON days ending at the
    origin -- still training data -- and `fit` is everything before it. The
    fold's own validation block (origin+1 .. origin+HORIZON) is never seen
    during fitting or stopping.
    """
    train = feat[(feat["day_idx"] <= fold.origin) & feat["trainable"]
                 & ~feat["id"].isin(exclude)]
    stop_start = fold.origin - config.HORIZON + 1
    fit_rows = train[train["day_idx"] < stop_start]
    stop_rows = train[train["day_idx"] >= stop_start]
    val = feat[(feat["day_idx"] >= fold.val_start) & (feat["day_idx"] <= fold.val_end)
               & feat["trainable"] & ~feat["id"].isin(exclude)]
    return fit_rows, stop_rows, val


def fit_predict_fold(
    feat: pd.DataFrame, fold: Fold, entrant: Entrant, exclude: set[str]
) -> pd.DataFrame:
    """Fit one entrant on one fold and predict its validation block.

    Forecasts are clamped to >= 0. Tweedie and Poisson are non-negative by their
    log link, but L2 and quantile are not, so the clamp is load-bearing for E1/E4
    -- and it is the Phase-4 critic's `min_forecast` bound applied at source,
    consistent with Phase 2.
    """
    fit_rows, stop_rows, val = _train_val_rows(feat, fold, exclude)
    if fit_rows.empty or val.empty:
        return pd.DataFrame(columns=["id", "day_idx", "actual", "forecast", "fold"])
    cols = ft.feature_columns(feat)

    dtrain = lgb.Dataset(fit_rows[cols], label=fit_rows["units"], free_raw_data=False)
    callbacks, valid_sets = [], []
    if not stop_rows.empty:
        valid_sets = [lgb.Dataset(stop_rows[cols], label=stop_rows["units"],
                                  reference=dtrain, free_raw_data=False)]
        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]

    booster = lgb.train(entrant.params, dtrain, num_boost_round=MAX_ROUNDS,
                        valid_sets=valid_sets, callbacks=callbacks)
    pred = np.clip(booster.predict(val[cols], num_iteration=booster.best_iteration), 0.0, None)
    return pd.DataFrame({
        "id": val["id"].to_numpy(), "day_idx": val["day_idx"].to_numpy(),
        "actual": val["units"].to_numpy(), "forecast": pred, "fold": fold.index,
    })


def predict_all_folds(
    feat: pd.DataFrame, folds: list[Fold], entrant: Entrant, exclude: set[str]
) -> pd.DataFrame:
    """Every fold's out-of-sample predictions for one entrant config."""
    frames = [fit_predict_fold(feat, f, entrant, exclude) for f in folds]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["id", "day_idx", "actual", "forecast", "fold"])
    return pd.concat(frames, ignore_index=True)
