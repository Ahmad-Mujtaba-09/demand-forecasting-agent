"""Central config for the demand-forecasting agent.

Single source of truth for paths and the day-window boundaries. The train/eval
split is defined here so no other module hard-codes 1913/1941.
"""

from __future__ import annotations

from pathlib import Path

# --- paths ---
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data" / "m5-forecasting-accuracy"
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"

CALENDAR_CSV: Path = DATA_DIR / "calendar.csv"
SELL_PRICES_CSV: Path = DATA_DIR / "sell_prices.csv"
# We load the *evaluation* file: identical to validation up to d_1913, but it
# also carries the sealed d_1914..d_1941 labels used only in Phase 5.
SALES_CSV: Path = DATA_DIR / "sales_train_evaluation.csv"

# --- day-window boundaries (per the Phase 1 plan) ---
DATASET_START_DATE: str = "2011-01-29"  # d_1
TRAIN_END_DAY: int = 1913               # last day used for EDA / training
EVAL_END_DAY: int = 1941                # last day in the evaluation file
HORIZON: int = EVAL_END_DAY - TRAIN_END_DAY  # 28 sealed days

# series identity columns in the wide sales file
ID_COLS: tuple[str, ...] = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
