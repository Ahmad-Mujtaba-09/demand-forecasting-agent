"""Profile (store, dept) groups and select the four sub-datasets.

Deterministic, defensible selection rules (per the Phase 1 plan). Each pick maps
to one branch and is chosen at a pattern extreme from the observed profiles:

  A dense/fast     -> min median zero_share            -> B3 + fairest baseline
  B intermittent   -> most intermittent-class-dominant -> B1 (Tweedie)
                      among still-modelable groups
  C slow/sparse    -> max median zero_share            -> B2 (baseline fallback)
  D held-out       -> most 'representative' group       -> Phase 5 general test

Exclusions are **explicit and symmetric**, applied in pick order **A -> C -> B
-> D**. Each pick removes its (store, dept) *cell* from the pool before the next
pick, which guarantees the four sub-datasets share **no series** (the hard
constraint). Reusing a *department* is a soft, cosmetic preference (product-
family diversity), relaxed before disjointness is ever touched.

Graceful degradation, not silent forcing:
- **B** relaxes in loosest-cost order if its pool is empty -- drop distinct-dept,
  then drop n>=SAMPLE_N (down to MIN_EVAL_SERIES), then drop the modelable floor
  (which blurs B toward C and is flagged). If even the fully-relaxed pool is
  empty, the dataset has no clean B1 case -> raise, don't force a pick.
- **D** prefers an unseen department; if departments run out it falls back to a
  distinct *cell* in a reused department (still unseen series, flagged as not an
  unseen product family). If no disjoint cell exists at all -> raise.
Every empty-pool situation raises `DatasetSelectionError` naming the pick and the
reason, instead of an opaque `IndexError` from `.iloc[0]`.

Each selected group is then stratified-sampled to <=N series to keep later
phases fast, preserving the group's SB-class mix.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

SAMPLE_N: int = 250
SEED: int = 42
MIN_EVAL_SERIES: int = 100   # a group must have this many series to be evaluable
MODELABLE_NONZERO: float = 1.4  # median non-zero floor for the "modelable" B pick


class DatasetSelectionError(RuntimeError):
    """A required sub-dataset pick could not be made.

    Carries which pick failed (A/B/C/D) and why, so selection breaks with a
    located, actionable message instead of a bare IndexError from an empty
    `.iloc[0]`.
    """


def profile_groups(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (store_id, dept_id) with median signal profile + class mix."""
    g = table.groupby(["store_id", "dept_id"], observed=True)
    prof = g.agg(
        n=("id", "size"),
        med_zero=("zero_share", "median"),
        med_adi=("adi", "median"),
        med_cv2=("cv2", "median"),
        med_nonzero=("mean_nonzero", "median"),
        med_dow=("dow_season", "median"),
        med_spike_ratio=("spike_ratio", "median"),
        pct_intermittent=("sb_class", lambda s: (s == "intermittent").mean()),
        pct_lumpy=("sb_class", lambda s: (s == "lumpy").mean()),
        pct_smooth=("sb_class", lambda s: (s == "smooth").mean()),
        pct_erratic=("sb_class", lambda s: (s == "erratic").mean()),
    ).reset_index()

    # distance to the global median-of-group profile (for the 'representative' D)
    gm_zero = prof["med_zero"].median()
    gm_nz = prof["med_nonzero"].median()
    prof["dist_to_median"] = (
        ((prof["med_zero"] - gm_zero) / gm_zero).abs()
        + ((prof["med_nonzero"] - gm_nz) / gm_nz).abs()
    )
    return prof


def _drop_cells(df: pd.DataFrame, used: set[tuple[str, str]]) -> pd.DataFrame:
    """Rows whose (store_id, dept_id) cell is not already picked.

    Cell-level exclusion is the hard guarantee that picks share no series.
    """
    if not used:
        return df
    keep = [c not in used for c in zip(df["store_id"], df["dept_id"])]
    return df[keep]


def _first(df: pd.DataFrame, by: list[str], ascending: list[bool], label: str, why: str):
    """Sort and take the top row, raising a located error if the pool is empty."""
    if len(df) == 0:
        raise DatasetSelectionError(f"cannot pick {label}: {why}")
    return df.sort_values(by, ascending=ascending).iloc[0]


def _pick_b(evaluable: pd.DataFrame, used_cells, used_depts) -> tuple[pd.Series, str]:
    """B = most intermittent-class-dominant modelable group, cell-disjoint.

    If the strict pool is empty, relax loosest-cost first: distinct-dept, then
    group size, then the modelable floor (last -- it blurs B toward C). Raise if
    even the fully-relaxed pool is empty (no clean B1 case in this dataset).
    """
    base = _drop_cells(evaluable, used_cells)  # never reuse A/C cell (hard)
    modelable = base["med_nonzero"] >= MODELABLE_NONZERO
    big = base["n"] >= SAMPLE_N
    distinct_dept = ~base["dept_id"].isin(used_depts)

    ladder = [
        (base[modelable & big & distinct_dept],
         "max pct_intermittent; modelable, n>=SAMPLE_N, distinct dept"),
        (base[modelable & big],
         "max pct_intermittent; modelable, n>=SAMPLE_N (RELAXED: dept overlaps A/C)"),
        (base[modelable],
         "max pct_intermittent; modelable, n>=MIN_EVAL_SERIES (RELAXED: dept overlap + smaller group)"),
        (base,
         "max pct_intermittent; n>=MIN_EVAL_SERIES only (RELAXED: modelable floor dropped -- B blurring toward C)"),
    ]
    for pool, note in ladder:
        if len(pool):
            b = pool.sort_values(
                ["pct_intermittent", "store_id", "dept_id"], ascending=[False, True, True]
            ).iloc[0]
            return b, note
    raise DatasetSelectionError(
        "cannot pick B_intermittent: no group survives after excluding A/C cells and "
        "relaxing distinct-dept, size, and the modelable floor -- this dataset has no clean "
        "B1 (intermittent-but-modelable) case; flagging rather than forcing a pick."
    )


def _pick_d(evaluable: pd.DataFrame, used_cells, used_depts) -> tuple[pd.Series, str]:
    """D = most representative held-out group.

    Prefer an unseen department (unseen product family). If departments are
    exhausted, fall back to a distinct *cell* in a reused department (still unseen
    series, flagged as not an unseen product family). Raise if no disjoint cell
    remains at all.
    """
    novel = _drop_cells(evaluable[~evaluable["dept_id"].isin(used_depts)], used_cells)
    if len(novel):
        d = novel.sort_values(["dist_to_median", "store_id", "dept_id"]).iloc[0]
        return d, "min dist_to_global-median, distinct dept (unseen product family)"

    distinct_cell = _drop_cells(evaluable, used_cells)
    if len(distinct_cell):
        d = distinct_cell.sort_values(["dist_to_median", "store_id", "dept_id"]).iloc[0]
        return d, ("min dist_to_global-median, distinct CELL only (DEGRADED: dept also seen in "
                   "training -- held-out tests branch selection on unseen series but NOT an unseen "
                   "product family)")
    raise DatasetSelectionError(
        "cannot pick D_heldout: every evaluable group's cell is already used by A/B/C; "
        "no disjoint held-out set can be formed from this dataset."
    )


def select_groups(prof: pd.DataFrame) -> dict[str, dict]:
    """Apply the deterministic rules with explicit, symmetric exclusions (A->C->B->D)."""
    def cell(row) -> dict:
        return {"store_id": row["store_id"], "dept_id": row["dept_id"]}

    evaluable = prof[prof["n"] >= MIN_EVAL_SERIES].copy()
    if evaluable.empty:
        raise DatasetSelectionError(
            f"no (store, dept) group clears MIN_EVAL_SERIES={MIN_EVAL_SERIES} "
            f"(largest group has n={int(prof['n'].max())}); dataset too small to select "
            f"any sub-dataset."
        )

    used_cells: set[tuple[str, str]] = set()   # hard: series-disjointness
    used_depts: set[str] = set()               # soft: product-family diversity

    def take(row: pd.Series) -> pd.Series:
        used_cells.add((row["store_id"], row["dept_id"]))
        used_depts.add(row["dept_id"])
        return row

    # A: densest group with a full training population (no cells used yet)
    a = take(_first(
        _drop_cells(evaluable[evaluable["n"] >= SAMPLE_N], used_cells),
        by=["med_zero", "store_id", "dept_id"], ascending=[True, True, True],
        label="A_dense",
        why=f"no evaluable group has n>=SAMPLE_N={SAMPLE_N} for a full training population",
    ))

    # C: sparsest evaluable group, excluding A's cell
    c = take(_first(
        _drop_cells(evaluable, used_cells),
        by=["med_zero", "store_id", "dept_id"], ascending=[False, True, True],
        label="C_sparse",
        why="no evaluable group remains after excluding A's cell",
    ))

    # B: intermittent-but-modelable, excluding A and C cells (graceful ladder)
    b, b_rule = _pick_b(evaluable, used_cells, used_depts)
    take(b)

    # D: representative held-out, excluding A/B/C cells (graceful ladder)
    d, d_rule = _pick_d(evaluable, used_cells, used_depts)

    return {
        "A_dense": {**cell(a), "rule": "min median zero_share, n>=SAMPLE_N, cell-disjoint", "branch": "B3 + baseline"},
        "B_intermittent": {**cell(b), "rule": b_rule, "branch": "B1 Tweedie"},
        "C_sparse": {**cell(c), "rule": "max median zero_share, evaluable, cell-disjoint", "branch": "B2 fallback"},
        "D_heldout": {**cell(d), "rule": d_rule, "branch": "Phase 5 test"},
    }


def stratified_sample_ids(
    table: pd.DataFrame, store_id: str, dept_id: str, n: int = SAMPLE_N, seed: int = SEED
) -> list[str]:
    """<=n series ids from a group, allocated across SB classes by largest remainder."""
    grp = table[(table["store_id"] == store_id) & (table["dept_id"] == dept_id)]
    if len(grp) <= n:
        return grp["id"].tolist()

    counts = grp["sb_class"].value_counts()
    exact = counts / counts.sum() * n
    alloc = np.floor(exact).astype(int)
    remainder = n - alloc.sum()
    # hand out the remaining slots to the largest fractional parts
    for cls in (exact - alloc).sort_values(ascending=False).index[:remainder]:
        alloc[cls] += 1

    picks: list[str] = []
    for cls, k in alloc.items():
        if k <= 0:
            continue
        sub = grp[grp["sb_class"] == cls]
        picks.extend(sub.sample(min(k, len(sub)), random_state=seed)["id"].tolist())
    return picks


def _profile_of_ids(table: pd.DataFrame, ids: list[str]) -> dict:
    sub = table[table["id"].isin(ids)]
    return {
        "n": len(sub),
        "med_zero": round(float(sub["zero_share"].median()), 4),
        "med_dow": round(float(sub["dow_season"].median()), 4),
        "med_nonzero": round(float(sub["mean_nonzero"].median()), 4),
        "class_mix": sub["sb_class"].value_counts(normalize=True).round(3).to_dict(),
    }


def build_selection(table: pd.DataFrame) -> dict:
    prof = profile_groups(table)
    groups = select_groups(prof)

    selection: dict = {"seed": SEED, "sample_n": SAMPLE_N, "datasets": {}}
    for label, meta in groups.items():
        ids = stratified_sample_ids(table, meta["store_id"], meta["dept_id"])
        full_prof = _profile_of_ids(
            table[(table["store_id"] == meta["store_id"]) & (table["dept_id"] == meta["dept_id"])],
            table[(table["store_id"] == meta["store_id"]) & (table["dept_id"] == meta["dept_id"])]["id"].tolist(),
        )
        samp_prof = _profile_of_ids(table, ids)
        selection["datasets"][label] = {
            **meta,
            "n_full": full_prof["n"],
            "n_sampled": samp_prof["n"],
            "full_profile": full_prof,
            "sample_profile": samp_prof,
            # fidelity check: sampled medians must track the full group
            "fidelity_ok": (
                abs(full_prof["med_zero"] - samp_prof["med_zero"]) < 0.03
                and abs(full_prof["med_dow"] - samp_prof["med_dow"]) < 0.03
            ),
            "ids": sorted(ids),
        }
    return selection


def main() -> None:
    table = pd.read_parquet(config.ARTIFACTS_DIR / "signal_table.parquet")
    selection = build_selection(table)

    out_dir = config.ARTIFACTS_DIR / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selection.json"
    out_path.write_text(json.dumps(selection, indent=2))

    print(f"wrote {out_path}\n")
    degraded = []
    for label, d in selection["datasets"].items():
        fp, sp = d["full_profile"], d["sample_profile"]
        print(f"{label:16} {d['store_id']:>5} x {d['dept_id']:<12} "
              f"n_full={d['n_full']:>3} sampled={d['n_sampled']:>3} "
              f"med_zero {fp['med_zero']:.3f}->{sp['med_zero']:.3f} "
              f"med_dow {fp['med_dow']:.3f}->{sp['med_dow']:.3f} "
              f"fidelity_ok={d['fidelity_ok']}  [{d['branch']}]")
        if "RELAXED" in d["rule"] or "DEGRADED" in d["rule"]:
            degraded.append((label, d["rule"]))

    if degraded:
        print("\n!! degraded picks (a constraint was relaxed -- stated honestly):")
        for label, rule in degraded:
            print(f"   {label}: {rule}")

    # held-out integrity: A/B/C/D must be pairwise disjoint (cell exclusion guarantees it)
    ids = {k: set(v["ids"]) for k, v in selection["datasets"].items()}
    d_ids = ids["D_heldout"]
    overlap = any(d_ids & ids[k] for k in ["A_dense", "B_intermittent", "C_sparse"])
    print(f"\nD overlaps A/B/C series: {overlap} (must be False)")


if __name__ == "__main__":
    main()
