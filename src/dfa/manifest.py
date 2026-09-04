"""Run provenance: what produced a results file (Phase 3 sub-plan sec 6b).

A results table with no provenance cannot be audited. Phase 5's sealed run is
only meaningful if the configuration that produced it is pinned to a commit, and
Phase 4's Report node has to surface the same payload. So every artifact this
project writes carries a manifest.

Deliberately cheap and dependency-free: it reads package metadata and shells out
to git once. Nothing here should ever fail a run -- an unavailable git SHA (a
tarball export, a CI checkout without history) degrades to "unknown" rather than
raising, because provenance is evidence about a run, not a precondition for one.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as _md
import platform
import subprocess
from typing import Iterable

from . import config

# Tracked because each can change a number without changing a line of our code.
TRACKED_PACKAGES: tuple[str, ...] = (
    "pandas", "numpy", "pyarrow", "lightgbm", "scikit-learn",
)

# The project-wide RNG seed. One constant so every fit, sample and shuffle in
# the pipeline is traceable to a single documented value.
SEED: int = 20240229
# Fixed thread count: LightGBM's histogram construction is thread-count-dependent,
# so leaving this to the machine makes the *same code on the same data* return
# different numbers on a different host. Pinned low enough to be reproducible on
# any dev box; raise it only with the knowledge that results shift.
NUM_THREADS: int = 4


def _git_sha() -> str:
    """Short SHA of HEAD, plus '-dirty' when the tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(config.REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(config.REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        # provenance is evidence, not a precondition -- never fail a run for it
        return "unknown"


def _versions() -> dict[str, str]:
    out = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def hash_ids(ids: Iterable[str]) -> str:
    """Order-independent digest of a series-id list.

    Identifies *which series* a result covers without inlining hundreds of ids
    into every artifact. Sorted first so a reordered list hashes identically --
    the set is what matters, not the order it arrived in.
    """
    joined = "\n".join(sorted(str(i) for i in ids))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def hash_columns(columns: Iterable[str]) -> str:
    """Digest of a feature-column manifest, ORDER-SENSITIVE.

    Unlike `hash_ids`, order is preserved: a design matrix whose columns are
    permuted is a different design matrix, and the point of this hash is to make
    "the feature set is frozen" (Phase 3 sub-plan sec 4.4) mechanically checkable
    rather than a claim in prose.
    """
    return hashlib.sha256("\n".join(columns).encode()).hexdigest()[:16]


def run_manifest(**extra) -> dict:
    """Provenance stamp for a results artifact. Extra keys are merged in."""
    return {
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _versions(),
        "seed": SEED,
        "num_threads": NUM_THREADS,
        **extra,
    }
