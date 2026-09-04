"""Increment 0 tests: run provenance."""

from __future__ import annotations

from dfa import manifest as mf


def test_manifest_has_required_provenance_keys():
    m = mf.run_manifest()
    for key in ("git_sha", "python", "platform", "packages", "seed", "num_threads"):
        assert key in m, key
    assert m["seed"] == mf.SEED
    for pkg in mf.TRACKED_PACKAGES:
        assert pkg in m["packages"]


def test_manifest_merges_extra_keys():
    m = mf.run_manifest(dataset="A", n_series=250)
    assert m["dataset"] == "A" and m["n_series"] == 250


def test_manifest_never_raises_without_git(monkeypatch):
    """Provenance is evidence about a run, not a precondition for one."""
    monkeypatch.setattr(mf.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert mf.run_manifest()["git_sha"] == "unknown"


def test_hash_ids_is_order_independent_but_set_sensitive():
    assert mf.hash_ids(["b", "a"]) == mf.hash_ids(["a", "b"])
    assert mf.hash_ids(["a", "b"]) != mf.hash_ids(["a", "c"])


def test_hash_columns_is_order_SENSITIVE():
    """A permuted design matrix is a different design matrix."""
    assert mf.hash_columns(["a", "b"]) != mf.hash_columns(["b", "a"])
    assert mf.hash_columns(["a", "b"]) == mf.hash_columns(["a", "b"])
