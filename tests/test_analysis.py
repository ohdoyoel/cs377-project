"""Sanity tests for the Phase F analysis pipeline.

These tests verify that ``experiments/results/summary.parquet`` is loadable
and has the expected schema once the runner has finished. They also poke
at the per-run artifact layout (``experiments/runs/<id>/...``).

The runner (task #1) writes those files; if it has not finished yet we
``skip`` rather than fail so the suite stays green during scaffolding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _REPO_ROOT / "experiments" / "results"
_RUNS = _REPO_ROOT / "experiments" / "runs"
_SUMMARY = _RESULTS / "summary.parquet"

_REQUIRED_COLS = {"alpha", "seed", "true_score_rate", "rm_score_norm", "hacking_gap"}
_OPTIONAL_AGG_COLS = {"foul_rate", "mean_cushions", "run_id"}
_ATTR_COMPONENTS = {
    "score_term",
    "foul_term",
    "cushion_term",
    "red_contact_term",
    "opp_contact_term",
    "position_term",
    "duration_term",
}


def _summary_or_skip() -> pd.DataFrame:
    if not _SUMMARY.exists():
        pytest.skip(f"runner has not produced {_SUMMARY} yet")
    return pd.read_parquet(_SUMMARY)


def test_summary_loadable() -> None:
    df = _summary_or_skip()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "summary.parquet is empty"


def test_summary_has_required_columns() -> None:
    df = _summary_or_skip()
    missing = _REQUIRED_COLS - set(df.columns)
    assert not missing, f"summary.parquet missing required columns: {sorted(missing)}"


def test_summary_alpha_range() -> None:
    df = _summary_or_skip()
    alphas = set(df["alpha"].unique())
    assert all(0.0 <= float(a) <= 1.0 for a in alphas), f"α out of [0,1]: {alphas}"
    # Need at least the endpoints to compare RM-only vs env-only.
    assert {0.0, 1.0}.issubset(alphas), f"α endpoints missing: {sorted(alphas)}"


def test_summary_seed_count_per_alpha() -> None:
    df = _summary_or_skip()
    counts = df.groupby("alpha")["seed"].nunique()
    assert (counts >= 1).all(), f"some α have zero seeds: {counts.to_dict()}"


def test_runs_dir_has_artifacts_when_summary_present() -> None:
    if not _SUMMARY.exists():
        pytest.skip(f"runner has not produced {_SUMMARY} yet")
    assert _RUNS.exists(), f"{_RUNS} missing although summary parquet exists"
    run_dirs = [p for p in _RUNS.iterdir() if p.is_dir()]
    assert run_dirs, "no per-run directories found under experiments/runs/"


def test_per_run_artifact_shapes() -> None:
    if not _SUMMARY.exists():
        pytest.skip(f"runner has not produced {_SUMMARY} yet")
    if not _RUNS.exists():
        pytest.skip(f"{_RUNS} missing")
    saw_eval = saw_attr = saw_curve = False
    for run_dir in _RUNS.iterdir():
        if not run_dir.is_dir():
            continue
        ep = run_dir / "eval.parquet"
        ap = run_dir / "attribution.json"
        cp = run_dir / "training_curve.csv"
        if ep.exists():
            ev = pd.read_parquet(ep)
            assert len(ev) > 0, f"{ep} is empty"
            saw_eval = True
        if ap.exists():
            payload = json.loads(ap.read_text())
            # attribution.json should expose at least one heuristic component.
            assert isinstance(payload, dict), f"{ap} not a JSON object"
            keys = set(payload.keys())
            assert keys & _ATTR_COMPONENTS, (
                f"{ap} has no recognized component keys; got {sorted(keys)}"
            )
            saw_attr = True
        if cp.exists():
            cv = pd.read_csv(cp)
            assert "timesteps" in {c.lower() for c in cv.columns}, (
                f"{cp} missing 'timesteps' column"
            )
            saw_curve = True
    # Don't fail if a particular optional artifact is missing for *some* runs;
    # we only require that the runner produced at least one of each kind.
    assert saw_eval, "no eval.parquet found in any run dir"
    assert saw_attr, "no attribution.json found in any run dir"
    assert saw_curve, "no training_curve.csv found in any run dir"
