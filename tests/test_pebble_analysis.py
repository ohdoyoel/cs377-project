"""Sanity tests for Phase G PEBBLE analysis pipeline.

Mirrors ``tests/test_analysis.py`` but targets the Phase G summary parquet
written by the impl runner (task #1) at
``experiments/results/pebble_summary.parquet``. Tests skip if the runner
has not yet produced the artifact, so the suite stays green during
scaffolding.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _REPO_ROOT / "experiments" / "results"
_PEBBLE_SUMMARY = _RESULTS / "pebble_summary.parquet"

_REQUIRED_COLS = {
    "method",
    "seed",
    "run_id",
    "true_score_rate",
    "foul_rate",
    "mean_cushions",
    "queries_used",
    "rm_score_mean",
    "last_relabel_delta_mean",
}
# Methods the analyst will plot. The runner is allowed to emit a superset, but
# at least these must be present so figs 5-8 have something to render.
_REQUIRED_METHODS = {"sac_env", "sac_rm_frozen", "pebble_full", "pebble_disagree"}


def _summary_or_skip() -> pd.DataFrame:
    if not _PEBBLE_SUMMARY.exists():
        pytest.skip(f"Phase G runner has not produced {_PEBBLE_SUMMARY} yet")
    return pd.read_parquet(_PEBBLE_SUMMARY)


def test_pebble_summary_loadable_and_schema() -> None:
    df = _summary_or_skip()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "pebble_summary.parquet is empty"

    missing_required = _REQUIRED_COLS - set(df.columns)
    assert not missing_required, (
        f"pebble_summary.parquet missing required columns: "
        f"{sorted(missing_required)}; have {sorted(df.columns)}"
    )

    methods = set(df["method"].unique())
    missing_methods = _REQUIRED_METHODS - methods
    assert not missing_methods, (
        f"pebble_summary.parquet missing required methods: "
        f"{sorted(missing_methods)}; have {sorted(methods)}"
    )

    counts = df.groupby("method")["seed"].nunique()
    assert (counts >= 1).all(), (
        f"some methods have zero seeds: {counts.to_dict()}"
    )

    ts = df["true_score_rate"].astype(float)
    assert ts.notna().all(), "true_score_rate has NaNs"
    assert ((ts >= 0.0) & (ts <= 100.0)).all(), (
        f"true_score_rate out of [0,100]: range=[{ts.min()}, {ts.max()}]"
    )
