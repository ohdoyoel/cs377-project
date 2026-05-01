"""Smoke + identity tests for the preference data pipeline.

Designed to run *without* an Anthropic API key — the AI labeler is
imported but its network-touching code path is never invoked here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from billiards.preference import (
    PreferencePair,
    append_pair,
    generate_random_pairs,
    load_pairs,
    summary_stats,
)
from billiards.preference.labeler_heuristic import (
    heuristic_score,
    label_pair_heuristic,
)
from billiards.render.dual_replay import render_dual_html


def _make_demo_pair(pair_id: str = "test-pair-001") -> PreferencePair:
    return PreferencePair(
        pair_id=pair_id,
        initial_state=[0.0] * 28,
        action_A=[1.5707, 0.5, 0.0, 0.0],
        action_B=[3.1416, 0.7, 0.1, -0.2],
        result_A={
            "score": 0, "fouled": False, "cushion_hits": 2,
            "duration": 3.5, "trajectory_len": 40, "n_events": 5,
            "event_types": ["cue_hit_red", "cue_hit_cushion"],
        },
        result_B={
            "score": 1, "fouled": False, "cushion_hits": 1,
            "duration": 2.1, "trajectory_len": 20, "n_events": 4,
            "event_types": ["cue_hit_red", "cue_hit_red", "cue_hit_cushion"],
        },
        preference=None,
    )


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


def test_preference_pair_jsonl_roundtrip(tmp_path: Path) -> None:
    pair = _make_demo_pair()
    target = tmp_path / "pairs.jsonl"

    append_pair(pair, path=target)
    append_pair(pair, path=target)

    loaded = load_pairs(target)
    assert len(loaded) == 2
    assert loaded[0].to_dict() == pair.to_dict()
    assert loaded[1].to_dict() == pair.to_dict()

    # JSON content sanity: each line parses to the same dict.
    raw_lines = [
        json.loads(line) for line in target.read_text().splitlines() if line.strip()
    ]
    assert raw_lines == [pair.to_dict(), pair.to_dict()]


def test_load_pairs_missing_file(tmp_path: Path) -> None:
    assert load_pairs(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# generate_random_pairs
# ---------------------------------------------------------------------------


def test_generate_random_pairs_shapes() -> None:
    from billiards.env import Billiards4BallEnv

    pairs, trajs = generate_random_pairs(
        env_factory_or_env=lambda: Billiards4BallEnv(),
        n_pairs=3,
        seed=100,
    )
    assert len(pairs) == 3
    assert len(trajs) == 3
    for pair, (tA, tB) in zip(pairs, trajs):
        assert isinstance(pair, PreferencePair)
        assert len(pair.initial_state) == 28
        assert len(pair.action_A) == 4
        assert len(pair.action_B) == 4
        # Trajectories are non-empty (post-reset there's at least the t0 snapshot).
        assert len(tA) > 0 and len(tB) > 0
        # Each frame is (t, (4,7) ndarray).
        t0, arr0 = tA[0]
        assert isinstance(t0, float)
        assert np.asarray(arr0).shape == (4, 7)


# ---------------------------------------------------------------------------
# Heuristic labeler
# ---------------------------------------------------------------------------


def test_heuristic_score_components() -> None:
    # Scoring shot, no foul, 2 cushions, 2 red contacts:
    # +5 (score) + 0.4*2 (cushion) + 0.6*2 (red) - 0.05*0 (duration) = 7.0
    s = heuristic_score({
        "score": 1, "fouled": False, "cushion_hits": 2, "duration": 0.0,
        "event_types": ["cue_hit_red", "cue_hit_red", "cue_hit_cushion", "cue_hit_cushion"],
    })
    assert abs(s - (5.0 + 0.4 * 2 + 0.6 * 2)) < 1e-9

    # Foul: -5, with one cushion + one red contact + one opp contact
    # = -5 + 0.4*1 + 0.6*1 - 0.3*1 - 0.05*0 = -4.3
    s_foul = heuristic_score({
        "score": 0, "fouled": True, "cushion_hits": 1, "duration": 0.0,
        "event_types": ["cue_hit_red", "cue_hit_opp"],
    })
    assert abs(s_foul - (-5.0 + 0.4 + 0.6 - 0.3)) < 1e-9


def test_heuristic_label_is_deterministic() -> None:
    pair = _make_demo_pair()
    a = label_pair_heuristic(pair)
    b = label_pair_heuristic(pair)
    assert a.preference == b.preference == "B"
    assert a.labeler == "heuristic"
    assert a.label_meta["score_A"] == b.label_meta["score_A"]
    assert a.label_meta["score_B"] == b.label_meta["score_B"]


def test_heuristic_label_tie_band() -> None:
    pair = _make_demo_pair()
    # Force result_A and result_B to be effectively identical → tie.
    pair.result_B = dict(pair.result_A)
    out = label_pair_heuristic(pair)
    assert out.preference == "tie"


# ---------------------------------------------------------------------------
# Dual replay HTML — content sanity
# ---------------------------------------------------------------------------


def _fake_traj(seed: int, n: int = 16) -> list[tuple[float, np.ndarray]]:
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0.3, 1.2, size=(4, 2))
    v = rng.uniform(-0.4, 0.4, size=(4, 2))
    out = []
    for k in range(n):
        t = k * 0.04
        a = np.zeros((4, 7))
        a[:, 0:2] = p0 + v * t
        out.append((t, a))
    return out


def test_render_dual_html_contains_required_markers(tmp_path: Path) -> None:
    out = render_dual_html(
        pair_id="render-smoke-pair",
        traj_A=_fake_traj(0),
        traj_B=_fake_traj(1),
        spec={"width_m": 2.540, "height_m": 1.270, "ball_radius_m": 0.03275},
        descriptors_A={"score": 0, "cushion_hits": 1, "note": "panel A"},
        descriptors_B={"score": 1, "cushion_hits": 2, "note": "panel B"},
        save_path=tmp_path / "pair.html",
    )
    s = out.data if hasattr(out, "data") and isinstance(out.data, str) else str(out)
    assert "translate3d" in s
    assert s.count("data:image/png;base64,") >= 3
    assert "render-smoke-pair" in s
    # Persisted to disk too.
    assert (tmp_path / "pair.html").exists()
    on_disk = (tmp_path / "pair.html").read_text(encoding="utf-8")
    assert "render-smoke-pair" in on_disk


# ---------------------------------------------------------------------------
# AI labeler — import-only test (no API call)
# ---------------------------------------------------------------------------


def test_labeler_ai_imports_and_descriptor() -> None:
    # Import is safe (no network) — we only invoke pure-Python helpers.
    from billiards.preference import labeler_ai

    desc = labeler_ai.build_user_descriptor(_make_demo_pair())
    assert "Pair test-pair-001" in desc
    assert "score=0" in desc and "score=1" in desc
    assert "PREFERENCE" in desc

    # Parse helper should pull the structured tail line.
    assert labeler_ai._parse_preference("blah blah\nPREFERENCE: A\n") == "A"
    assert labeler_ai._parse_preference("foo\nPREFERENCE: tie") == "tie"
    assert labeler_ai._parse_preference("no tail line here") is None


# ---------------------------------------------------------------------------
# summary_stats
# ---------------------------------------------------------------------------


def test_summary_stats_basic() -> None:
    pair = _make_demo_pair()
    a = label_pair_heuristic(pair)
    stats = summary_stats([a, a])
    assert stats["n"] == 2
    assert stats["by_labeler"] == {"heuristic": 2}
    assert "B" in stats["by_preference"]
