"""Tests for billiards.inning_env.Billiards4BallInningEnv."""

from __future__ import annotations

import math

import numpy as np
import pytest

from billiards.inning_env import Billiards4BallInningEnv
from billiards.physics import BallRole, TableSpec


# ---------------------------------------------------------------- helpers

# Earlier work showed this CueAction scores from the standard initial
# layout (cue_id=CUE_WHITE). theta=π*0.6 aims roughly toward the reds at
# (W/2, H/2) and (W/4, H/2) from the head spot at (3W/4, H/2).
SCORING_ACTION = np.array([math.pi * 0.6, 1.0, 0.6, 0.3], dtype=np.float32)
MISS_ACTION = np.array([math.pi, 0.001, 0.0, 0.0], dtype=np.float32)


def _aim_at(env: Billiards4BallInningEnv, target_xy: tuple[float, float],
            power: float = 0.5) -> np.ndarray:
    """Build an action whose theta points the cue ball at ``target_xy``."""
    state = env._state
    assert state is not None
    cue = state.cue_ball
    dx = target_xy[0] - cue.x
    dy = target_xy[1] - cue.y
    theta = math.atan2(dy, dx) % (2.0 * math.pi)
    return np.array([theta, power, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------- tests


def test_reset_obs_shape():
    env = Billiards4BallInningEnv()
    obs, info = env.reset()
    assert obs.shape == (28,)
    assert obs.dtype == np.float32
    assert "spec" in info
    assert info["shot_index"] == 0
    assert info["cumulative_score"] == 0


def test_score_then_miss():
    env = Billiards4BallInningEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(SCORING_ACTION)
    assert info["score"] == 1, f"SCORING_ACTION did not score: {info}"
    assert reward == 1.0
    assert terminated is False, "scoring shot should not terminate the inning"
    assert truncated is False
    assert info["shot_index"] == 1
    assert info["cumulative_score"] == 1

    # Now a tiny power "miss" — barely moves cue ball, no reds touched.
    obs, reward, terminated, truncated, info = env.step(MISS_ACTION)
    assert info["score"] == 0
    assert reward == 0.0
    assert terminated is True
    assert info["cumulative_score"] == 1


def test_foul_action():
    env = Billiards4BallInningEnv()
    env.reset()
    spec = env._spec
    # Opponent yellow sits at (3W/4, H/2 - 0.1825) when cue_id == CUE_WHITE.
    yellow_xy = (3 * spec.width / 4, spec.height / 2 - 0.1825)
    action = _aim_at(env, yellow_xy, power=0.5)
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["fouled"] is True, f"expected foul, got {info}"
    assert terminated is True
    assert reward == 0.0


def test_max_shots_truncates(monkeypatch):
    env = Billiards4BallInningEnv(max_shots=2)
    env.reset()

    # Force every shot to score, regardless of physics: stub simulate_shot
    # so we can drive the truncation path deterministically.
    from billiards import inning_env as ie_module

    fake_traj = [(0.0, env._state.to_array()), (1.0, env._state.to_array())]

    def fake_simulate_shot(state, action, t_max=12.0):
        state.t = 1.0
        return {
            "trajectory": list(fake_traj),
            "events": [],
            "score": 1,
            "fouled": False,
            "cushion_hits": 0,
            "duration": 1.0,
        }

    monkeypatch.setattr(ie_module, "simulate_shot", fake_simulate_shot)

    _, _, terminated, truncated, info = env.step(SCORING_ACTION)
    assert terminated is False and truncated is False
    assert info["shot_index"] == 1

    _, _, terminated, truncated, info = env.step(SCORING_ACTION)
    assert terminated is False
    assert truncated is True
    assert info["shot_index"] == 2
    assert info["cumulative_score"] == 2


def test_full_trajectory_continuous():
    env = Billiards4BallInningEnv()
    env.reset()
    # Two shots — even if the second one terminates the inning, we should
    # still have stitched both into the global timeline.
    env.step(SCORING_ACTION)
    env.step(MISS_ACTION)

    full = env.full_trajectory
    assert len(full) >= 2
    times = [t for t, _ in full]
    assert times[0] == pytest.approx(0.0, abs=1e-9)
    for a, b in zip(times, times[1:]):
        assert b >= a - 1e-9, f"trajectory timestamps must be monotonic: {a} > {b}"
    # Strictly increasing across shot boundaries (loose: at least once).
    assert times[-1] > times[0]
