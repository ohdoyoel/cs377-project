"""Unit tests for the baseline policies."""

from __future__ import annotations

import math

import numpy as np
import pytest

from billiards.env import Billiards4BallEnv
from billiards.physics.state import BallRole
from policies.geometric_aim import GeometricAimPolicy
from policies.random_policy import RandomPolicy


def test_random_action_in_bounds() -> None:
    policy = RandomPolicy(seed=0)
    obs = np.zeros(28, dtype=np.float32)
    actions = np.stack([policy.act(obs) for _ in range(1000)])
    theta = actions[:, 0]
    power = actions[:, 1]
    a = actions[:, 2]
    b = actions[:, 3]
    assert np.all(theta >= 0.0) and np.all(theta <= 2.0 * math.pi)
    assert np.all(power >= 0.0) and np.all(power <= 1.0)
    assert np.all(a * a + b * b <= 1.0001)


def test_geometric_aims_at_red() -> None:
    env = Billiards4BallEnv()
    obs, _ = env.reset(seed=0)
    policy = GeometricAimPolicy(target="red1", power=0.4, avoid_opp=False)
    action = policy.act(obs)
    theta = float(action[0])

    balls = obs.reshape(4, 7)
    cue = balls[int(BallRole.CUE_WHITE), :2]
    red1 = balls[int(BallRole.RED_1), :2]
    expected = math.atan2(red1[1] - cue[1], red1[0] - cue[0]) % (2.0 * math.pi)
    diff = abs(((theta - expected) + math.pi) % (2.0 * math.pi) - math.pi)
    assert diff < 0.05, f"theta {theta} differs from {expected} by {diff} rad"


def test_random_rollout_runs() -> None:
    env = Billiards4BallEnv()
    policy = RandomPolicy(seed=123)
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        action = policy.act(obs)
        _, _, terminated, _, info = env.step(action)
        assert terminated is True
        assert "score" in info and "fouled" in info


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
