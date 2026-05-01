"""Tests for billiards.wrappers.MixedRewardEnv.

Skips gracefully when either the reward model or the normalization stats
are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from billiards.env import Billiards4BallEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "reward_model.pt"
NORM_PATH = REPO_ROOT / "experiments" / "rm_normalization.json"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and NORM_PATH.exists()),
    reason=(
        f"need reward model at {MODEL_PATH} and normalization at {NORM_PATH} "
        f"(run experiments/normalize_rm.py first)"
    ),
)


def _make(alpha: float):
    from billiards.wrappers import MixedRewardEnv

    return MixedRewardEnv(
        Billiards4BallEnv(),
        alpha=alpha,
        model_path=str(MODEL_PATH),
        norm_path=str(NORM_PATH),
    )


def _step_with(env, seed: int, action: np.ndarray):
    env.reset(seed=seed)
    return env.step(action)


def test_alpha_one_equals_env_score() -> None:
    env = _make(alpha=1.0)
    action = np.array([1.5, 0.6, 0.0, 0.0], dtype=np.float32)
    _, r, _, _, info = _step_with(env, seed=42, action=action)
    assert r == pytest.approx(info["env_reward"], abs=1e-9)


def test_alpha_zero_equals_rm_norm() -> None:
    env = _make(alpha=0.0)
    action = np.array([2.0, 0.5, 0.1, -0.1], dtype=np.float32)
    _, r, _, _, info = _step_with(env, seed=7, action=action)
    assert r == pytest.approx(info["rm_reward_norm"], abs=1e-9)


def test_info_keys_present() -> None:
    env = _make(alpha=0.5)
    action = np.array([0.5, 0.4, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info = _step_with(env, seed=1, action=action)
    for key in ("env_reward", "rm_reward_raw", "rm_reward_norm", "mixed_reward"):
        assert key in info, f"missing info key: {key}"


def test_rm_norm_is_clipped() -> None:
    """The clip to [-2, 4] must hold for whatever raw value falls out."""
    env = _make(alpha=0.5)
    actions = [
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([2.0 * np.pi - 1e-3, 1.0, 1.0, 1.0], dtype=np.float32),
        np.array([np.pi, 0.5, -0.5, 0.5], dtype=np.float32),
        np.array([3.0, 0.9, 0.9, -0.9], dtype=np.float32),
    ]
    for i, a in enumerate(actions):
        _, _, _, _, info = _step_with(env, seed=100 + i, action=a)
        assert -2.0 - 1e-9 <= info["rm_reward_norm"] <= 4.0 + 1e-9


def test_mixed_reward_formula() -> None:
    """r = alpha * env + (1-alpha) * rm_norm, exactly."""
    alpha = 0.3
    env = _make(alpha=alpha)
    action = np.array([1.0, 0.7, 0.2, -0.2], dtype=np.float32)
    _, r, _, _, info = _step_with(env, seed=99, action=action)
    expected = alpha * info["env_reward"] + (1.0 - alpha) * info["rm_reward_norm"]
    assert r == pytest.approx(expected, abs=1e-9)


def test_normalization_stats_loaded() -> None:
    env = _make(alpha=0.5)
    with NORM_PATH.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    assert env.mu == pytest.approx(stats["mu"], abs=1e-12)
    assert env.sigma == pytest.approx(stats["sigma"], abs=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
