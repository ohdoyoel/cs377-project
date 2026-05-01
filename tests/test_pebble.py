"""Smoke tests for the PEBBLE module.

Aim: <60s wall under uv. We intentionally use tiny shapes (small buffer,
short SAC.learn smoke) to keep this fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from billiards.env import Billiards4BallEnv
from billiards.pebble import PEBBLEAgent, PEBBLEBuffer, RMEnsemble
from billiards.reward_model.network import RewardMLP


def _fill_buffer(buf: PEBBLEBuffer, n_steps: int, seed: int = 0) -> None:
    """Push ``n_steps`` random transitions through the buffer."""
    env = Billiards4BallEnv(t_max=4.0)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        action = np.array([
            rng.uniform(0.0, 2 * np.pi),
            rng.uniform(0.2, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        ], dtype=np.float32)
        next_obs, reward, term, trunc, info = env.step(action)
        buf.add(
            obs=np.asarray(obs, dtype=np.float32),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            action=action,
            reward=np.array([reward], dtype=np.float32),
            done=np.array([term or trunc], dtype=np.float32),
            infos=[info],
        )
        obs, _ = env.reset()


def test_buffer_relabel_changes_rewards() -> None:
    env = Billiards4BallEnv(t_max=4.0)
    buf = PEBBLEBuffer(
        buffer_size=32,
        observation_space=env.observation_space,
        action_space=env.action_space,
        n_envs=1,
    )
    _fill_buffer(buf, n_steps=20, seed=0)
    assert buf.num_transitions() == 20

    # Use a randomly-init RewardMLP -- outputs should be ~0 noise, very different
    # from the env's integer 0/1 rewards. Relabel must change the rewards array.
    torch.manual_seed(7)
    rm = RewardMLP(state_dim=28, action_dim=4, hidden=64)
    # bias the head so predictions are noticeably non-zero
    with torch.no_grad():
        rm.head.bias.fill_(0.5)

    before = buf.rewards.copy()
    delta = buf.relabel(rm)
    after = buf.rewards.copy()

    assert delta["n"] == 20
    assert delta["mean_abs_delta"] > 0.0
    # Confirm at least some entries changed.
    assert not np.allclose(before[:20], after[:20])


def test_sample_pairs_uniform_disjoint_in_range() -> None:
    env = Billiards4BallEnv(t_max=4.0)
    buf = PEBBLEBuffer(
        buffer_size=64, observation_space=env.observation_space,
        action_space=env.action_space, n_envs=1,
    )
    _fill_buffer(buf, n_steps=16, seed=1)
    pairs = buf.sample_pairs(n_pairs=8, strategy="uniform",
                             rng=np.random.default_rng(0))
    assert len(pairs) == 8
    valid_max = buf.num_transitions()
    for (a, b) in pairs:
        assert a != b
        pos_a, env_a = a
        pos_b, env_b = b
        assert 0 <= pos_a < valid_max
        assert 0 <= pos_b < valid_max
        assert env_a == 0 and env_b == 0


def test_sample_pairs_disagreement_requires_ensemble() -> None:
    env = Billiards4BallEnv(t_max=4.0)
    buf = PEBBLEBuffer(
        buffer_size=64, observation_space=env.observation_space,
        action_space=env.action_space, n_envs=1,
    )
    _fill_buffer(buf, n_steps=16, seed=2)

    # No ensemble -> falls back to uniform with a warning.
    with pytest.warns(UserWarning):
        pairs = buf.sample_pairs(
            n_pairs=4, strategy="disagreement", ensemble=None,
            rng=np.random.default_rng(0),
        )
    assert len(pairs) == 4

    # 1-member ensemble also falls back.
    one = RMEnsemble(n_models=1, seed_base=3)
    with pytest.warns(UserWarning):
        pairs = buf.sample_pairs(
            n_pairs=4, strategy="disagreement", ensemble=one,
            rng=np.random.default_rng(0),
        )
    assert len(pairs) == 4

    # Real ensemble -> no warning, 4 pairs.
    ens = RMEnsemble(n_models=2, seed_base=5)
    pairs = buf.sample_pairs(
        n_pairs=4, strategy="disagreement", ensemble=ens,
        rng=np.random.default_rng(0),
    )
    assert len(pairs) == 4


def test_ensemble_forward_returns_mean_std() -> None:
    ens = RMEnsemble(n_models=2, seed_base=11)
    s = torch.zeros(8, 28, dtype=torch.float32)
    a = torch.zeros(8, 4, dtype=torch.float32)
    mean, std = ens(s, a)
    assert mean.shape == (8,)
    assert std.shape == (8,)
    # Different random inits -> non-zero disagreement somewhere.
    assert float(std.abs().sum().item()) > 0.0


def test_pebble_agent_smoke() -> None:
    """End-to-end: very small phases, expect no exceptions."""
    agent = PEBBLEAgent(
        env_factory=lambda: Billiards4BallEnv(t_max=4.0),
        total_steps=2048,
        query_phase_steps=512,
        queries_per_phase=4,
        seed=0,
        ensemble_train_epochs=2,
    )
    summary = agent.learn()
    assert summary["queries_used"] >= 4
    rows = agent.evaluate(n_episodes=4, seed_base=1234)
    assert len(rows) == 4
    for row in rows:
        assert {"score", "fouled", "cushion_hits", "rm_reward"}.issubset(row.keys())
