"""Smoke tests for the Bradley-Terry reward model."""

from __future__ import annotations

import torch

from billiards.preference.dataset import PreferencePair
from billiards.reward_model import RewardMLP, make_dataset, train


def _synth_pair(pid: int, pref: str) -> PreferencePair:
    return PreferencePair(
        pair_id=f"synth-{pid:04d}",
        initial_state=[0.01 * pid] * 28,
        action_A=[0.1, 0.5, 0.0, 0.0],
        action_B=[2.0, 0.7, 0.1, -0.1],
        result_A={"score": 0, "fouled": False, "cushion_hits": 0,
                  "duration": 1.0, "trajectory_len": 1, "n_events": 0,
                  "event_types": []},
        result_B={"score": 0, "fouled": False, "cushion_hits": 0,
                  "duration": 1.0, "trajectory_len": 1, "n_events": 0,
                  "event_types": []},
        preference=pref,
    )


def _make_synth_pairs(n: int = 10) -> list[PreferencePair]:
    out: list[PreferencePair] = []
    for k in range(n):
        if k % 3 == 0:
            pref = "tie"
        elif k % 2 == 0:
            pref = "A"
        else:
            pref = "B"
        out.append(_synth_pair(k, pref))
    return out


# ---------------------------------------------------------------------------
# Dataset round-trip
# ---------------------------------------------------------------------------


def test_make_dataset_shapes() -> None:
    pairs = _make_synth_pairs(10)
    s, aA, aB, y = make_dataset(pairs)
    assert s.shape == (10, 28)
    assert aA.shape == (10, 4)
    assert aB.shape == (10, 4)
    assert y.shape == (10,)
    # Labels match preferences.
    assert ((y == 1.0) | (y == 0.0) | (y == 0.5)).all()


def test_make_dataset_drops_unlabeled() -> None:
    pairs = _make_synth_pairs(5)
    pairs.append(_synth_pair(99, None))   # type: ignore[arg-type]
    s, *_ = make_dataset(pairs)
    assert s.shape[0] == 5  # dropped the None-pref pair


# ---------------------------------------------------------------------------
# Model forward / backward
# ---------------------------------------------------------------------------


def test_reward_mlp_batched_output_shape() -> None:
    model = RewardMLP(state_dim=28, action_dim=4, hidden=32)
    s = torch.randn(7, 28)
    a = torch.randn(7, 4)
    out = model(s, a)
    assert out.shape == (7, 1)


def test_reward_mlp_unbatched_promoted() -> None:
    model = RewardMLP(state_dim=28, action_dim=4, hidden=32)
    s = torch.randn(28)
    a = torch.randn(4)
    out = model(s, a)
    # Single sample is promoted to (1, 1).
    assert out.shape == (1, 1)


def test_one_step_forward_backward_runs() -> None:
    model = RewardMLP(state_dim=28, action_dim=4, hidden=16)
    s = torch.randn(4, 28); aA = torch.randn(4, 4); aB = torch.randn(4, 4)
    y = torch.tensor([1.0, 0.0, 0.5, 1.0])
    rA = model(s, aA).squeeze(-1)
    rB = model(s, aB).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(rA - rB, y)
    loss.backward()
    # All trainable parameters should now have gradients.
    for p in model.parameters():
        assert p.grad is not None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_train_is_deterministic_with_same_seed() -> None:
    pairs = _make_synth_pairs(20)
    m1, _ = train(pairs, epochs=1, lr=1e-3, batch_size=8,
                  val_frac=0.25, seed=42, verbose=False)
    m2, _ = train(pairs, epochs=1, lr=1e-3, batch_size=8,
                  val_frac=0.25, seed=42, verbose=False)
    sd1 = m1.state_dict(); sd2 = m2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k]), f"weights diverged at {k}"
