"""Reward model package for Phase-D preference learning.

Exposes :class:`RewardMLP` and :func:`train` for Bradley-Terry training
on preference pairs produced by ``billiards.preference``.
"""

from billiards.reward_model.network import RewardMLP
from billiards.reward_model.train import make_dataset, train

__all__ = ["RewardMLP", "make_dataset", "train"]
