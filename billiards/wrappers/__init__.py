"""Gym wrappers for the Billiards4BallEnv (RLHF training).

Currently provides ``RewardModelEnv`` — substitute the env's true integer
score with a learned scalar reward from a Bradley-Terry preference model.
"""

from billiards.wrappers.mixed_reward_env import MixedRewardEnv
from billiards.wrappers.random_start_env import RandomStartInningEnv
from billiards.wrappers.reward_model_env import RewardModelEnv

__all__ = ["MixedRewardEnv", "RandomStartInningEnv", "RewardModelEnv"]
