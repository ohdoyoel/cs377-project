"""Reward-model wrapper that overrides env_reward with a learned scalar.

The wrapped env still drives all physics; we only swap the reward signal
returned by ``step`` to the value of a Bradley-Terry MLP evaluated on the
*pre-shot* state and the action that was executed. The original integer
score is preserved in ``info['env_reward']``.

The wrapper does NOT modify the underlying env's observation / action
spaces, episode termination, or info dict (besides adding ``env_reward``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch


def _load_reward_model(model_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """Load the trained Bradley-Terry reward MLP saved by Phase D.

    Phase D's training script is expected to ``torch.save`` either the
    full model (``torch.nn.Module``) or a dict with key ``"model"``.
    Both shapes are accepted here; if a state_dict is supplied, this
    fails fast — callers must persist a full model so this wrapper
    needn't know the architecture.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"reward model not found at {path}; train Phase D first "
            f"(scripts/train_reward_model.py)"
        )
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
    else:
        model = obj
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"loaded object at {path} is not a torch.nn.Module "
            f"(got {type(model).__name__}); Phase D must persist the full model"
        )
    model.to(device)
    model.eval()
    return model


class RewardModelEnv(gym.Wrapper):
    """Replace env reward with the reward model's scalar value.

    Parameters
    ----------
    env : gym.Env
        Underlying env (typically ``Billiards4BallEnv``).
    model_path : str | Path
        Path to the reward model (default ``models/reward_model.pt``).
    device : str
        torch device for inference.
    """

    def __init__(
        self,
        env: gym.Env,
        model_path: str | Path = "models/reward_model.pt",
        device: str = "cpu",
    ) -> None:
        super().__init__(env)
        self.device = str(device)
        self.model = _load_reward_model(model_path, device=self.device)
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("RewardModelEnv.step called before reset")
        # Snapshot pre-shot state + action used for reward inference.
        state_for_reward = self._last_obs
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)

        obs, env_reward, terminated, truncated, info = self.env.step(action)

        with torch.no_grad():
            s_t = torch.as_tensor(state_for_reward, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            a_t = torch.as_tensor(action_arr, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            r_pred = self.model(s_t, a_t)
            r_value = float(r_pred.squeeze().item())

        new_info = dict(info) if info is not None else {}
        new_info["env_reward"] = float(env_reward)
        new_info["reward_model"] = r_value

        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        return obs, r_value, terminated, truncated, new_info
