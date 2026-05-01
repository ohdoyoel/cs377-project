"""Mixed-reward wrapper: r = alpha * env_score + (1 - alpha) * normalized_rm.

Combines the underlying env's true integer score with a learned Bradley-Terry
reward, normalized to roughly unit scale via a precomputed (mu, sigma).

The normalization stats come from a one-shot warm-up over 1000 random
(state, action) pairs and are persisted to ``experiments/rm_normalization.json``
so every alpha-sweep run uses the same scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from billiards.wrappers.reward_model_env import _load_reward_model


_RM_CLIP_LO = -2.0
_RM_CLIP_HI = 4.0


class MixedRewardEnv(gym.Wrapper):
    """Mix env reward with normalized reward-model output.

    Parameters
    ----------
    env : gym.Env
        Underlying single-shot env (typically ``Billiards4BallEnv``).
    alpha : float
        Weight on the env score. ``r = alpha*env_r + (1-alpha)*rm_norm``.
    model_path : str | Path
        Path to the trained reward MLP saved by Phase D.
    norm_path : str | Path
        Path to the JSON with ``{"mu": ..., "sigma": ...}`` produced by
        ``experiments/normalize_rm.py``.
    device : str
        torch device for the reward model (default ``cpu``).
    """

    def __init__(
        self,
        env: gym.Env,
        alpha: float,
        model_path: str | Path = "models/reward_model.pt",
        norm_path: str | Path = "experiments/rm_normalization.json",
        device: str = "cpu",
    ) -> None:
        super().__init__(env)
        self.alpha = float(alpha)
        self.device = str(device)
        self.model = _load_reward_model(model_path, device=self.device)

        norm_path = Path(norm_path)
        if not norm_path.exists():
            raise FileNotFoundError(
                f"reward-model normalization stats not found at {norm_path}; "
                f"run experiments/normalize_rm.py first."
            )
        with norm_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        self.mu = float(stats["mu"])
        self.sigma = float(stats["sigma"])
        if self.sigma <= 0.0:
            raise ValueError(f"non-positive sigma in {norm_path}: {self.sigma}")

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
            raise RuntimeError("MixedRewardEnv.step called before reset")
        obs_init = self._last_obs
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)

        obs, env_r, term, trunc, info = self.env.step(action)

        with torch.no_grad():
            s_t = torch.as_tensor(obs_init, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            a_t = torch.as_tensor(action_arr, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            rm_raw = float(self.model(s_t, a_t).squeeze().item())

        rm_norm = max(_RM_CLIP_LO, min(_RM_CLIP_HI, (rm_raw - self.mu) / self.sigma))
        r_mixed = self.alpha * float(env_r) + (1.0 - self.alpha) * rm_norm

        new_info = dict(info) if info is not None else {}
        new_info["env_reward"] = float(env_r)
        new_info["rm_reward_raw"] = rm_raw
        new_info["rm_reward_norm"] = rm_norm
        new_info["mixed_reward"] = float(r_mixed)

        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        return obs, float(r_mixed), term, trunc, new_info
