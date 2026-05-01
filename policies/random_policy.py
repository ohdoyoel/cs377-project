"""Uniform random policy over the 4-D cue action space."""

from __future__ import annotations

import math

import numpy as np


class RandomPolicy:
    """Sample (theta, power, a, b) uniformly with (a, b) on the unit disk."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        power = self.rng.uniform(0.0, 1.0)
        # Uniform on unit disk via inverse-CDF: r = sqrt(u), φ = 2π v.
        r = math.sqrt(self.rng.uniform(0.0, 1.0))
        phi = self.rng.uniform(0.0, 2.0 * math.pi)
        a = r * math.cos(phi)
        b = r * math.sin(phi)
        return np.array([theta, power, a, b], dtype=np.float32)
