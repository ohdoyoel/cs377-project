"""Deterministic geometric aim policy: cue points at a chosen red ball."""

from __future__ import annotations

import math

import numpy as np

from billiards.physics.state import BallRole, TableSpec


class GeometricAimPolicy:
    """Aim cue ball directly at a red target.

    Parameters
    ----------
    target : str
        'nearest_red'  → red whose center is closer to cue
        'farther_red'  → red whose center is farther from cue
        'red1'         → BallRole.RED_1 (foot spot)
        'red2'         → BallRole.RED_2 (center spot)
    power : float
        Shot strength in [0, 1].
    spin : tuple[float, float]
        (a, b) cue contact offset; a²+b² ≤ 1.
    avoid_opp : bool
        If True, perturb theta by +5° when the opponent ball lies near
        the aiming line within `opp_tolerance` ball radii.
    opp_tolerance : float
        Lateral distance threshold (in ball radii) for the avoidance trigger.
    """

    _PERTURB_RAD = math.radians(5.0)

    def __init__(
        self,
        target: str = "nearest_red",
        power: float = 0.55,
        spin: tuple[float, float] = (0.0, 0.0),
        avoid_opp: bool = True,
        opp_tolerance: float = 2.5,
    ) -> None:
        if target not in {"nearest_red", "farther_red", "red1", "red2"}:
            raise ValueError(f"unknown target: {target!r}")
        self.target = target
        self.power = float(power)
        self.spin = (float(spin[0]), float(spin[1]))
        self.avoid_opp = bool(avoid_opp)
        self.opp_tolerance = float(opp_tolerance)
        self._ball_radius = TableSpec().ball_radius

    def _select_red(
        self,
        cue_xy: np.ndarray,
        red1_xy: np.ndarray,
        red2_xy: np.ndarray,
    ) -> np.ndarray:
        if self.target == "red1":
            return red1_xy
        if self.target == "red2":
            return red2_xy
        d1 = float(np.linalg.norm(red1_xy - cue_xy))
        d2 = float(np.linalg.norm(red2_xy - cue_xy))
        if self.target == "nearest_red":
            return red1_xy if d1 <= d2 else red2_xy
        return red1_xy if d1 >= d2 else red2_xy  # farther_red

    def _opp_blocks(
        self,
        cue_xy: np.ndarray,
        red_xy: np.ndarray,
        opp_xy: np.ndarray,
    ) -> bool:
        """True if opponent center lies within tolerance of segment cue→red.

        Lateral distance is the perpendicular distance from opp to the
        infinite line, only counted when its projection lies between the
        endpoints.
        """
        seg = red_xy - cue_xy
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            return False
        u = seg / seg_len
        rel = opp_xy - cue_xy
        proj = float(np.dot(rel, u))
        if proj <= 0.0 or proj >= seg_len:
            return False
        lateral = float(np.linalg.norm(rel - proj * u))
        return lateral < self.opp_tolerance * self._ball_radius

    def act(self, obs: np.ndarray) -> np.ndarray:
        balls = np.asarray(obs, dtype=np.float64).reshape(4, 7)
        cue_xy = balls[int(BallRole.CUE_WHITE), :2]
        opp_xy = balls[int(BallRole.CUE_YELLOW), :2]
        red1_xy = balls[int(BallRole.RED_1), :2]
        red2_xy = balls[int(BallRole.RED_2), :2]

        target_xy = self._select_red(cue_xy, red1_xy, red2_xy)
        dy = target_xy[1] - cue_xy[1]
        dx = target_xy[0] - cue_xy[0]
        theta = math.atan2(dy, dx)

        if self.avoid_opp and self._opp_blocks(cue_xy, target_xy, opp_xy):
            theta += self._PERTURB_RAD

        theta = theta % (2.0 * math.pi)
        a, b = self.spin
        return np.array([theta, self.power, a, b], dtype=np.float32)
