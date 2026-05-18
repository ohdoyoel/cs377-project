"""Multi-shot inning environment for the 4-ball carom simulator.

Single-shot ``Billiards4BallEnv`` terminates after one cue impulse. This
wrapper keeps the table state alive across shots so a policy plays an
*inning* — a streak of shots that ends on a miss, foul, or shot-cap.

Termination semantics:
    terminated:  the inning ended due to game rules (no score or foul)
    truncated:   the per-inning shot budget was exhausted
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np

from .physics import (
    BallRole,
    CueAction,
    TableSpec,
    TableState,
    simulate_shot,
)

OBS_DIM = 4 * 7  # 28


def _spec_to_dict(spec: TableSpec) -> dict[str, float]:
    return asdict(spec)


def _project_action(a: np.ndarray) -> CueAction:
    """Same projection as Billiards4BallEnv: wrap theta, clip power,
    radial-clip (a, b) into the unit disk."""
    theta = float(a[0]) % (2.0 * np.pi)
    power = float(np.clip(a[1], 0.0, 1.0))
    ax = float(np.clip(a[2], -1.0, 1.0))
    ay = float(np.clip(a[3], -1.0, 1.0))
    r = np.hypot(ax, ay)
    if r > 1.0:
        ax /= r
        ay /= r
    return CueAction(theta=theta, power=power, a=ax, b=ay)


class Billiards4BallInningEnv(gym.Env):
    """Korean 4-ball carom: continue shooting until a miss / foul / cap."""

    metadata = {"render_modes": ["trajectory"]}

    def __init__(
        self,
        spec: TableSpec | None = None,
        cue_id: int = int(BallRole.CUE_WHITE),
        t_max: float = 12.0,
        max_shots: int = 50,
        continue_on_miss: bool = False,
        foul_penalty: float = 0.1,
        ignore_opponent: bool = False,
    ) -> None:
        super().__init__()
        self._spec = spec or TableSpec()
        self._cue_id = cue_id
        self._t_max = t_max
        self._max_shots = int(max_shots)
        # When True, the episode never terminates on miss/foul; the policy
        # keeps shooting from whatever ball configuration the previous shot
        # produced until ``max_shots`` is reached. Used to expose the policy
        # to a wide distribution of in-play states during training.
        self._continue_on_miss = bool(continue_on_miss)
        self._foul_penalty = float(foul_penalty)
        # Curriculum stage 1: pretend the opponent ball doesn't exist for
        # scoring/foul purposes. Physics still simulates all 4 balls (so the
        # observation space is unchanged), but score depends only on the cue
        # ball hitting both reds and ``fouled`` is forced False.
        self._ignore_opponent = bool(ignore_opponent)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([2.0 * np.pi, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._state: TableState | None = None
        self._cumulative_score: int = 0
        self._shot_index: int = 0
        self._cumulative_t: float = 0.0
        # Per-shot trajectories with their original (local) timestamps.
        self._shot_trajectories: list[list[tuple[float, np.ndarray]]] = []
        # Per-shot offsets so we can reconstruct an inning-global timeline.
        self._shot_offsets: list[float] = []
        self._inning_log_records: list[dict[str, Any]] = []
        self._last_info: dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers

    def _obs(self) -> np.ndarray:
        assert self._state is not None
        return self._state.to_array().reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------ Gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = TableState.initial_4ball(cue_id=self._cue_id, spec=self._spec)
        self._cumulative_score = 0
        self._shot_index = 0
        self._cumulative_t = 0.0
        self._shot_trajectories = []
        self._shot_offsets = []
        self._inning_log_records = []
        self._last_info = None
        info: dict[str, Any] = {
            "spec": _spec_to_dict(self._spec),
            "shot_index": 0,
            "cumulative_score": 0,
        }
        return self._obs(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("env.step called before env.reset")

        cue_action = _project_action(np.asarray(action, dtype=np.float64))

        # simulate_shot mutates state in place and resets per-shot t origin
        # via state.t (which is currently whatever we left it at). We snap
        # the local clock to 0 before each shot so each trajectory starts at
        # t=0 locally; the inning timeline is rebuilt via _shot_offsets.
        offset = self._cumulative_t
        self._state.t = 0.0
        result = simulate_shot(self._state, cue_action, t_max=self._t_max)

        # Record per-shot trajectory (raw, local-time).
        shot_traj: list[tuple[float, np.ndarray]] = list(result["trajectory"])
        self._shot_trajectories.append(shot_traj)
        self._shot_offsets.append(offset)
        self._cumulative_t = offset + float(result["duration"])

        if self._ignore_opponent:
            reds: set[int] = set()
            for ev in result["events"]:
                if ev["type"] == "cue_hit_red":
                    i, j = ev["detail"]["balls"]
                    red_idx = j if i == self._cue_id else i
                    reds.add(red_idx)
            score = 1 if len(reds) >= 2 else 0
            fouled = False
        else:
            score = int(result["score"])
            fouled = bool(result["fouled"])
        self._cumulative_score += score
        self._shot_index += 1

        # Stash a result record sans heavyweight trajectory for analysis.
        record = {
            "shot_index": self._shot_index,
            "score": score,
            "fouled": fouled,
            "cushion_hits": int(result["cushion_hits"]),
            "duration": float(result["duration"]),
            "events": result["events"],
            "offset": offset,
        }
        self._inning_log_records.append(record)

        if self._continue_on_miss:
            terminated = False
            truncated = self._shot_index >= self._max_shots
            reward = float(score) - (self._foul_penalty if fouled else 0.0)
            # simulate_shot only zero-clamps balls that ended at rest. When
            # t_max truncates a still-rolling shot the cue ball keeps a
            # residual velocity and the next apply_cue() raises. In
            # continue_on_miss mode we force the table to a halt between
            # shots so the next shot is always well-defined.
            if not truncated:
                for b in self._state.balls:
                    b.vx = b.vy = 0.0
                    b.wx = b.wy = 0.0
                    b.wz = 0.0
        else:
            terminated = (score == 0) or fouled
            truncated = (self._shot_index >= self._max_shots) and not terminated
            reward = float(score)

        info: dict[str, Any] = {
            "event_log": result["events"],
            "cushion_hits": result["cushion_hits"],
            "fouled": fouled,
            "score": score,
            "duration": result["duration"],
            "trajectory": shot_traj,
            "spec": _spec_to_dict(self._spec),
            "cue_id": self._cue_id,
            "shot_index": self._shot_index,
            "cumulative_score": self._cumulative_score,
        }
        self._last_info = info
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> Any:
        if self._last_info is None:
            return None
        return self._last_info["trajectory"]

    # ------------------------------------------------------------------ inning-level views

    @property
    def full_trajectory(self) -> list[tuple[float, np.ndarray]]:
        """All shots stitched into one (t_global, (4,7)) timeline.

        Each shot's local t is shifted by its inning offset so timestamps
        are globally monotonic. The very first sample is forced to t=0.
        """
        out: list[tuple[float, np.ndarray]] = []
        for traj, offset in zip(self._shot_trajectories, self._shot_offsets):
            for t_local, arr in traj:
                out.append((float(t_local) + float(offset), arr))
        out.sort(key=lambda tp: tp[0])
        return out

    @property
    def inning_log(self) -> list[dict[str, Any]]:
        """Per-shot result records (no trajectory payload)."""
        return list(self._inning_log_records)

    @property
    def shot_trajectories(self) -> list[list[tuple[float, np.ndarray]]]:
        """Raw per-shot trajectories with local-time origins (one list per shot)."""
        return [list(traj) for traj in self._shot_trajectories]

    @property
    def cumulative_score(self) -> int:
        return self._cumulative_score

    @property
    def shot_index(self) -> int:
        return self._shot_index
