"""Time-budget "solo game" wrapper for the inning env.

Turns the multi-shot inning env into a single episode that spans a fixed
budget of *simulated table time*. Within a game the agent plays innings back
to back: an inning ends on the first miss/foul (the wrapped env must use
``continue_on_miss=False``), and this wrapper silently resets to a fresh random
rack and keeps the game going — accumulating reward — until the cumulative shot
duration reaches ``budget_s``, at which point the gym episode is *truncated*.

Reward is passed through unchanged (score + whatever shaping the inner env
applies), so maximizing return == maximizing points scored within the time
budget. Faster shots (shorter ball travel) and positions that set up quick
follow-ups therefore score higher *structurally*, with no explicit time bonus —
this is the difference vs. the ``--time_reward`` shaping on the inning env.

Observation is left UNCHANGED (no remaining-time feature). This keeps the obs
dim identical to the inning env so an inning-trained policy can be warm-started
without an input-layer change (design Option 2). If end-of-budget value
estimation turns out to matter, add a normalized remaining-time feature later
(Option 3) via a zero-initialized input column on the warm-start weights.

Episode semantics for SB3:
  - terminated is always False (the game never reaches a true terminal state;
    a miss just resets the rack).
  - truncated becomes True only when the time budget is reached, so SB3
    bootstraps the value of the final observation (correct for a time limit).
  - A within-game miss/foul folds the rack reset into the same step with
    done=False, so Q(s_miss, a) = r_miss + gamma * V(fresh_rack): the cost of
    missing is losing the chained position and drawing a new random rack.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class TimeBudgetGameEnv(gym.Wrapper):
    """Wrap an inning env (``continue_on_miss=False``, ideally random-start) so
    one episode is a fixed simulated-time-budget game across multiple innings.

    Parameters
    ----------
    env : gym.Env
        Inning env, typically ``RandomStartInningEnv(Billiards4BallInningEnv(
        continue_on_miss=False, ...))``.
    budget_s : float
        Simulated table-time budget per game, in seconds (sum of shot
        durations). The episode truncates once this is reached.
    """

    def __init__(self, env: gym.Env, budget_s: float = 120.0) -> None:
        super().__init__(env)
        self._budget_s = float(budget_s)
        self._game_t = 0.0
        self._game_score = 0
        self._n_innings = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._game_t = 0.0
        self._game_score = 0
        self._n_innings = 1
        return self.env.reset(seed=seed, options=options)

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._game_t += float(info.get("duration", 0.0))
        self._game_score += int(info.get("score", 0))
        budget_reached = self._game_t >= self._budget_s

        # An inning ended (miss/foul, or inner max_shots cap) but the game
        # budget remains: draw a fresh random rack and keep playing. The
        # reward for THIS transition stays the inner reward (0 on a miss,
        # -foul_penalty on a foul); only the next observation changes.
        if (terminated or truncated) and not budget_reached:
            obs, _ = self.env.reset()
            self._n_innings += 1

        info = dict(info)
        info["game_sim_time"] = self._game_t
        info["game_score"] = self._game_score
        info["n_innings"] = self._n_innings
        return obs, reward, False, budget_reached, info
