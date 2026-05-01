"""PEBBLE outer-loop agent: SAC + reward-model ensemble + relabeling.

Algorithmic loop (Lee et al., ICML 2021), adapted to our single-shot env:

    init: empty replay R, empty preference dataset D, RMEnsemble (N=2)
    for phase in 0..n_phases:
        SAC.learn(query_phase_steps)        # SAC trained against current RM
        pairs = R.sample_pairs(...)         # uniform / disagreement
        for (i_A, i_B) in pairs:
            pref = label_pair_heuristic(make_pair(meta[i_A], meta[i_B]))
            D.append(pref)
        ensemble.train_on_pairs(D)
        R.relabel(ensemble.mean)            # KEY PEBBLE STEP

Three reward modes share the same outer loop:
    'rm_only'    -- SAC sees rm(s, a) only.
    'env_only'   -- SAC sees env score; RM/ensemble unused.
    'mix_alpha'  -- SAC sees alpha*env + (1-alpha)*rm(s, a).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from billiards.preference.dataset import PreferencePair
from billiards.preference.labeler_heuristic import label_pair_heuristic

from .buffer import PEBBLEBuffer
from .ensemble import RMEnsemble


# ---------------------------------------------------------------------------
# Reward-injection wrapper. Reads from a callable supplied by the agent so the
# wrapper picks up the *current* ensemble parameters every step (no caching).
# ---------------------------------------------------------------------------


class _RewardInjectionEnv(gym.Wrapper):
    """Substitute env reward with a callable on (pre-shot state, action).

    ``mode``:
        'rm_only'   r = rm(s, a)
        'env_only'  r = env_score
        'mix_alpha' r = alpha*env + (1-alpha)*rm(s, a)
    """

    def __init__(
        self,
        env: gym.Env,
        rm_callable: Callable[[np.ndarray, np.ndarray], float],
        mode: str = "rm_only",
        alpha: float = 0.0,
    ) -> None:
        super().__init__(env)
        self._rm_callable = rm_callable
        self._mode = str(mode)
        self._alpha = float(alpha)
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        return obs, info

    def step(self, action: np.ndarray):  # noqa: ANN201
        if self._last_obs is None:
            raise RuntimeError("_RewardInjectionEnv.step before reset")
        s_pre = self._last_obs
        a_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, env_r, term, trunc, info = self.env.step(action)

        if self._mode == "env_only":
            r = float(env_r)
            rm_val = float("nan")
        else:
            rm_val = float(self._rm_callable(s_pre, a_arr))
            if self._mode == "rm_only":
                r = rm_val
            elif self._mode == "mix_alpha":
                r = self._alpha * float(env_r) + (1.0 - self._alpha) * rm_val
            else:
                raise ValueError(f"unknown reward mode: {self._mode!r}")

        new_info = dict(info) if info is not None else {}
        new_info["env_reward"] = float(env_r)
        new_info["rm_reward"] = rm_val
        new_info["pebble_reward"] = float(r)
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        return obs, float(r), term, trunc, new_info


# ---------------------------------------------------------------------------
# Curve callback — appends one row per `record_freq` steps to a list. The
# outer agent flushes this to CSV after each phase.
# ---------------------------------------------------------------------------


class _CurveCallback(BaseCallback):
    """Collect per-rollout aggregates from the SAC monitor info_keywords."""

    def __init__(
        self,
        sink: list[dict[str, float]],
        queries_used: Callable[[], int],
        rm_train_loss: Callable[[], float],
        relabel_delta: Callable[[], tuple[float, float]],
        record_every: int = 256,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._sink = sink
        self._queries_used = queries_used
        self._rm_train_loss = rm_train_loss
        self._relabel_delta = relabel_delta
        self._record_every = int(record_every)
        self._last_recorded = 0

    def _on_step(self) -> bool:  # noqa: D401
        if self.num_timesteps - self._last_recorded < self._record_every:
            return True
        self._last_recorded = int(self.num_timesteps)
        ep_buf = list(self.model.ep_info_buffer or [])

        def _meanof(key: str) -> float:
            vals = [e.get(key) for e in ep_buf if key in e]
            if not vals:
                return float("nan")
            return float(np.asarray(vals, dtype=np.float64).mean())

        env_score = _meanof("env_reward")
        rm_score = _meanof("rm_reward")
        cushion = _meanof("cushion_hits")
        fouled_vals = [
            1.0 if bool(e["fouled"]) else 0.0
            for e in ep_buf if "fouled" in e
        ]
        foul = float(np.mean(fouled_vals)) if fouled_vals else float("nan")
        d_mean, d_std = self._relabel_delta()
        self._sink.append({
            "timesteps": int(self.num_timesteps),
            "ep_env_score_mean": env_score,
            "ep_rm_score_mean": rm_score,
            "ep_cushion_mean": cushion,
            "ep_foul_rate": foul,
            "queries_used_so_far": int(self._queries_used()),
            "rm_train_loss": float(self._rm_train_loss()),
            "relabel_delta_mean": float(d_mean),
            "relabel_delta_std": float(d_std),
        })
        return True


# ---------------------------------------------------------------------------
# PEBBLEAgent
# ---------------------------------------------------------------------------


class PEBBLEAgent:
    """Driver for one PEBBLE training run.

    Notes
    -----
    * ``env_factory`` should be a no-arg callable returning a fresh
      :class:`gymnasium.Env` (typically :class:`Billiards4BallEnv`).
    * ``reward_mode='env_only'`` skips ensemble training/relabel entirely.
    * Pairs are generated from buffer transitions; the heuristic labeler
      converts each (meta_A, meta_B) into a synthetic preference. We use
      the buffer's stored ``initial_state`` as the pair's shared anchor —
      the two transitions need not have started from the *same* env state
      for SB3-flavored PEBBLE; the BT loss compares (s_A, a_A) vs (s_B, a_B)
      directly, which is the convention adopted in the original code-base.
    """

    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        total_steps: int = 30_000,
        query_phase_steps: int = 5_000,
        queries_per_phase: int = 50,
        query_strategy: str = "uniform",
        relabel_after_query: bool = True,
        ensemble_size: int = 2,
        reward_mode: str = "rm_only",
        alpha: float = 0.0,
        seed: int = 0,
        device: str = "cpu",
        sac_kwargs: dict[str, Any] | None = None,
        ensemble_train_epochs: int = 10,
        rm_pretrain_pairs: int = 100,
    ) -> None:
        self.env_factory = env_factory
        self.total_steps = int(total_steps)
        self.query_phase_steps = int(query_phase_steps)
        self.queries_per_phase = int(queries_per_phase)
        self.query_strategy = str(query_strategy)
        self.relabel_after_query = bool(relabel_after_query)
        self.ensemble_size = int(ensemble_size)
        self.reward_mode = str(reward_mode)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.device = str(device)
        self.sac_kwargs = dict(sac_kwargs) if sac_kwargs else {}
        self.ensemble_train_epochs = int(ensemble_train_epochs)
        self.rm_pretrain_pairs = int(rm_pretrain_pairs)

        # Lazy-initialized when learn() runs.
        self.ensemble: RMEnsemble | None = None
        self.preference_dataset: list[PreferencePair] = []
        self._sac: SAC | None = None
        self._buffer: PEBBLEBuffer | None = None
        self._curve_rows: list[dict[str, float]] = []
        self._queries_used = 0
        self._last_rm_train_loss = float("nan")
        self._last_relabel_delta = (0.0, 0.0)
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ learn

    def _rm_callable(self, s_pre: np.ndarray, a_arr: np.ndarray) -> float:
        """Mean-of-ensemble reward, or 0 when ensemble is uninitialized."""
        if self.ensemble is None:
            return 0.0
        with torch.no_grad():
            s_t = torch.as_tensor(s_pre, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            a_t = torch.as_tensor(a_arr, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            mean, _ = self.ensemble(s_t, a_t)
            return float(mean.squeeze().item())

    def _wrap_env(self, env: gym.Env) -> gym.Env:
        from stable_baselines3.common.monitor import Monitor
        env = _RewardInjectionEnv(
            env,
            rm_callable=self._rm_callable,
            mode=self.reward_mode,
            alpha=self.alpha,
        )
        env = Monitor(
            env,
            info_keywords=("env_reward", "rm_reward", "cushion_hits", "fouled"),
        )
        return env

    def _make_sac(self, env: gym.Env) -> SAC:
        defaults = dict(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=max(self.total_steps, 10_000),
            learning_starts=min(256, max(64, self.total_steps // 50)),
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=0,
            seed=self.seed,
            device=self.device,
        )
        defaults.update(self.sac_kwargs)
        defaults["replay_buffer_class"] = PEBBLEBuffer
        sac = SAC(**defaults)
        return sac

    def _generate_pairs_from_buffer(
        self, n_pairs: int
    ) -> list[PreferencePair]:
        """Pull index-pairs from the buffer, build PreferencePair records."""
        assert self._buffer is not None
        idx_pairs = self._buffer.sample_pairs(
            n_pairs=int(n_pairs),
            strategy=self.query_strategy,
            ensemble=self.ensemble,
            rng=self._rng,
        )
        pairs: list[PreferencePair] = []
        for (pos_a, env_a), (pos_b, env_b) in idx_pairs:
            obs_a = self._buffer.observations[pos_a, env_a].astype(np.float32)
            act_a = self._buffer.actions[pos_a, env_a].astype(np.float32)
            obs_b = self._buffer.observations[pos_b, env_b].astype(np.float32)
            act_b = self._buffer.actions[pos_b, env_b].astype(np.float32)
            meta_a = self._buffer.get_meta(pos_a, env_a) or {}
            meta_b = self._buffer.get_meta(pos_b, env_b) or {}

            # Use obs_a as the anchor "initial_state" for the pair.
            pair = PreferencePair(
                pair_id=uuid.uuid4().hex,
                initial_state=[float(v) for v in obs_a.reshape(-1).tolist()],
                action_A=[float(v) for v in act_a.reshape(-1).tolist()],
                action_B=[float(v) for v in act_b.reshape(-1).tolist()],
                result_A=dict(meta_a),
                result_B=dict(meta_b),
                preference=None,
                labeler="unlabeled",
                label_meta={
                    "anchor_obs_b": [float(v) for v in obs_b.reshape(-1).tolist()],
                    "buffer_idx_A": [int(pos_a), int(env_a)],
                    "buffer_idx_B": [int(pos_b), int(env_b)],
                },
            )
            labeled = label_pair_heuristic(pair)
            pairs.append(labeled)
        return pairs

    def learn(self) -> dict[str, Any]:
        """Run the PEBBLE outer loop; return a small summary dict."""
        t0 = time.perf_counter()

        # Single-env training (per task spec n_envs=1 for stability).
        train_env = self._wrap_env(self.env_factory())

        if self.reward_mode != "env_only":
            self.ensemble = RMEnsemble(
                n_models=self.ensemble_size,
                state_dim=28,
                action_dim=4,
                hidden=256,
                device=self.device,
                seed_base=42 + self.seed,
            )
        else:
            self.ensemble = None

        sac = self._make_sac(train_env)
        self._sac = sac
        # SB3's SAC stashes its replay_buffer post-init.
        self._buffer = sac.replay_buffer  # type: ignore[assignment]
        assert isinstance(self._buffer, PEBBLEBuffer)

        # Phase scheduling: split total_steps into chunks of query_phase_steps.
        n_phases = max(1, self.total_steps // self.query_phase_steps)
        remainder = self.total_steps - n_phases * self.query_phase_steps
        phase_steps = [self.query_phase_steps] * n_phases
        if remainder > 0:
            phase_steps[-1] += remainder

        callback = _CurveCallback(
            sink=self._curve_rows,
            queries_used=lambda: self._queries_used,
            rm_train_loss=lambda: self._last_rm_train_loss,
            relabel_delta=lambda: self._last_relabel_delta,
            record_every=max(64, self.query_phase_steps // 16),
        )

        for phase, ps in enumerate(phase_steps):
            sac.learn(
                total_timesteps=int(ps),
                reset_num_timesteps=(phase == 0),
                callback=callback,
                progress_bar=False,
            )
            if self.reward_mode == "env_only":
                continue

            # Query + label.
            new_pairs = self._generate_pairs_from_buffer(self.queries_per_phase)
            self.preference_dataset.extend(new_pairs)
            self._queries_used = len(self.preference_dataset)

            # Train ensemble on accumulated dataset.
            if self.preference_dataset and self.ensemble is not None:
                history = self.ensemble.train_on_pairs(
                    self.preference_dataset,
                    epochs=self.ensemble_train_epochs,
                    lr=3e-4,
                    batch_size=128,
                    verbose=False,
                )
                # Track the mean of last-epoch train losses across members.
                last_losses: list[float] = []
                for m in history.get("members", []):
                    tl = m.get("train_loss") if isinstance(m, dict) else None
                    if tl:
                        last_losses.append(float(tl[-1]))
                self._last_rm_train_loss = (
                    float(np.mean(last_losses)) if last_losses else float("nan")
                )

                if self.relabel_after_query:
                    delta = self._buffer.relabel(self.ensemble.mean_callable())
                    self._last_relabel_delta = (
                        float(delta["mean_abs_delta"]),
                        float(delta["std_abs_delta"]),
                    )

        wall = time.perf_counter() - t0
        try:
            train_env.close()
        except Exception:  # noqa: BLE001
            pass

        return {
            "wall_s": float(wall),
            "n_phases": int(n_phases),
            "queries_used": int(self._queries_used),
            "preference_dataset_size": int(len(self.preference_dataset)),
            "last_rm_train_loss": float(self._last_rm_train_loss),
            "last_relabel_delta_mean": float(self._last_relabel_delta[0]),
            "last_relabel_delta_std": float(self._last_relabel_delta[1]),
            "curve_rows": list(self._curve_rows),
        }

    # ------------------------------------------------------------------ evaluate

    def evaluate(self, n_episodes: int = 500, seed_base: int = 10_000) -> list[dict[str, Any]]:
        """Roll the trained policy on the BASE env, one row per episode."""
        if self._sac is None:
            raise RuntimeError("PEBBLEAgent.evaluate called before learn()")
        env = self.env_factory()
        rows: list[dict[str, Any]] = []
        for ep in range(int(n_episodes)):
            obs, _ = env.reset(seed=int(seed_base) + ep)
            s_pre = np.asarray(obs, dtype=np.float32).reshape(-1)
            action, _ = self._sac.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            next_obs, _, _, _, info = env.step(action)
            rm_val = float("nan")
            if self.ensemble is not None:
                with torch.no_grad():
                    s_t = torch.as_tensor(s_pre, dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
                    a_t = torch.as_tensor(action, dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
                    mean, _std = self.ensemble(s_t, a_t)
                    rm_val = float(mean.squeeze().item())
            events = info.get("event_log", []) or []
            event_types = [str(e.get("type", "")) for e in events]
            rows.append({
                "ep_idx": int(ep),
                "seed": int(seed_base) + ep,
                "theta": float(action[0]),
                "power": float(action[1]),
                "a": float(action[2]),
                "b": float(action[3]),
                "score": int(info.get("score", 0)),
                "fouled": bool(info.get("fouled", False)),
                "cushion_hits": int(info.get("cushion_hits", 0)),
                "duration": float(info.get("duration", 0.0)),
                "n_events": int(len(events)),
                "event_types": event_types,
                "rm_reward": rm_val,
                "final_state": np.asarray(next_obs, dtype=np.float32).reshape(-1).tolist(),
            })
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        return rows
