"""PEBBLE replay buffer.

Wraps SB3's :class:`stable_baselines3.common.buffers.ReplayBuffer` with two
extra capabilities required by PEBBLE:

* **Per-transition meta** — single-shot info dicts (cushion_hits, fouled,
  score, event_types, final_state, duration, spec, cue_id) that the
  heuristic labeler reads when constructing a synthetic preference.

* **Relabel** — replace ``self.rewards`` in place with ``rm(s, a)``, so
  SAC's next gradient step sees the new reward model output.

Single-shot env: ``done = True`` after every step, so successive buffer
positions are independent. The reward field is what gets relabeled — not
the value bootstrap (terminal anyway).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.buffers import ReplayBuffer


class PEBBLEBuffer(ReplayBuffer):
    """SB3 ReplayBuffer + parallel meta storage + relabel/sample-pairs.

    Parameters mirror :class:`ReplayBuffer`. The extra ``meta_dict`` array
    stays aligned with ``self.observations[pos, env_idx]`` so the heuristic
    labeler can read full info dicts back out at query time.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: str | torch.device = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        # buffer_size after super().__init__ is the per-env capacity.
        self.meta_dict: list[list[dict[str, Any] | None]] = [
            [None for _ in range(self.n_envs)] for _ in range(self.buffer_size)
        ]

    # ------------------------------------------------------------------ add

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Store meta at current pos before super().add advances it.
        write_pos = self.pos
        for env_idx in range(self.n_envs):
            info = infos[env_idx] if env_idx < len(infos) else {}
            self.meta_dict[write_pos][env_idx] = self._extract_meta(info)
        super().add(obs, next_obs, action, reward, done, infos)

    @staticmethod
    def _extract_meta(info: dict[str, Any]) -> dict[str, Any]:
        """Strip a step() info dict to keys the heuristic labeler uses."""
        events = info.get("event_log", []) or []
        out: dict[str, Any] = {
            "score": int(info.get("score", 0)),
            "fouled": bool(info.get("fouled", False)),
            "cushion_hits": int(info.get("cushion_hits", 0)),
            "duration": float(info.get("duration", 0.0)),
            "n_events": int(len(events)),
            "event_types": [str(e.get("type", "")) for e in events],
        }
        spec = info.get("spec")
        if isinstance(spec, dict):
            slim = {
                "width": float(spec.get("width", 0.0) or 0.0),
                "height": float(spec.get("height", 0.0) or 0.0),
                "ball_radius": float(spec.get("ball_radius", 0.0) or 0.0),
            }
            cue_id = info.get("cue_id", spec.get("cue_id"))
            if cue_id is not None:
                slim["cue_id"] = int(cue_id)
            out["spec"] = slim
        traj = info.get("trajectory")
        if traj:
            try:
                last = traj[-1]
                if isinstance(last, tuple) and len(last) >= 2:
                    arr = np.asarray(last[1], dtype=np.float64)
                    if arr.size == 28:
                        out["final_state"] = [float(v) for v in arr.reshape(-1).tolist()]
            except Exception:  # noqa: BLE001
                pass
        return out

    # ------------------------------------------------------------------ size

    def num_transitions(self) -> int:
        """Number of valid stored (pos, env_idx) transitions."""
        per_pos = self.buffer_size if self.full else self.pos
        return int(per_pos) * int(self.n_envs)

    def _valid_positions(self) -> list[tuple[int, int]]:
        """List of (pos, env_idx) for every stored transition."""
        per_pos = self.buffer_size if self.full else self.pos
        return [(p, e) for p in range(per_pos) for e in range(self.n_envs)]

    def get_meta(self, pos: int, env_idx: int = 0) -> dict[str, Any] | None:
        return self.meta_dict[pos][env_idx]

    # ------------------------------------------------------------------ relabel

    def relabel(
        self,
        rm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | nn.Module,
        batch_size: int = 512,
    ) -> dict[str, float]:
        """Recompute ``self.rewards[pos, env_idx] = rm(obs, action)`` in place.

        ``rm`` may be an ``nn.Module`` (called with ``(state, action)``) or a
        callable returning a tensor. Returns ``{mean_abs_delta, std_abs_delta,
        n}`` so callers can log how much the reward landscape shifted.
        """
        n_total = self.num_transitions()
        if n_total == 0:
            return {"mean_abs_delta": 0.0, "std_abs_delta": 0.0, "n": 0}

        per_pos = self.buffer_size if self.full else self.pos
        device = "cpu"
        if isinstance(rm, nn.Module):
            try:
                device = next(rm.parameters()).device
            except StopIteration:
                device = "cpu"

        deltas: list[np.ndarray] = []
        idx_pairs: list[tuple[int, int]] = [(p, e) for p in range(per_pos) for e in range(self.n_envs)]

        with torch.no_grad():
            for start in range(0, len(idx_pairs), batch_size):
                chunk = idx_pairs[start:start + batch_size]
                obs_batch = np.stack([self.observations[p, e] for p, e in chunk]).astype(np.float32)
                act_batch = np.stack([self.actions[p, e] for p, e in chunk]).astype(np.float32)
                s_t = torch.as_tensor(obs_batch, device=device)
                a_t = torch.as_tensor(act_batch, device=device)
                preds = rm(s_t, a_t)
                if isinstance(preds, tuple):
                    preds = preds[0]
                preds = preds.detach().cpu().reshape(-1).numpy().astype(np.float32)
                old = np.array([self.rewards[p, e] for p, e in chunk], dtype=np.float32)
                deltas.append(np.abs(preds - old))
                for (p, e), r in zip(chunk, preds):
                    self.rewards[p, e] = float(r)

        flat = np.concatenate(deltas) if deltas else np.zeros(0, dtype=np.float32)
        return {
            "mean_abs_delta": float(flat.mean()) if flat.size else 0.0,
            "std_abs_delta": float(flat.std(ddof=0)) if flat.size else 0.0,
            "n": int(flat.size),
        }

    # ------------------------------------------------------------------ sample_pairs

    def sample_pairs(
        self,
        n_pairs: int,
        strategy: str = "uniform",
        ensemble: Any = None,
        rng: np.random.Generator | None = None,
        candidate_factor: int = 4,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Sample ``n_pairs`` index-pairs from the buffer.

        Each pair is ``((pos_A, env_A), (pos_B, env_B))``; same shape works
        whether ``n_envs == 1`` or not. ``strategy``:

        * ``"uniform"`` — independent uniform samples, with replacement
          across pairs but never ``A == B`` within a pair.
        * ``"disagreement"`` — needs an ``ensemble`` with ``>= 2`` members;
          score ``|rm1(s,a) - rm2(s,a)|`` for every transition, take the
          top ``2 * n_pairs * candidate_factor`` candidates, shuffle, then
          chunk into pairs. Falls back to ``"uniform"`` (with a warning) if
          ``ensemble`` is missing or has only one member.
        """
        valid = self._valid_positions()
        if len(valid) < 2:
            return []
        rng = rng if rng is not None else np.random.default_rng()
        n_pairs = int(n_pairs)
        if n_pairs <= 0:
            return []

        if strategy == "disagreement":
            if ensemble is None or getattr(ensemble, "n_models", 1) < 2:
                warnings.warn(
                    "sample_pairs(strategy='disagreement') needs an ensemble with"
                    " >=2 members; falling back to uniform.",
                    stacklevel=2,
                )
                strategy = "uniform"

        if strategy == "uniform":
            return self._sample_uniform(valid, n_pairs, rng)
        if strategy == "disagreement":
            return self._sample_disagreement(valid, n_pairs, ensemble, rng,
                                             candidate_factor=candidate_factor)
        raise ValueError(f"unknown strategy: {strategy!r}")

    def _sample_uniform(
        self,
        valid: list[tuple[int, int]],
        n_pairs: int,
        rng: np.random.Generator,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        n = len(valid)
        out: list[tuple[tuple[int, int], tuple[int, int]]] = []
        max_tries = n_pairs * 10
        tries = 0
        while len(out) < n_pairs and tries < max_tries:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            tries += 1
            if i == j:
                continue
            out.append((valid[i], valid[j]))
        return out

    def _sample_disagreement(
        self,
        valid: list[tuple[int, int]],
        n_pairs: int,
        ensemble: Any,
        rng: np.random.Generator,
        candidate_factor: int = 4,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        # Score every valid transition by ensemble disagreement.
        obs = np.stack([self.observations[p, e] for p, e in valid]).astype(np.float32)
        act = np.stack([self.actions[p, e] for p, e in valid]).astype(np.float32)
        device = getattr(ensemble, "device", "cpu")
        with torch.no_grad():
            s_t = torch.as_tensor(obs, device=device)
            a_t = torch.as_tensor(act, device=device)
            mean, std = ensemble(s_t, a_t)
            disagreement = std.detach().cpu().reshape(-1).numpy()

        n_top = min(len(valid), 2 * n_pairs * max(1, candidate_factor))
        top_idx = np.argsort(-disagreement)[:n_top]
        rng.shuffle(top_idx)
        out: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for k in range(0, len(top_idx) - 1, 2):
            a = valid[int(top_idx[k])]
            b = valid[int(top_idx[k + 1])]
            if a == b:
                continue
            out.append((a, b))
            if len(out) >= n_pairs:
                break
        # Top-up with uniform if disagreement pool was too thin.
        if len(out) < n_pairs:
            out.extend(self._sample_uniform(valid, n_pairs - len(out), rng))
        return out
