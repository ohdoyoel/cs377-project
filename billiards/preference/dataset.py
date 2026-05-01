"""Preference-pair dataclass + JSONL persistence + random-pair generation.

A ``PreferencePair`` is a *compact* record of two single-shot rollouts from
the same initial state: just the actions and an event-summary digest of
each result. Full trajectories are intentionally **not** stored on the
dataclass — we keep them in a parallel list returned by
``generate_random_pairs`` so renderers can show A/B animations without
bloating ``data/preference_pairs.jsonl``.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=False)
class PreferencePair:
    """A pair of shots judged from the same initial state.

    Attributes match the JSONL schema 1:1 so round-tripping is trivial.
    """

    pair_id: str
    initial_state: list[float]    # length 28 (4 balls × 7 dims), pre-shot
    action_A: list[float]         # length 4 (theta, power, a, b)
    action_B: list[float]         # length 4
    result_A: dict[str, Any]
    result_B: dict[str, Any]
    preference: str | None = None    # 'A' | 'B' | 'tie' | None
    labeler: str = "unlabeled"       # 'human' | 'heuristic' | 'claude:<model>' | ...
    label_meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc)
                            .replace(microsecond=0).isoformat())

    # ------------------------------------------------------------------ I/O

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreferencePair":
        return cls(
            pair_id=d["pair_id"],
            initial_state=list(d["initial_state"]),
            action_A=list(d["action_A"]),
            action_B=list(d["action_B"]),
            result_A=dict(d["result_A"]),
            result_B=dict(d["result_B"]),
            preference=d.get("preference"),
            labeler=d.get("labeler", "unlabeled"),
            label_meta=dict(d.get("label_meta", {})),
            created_at=d.get(
                "created_at",
                datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            ),
        )


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


def append_pair(
    pair: PreferencePair,
    path: str | Path = "data/preference_pairs.jsonl",
) -> Path:
    """Append a single ``PreferencePair`` to ``path`` (creates parents)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(pair.to_dict(), separators=(",", ":")) + "\n")
    return p


def load_pairs(
    path: str | Path = "data/preference_pairs.jsonl",
) -> list[PreferencePair]:
    """Read all rows from a JSONL preference file (missing file → ``[]``)."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[PreferencePair] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(PreferencePair.from_dict(json.loads(line)))
    return out


def summary_stats(pairs: Iterable[PreferencePair]) -> dict[str, Any]:
    """Return ``{n, by_labeler, by_preference, agreement}`` digest.

    ``agreement`` is keyed by labeler-pair (e.g. ``'heuristic_vs_claude'``)
    when the same ``pair_id`` is labeled by more than one labeler — but
    typically a pair carries only one labeler at a time, so this dict will
    often be empty.
    """
    pairs = list(pairs)
    n = len(pairs)
    by_labeler = Counter(p.labeler for p in pairs)
    by_pref = Counter((p.preference or "unlabeled") for p in pairs)

    # Track multiple labels per pair_id (rare but supported).
    by_id: dict[str, list[PreferencePair]] = {}
    for p in pairs:
        by_id.setdefault(p.pair_id, []).append(p)
    agreement: dict[str, dict[str, float | int]] = {}
    for pid, group in by_id.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                key = "_vs_".join(sorted([a.labeler, b.labeler]))
                slot = agreement.setdefault(key, {"n": 0, "match": 0})
                slot["n"] = int(slot["n"]) + 1
                if a.preference == b.preference and a.preference is not None:
                    slot["match"] = int(slot["match"]) + 1
    for k, slot in agreement.items():
        n_k = int(slot["n"])
        slot["rate"] = (int(slot["match"]) / n_k) if n_k else 0.0

    return {
        "n": n,
        "by_labeler": dict(by_labeler),
        "by_preference": dict(by_pref),
        "agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Random-pair generation
# ---------------------------------------------------------------------------


def _slim_spec(spec: Any) -> dict[str, Any] | None:
    """Keep only the fields position_bonus actually reads."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        out = {
            "width": float(spec.get("width", 0.0) or 0.0),
            "height": float(spec.get("height", 0.0) or 0.0),
            "ball_radius": float(spec.get("ball_radius", 0.0) or 0.0),
        }
        if "cue_id" in spec:
            out["cue_id"] = int(spec["cue_id"])
        return out
    return None


def _final_state_from_traj(trajectory: Any) -> list[float] | None:
    """Flatten the last (4,7) frame from a trajectory list to a list[28]."""
    if not trajectory:
        return None
    last = trajectory[-1]
    if not isinstance(last, tuple) or len(last) < 2:
        return None
    arr = np.asarray(last[1], dtype=np.float64)
    if arr.size != 28:
        return None
    return [float(v) for v in arr.reshape(-1).tolist()]


def _result_digest(info: dict[str, Any]) -> dict[str, Any]:
    """Strip a step() info-dict down to the keys we actually persist.

    We deliberately drop ``trajectory`` (heavy) and ``event_log`` (only
    needed for the heuristic + AI labelers, which see the *fresh* info
    dict at generation time, not the persisted record).

    ``final_state`` (length-28 last-frame snapshot) and ``spec`` (slim
    dict of {width, height, ball_radius, cue_id}) are kept so position-
    aware labelers can read them off the persisted record.
    """
    events = info.get("event_log", []) or []
    out: dict[str, Any] = {
        "score": int(info.get("score", 0)),
        "fouled": bool(info.get("fouled", False)),
        "cushion_hits": int(info.get("cushion_hits", 0)),
        "duration": float(info.get("duration", 0.0)),
        "trajectory_len": int(len(info.get("trajectory", []) or [])),
        "n_events": int(len(events)),
        # event_types kept compactly for downstream heuristic / AI use
        "event_types": [str(e.get("type", "")) for e in events],
    }
    final_state = _final_state_from_traj(info.get("trajectory"))
    if final_state is not None:
        out["final_state"] = final_state
    spec = _slim_spec(info.get("spec"))
    if spec is not None:
        if "cue_id" not in spec and "cue_id" in info:
            spec["cue_id"] = int(info["cue_id"])
        out["spec"] = spec
    return out


def _sample_random_action(rng: np.random.Generator) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    power = float(rng.uniform(0.2, 1.0))     # avoid degenerate near-zero shots
    a = float(rng.uniform(-1.0, 1.0))
    b = float(rng.uniform(-1.0, 1.0))
    r = (a * a + b * b) ** 0.5
    if r > 1.0:
        a /= r
        b /= r
    return np.array([theta, power, a, b], dtype=np.float64)


def _make_env(env_factory_or_env, seed: int):
    """Either call ``env_factory_or_env(seed)`` or ``copy.deepcopy`` reset."""
    if callable(env_factory_or_env):
        env = env_factory_or_env()
    else:
        # Re-use a single env instance — reset() will rewind it.
        env = env_factory_or_env
    env.reset(seed=seed)
    return env


def generate_random_pairs(
    env_factory_or_env,
    n_pairs: int,
    seed: int = 0,
) -> tuple[list[PreferencePair], list[tuple[list, list]]]:
    """Build ``n_pairs`` un-labeled preference pairs from random shots.

    For each pair we:
      1. construct two fresh env instances by re-applying ``seed_i``,
      2. sample two independent random actions from a child RNG,
      3. step each env once,
      4. snapshot the *initial* state (post-reset, pre-step) and digest
         the two ``info`` dicts into ``result_A`` / ``result_B``.

    Returns
    -------
    (pairs, trajectories)
        ``pairs[i]`` is a ``PreferencePair`` with ``preference=None``;
        ``trajectories[i]`` is ``(traj_A, traj_B)`` for renderer use only,
        each in the standard ``list[(t, (N,7) ndarray)]`` form.
    """
    if n_pairs <= 0:
        return [], []
    base_rng = np.random.default_rng(int(seed))
    pairs: list[PreferencePair] = []
    trajectories: list[tuple[list, list]] = []

    for k in range(int(n_pairs)):
        env_seed = int(seed) + k
        action_rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))

        env_A = _make_env(env_factory_or_env, env_seed)
        initial_obs, _info0 = env_A.reset(seed=env_seed)
        initial_state = [float(v) for v in np.asarray(initial_obs).reshape(-1).tolist()]
        action_A = _sample_random_action(action_rng)
        _, _, _, _, info_A = env_A.step(action_A)

        env_B = _make_env(env_factory_or_env, env_seed)
        env_B.reset(seed=env_seed)
        action_B = _sample_random_action(action_rng)
        _, _, _, _, info_B = env_B.step(action_B)

        traj_A = info_A.get("trajectory", []) or []
        traj_B = info_B.get("trajectory", []) or []

        pair = PreferencePair(
            pair_id=uuid.uuid4().hex,
            initial_state=initial_state,
            action_A=[float(v) for v in action_A.tolist()],
            action_B=[float(v) for v in action_B.tolist()],
            result_A=_result_digest(info_A),
            result_B=_result_digest(info_B),
            preference=None,
            labeler="unlabeled",
            label_meta={"seed": env_seed},
        )
        pairs.append(pair)
        trajectories.append((list(traj_A), list(traj_B)))

    return pairs, trajectories
