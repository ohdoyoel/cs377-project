"""Cheap deterministic labeler for preference pairs.

Phase-D scorer adds position-aware components on top of the event-based
signals from Phase C. Each shot's scalar score is a weighted sum:

    +5.0  shot scored (≥ 1 point)
    -5.0  cue ball touched the opponent ball (foul)
    +0.4 * min(cushion_hits, 5)        cushion-control bonus, capped
    +0.6 * (# cue_hit_red events)      progress toward score
    -0.3 * (# cue_hit_opp events)      mirrors the foul axis
    +position_bonus(final_state, spec) ≤ 1.0 — clustered reds & cue near reds
    -0.05 * duration                   penalize stalling

The pair gets ``preference='A'`` if ``score_A - score_B > 0.5``,
``'B'`` if the inverse, else ``'tie'``. The 0.5 dead-band keeps small
control-bonus differences from flipping the label.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from billiards.preference.dataset import PreferencePair


_TIE_BAND = 0.5


def _count_event_types(result: dict[str, Any]) -> dict[str, int]:
    types = result.get("event_types", []) or []
    out: dict[str, int] = {}
    for t in types:
        out[t] = out.get(t, 0) + 1
    return out


def position_bonus(
    final_state: Sequence[float] | np.ndarray | None,
    spec: dict[str, Any] | None,
) -> float:
    """Bonus reward for ending the shot with reds clustered + cue nearby.

    ``final_state`` is the flattened (4, 7) post-shot snapshot used as the
    next observation. Index layout matches ``BallRole``:
        0: white cue, 1: yellow cue, 2: red 1, 3: red 2.
    """
    if final_state is None or spec is None:
        return 0.0
    arr = np.asarray(final_state, dtype=np.float64)
    if arr.size != 28:
        return 0.0
    arr = arr.reshape(4, 7)
    # Use the actual cue ball position; default to white if cue id absent.
    cue_id = int(spec.get("cue_id", 0)) if isinstance(spec, dict) else 0
    if cue_id not in (0, 1):
        cue_id = 0
    cue_pos = arr[cue_id, 0:2]
    red1_pos = arr[2, 0:2]
    red2_pos = arr[3, 0:2]

    d_red = float(np.linalg.norm(red1_pos - red2_pos))
    d_cue_to_nearest_red = float(min(
        np.linalg.norm(cue_pos - red1_pos),
        np.linalg.norm(cue_pos - red2_pos),
    ))

    cluster_score = max(0.0, min(1.0, 1.0 - d_red / 0.50))
    proximity_score = max(0.0, min(1.0, 1.0 - d_cue_to_nearest_red / 0.80))
    return 0.6 * cluster_score + 0.4 * proximity_score


def heuristic_score(
    result: dict[str, Any],
    final_state: Sequence[float] | np.ndarray | None = None,
    spec: dict[str, Any] | None = None,
) -> float:
    """Score a single shot's result digest. Higher == better."""
    score = float(result.get("score", 0))
    fouled = bool(result.get("fouled", False))
    cushion_hits = int(result.get("cushion_hits", 0))
    duration = float(result.get("duration", 0.0))
    counts = _count_event_types(result)
    n_red_contacts = int(counts.get("cue_hit_red", 0))
    n_opp_contacts = int(counts.get("cue_hit_opp", 0))

    s = 0.0
    s += 5.0 * (1.0 if score >= 1 else 0.0)
    if fouled:
        s -= 5.0
    s += 0.4 * min(cushion_hits, 5)
    s += 0.6 * n_red_contacts
    s -= 0.3 * n_opp_contacts
    if final_state is not None and spec is not None:
        s += position_bonus(final_state, spec)
    s -= 0.05 * duration
    return float(s)


def label_pair_heuristic(pair: PreferencePair) -> PreferencePair:
    """Return a copy of ``pair`` with the heuristic preference set."""
    sa = heuristic_score(
        pair.result_A,
        final_state=pair.result_A.get("final_state"),
        spec=pair.result_A.get("spec"),
    )
    sb = heuristic_score(
        pair.result_B,
        final_state=pair.result_B.get("final_state"),
        spec=pair.result_B.get("spec"),
    )
    if sa - sb > _TIE_BAND:
        pref: str | None = "A"
    elif sb - sa > _TIE_BAND:
        pref = "B"
    else:
        pref = "tie"

    return replace(
        pair,
        preference=pref,
        labeler="heuristic",
        label_meta={
            **pair.label_meta,
            "score_A": sa,
            "score_B": sb,
            "tie_band": _TIE_BAND,
        },
    )
