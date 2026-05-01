"""Preference data pipeline for billiards RLHF.

Builds (action_A, action_B) shot pairs from a fixed initial state, stores
them as JSONL on disk (without bulky trajectories), and provides labelers:
heuristic (cheap, deterministic) and Anthropic Claude (high-quality).
"""

from billiards.preference.dataset import (
    PreferencePair,
    append_pair,
    generate_random_pairs,
    load_pairs,
    summary_stats,
)

__all__ = [
    "PreferencePair",
    "append_pair",
    "generate_random_pairs",
    "load_pairs",
    "summary_stats",
]
