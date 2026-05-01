"""PEBBLE — preference-based RL with relabeled SAC replay buffer.

Public API:
    PEBBLEBuffer  -- SB3 ReplayBuffer + per-transition meta + relabel.
    RMEnsemble    -- ensemble of RewardMLPs for disagreement queries.
    PEBBLEAgent   -- SAC + ensemble + outer query/relabel loop.
"""

from billiards.pebble.agent import PEBBLEAgent
from billiards.pebble.buffer import PEBBLEBuffer
from billiards.pebble.ensemble import RMEnsemble

__all__ = ["PEBBLEBuffer", "RMEnsemble", "PEBBLEAgent"]
