"""Reward MLP for Bradley-Terry preference learning.

Tiny architecture — (state_dim + action_dim) → hidden → hidden → 1 — is
plenty of capacity for a 5k-pair toy dataset and trains in seconds on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RewardMLP(nn.Module):
    """Score (state, action) → scalar reward."""

    def __init__(
        self,
        state_dim: int = 28,
        action_dim: int = 4,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.fc1 = nn.Linear(self.state_dim + self.action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.act = nn.ReLU()
        # Small init on the head for early-training stability.
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.head(x)
