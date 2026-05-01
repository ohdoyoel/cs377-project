"""Ensemble of RewardMLPs for PEBBLE disagreement-based queries.

Random init seeds vary across members so disagreement is non-zero before
training. After training each member on the same preference dataset, the
ensemble mean is used as SAC's reward (and for buffer relabeling) while
the per-batch std becomes the disagreement signal for active queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

from billiards.preference.dataset import PreferencePair
from billiards.reward_model.network import RewardMLP
from billiards.reward_model.train import train as train_single_rm


class RMEnsemble:
    """N independently-initialized RewardMLPs trained on the same pairs."""

    def __init__(
        self,
        n_models: int = 2,
        state_dim: int = 28,
        action_dim: int = 4,
        hidden: int = 256,
        device: str = "cpu",
        seed_base: int = 42,
    ) -> None:
        self.n_models = int(n_models)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.device = str(device)
        self.seed_base = int(seed_base)
        self.models: list[RewardMLP] = []
        for k in range(self.n_models):
            torch.manual_seed(self.seed_base + k)
            m = RewardMLP(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden=self.hidden,
            ).to(self.device)
            self.models.append(m)

    # ------------------------------------------------------------------ forward

    def __call__(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(state, action)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.to(self.device)
        action = action.to(self.device)
        preds = []
        for m in self.models:
            m.eval()
            preds.append(m(state, action).squeeze(-1))
        stacked = torch.stack(preds, dim=0)  # (N, B)
        mean = stacked.mean(dim=0)
        if self.n_models > 1:
            std = stacked.std(dim=0, unbiased=False)
        else:
            std = torch.zeros_like(mean)
        return mean, std

    # ------------------------------------------------------------------ training

    def train_on_pairs(
        self,
        pairs: list[PreferencePair],
        epochs: int = 10,
        lr: float = 3e-4,
        batch_size: int = 128,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Train every member on ``pairs``. Returns per-member history."""
        out: dict[str, Any] = {"members": []}
        if not pairs:
            return out
        for k, m in enumerate(self.models):
            seed = self.seed_base + 1000 + k
            trained, hist = train_single_rm(
                pairs=pairs,
                epochs=int(epochs),
                lr=float(lr),
                batch_size=int(batch_size),
                device=self.device,
                seed=int(seed),
                save_dir=None,
                verbose=verbose,
            )
            # Replace member weights with trained module.
            self.models[k] = trained.to(self.device)
            out["members"].append(hist)
        return out

    # ------------------------------------------------------------------ helpers

    def mean_callable(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return a callable ``(s, a) -> mean_pred`` for buffer relabeling."""

        def _call(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            mean, _ = self.forward(state, action)
            return mean

        return _call

    # ------------------------------------------------------------------ persistence

    def save_dir(self, out_dir: str | Path) -> None:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        for k, m in enumerate(self.models):
            torch.save(
                {"model": m, "state_dict": m.state_dict(),
                 "state_dim": m.state_dim, "action_dim": m.action_dim},
                d / f"member_{k}.pt",
            )

    def load_dir(self, in_dir: str | Path) -> None:
        d = Path(in_dir)
        for k in range(self.n_models):
            obj = torch.load(d / f"member_{k}.pt", map_location=self.device,
                             weights_only=False)
            if isinstance(obj, dict) and "model" in obj:
                self.models[k] = obj["model"].to(self.device)
            elif isinstance(obj, nn.Module):
                self.models[k] = obj.to(self.device)
            else:
                raise TypeError(f"member_{k}.pt has unsupported payload type")
