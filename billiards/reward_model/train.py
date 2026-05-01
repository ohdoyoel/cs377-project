"""Bradley-Terry training loop for the preference reward model.

The pairwise loss is the binary cross entropy between
``sigmoid(r_A - r_B)`` and the soft label ``y`` ∈ {1, 0, 0.5}.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from billiards.preference.dataset import PreferencePair
from billiards.reward_model.network import RewardMLP


def _label_for(pair: PreferencePair) -> float | None:
    if pair.preference == "A":
        return 1.0
    if pair.preference == "B":
        return 0.0
    if pair.preference == "tie":
        return 0.5
    return None


def make_dataset(
    pairs: Iterable[PreferencePair],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a list of pairs to (state, action_A, action_B, label) tensors."""
    states: list[list[float]] = []
    actions_A: list[list[float]] = []
    actions_B: list[list[float]] = []
    labels: list[float] = []
    for p in pairs:
        y = _label_for(p)
        if y is None:
            continue
        states.append(list(p.initial_state))
        actions_A.append(list(p.action_A))
        actions_B.append(list(p.action_B))
        labels.append(float(y))
    if not states:
        return (
            torch.zeros((0, 28)),
            torch.zeros((0, 4)),
            torch.zeros((0, 4)),
            torch.zeros((0,)),
        )
    s = torch.tensor(states, dtype=torch.float32)
    aA = torch.tensor(actions_A, dtype=torch.float32)
    aB = torch.tensor(actions_B, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return s, aA, aB, y


def _eval_split(
    model: RewardMLP,
    s: torch.Tensor,
    aA: torch.Tensor,
    aB: torch.Tensor,
    y: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    """Return (loss, accuracy) on the given tensors."""
    if s.shape[0] == 0:
        return float("nan"), float("nan")
    model.eval()
    with torch.no_grad():
        s_d = s.to(device)
        aA_d = aA.to(device)
        aB_d = aB.to(device)
        y_d = y.to(device)
        rA = model(s_d, aA_d).squeeze(-1)
        rB = model(s_d, aB_d).squeeze(-1)
        diff = rA - rB
        loss = F.binary_cross_entropy_with_logits(diff, y_d, reduction="mean").item()
        # Predictive label: tie if |diff|<1e-3 else 1 if rA>rB else 0
        pred = torch.where(
            diff.abs() < 1e-3,
            torch.full_like(diff, 0.5),
            (diff > 0).float(),
        )
        match = (pred == y_d).float().mean().item()
    return float(loss), float(match)


def train(
    pairs: list[PreferencePair],
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 256,
    device: str = "cpu",
    val_frac: float = 0.15,
    seed: int = 0,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[RewardMLP, dict[str, list[float]]]:
    """Train a :class:`RewardMLP` with Bradley-Terry loss.

    Returns the trained model and a history dict with per-epoch
    ``train_loss``, ``val_loss`` and ``val_acc`` lists.
    """
    s, aA, aB, y = make_dataset(pairs)
    n = s.shape[0]
    if n == 0:
        raise ValueError("No labeled pairs to train on.")

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    s_tr, aA_tr, aB_tr, y_tr = s[train_idx], aA[train_idx], aB[train_idx], y[train_idx]
    s_va, aA_va, aB_va, y_va = s[val_idx], aA[val_idx], aB[val_idx], y[val_idx]

    torch.manual_seed(int(seed))
    model = RewardMLP(state_dim=s.shape[1], action_dim=aA.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(s_tr, aA_tr, aB_tr, y_tr)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_acc": [],
    }

    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        n_batches = 0
        for s_b, aA_b, aB_b, y_b in loader:
            s_b = s_b.to(device); aA_b = aA_b.to(device)
            aB_b = aB_b.to(device); y_b = y_b.to(device)
            rA = model(s_b, aA_b).squeeze(-1)
            rB = model(s_b, aB_b).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(rA - rB, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        train_loss = running / max(1, n_batches)
        val_loss, val_acc = _eval_split(model, s_va, aA_va, aB_va, y_va, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(
                f"epoch {epoch + 1:>2}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}",
                flush=True,
            )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Persist the full model object so downstream wrappers can load it
        # without needing to know the architecture.
        torch.save({
            "model": model,
            "state_dict": model.state_dict(),
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
        }, save_dir / "reward_model.pt")
        with (save_dir / "reward_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return model, history
