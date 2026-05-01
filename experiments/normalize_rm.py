"""Compute (mu, sigma) over 1000 random (state, action) pairs.

Run once before the alpha-sweep. Persists to
``experiments/rm_normalization.json`` for reuse by ``MixedRewardEnv``.

Usage:
    uv run python experiments/normalize_rm.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from billiards.env import Billiards4BallEnv  # noqa: E402
from billiards.wrappers.reward_model_env import _load_reward_model  # noqa: E402


def compute_norm_stats(
    model_path: str | Path,
    n_samples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Sample ``n_samples`` random (s, a) pairs, return (mu, sigma, raw_outputs).

    ``s`` is drawn from ``Billiards4BallEnv.reset(seed=...)`` (initial state
    is deterministic but identical, so we re-roll seed each sample) and
    ``a`` is drawn uniformly from the 4-D action space.
    """
    rng = np.random.default_rng(seed)
    env = Billiards4BallEnv()
    model = _load_reward_model(model_path, device="cpu")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for i in range(n_samples):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        a_low = env.action_space.low
        a_high = env.action_space.high
        a = rng.uniform(a_low, a_high).astype(np.float32)
        actions.append(a)

    s_t = torch.tensor(np.stack(states), dtype=torch.float32)
    a_t = torch.tensor(np.stack(actions), dtype=torch.float32)
    with torch.no_grad():
        out = model(s_t, a_t).squeeze(-1).cpu().numpy()
    out = out.reshape(-1).astype(np.float64)

    mu = float(out.mean())
    sigma = float(out.std(ddof=0))
    if sigma <= 0.0:
        sigma = 1.0  # safety: degenerate constant model
    return mu, sigma, out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute reward-model normalization stats."
    )
    parser.add_argument("--model_path", type=str, default="models/reward_model.pt")
    parser.add_argument(
        "--out_path", type=str, default="experiments/rm_normalization.json"
    )
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    mu, sigma, raw = compute_norm_stats(
        args.model_path, n_samples=args.n_samples, seed=args.seed
    )
    elapsed = time.perf_counter() - t0

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mu": mu,
        "sigma": sigma,
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "model_path": str(args.model_path),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "elapsed_s": elapsed,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[normalize_rm] n={args.n_samples} mu={mu:.4f} sigma={sigma:.4f} "
        f"min={raw.min():.4f} max={raw.max():.4f} elapsed={elapsed:.1f}s"
    )
    print(f"[normalize_rm] wrote -> {out_path}")


if __name__ == "__main__":
    main()
