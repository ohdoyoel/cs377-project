"""Evaluate Random / GeometricAim / PPO policies on Billiards4BallEnv.

Each policy plays ``--n_episodes`` deterministic-seeded episodes; per-episode
metrics (action, score, fouled, cushion_hits, duration) are written to a
single parquet file, with a ``policy`` column distinguishing the runs.

Usage:
    uv run python scripts/eval_policies.py --n_episodes 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from billiards.env import Billiards4BallEnv  # noqa: E402
from policies.geometric_aim import GeometricAimPolicy  # noqa: E402
from policies.random_policy import RandomPolicy  # noqa: E402


def _ppo_act_fn(policy_path: Path):
    """Return ``act(obs) -> action`` backed by a saved SB3 PPO policy."""
    from stable_baselines3 import PPO  # local import — only needed for ppo

    model = PPO.load(str(policy_path), device="cpu")

    def _act(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    return _act


def _eval_policy(name: str, act_fn, n_episodes: int, base_seed: int) -> list[dict]:
    env = Billiards4BallEnv()
    rows: list[dict] = []
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        action = act_fn(obs)
        _, _, _, _, info = env.step(action)
        rows.append(
            {
                "policy": name,
                "ep": int(ep),
                "seed": int(base_seed + ep),
                "theta": float(action[0]),
                "power": float(action[1]),
                "a": float(action[2]),
                "b": float(action[3]),
                "score": int(info["score"]),
                "fouled": bool(info["fouled"]),
                "cushion_hits": int(info["cushion_hits"]),
                "duration": float(info["duration"]),
            }
        )
        if (ep + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            fps = (ep + 1) / max(1e-9, elapsed)
            print(f"[eval_policies] {name}: {ep + 1}/{n_episodes} ({fps:.1f} eps/s)")
    print(f"[eval_policies] {name} done in {time.perf_counter() - t0:.1f}s")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline + PPO policies.")
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--out_path", type=str, default="data/eval_results.parquet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppo_path", type=str, default="models/ppo_policy.zip")
    args = parser.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ppo_path = Path(args.ppo_path)

    rows: list[dict] = []

    print(f"[eval_policies] random policy ({args.n_episodes} eps)...")
    random_policy = RandomPolicy(seed=args.seed)
    rows.extend(_eval_policy("random", random_policy.act,
                             args.n_episodes, args.seed))

    print(f"[eval_policies] geometric policy ({args.n_episodes} eps)...")
    geo_policy = GeometricAimPolicy()
    rows.extend(_eval_policy("geometric", geo_policy.act,
                             args.n_episodes, args.seed))

    if ppo_path.exists():
        print(f"[eval_policies] ppo policy ({args.n_episodes} eps) ← {ppo_path}")
        ppo_act = _ppo_act_fn(ppo_path)
        rows.extend(_eval_policy("ppo", ppo_act, args.n_episodes, args.seed))
    else:
        print(f"[eval_policies] WARNING: PPO policy not found at {ppo_path}; skipping")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"[eval_policies] wrote {len(df)} rows → {out_path}")

    # Summary table
    print("\n[eval_policies] summary:")
    print(f"  {'policy':<10} {'n':>6} {'score%':>8} {'foul%':>8} "
          f"{'mean_cush':>10} {'mean_dur':>10}")
    for policy in df["policy"].unique():
        sub = df[df["policy"] == policy]
        n = len(sub)
        sr = 100.0 * sub["score"].mean()
        fr = 100.0 * sub["fouled"].mean()
        mc = sub["cushion_hits"].mean()
        md = sub["duration"].mean()
        print(f"  {policy:<10} {n:>6} {sr:>7.1f}% {fr:>7.1f}% "
              f"{mc:>10.2f} {md:>10.2f}")


if __name__ == "__main__":
    main()
