"""Train a PPO policy on Billiards4BallEnv against the learned reward model.

Usage:
    uv run python scripts/train_ppo.py --total_steps 100000 --n_envs 8

After training, runs a 200-episode true-score evaluation on the
*unwrapped* env and prints the mean true score rate, foul rate, and
mean cushion hits.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow `uv run python scripts/train_ppo.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from billiards.env import Billiards4BallEnv  # noqa: E402
from billiards.wrappers import RewardModelEnv  # noqa: E402

REWARD_CHOICES = ("model", "env")


class HistoryCallback(BaseCallback):
    """Append per-rollout aggregates to a CSV after every PPO update.

    SB3 emits its own scalars to the logger; we ride the same updates
    by overriding ``_on_rollout_end`` and writing ep_rew_mean / fps /
    elapsed_steps for later plotting from a single deterministic file.
    """

    def __init__(self, csv_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self._t0 = time.perf_counter()
        # Open in 'w' mode at the start of training, with header.
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                ["timesteps", "wall_seconds", "ep_rew_mean", "ep_len_mean", "fps"]
            )

    def _on_step(self) -> bool:  # noqa: D401 - SB3 hook
        return True

    def _on_rollout_end(self) -> None:
        ep_buf = list(self.model.ep_info_buffer or [])
        if ep_buf:
            ep_rew = float(np.mean([e["r"] for e in ep_buf]))
            ep_len = float(np.mean([e["l"] for e in ep_buf]))
        else:
            ep_rew = float("nan")
            ep_len = float("nan")
        elapsed = max(1e-9, time.perf_counter() - self._t0)
        fps = float(self.num_timesteps) / elapsed
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                [int(self.num_timesteps), f"{elapsed:.2f}", f"{ep_rew:.4f}",
                 f"{ep_len:.2f}", f"{fps:.1f}"]
            )


def _make_env_fn(
    reward_kind: str,
    model_path: str,
    t_max: float,
    seed: int,
):
    """Return a no-arg callable usable by SB3 vec-env constructors."""

    def _factory():
        env = Billiards4BallEnv(t_max=t_max)
        if reward_kind == "model":
            env = RewardModelEnv(env, model_path=model_path, device="cpu")
        env.reset(seed=seed)
        return env

    return _factory


def _make_vec_env(
    n_envs: int,
    reward_kind: str,
    model_path: str,
    t_max: float,
    seed: int,
    use_subproc: bool,
):
    fns = [
        _make_env_fn(reward_kind, model_path, t_max, seed + i) for i in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)


def _evaluate_true_score(model: PPO, n_episodes: int, t_max: float, seed: int) -> dict:
    """Roll the trained policy on the *unwrapped* env and report metrics."""
    env = Billiards4BallEnv(t_max=t_max)
    scores = []
    fouls = []
    cushions = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        scores.append(int(info["score"]))
        fouls.append(bool(info["fouled"]))
        cushions.append(int(info["cushion_hits"]))
    n = max(1, len(scores))
    return {
        "n": n,
        "score_rate": 100.0 * sum(scores) / n,
        "foul_rate": 100.0 * sum(1 for f in fouls if f) / n,
        "mean_cushions": float(np.mean(cushions)) if cushions else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Billiards4BallEnv.")
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--reward", choices=REWARD_CHOICES, default="model")
    parser.add_argument("--model_path", type=str, default="models/reward_model.pt")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t_max", type=float, default=12.0)
    parser.add_argument("--no_subproc", action="store_true",
                        help="Force DummyVecEnv even with n_envs>1 (debug).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_csv = out_dir / "ppo_history.csv"
    policy_path = out_dir / "ppo_policy.zip"

    if args.reward == "model" and not Path(args.model_path).exists():
        raise SystemExit(
            f"reward model not found at {args.model_path}; train Phase D first or "
            f"pass --reward env to fall back to the integer score signal."
        )

    print(
        f"[train_ppo] reward={args.reward} n_envs={args.n_envs} "
        f"total_steps={args.total_steps} seed={args.seed} t_max={args.t_max}"
    )

    use_subproc = (not args.no_subproc) and (args.n_envs > 1)
    vec_env = _make_vec_env(
        n_envs=args.n_envs,
        reward_kind=args.reward,
        model_path=args.model_path,
        t_max=args.t_max,
        seed=args.seed,
        use_subproc=use_subproc,
    )

    # n_steps × n_envs == rollout buffer size; keep batch_size dividing it.
    n_steps = 512
    batch_size = 512
    rollout = n_steps * args.n_envs
    if rollout % batch_size != 0:
        # Find the largest divisor of rollout that ≤ 512 to avoid SB3 warnings.
        for cand in (512, 256, 128, 64):
            if rollout % cand == 0:
                batch_size = cand
                break

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    callback = HistoryCallback(csv_path=history_csv)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.total_steps, callback=callback,
                progress_bar=False)
    train_secs = time.perf_counter() - t0
    print(f"[train_ppo] training done in {train_secs:.1f}s")

    model.save(str(policy_path))
    print(f"[train_ppo] saved policy → {policy_path}")
    print(f"[train_ppo] history CSV → {history_csv}")

    # Final ep_rew_mean from the buffer. Single-shot env: SB3 only writes to
    # ep_info_buffer when episode termination is detected via the Monitor
    # wrapper, which we don't add — so this is usually nan. Fall back to
    # sampling rewards directly from the trained policy.
    final_ep_rew = float("nan")
    if model.ep_info_buffer:
        final_ep_rew = float(np.mean([e["r"] for e in model.ep_info_buffer]))
    if not (final_ep_rew == final_ep_rew):  # NaN guard
        env_probe = Billiards4BallEnv(t_max=args.t_max)
        if args.reward == "model":
            env_probe = RewardModelEnv(env_probe, model_path=args.model_path,
                                       device="cpu")
        rewards = []
        for ep in range(50):
            obs, _ = env_probe.reset(seed=args.seed + 5000 + ep)
            action, _ = model.predict(obs, deterministic=True)
            _, r, _, _, _ = env_probe.step(action)
            rewards.append(float(r))
        final_ep_rew = float(np.mean(rewards)) if rewards else float("nan")
    print(f"[train_ppo] final ep_rew_mean (model reward): {final_ep_rew:.4f}")

    # 200-episode true-score evaluation on the *unwrapped* env.
    eval_result = _evaluate_true_score(
        model, n_episodes=200, t_max=args.t_max, seed=args.seed + 10_000
    )
    print(
        f"[train_ppo] eval n={eval_result['n']} "
        f"true_score_rate={eval_result['score_rate']:.1f}% "
        f"foul_rate={eval_result['foul_rate']:.1f}% "
        f"mean_cushions={eval_result['mean_cushions']:.2f}"
    )

    # Cleanup vec env (subproc workers).
    try:
        vec_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
