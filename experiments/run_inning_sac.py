"""Train + evaluate SAC or TD3 on the multi-shot inning env.

Reward per env step = score for that shot (0 or 1). The episode is a
Korean-4-ball *inning*: cumulative reward across shots until the policy
misses, fouls, or hits the per-inning cap. Compared to the single-shot
``Billiards4BallEnv`` used in Phase F/G, this exposes credit for
*chaining* scoring shots in one possession.

PPO is kept in commented-out form for reference (Phase II_b excludes it
from training since SAC dominated 100% vs PPO 0% in the prior matrix).

Outputs under ``{out_dir}/{run_id}/``:
    training_curve.csv  per-rollout aggregates (ep_return = inning score)
    eval.parquet        one row per evaluation inning (200 by default)
    policy.zip
    config.json
    run.log

Usage:
    uv run python experiments/run_inning_sac.py \\
        --algo sac --seed 0 --total_steps 50000 --eval_episodes 200 \\
        --out_dir experiments/runs_inning/sac_s0

    uv run python experiments/run_inning_sac.py \\
        --algo td3 --seed 0 --total_steps 50000 --eval_episodes 200 \\
        --out_dir experiments/runs_inning/td3_s0
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3 import PPO, SAC, TD3  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402


# ---- shared hyperparameters ----------------------------------------------
T_MAX = 12.0
GAMMA = 0.99
LR = 3e-4

# SAC
SAC_BATCH = 256
SAC_BUFFER = 200_000
SAC_LEARNING_STARTS = 1_000

# TD3
TD3_BATCH = 256
TD3_BUFFER = 200_000
TD3_LEARNING_STARTS = 1_000
TD3_ACTION_NOISE_SIGMA = 0.1  # exploration noise stddev (action space scale)

# PPO (excluded from training — kept for reference)
# PPO_N_STEPS = 512
# PPO_BATCH = 512
# PPO_N_EPOCHS = 4
# PPO_GAE = 0.95
# PPO_CLIP = 0.2
# PPO_VF = 0.5
# PPO_ENT = 0.01


# ---------------------------------------------------------------- IO helpers


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
                st.flush()
            except Exception:  # noqa: BLE001
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------- env factory


def _make_train_env(max_shots: int, seed: int) -> Monitor:
    """Single-env wrapper: Monitor records ep_return / ep_length and
    forwards per-step ``cushion_hits`` / ``fouled`` so we can mean them
    over an inning."""
    env = Billiards4BallInningEnv(t_max=T_MAX, max_shots=max_shots)
    env = Monitor(env, info_keywords=("cushion_hits", "fouled", "score"))
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------- callbacks


class InningCurveCallback(BaseCallback):
    """Append rollout aggregates to ``training_curve.csv``.

    The Monitor wraps the inning env with terminated=True at the end of
    each inning, so SB3's ``ep_info_buffer`` collects one entry per
    inning (with ``r`` = cumulative inning score = inning_score and ``l``
    = number of shots in the inning). We additionally average our
    ``info_keywords`` across the steps that fed this inning into the
    buffer; with Monitor these are recorded only at episode end as the
    *last* shot's value, so we additionally tally them ourselves on
    ``_on_step``.
    """

    HEADER = [
        "timesteps",
        "ep_return_mean", "ep_length_mean",
        "ep_cushion_mean", "ep_foul_rate",
        "n_episodes_in_window",
    ]

    def __init__(self, csv_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(self.HEADER)
        # Per-shot tallies, flushed at end of each inning.
        self._cur_cushions: list[int] = []
        self._cur_score: int = 0
        self._cur_shots: int = 0
        self._cur_fouls: int = 0
        # Completed-inning stats since last write (window).
        self._win_returns: list[float] = []
        self._win_lengths: list[int] = []
        self._win_cushion_means: list[float] = []
        self._win_foul_any: list[float] = []
        self._next_log = 0
        self._log_every = 2048  # flush every ~2k env steps

    def _on_step(self) -> bool:  # noqa: D401
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            dones = [False] * len(infos)
        for info, done in zip(infos, dones):
            cushion = int(info.get("cushion_hits", 0)) if info else 0
            self._cur_cushions.append(cushion)
            self._cur_score += int(info.get("score", 0)) if info else 0
            self._cur_shots += 1
            if info and bool(info.get("fouled", False)):
                self._cur_fouls += 1
            if done:
                self._win_returns.append(float(self._cur_score))
                self._win_lengths.append(int(self._cur_shots))
                self._win_cushion_means.append(
                    float(np.mean(self._cur_cushions)) if self._cur_cushions else 0.0
                )
                self._win_foul_any.append(1.0 if self._cur_fouls > 0 else 0.0)
                self._cur_cushions = []
                self._cur_score = 0
                self._cur_shots = 0
                self._cur_fouls = 0

        if self.num_timesteps - self._next_log >= self._log_every:
            self._flush_window()
            self._next_log = int(self.num_timesteps)
        return True

    def _flush_window(self) -> None:
        if not self._win_returns:
            return
        row = [
            int(self.num_timesteps),
            f"{float(np.mean(self._win_returns)):.4f}",
            f"{float(np.mean(self._win_lengths)):.4f}",
            f"{float(np.mean(self._win_cushion_means)):.4f}",
            f"{float(np.mean(self._win_foul_any)):.4f}",
            int(len(self._win_returns)),
        ]
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
        self._win_returns.clear()
        self._win_lengths.clear()
        self._win_cushion_means.clear()
        self._win_foul_any.clear()

    def _on_training_end(self) -> None:
        self._flush_window()


# ---------------------------------------------------------------- evaluation


def _evaluate(model, n_episodes: int, seed_base: int, max_shots: int) -> pd.DataFrame:
    """Run ``n_episodes`` deterministic innings; one row per inning."""
    env = Billiards4BallInningEnv(t_max=T_MAX, max_shots=max_shots)
    rows: list[dict] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        cushions = 0
        n_shots = 0
        fouled = False
        total_dur = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            obs, _, terminated, truncated, info = env.step(action)
            cushions += int(info.get("cushion_hits", 0))
            total_dur += float(info.get("duration", 0.0))
            n_shots += 1
            if bool(info.get("fouled", False)):
                fouled = True
            if terminated or truncated:
                break
        rows.append({
            "ep_idx": int(ep),
            "seed": int(seed_base + ep),
            "inning_score": int(env.cumulative_score),
            "n_shots": int(n_shots),
            "mean_cushions": float(cushions / max(1, n_shots)),
            "fouled": bool(fouled),
            "total_duration": float(total_dur),
            "truncated": bool(truncated),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Inning-env SAC/PPO baseline.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--max_shots", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--algo", type=str, choices=("sac", "td3"), default="sac")
    parser.add_argument("--out_dir", type=str, default="experiments/runs_inning")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_f = log_path.open("w", encoding="utf-8")
    tee = _Tee(sys.__stdout__, log_f)
    err_tee = _Tee(sys.__stderr__, log_f)

    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err_tee):
        t0 = time.perf_counter()
        config = {
            "algo": args.algo,
            "seed": int(args.seed),
            "total_steps": int(args.total_steps),
            "max_shots": int(args.max_shots),
            "eval_episodes": int(args.eval_episodes),
            "t_max": T_MAX,
            "gamma": GAMMA,
            "learning_rate": LR,
        }
        if args.algo == "sac":
            config.update({
                "batch_size": SAC_BATCH,
                "buffer_size": SAC_BUFFER,
                "learning_starts": SAC_LEARNING_STARTS,
            })
        elif args.algo == "td3":
            config.update({
                "batch_size": TD3_BATCH,
                "buffer_size": TD3_BUFFER,
                "learning_starts": TD3_LEARNING_STARTS,
                "action_noise_sigma": TD3_ACTION_NOISE_SIGMA,
            })
        # else:
        #     config.update({
        #         "n_steps": PPO_N_STEPS,
        #         "batch_size": PPO_BATCH,
        #         "n_epochs": PPO_N_EPOCHS,
        #         "gae_lambda": PPO_GAE,
        #         "clip_range": PPO_CLIP,
        #         "vf_coef": PPO_VF,
        #         "ent_coef": PPO_ENT,
        #     })
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"[run_inning] algo={args.algo} seed={args.seed} "
              f"total_steps={args.total_steps} max_shots={args.max_shots} "
              f"out_dir={out_dir}")

        set_random_seed(int(args.seed))
        env = _make_train_env(max_shots=int(args.max_shots), seed=int(args.seed))

        if args.algo == "sac":
            model = SAC(
                policy="MlpPolicy",
                env=env,
                learning_rate=LR,
                buffer_size=SAC_BUFFER,
                batch_size=SAC_BATCH,
                gamma=GAMMA,
                learning_starts=SAC_LEARNING_STARTS,
                seed=int(args.seed),
                verbose=0,
                device="cpu",
            )
        elif args.algo == "td3":
            n_actions = env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions, dtype=np.float32),
                sigma=TD3_ACTION_NOISE_SIGMA * np.ones(n_actions, dtype=np.float32),
            )
            model = TD3(
                policy="MlpPolicy",
                env=env,
                learning_rate=LR,
                buffer_size=TD3_BUFFER,
                batch_size=TD3_BATCH,
                gamma=GAMMA,
                learning_starts=TD3_LEARNING_STARTS,
                action_noise=action_noise,
                seed=int(args.seed),
                verbose=0,
                device="cpu",
            )
        # else:
        #     model = PPO(
        #         policy="MlpPolicy",
        #         env=env,
        #         learning_rate=LR,
        #         n_steps=PPO_N_STEPS,
        #         batch_size=PPO_BATCH,
        #         n_epochs=PPO_N_EPOCHS,
        #         gamma=GAMMA,
        #         gae_lambda=PPO_GAE,
        #         clip_range=PPO_CLIP,
        #         vf_coef=PPO_VF,
        #         ent_coef=PPO_ENT,
        #         seed=int(args.seed),
        #         verbose=0,
        #         device="cpu",
        #     )

        try:
            cb = InningCurveCallback(out_dir / "training_curve.csv")
            t_train0 = time.perf_counter()
            model.learn(total_timesteps=int(args.total_steps), callback=cb,
                        progress_bar=False)
            train_wall = time.perf_counter() - t_train0
            print(f"[run_inning] train done in {train_wall:.1f}s")

            policy_path = out_dir / "policy.zip"
            model.save(str(policy_path))
            print(f"[run_inning] saved policy -> {policy_path}")

            t_eval0 = time.perf_counter()
            eval_df = _evaluate(
                model,
                n_episodes=int(args.eval_episodes),
                seed_base=int(args.seed) + 10_000,
                max_shots=int(args.max_shots),
            )
            eval_wall = time.perf_counter() - t_eval0
            eval_path = out_dir / "eval.parquet"
            eval_df.to_parquet(eval_path, engine="pyarrow", index=False)
            print(f"[run_inning] eval n={len(eval_df)} done in {eval_wall:.1f}s "
                  f"-> {eval_path}")

            mean_inning = float(eval_df["inning_score"].mean())
            max_inning = int(eval_df["inning_score"].max())
            p_ge1 = 100.0 * float((eval_df["inning_score"] >= 1).mean())
            p_ge3 = 100.0 * float((eval_df["inning_score"] >= 3).mean())
            p_ge5 = 100.0 * float((eval_df["inning_score"] >= 5).mean())
            mean_shots = float(eval_df["n_shots"].mean())
            foul_rate = 100.0 * float(eval_df["fouled"].mean())
            mean_cushions = float(eval_df["mean_cushions"].mean())
            wall = time.perf_counter() - t0
            summary = {
                "algo": args.algo,
                "seed": int(args.seed),
                "train_wall_s": float(train_wall),
                "eval_wall_s": float(eval_wall),
                "wall_s": float(wall),
                "mean_inning_score": mean_inning,
                "max_inning_score": max_inning,
                "p_score_ge1": p_ge1,
                "p_score_ge3": p_ge3,
                "p_score_ge5": p_ge5,
                "mean_shots": mean_shots,
                "foul_rate": foul_rate,
                "mean_cushions": mean_cushions,
            }
            with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(
                f"[run_inning] DONE algo={args.algo} seed={args.seed} "
                f"mean_inning={mean_inning:.3f} max={max_inning} "
                f"p>=1={p_ge1:.1f}% p>=3={p_ge3:.1f}% p>=5={p_ge5:.1f}% "
                f"mean_shots={mean_shots:.2f} foul%={foul_rate:.2f} "
                f"wall={wall:.1f}s"
            )
        finally:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
            log_f.close()


if __name__ == "__main__":
    main()
