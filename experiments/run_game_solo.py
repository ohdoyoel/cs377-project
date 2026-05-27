"""Train + evaluate SAC on the time-budget "solo game" env.

Unlike ``run_inning_sac.py`` (episode = one inning, ends on miss/foul or a
shot-count cap), here an episode is a *game* bounded by a fixed budget of
simulated table time (``--budget_s``). The agent plays innings back to back —
a miss resets the rack — and reward is the points scored within the budget.
Faster shots and quick-follow-up positions therefore score higher
*structurally*, with no explicit time bonus needed. See
``billiards/wrappers/time_budget_game_env.py``.

Design Option 2: observation is unchanged (no remaining-time feature) so we can
warm-start an inning-trained policy without an input-dim change. Recommended
start is the *non-time* SOTA so the new objective shapes a neutral base:

    python experiments/run_game_solo.py --seed 4 \\
        --load_policy experiments/runs_inning_v2/fast_long_fp02_s4/policy.zip \\
        --out_dir experiments/runs_game_solo/game_s4 \\
        --total_steps 150000 --budget_s 120 \\
        --constrain_aim --extra_features --gentle_shot --setup_shaping \\
        --n_envs 8 --gradient_steps 2   # foul_penalty defaults to 1.0 (-1 per foul)

Outputs under ``{out_dir}/``: policy.zip, config.json, summary.json,
training_curve.csv, run.log.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402
from billiards.wrappers.time_budget_game_env import TimeBudgetGameEnv  # noqa: E402
# Reuse logging tee, the per-episode CSV callback, and shared hyperparams.
from experiments.run_inning_sac import (  # noqa: E402
    _Tee, InningCurveCallback, T_MAX, GAMMA, LR,
    SAC_BATCH, SAC_BUFFER, SAC_LEARNING_STARTS,
)

_INFO_KEYS = ("cushion_hits", "fouled", "score",
              "game_score", "game_sim_time", "n_innings")


# ---------------------------------------------------------------- env factory


def _game_env_factory(
    budget_s: float,
    inner_max_shots: int,
    seed: int,
    constrain_aim: bool,
    extra_features: bool,
    foul_penalty: float,
    gentle_shot: bool,
    setup_shaping: bool,
    setup_alpha: float,
    setup_scale: float,
    time_reward: bool,
    time_alpha: float,
    time_scale: float,
):
    def _thunk():
        env = Billiards4BallInningEnv(
            t_max=T_MAX,
            max_shots=inner_max_shots,      # large: an inning ends on miss, not cap
            continue_on_miss=False,         # required by TimeBudgetGameEnv
            constrain_aim=constrain_aim,
            extra_features=extra_features,
            foul_penalty=foul_penalty,
            gentle_shot=gentle_shot,
            setup_shaping=setup_shaping,
            setup_alpha=setup_alpha,
            setup_scale=setup_scale,
            time_reward=time_reward,
            time_alpha=time_alpha,
            time_scale=time_scale,
        )
        env = RandomStartInningEnv(env)              # fresh random rack per inning
        env = TimeBudgetGameEnv(env, budget_s=budget_s)
        env = Monitor(env, info_keywords=_INFO_KEYS)
        env.reset(seed=seed)
        return env
    return _thunk


# ---------------------------------------------------------------- evaluation


def _evaluate_game(
    model,
    n_games: int,
    seed_base: int,
    budget_s: float,
    inner_max_shots: int,
    constrain_aim: bool,
    extra_features: bool,
) -> pd.DataFrame:
    """Run ``n_games`` deterministic time-budget games; one row per game.

    Metric of interest is ``game_score`` = pure points scored within budget
    (shaping excluded). Also reports innings played, shots, sim-time, fouls.
    """
    base = Billiards4BallInningEnv(
        t_max=T_MAX, max_shots=inner_max_shots, continue_on_miss=False,
        constrain_aim=constrain_aim, extra_features=extra_features,
    )
    env = TimeBudgetGameEnv(RandomStartInningEnv(base), budget_s=budget_s)
    rows: list[dict] = []
    for g in range(n_games):
        obs, _ = env.reset(seed=seed_base + g)
        n_shots, cushions, fouls = 0, 0, 0
        info: dict = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32).reshape(-1)
            )
            n_shots += 1
            cushions += int(info.get("cushion_hits", 0))
            if bool(info.get("fouled", False)):
                fouls += 1
            if terminated or truncated:
                break
        rows.append({
            "game_idx": int(g),
            "seed": int(seed_base + g),
            "game_score": int(info.get("game_score", 0)),
            "n_innings": int(info.get("n_innings", 0)),
            "n_shots": int(n_shots),
            "sim_time_s": float(info.get("game_sim_time", 0.0)),
            "mean_cushions": float(cushions / max(1, n_shots)),
            "foul_rate": float(fouls / max(1, n_shots)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description="Time-budget solo-game SAC trainer.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total_steps", type=int, default=150_000)
    p.add_argument("--budget_s", type=float, default=120.0,
                   help="Simulated-time budget per game (seconds).")
    p.add_argument("--inner_max_shots", type=int, default=1000,
                   help="Per-inning shot cap (large; budget is the real limit).")
    p.add_argument("--eval_episodes", type=int, default=50, help="Eval games.")
    p.add_argument("--out_dir", type=str, default="experiments/runs_game_solo")
    p.add_argument("--load_policy", type=str, default=None,
                   help="Warm-start policy .zip (recommend fast_long_fp02_s4).")
    p.add_argument("--constrain_aim", action="store_true")
    p.add_argument("--extra_features", action="store_true")
    # Game reward per shot: score (+ shaping) on a make, 0 on a miss, and
    # -foul_penalty on a foul. Default 1.0 => a foul costs a full point (-1),
    # both miss and foul then reset the rack (handled by TimeBudgetGameEnv).
    p.add_argument("--foul_penalty", type=float, default=1.0)
    p.add_argument("--gentle_shot", action="store_true")
    p.add_argument("--setup_shaping", action="store_true")
    p.add_argument("--setup_alpha", type=float, default=0.05)
    p.add_argument("--setup_scale", type=float, default=0.3)
    # time_reward shaping is OFF by default here: the budget structure replaces
    # it. Exposed so it can be ablated against the structural signal.
    p.add_argument("--time_reward", action="store_true")
    p.add_argument("--time_alpha", type=float, default=0.2)
    p.add_argument("--time_scale", type=float, default=3.0)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--gradient_steps", type=int, default=2)
    p.add_argument("--net_arch", type=str, default="")
    p.add_argument("--gamma", type=float, default=GAMMA,
                   help="Consider 0.995+ to approximate reward-rate (points/time).")
    p.add_argument("--buffer_size", type=int, default=SAC_BUFFER)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "run.log").open("w", encoding="utf-8")
    tee, err_tee = _Tee(sys.__stdout__, log_f), _Tee(sys.__stderr__, log_f)

    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err_tee):
        t0 = time.perf_counter()
        config = {
            "seed": int(args.seed), "total_steps": int(args.total_steps),
            "budget_s": float(args.budget_s), "inner_max_shots": int(args.inner_max_shots),
            "constrain_aim": bool(args.constrain_aim), "extra_features": bool(args.extra_features),
            "foul_penalty": float(args.foul_penalty), "gentle_shot": bool(args.gentle_shot),
            "setup_shaping": bool(args.setup_shaping), "setup_alpha": float(args.setup_alpha),
            "setup_scale": float(args.setup_scale), "time_reward": bool(args.time_reward),
            "time_alpha": float(args.time_alpha), "time_scale": float(args.time_scale),
            "n_envs": int(args.n_envs), "gradient_steps": int(args.gradient_steps),
            "net_arch": str(args.net_arch), "load_policy": args.load_policy,
            "eval_episodes": int(args.eval_episodes), "gamma": float(args.gamma),
            "buffer_size": int(args.buffer_size), "t_max": T_MAX, "learning_rate": LR,
        }
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"[run_game_solo] seed={args.seed} total_steps={args.total_steps} "
              f"budget_s={args.budget_s} time_reward={args.time_reward} "
              f"n_envs={args.n_envs} load_policy={args.load_policy} out_dir={out_dir}")

        set_random_seed(int(args.seed))
        env_kw = dict(
            budget_s=float(args.budget_s), inner_max_shots=int(args.inner_max_shots),
            constrain_aim=bool(args.constrain_aim), extra_features=bool(args.extra_features),
            foul_penalty=float(args.foul_penalty), gentle_shot=bool(args.gentle_shot),
            setup_shaping=bool(args.setup_shaping), setup_alpha=float(args.setup_alpha),
            setup_scale=float(args.setup_scale), time_reward=bool(args.time_reward),
            time_alpha=float(args.time_alpha), time_scale=float(args.time_scale),
        )
        n_envs = int(args.n_envs)
        factories = [_game_env_factory(seed=int(args.seed) + i, **env_kw)
                     for i in range(n_envs)]
        env = DummyVecEnv(factories) if n_envs <= 1 else SubprocVecEnv(factories)

        policy_kwargs = None
        if args.net_arch:
            policy_kwargs = {"net_arch": [int(x) for x in args.net_arch.split(",") if x.strip()]}

        if args.load_policy:
            model = SAC.load(args.load_policy, env=env, device="cpu")
            model.gamma = float(args.gamma)
            model.gradient_steps = int(args.gradient_steps)
            print(f"[run_game_solo] warm-start SAC <- {args.load_policy} "
                  f"(gamma={model.gamma}, gradient_steps={model.gradient_steps})")
        else:
            model = SAC(
                policy="MlpPolicy", env=env, learning_rate=LR,
                buffer_size=int(args.buffer_size), batch_size=SAC_BATCH,
                gamma=float(args.gamma), learning_starts=SAC_LEARNING_STARTS,
                gradient_steps=int(args.gradient_steps), policy_kwargs=policy_kwargs,
                seed=int(args.seed), verbose=0, device="cpu",
            )

        try:
            cb = InningCurveCallback(out_dir / "training_curve.csv")
            t_train0 = time.perf_counter()
            model.learn(total_timesteps=int(args.total_steps), callback=cb, progress_bar=False)
            train_wall = time.perf_counter() - t_train0
            model.save(str(out_dir / "policy.zip"))
            print(f"[run_game_solo] train done {train_wall:.1f}s -> {out_dir/'policy.zip'}")

            t_eval0 = time.perf_counter()
            eval_df = _evaluate_game(
                model, n_games=int(args.eval_episodes),
                seed_base=int(args.seed) + 10_000, budget_s=float(args.budget_s),
                inner_max_shots=int(args.inner_max_shots),
                constrain_aim=bool(args.constrain_aim), extra_features=bool(args.extra_features),
            )
            eval_wall = time.perf_counter() - t_eval0
            eval_df.to_parquet(out_dir / "eval_games.parquet", engine="pyarrow", index=False)

            summary = {
                "seed": int(args.seed), "budget_s": float(args.budget_s),
                "train_wall_s": float(train_wall), "eval_wall_s": float(eval_wall),
                "wall_s": float(time.perf_counter() - t0),
                "mean_game_score": float(eval_df["game_score"].mean()),
                "std_game_score": float(eval_df["game_score"].std()),
                "max_game_score": int(eval_df["game_score"].max()),
                "mean_innings": float(eval_df["n_innings"].mean()),
                "mean_shots": float(eval_df["n_shots"].mean()),
                "mean_sim_time_s": float(eval_df["sim_time_s"].mean()),
                "score_per_sim_s": float(eval_df["game_score"].sum()
                                         / max(1e-9, eval_df["sim_time_s"].sum())),
                "mean_cushions": float(eval_df["mean_cushions"].mean()),
                "foul_rate": float(eval_df["foul_rate"].mean()),
            }
            with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[run_game_solo] DONE seed={args.seed} "
                  f"mean_game_score={summary['mean_game_score']:.2f} "
                  f"score/sim_s={summary['score_per_sim_s']:.3f} "
                  f"mean_innings={summary['mean_innings']:.1f} "
                  f"mean_shots={summary['mean_shots']:.1f} "
                  f"wall={summary['wall_s']:.1f}s")
        finally:
            with contextlib.suppress(Exception):
                env.close()
            log_f.close()


if __name__ == "__main__":
    main()
