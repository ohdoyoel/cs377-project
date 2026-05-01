"""Train + evaluate one (alpha, seed) PPO config for the alpha-sweep.

Per-rollout training metrics, final policy, 500-episode true-score eval
parquet, heuristic component attribution JSON, and the run config are all
written under ``--out_dir``.

Usage (single run):
    uv run python experiments/run_one.py \\
        --alpha 0.5 --seed 0 --total_steps 50000 --eval_episodes 500 \\
        --out_dir experiments/runs/a0.5_s0
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from billiards.env import Billiards4BallEnv  # noqa: E402
from billiards.preference.labeler_heuristic import position_bonus  # noqa: E402
from billiards.wrappers import MixedRewardEnv  # noqa: E402
from billiards.wrappers.reward_model_env import _load_reward_model  # noqa: E402


# Hyperparameters fixed across the matrix (must match scripts/train_ppo.py).
TOTAL_STEPS_DEFAULT = 50_000
EVAL_EPISODES_DEFAULT = 500
N_ENVS = 8
N_STEPS = 512
BATCH_SIZE = 512
N_EPOCHS = 4
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
T_MAX = 12.0


def _make_env_fn(alpha: float, model_path: str, norm_path: str, seed: int):
    """Factory returning a no-arg callable for SubprocVecEnv."""

    def _factory():
        env = Billiards4BallEnv(t_max=T_MAX)
        env = MixedRewardEnv(env, alpha=alpha, model_path=model_path,
                             norm_path=norm_path)
        # Surface env_reward / rm_reward_norm / mixed_reward / cushion_hits /
        # fouled to the rollout via Monitor's info_keywords.
        env = Monitor(
            env,
            info_keywords=(
                "env_reward", "rm_reward_norm", "mixed_reward",
                "cushion_hits", "fouled",
            ),
        )
        env.reset(seed=seed)
        return env

    return _factory


def _make_vec_env(
    alpha: float, model_path: str, norm_path: str, seed: int, use_subproc: bool
):
    fns = [_make_env_fn(alpha, model_path, norm_path, seed + i) for i in range(N_ENVS)]
    if use_subproc and N_ENVS > 1:
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)


class TrainingCurveCallback(BaseCallback):
    """Append rollout aggregates to ``training_curve.csv``.

    Columns (analyst-aligned schema):
        timesteps, ep_env_score_mean, ep_rm_norm_mean, ep_mixed_reward_mean,
        ep_cushion_mean, ep_foul_rate, theta_std, abs_offset_mean
    """

    HEADER = [
        "timesteps",
        "ep_env_score_mean", "ep_rm_norm_mean", "ep_mixed_reward_mean",
        "ep_cushion_mean", "ep_foul_rate",
        "theta_std", "abs_offset_mean",
    ]

    def __init__(self, csv_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(self.HEADER)
        self._theta: list[float] = []
        self._abs_ab: list[float] = []

    def _on_step(self) -> bool:  # noqa: D401
        actions = self.locals.get("actions")
        if actions is not None:
            arr = np.asarray(actions, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 4:
                self._theta.extend(arr[:, 0].tolist())
                ab_abs = np.abs(arr[:, 2]) + np.abs(arr[:, 3])
                self._abs_ab.extend(ab_abs.tolist())
        return True

    def _on_rollout_end(self) -> None:
        ep_buf = list(self.model.ep_info_buffer or [])
        env_score = float("nan")
        rm_norm = float("nan")
        mixed = float("nan")
        cushion = float("nan")
        foul = float("nan")
        if ep_buf:
            def _meanof(key: str) -> float:
                vals = [e.get(key) for e in ep_buf if key in e]
                if not vals:
                    return float("nan")
                arr = np.asarray(vals, dtype=np.float64)
                return float(arr.mean())

            env_score = _meanof("env_reward")
            rm_norm = _meanof("rm_reward_norm")
            mixed = float(np.mean([e.get("r", float("nan")) for e in ep_buf]))
            cushion = _meanof("cushion_hits")
            # ``fouled`` is recorded as a python bool — coerce to float.
            fouled_vals = [
                1.0 if bool(e["fouled"]) else 0.0
                for e in ep_buf if "fouled" in e
            ]
            if fouled_vals:
                foul = float(np.mean(fouled_vals))
        theta_std = float(np.std(self._theta)) if self._theta else float("nan")
        abs_off = float(np.mean(self._abs_ab)) if self._abs_ab else float("nan")
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                int(self.num_timesteps),
                f"{env_score:.4f}", f"{rm_norm:.4f}", f"{mixed:.4f}",
                f"{cushion:.4f}", f"{foul:.4f}",
                f"{theta_std:.4f}", f"{abs_off:.4f}",
            ])
        self._theta.clear()
        self._abs_ab.clear()


def _act_array(action) -> np.ndarray:
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _evaluate(
    model: PPO,
    n_episodes: int,
    seed_base: int,
    rm_model: torch.nn.Module,
    mu: float,
    sigma: float,
) -> pd.DataFrame:
    """Roll the policy on the BASE env. One row per episode.

    Captures ``final_state``, the typed event list, and the (clipped)
    normalized reward-model output. ``rm_reward_norm`` mean across the 500
    eval eps is the X-axis of the analyst's Pareto figure.
    """
    env = Billiards4BallEnv(t_max=T_MAX)
    rows: list[dict] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        s_pre = np.asarray(obs, dtype=np.float32).reshape(-1)
        action, _ = model.predict(obs, deterministic=True)
        action = _act_array(action)
        next_obs, _, _, _, info = env.step(action)

        with torch.no_grad():
            s_t = torch.as_tensor(s_pre, dtype=torch.float32).unsqueeze(0)
            a_t = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            rm_raw = float(rm_model(s_t, a_t).squeeze().item())
        rm_norm = max(-2.0, min(4.0, (rm_raw - mu) / sigma))

        events = info["event_log"]
        event_types = [str(e.get("type", "")) for e in events]
        rows.append({
            "ep_idx": int(ep),
            "seed": int(seed_base + ep),
            "theta": float(action[0]),
            "power": float(action[1]),
            "a": float(action[2]),
            "b": float(action[3]),
            "score": int(info["score"]),
            "fouled": bool(info["fouled"]),
            "cushion_hits": int(info["cushion_hits"]),
            "duration": float(info["duration"]),
            "n_events": int(len(events)),
            "event_types": event_types,
            "rm_reward_raw": rm_raw,
            "rm_reward_norm": float(rm_norm),
            "final_state": np.asarray(next_obs, dtype=np.float32).reshape(-1).tolist(),
        })
    return pd.DataFrame(rows)


def _attribution(eval_df: pd.DataFrame, cue_id: int = 0) -> dict:
    """Per-episode breakdown of heuristic_score; flat mean+std keys.

    Schema (per analyst): for each of seven heuristic terms plus ``total``
    write ``<term>`` (mean) and ``<term>_std`` at the top level. ``n``
    holds the episode count.
    """
    cushion_term: list[float] = []
    score_term: list[float] = []
    foul_term: list[float] = []
    red_contact_term: list[float] = []
    opp_contact_term: list[float] = []
    position_term: list[float] = []
    duration_term: list[float] = []
    total: list[float] = []

    spec = {"cue_id": int(cue_id)}
    for _, row in eval_df.iterrows():
        cnt = Counter(row["event_types"])
        sc = 5.0 * float(row["score"])
        fl = -5.0 if bool(row["fouled"]) else 0.0
        cu = 0.4 * min(int(row["cushion_hits"]), 5)
        rc = 0.6 * int(cnt.get("cue_hit_red", 0))
        oc = -0.3 * int(cnt.get("cue_hit_opp", 0))
        pos = float(position_bonus(row["final_state"], spec))
        du = -0.05 * float(row["duration"])
        score_term.append(sc)
        foul_term.append(fl)
        cushion_term.append(cu)
        red_contact_term.append(rc)
        opp_contact_term.append(oc)
        position_term.append(pos)
        duration_term.append(du)
        total.append(sc + fl + cu + rc + oc + pos + du)

    out: dict = {"n": int(len(eval_df))}

    def _ms(name: str, v: list[float]) -> None:
        a = np.asarray(v, dtype=np.float64)
        out[name] = float(a.mean())
        out[f"{name}_std"] = float(a.std(ddof=0))

    _ms("score_term", score_term)
    _ms("foul_term", foul_term)
    _ms("cushion_term", cushion_term)
    _ms("red_contact_term", red_contact_term)
    _ms("opp_contact_term", opp_contact_term)
    _ms("position_term", position_term)
    _ms("duration_term", duration_term)
    _ms("total", total)
    return out


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="One alpha-sweep PPO run.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS_DEFAULT)
    parser.add_argument("--eval_episodes", type=int, default=EVAL_EPISODES_DEFAULT)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/reward_model.pt")
    parser.add_argument(
        "--norm_path", type=str, default="experiments/rm_normalization.json"
    )
    parser.add_argument("--no_subproc", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    set_random_seed(int(args.seed))
    log_f = log_path.open("w", encoding="utf-8")
    tee = _Tee(sys.__stdout__, log_f)
    err_tee = _Tee(sys.__stderr__, log_f)

    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err_tee):
        t0 = time.perf_counter()
        config = {
            "alpha": float(args.alpha),
            "seed": int(args.seed),
            "total_steps": int(args.total_steps),
            "eval_episodes": int(args.eval_episodes),
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "learning_rate": LR,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "vf_coef": VF_COEF,
            "ent_coef": ENT_COEF,
            "t_max": T_MAX,
            "model_path": args.model_path,
            "norm_path": args.norm_path,
        }
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"[run_one] alpha={args.alpha} seed={args.seed} "
              f"total_steps={args.total_steps} out_dir={out_dir}")

        use_subproc = (not args.no_subproc) and (N_ENVS > 1)
        vec_env = _make_vec_env(
            alpha=float(args.alpha),
            model_path=args.model_path,
            norm_path=args.norm_path,
            seed=int(args.seed),
            use_subproc=use_subproc,
        )

        try:
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=LR,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE,
                vf_coef=VF_COEF,
                ent_coef=ENT_COEF,
                seed=int(args.seed),
                verbose=0,
                device="cpu",
            )
            cb = TrainingCurveCallback(out_dir / "training_curve.csv")
            t_train0 = time.perf_counter()
            model.learn(total_timesteps=int(args.total_steps), callback=cb,
                        progress_bar=False)
            train_wall = time.perf_counter() - t_train0
            print(f"[run_one] train done in {train_wall:.1f}s")

            policy_path = out_dir / "policy.zip"
            model.save(str(policy_path))
            print(f"[run_one] saved policy -> {policy_path}")

            # Reward model + normalization for eval-time scoring.
            rm_model = _load_reward_model(args.model_path, device="cpu")
            with open(args.norm_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            mu = float(stats["mu"])
            sigma = float(stats["sigma"])

            t_eval0 = time.perf_counter()
            eval_df = _evaluate(
                model, n_episodes=int(args.eval_episodes),
                seed_base=int(args.seed) + 10_000,
                rm_model=rm_model, mu=mu, sigma=sigma,
            )
            eval_wall = time.perf_counter() - t_eval0

            eval_path = out_dir / "eval.parquet"
            eval_df.to_parquet(eval_path, engine="pyarrow", index=False)
            print(f"[run_one] eval n={len(eval_df)} done in {eval_wall:.1f}s "
                  f"-> {eval_path}")

            attribution = _attribution(eval_df, cue_id=0)
            with (out_dir / "attribution.json").open("w", encoding="utf-8") as f:
                json.dump(attribution, f, indent=2)

            true_score_rate = 100.0 * float(eval_df["score"].mean())
            foul_rate = 100.0 * float(eval_df["fouled"].mean())
            mean_cushions = float(eval_df["cushion_hits"].mean())
            rm_score_norm = float(eval_df["rm_reward_norm"].mean())
            wall = time.perf_counter() - t0
            summary = {
                "alpha": float(args.alpha),
                "seed": int(args.seed),
                "train_wall_s": float(train_wall),
                "eval_wall_s": float(eval_wall),
                "wall_s": float(wall),
                "true_score_rate": true_score_rate,
                "rm_score_norm": rm_score_norm,
                "foul_rate": foul_rate,
                "mean_cushions": mean_cushions,
                "attribution": attribution,
            }
            with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(
                f"[run_one] DONE alpha={args.alpha} seed={args.seed} "
                f"true_score%={true_score_rate:.2f} rm_norm={rm_score_norm:.3f} "
                f"foul%={foul_rate:.2f} cush={mean_cushions:.2f} wall={wall:.1f}s"
            )
        finally:
            try:
                vec_env.close()
            except Exception:
                pass
            log_f.close()


if __name__ == "__main__":
    main()
