"""Fixed sim-time-budget evaluation: how many points does a policy score
within a fixed budget of *simulated table time*?

Unlike ``eval_policy.py`` (per-inning score, capped by ``max_shots``), this
measures scoring throughput per unit simulated time. Within one rollout we
play innings back to back: an inning ends on the first miss/foul
(``continue_on_miss=False``), then we reset to a fresh random layout and keep
shooting, summing score, until the cumulative shot duration reaches
``--budget_s``. We repeat the whole rollout ``--n_repeats`` times (different
random sequences) and aggregate the cumulative-score-vs-time curve
(mean +/- std on a common time grid).

This pairs with ``--time_reward`` training: a policy that scores with shorter
ball travel fits more scoring shots into the same time budget, so its curve
should rise faster.

Standalone (one or more --run_dir to overlay time vs non-time):
    python experiments/eval_time_budget.py \\
        --run_dir experiments/runs_inning_v2/fast_long_fp02_s4 \\
        --run_dir experiments/runs_inning_v2/fast_time_fp02_s4 \\
        --budget_s 120 --n_repeats 30 \\
        --out experiments/artifacts/time_budget/s4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402

T_MAX = 12.0


def _rollout_curve(
    model,
    budget_s: float,
    n_repeats: int,
    max_shots: int,
    constrain_aim: bool,
    extra_features: bool,
    seed_base: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (times, cum_scores) step-functions, one per repeat.

    ``times[k]`` is the cumulative simulated time after shot k and
    ``cum_scores[k]`` the total valid points scored by then. Both arrays start
    with a (0.0, 0) anchor so interpolation before the first shot yields 0.
    """
    base = Billiards4BallInningEnv(
        t_max=T_MAX,
        max_shots=max_shots,
        continue_on_miss=False,  # inning ends on first miss/foul
        constrain_aim=constrain_aim,
        extra_features=extra_features,
    )
    env = RandomStartInningEnv(base)

    repeats: list[tuple[np.ndarray, np.ndarray]] = []
    for r in range(n_repeats):
        global_t = 0.0
        global_score = 0
        times = [0.0]
        scores = [0]
        # Seed the first inning of the repeat for reproducibility; later
        # innings call reset() without a seed so the wrapper's RNG advances
        # and yields a fresh random layout each time.
        first_inning = True
        while global_t < budget_s:
            if first_inning:
                obs, _ = env.reset(seed=seed_base + r)
                first_inning = False
            else:
                obs, _ = env.reset()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(
                    np.asarray(action, dtype=np.float32).reshape(-1)
                )
                global_t += float(info["duration"])
                if int(info["score"]) > 0 and not bool(info["fouled"]):
                    global_score += int(info["score"])
                times.append(global_t)
                scores.append(global_score)
                if global_t >= budget_s or terminated or truncated:
                    break
        repeats.append((np.asarray(times), np.asarray(scores)))
    return repeats


def _aggregate(
    repeats: list[tuple[np.ndarray, np.ndarray]],
    budget_s: float,
    grid_points: int,
) -> pd.DataFrame:
    """Interpolate each repeat's step-function onto a common time grid and
    return per-grid mean/std of cumulative score."""
    grid = np.linspace(0.0, budget_s, grid_points)
    mat = np.empty((len(repeats), grid_points), dtype=np.float64)
    for i, (times, scores) in enumerate(repeats):
        # Step function: score at time g = scores[last index with times <= g].
        idx = np.searchsorted(times, grid, side="right") - 1
        idx = np.clip(idx, 0, len(scores) - 1)
        mat[i] = scores[idx]
    return pd.DataFrame({
        "time_s": grid,
        "score_mean": mat.mean(axis=0),
        "score_std": mat.std(axis=0),
    })


def evaluate_run(
    run_dir: Path,
    budget_s: float,
    n_repeats: int,
    max_shots: int,
    grid_points: int,
    seed_base: int,
):
    from stable_baselines3 import SAC, TD3

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.json in {run_dir}")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    algo = cfg.get("algo", "sac")
    constrain_aim = bool(cfg.get("constrain_aim", False))
    extra_features = bool(cfg.get("extra_features", False))

    policy_path = run_dir / "policy.zip"
    if not policy_path.exists():
        raise FileNotFoundError(f"no policy.zip in {run_dir}")

    loader = SAC if algo == "sac" else TD3
    model = loader.load(str(policy_path), device="cpu")

    repeats = _rollout_curve(
        model,
        budget_s=budget_s,
        n_repeats=n_repeats,
        max_shots=max_shots,
        constrain_aim=constrain_aim,
        extra_features=extra_features,
        seed_base=seed_base,
    )
    curve = _aggregate(repeats, budget_s=budget_s, grid_points=grid_points)
    final_mean = float(curve["score_mean"].iloc[-1])
    final_std = float(curve["score_std"].iloc[-1])
    return curve, final_mean, final_std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", action="append", required=True,
                        help="Run directory (reads config.json + policy.zip). "
                             "Repeat to overlay multiple policies.")
    parser.add_argument("--budget_s", type=float, default=120.0,
                        help="Simulated-time budget per rollout (seconds).")
    parser.add_argument("--n_repeats", type=int, default=30,
                        help="Independent rollouts to average.")
    parser.add_argument("--max_shots", type=int, default=200,
                        help="Per-inning shot cap (a single scoring streak).")
    parser.add_argument("--grid_points", type=int, default=240)
    parser.add_argument("--seed_base", type=int, default=99000)
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for curves + comparison plot.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = None
    results: list[tuple[str, float, float]] = []
    for rd in args.run_dir:
        run_dir = Path(rd) if Path(rd).is_absolute() else _REPO_ROOT / rd
        label = run_dir.name
        curve, final_mean, final_std = evaluate_run(
            run_dir,
            budget_s=args.budget_s,
            n_repeats=args.n_repeats,
            max_shots=args.max_shots,
            grid_points=args.grid_points,
            seed_base=args.seed_base,
        )
        curve.to_csv(out_dir / f"{label}_curve.csv", index=False)
        results.append((label, final_mean, final_std))
        print(f"[time_budget] {label:28s}  "
              f"score@{args.budget_s:.0f}s = {final_mean:.2f} +/- {final_std:.2f}")
        col = curve[["time_s", "score_mean"]].rename(columns={"score_mean": label})
        combined = col if combined is None else combined.merge(col, on="time_s")

    if combined is not None:
        combined.to_csv(out_dir / "comparison.csv", index=False)

    summary = {
        "budget_s": args.budget_s,
        "n_repeats": args.n_repeats,
        "max_shots": args.max_shots,
        "final_scores": {lbl: {"mean": m, "std": s} for lbl, m, s in results},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optional overlay plot (skip silently if matplotlib is unavailable).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for rd in args.run_dir:
            label = (Path(rd) if Path(rd).is_absolute() else _REPO_ROOT / rd).name
            c = pd.read_csv(out_dir / f"{label}_curve.csv")
            ax.plot(c["time_s"], c["score_mean"], label=label)
            ax.fill_between(c["time_s"],
                            c["score_mean"] - c["score_std"],
                            c["score_mean"] + c["score_std"],
                            alpha=0.15)
        ax.set_xlabel("simulated table time (s)")
        ax.set_ylabel("cumulative score")
        ax.set_title(f"Score vs sim-time budget (n_repeats={args.n_repeats})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "comparison.png", dpi=120)
        print(f"[time_budget] plot -> {out_dir / 'comparison.png'}")
    except ImportError:
        print("[time_budget] matplotlib not available; skipped plot")

    print(f"[time_budget] curves -> {out_dir}")


if __name__ == "__main__":
    main()
