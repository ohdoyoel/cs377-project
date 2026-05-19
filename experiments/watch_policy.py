"""Load a saved policy and render its innings as a browser-viewable HTML replay.

Usage:
    uv run python experiments/watch_policy.py \\
        --policy experiments/runs_inning/sac_novec_s2/policy.zip \\
        --algo sac \\
        --n_innings 20 \\
        --seed 0

The script evaluates the policy for --n_innings innings (continue_on_miss=False,
so each episode ends on first miss/foul), prints a per-inning summary, then
renders the *longest* inning to HTML and opens it in the default browser.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3 import SAC, TD3  # noqa: E402

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.render.replay import render_inning_html  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402


def roll_inning(env: Billiards4BallInningEnv, model, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    fouled = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.float32).reshape(-1)
        )
        if info.get("fouled"):
            fouled = True
        if terminated or truncated:
            break
    inner = env.unwrapped
    return {
        "seed": seed,
        "score": inner.cumulative_score,
        "n_shots": inner.shot_index,
        "fouled": fouled,
        "truncated": truncated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to policy.zip")
    parser.add_argument("--algo", type=str, choices=("sac", "td3"), default="sac")
    parser.add_argument("--n_innings", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_shots", type=int, default=20,
                        help="Per-inning shot cap (continue_on_miss=False)")
    parser.add_argument("--no_constrain_aim", action="store_true",
                        help="Disable constrain_aim (default: on)")
    parser.add_argument("--no_extra_features", action="store_true",
                        help="Disable extra_features (default: on)")
    parser.add_argument("--random_start", action="store_true",
                        help="Randomize ball positions on each reset")
    parser.add_argument("--out", type=str, default=None,
                        help="HTML output path (default: artifacts/watch/<policy_dir>.html)")
    args = parser.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"[watch] ERROR: policy not found: {policy_path}")
        sys.exit(1)

    constrain_aim = not args.no_constrain_aim
    extra_features = not args.no_extra_features

    loader = SAC if args.algo == "sac" else TD3
    print(f"[watch] loading {args.algo.upper()} policy <- {policy_path}")
    model = loader.load(str(policy_path), device="cpu")

    base_env = Billiards4BallInningEnv(
        max_shots=args.max_shots,
        continue_on_miss=False,   # real inning: terminate on first miss/foul
        constrain_aim=constrain_aim,
        extra_features=extra_features,
    )
    env = RandomStartInningEnv(base_env) if args.random_start else base_env

    print(f"[watch] rolling {args.n_innings} innings "
          f"(constrain_aim={constrain_aim}, extra_features={extra_features}, "
          f"random_start={args.random_start}, continue_on_miss=False) ...")

    results = []
    shot_trajs_by_seed: dict[int, list] = {}
    inner_env = base_env  # always the unwrapped Billiards4BallInningEnv
    spec = inner_env._spec

    for i in range(args.n_innings):
        seed = args.seed + i
        rec = roll_inning(env, model, seed=seed)
        rec["idx"] = i
        results.append(rec)
        shot_trajs_by_seed[seed] = inner_env.shot_trajectories

    # Print summary table
    print(f"\n{'idx':>3} {'seed':>6} {'score':>5} {'shots':>5} {'foul':>5} {'trunc':>5}")
    print("-" * 35)
    total_score = 0
    for r in results:
        print(f"{r['idx']:>3} {r['seed']:>6} {r['score']:>5} {r['n_shots']:>5} "
              f"{'Y' if r['fouled'] else 'N':>5} {'Y' if r['truncated'] else 'N':>5}")
        total_score += r["score"]

    scores = [r["score"] for r in results]
    print(f"\nmean_score={np.mean(scores):.2f}  max={max(scores)}  "
          f"p>=1={100*np.mean([s>=1 for s in scores]):.0f}%  "
          f"p>=3={100*np.mean([s>=3 for s in scores]):.0f}%  "
          f"p>=5={100*np.mean([s>=5 for s in scores]):.0f}%")

    # Render the highest-scoring inning (tie-break: most shots)
    best = max(results, key=lambda r: (r["score"], r["n_shots"]))
    print(f"\n[watch] rendering best inning: seed={best['seed']} "
          f"score={best['score']} shots={best['n_shots']}")

    out_path = args.out
    if out_path is None:
        run_name = policy_path.parent.name
        out_dir = _REPO_ROOT / "artifacts" / "watch"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{run_name}.html")

    render_inning_html(
        shot_trajs_by_seed[best["seed"]],
        spec=spec,
        save_path=out_path,
    )
    print(f"[watch] saved -> {out_path}")
    webbrowser.open(f"file:///{Path(out_path).resolve()}")
    print("[watch] opened in browser")


if __name__ == "__main__":
    main()
