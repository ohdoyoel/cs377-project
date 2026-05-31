"""Unified evaluation for VALIDATION §1.2 learning-method comparison.

The 4 methods train under different episode structures (reset vs continue on
miss), so the trainer's own per-run eval is NOT comparable across them. This
script re-evaluates every saved policy under ONE fixed protocol:

    continue_on_miss=True, max_shots=10  ->  "score out of 10 shots"

so the inning score means the same thing for all methods. Each policy is scored
on both start conditions (random = generalization, canonical = the fixed rack),
paired across methods via a fixed eval seed base (identical racks for everyone).

Env stays PLAIN (constrain_aim=False, extra_features=False) to match how these
policies were trained.

Usage:
    uv run python experiments/eval_learnmethod.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stable_baselines3 import SAC  # noqa: E402

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402

RUN_ROOT = _REPO / "experiments" / "runs_inning_v2" / "valid_learnmethod"
OUT_DIR = _REPO / "experiments" / "artifacts" / "valid_learnmethod"

METHODS = ["canon_reset", "canon_cont", "rand_reset", "rand_cont"]
SEEDS = (0, 1, 2)
EVAL_N = 100
MAX_SHOTS = 10
T_MAX = 12.0
SEED_BASE = 20_000  # fixed -> every policy faces identical racks (paired)


def eval_policy(model, random_start: bool, n: int, seed_base: int) -> dict:
    """Run n innings under continue_on_miss=True, max_shots=10. Returns score
    stats: inning score = points scored across the (up to) 10 shots."""
    base = Billiards4BallInningEnv(
        t_max=T_MAX, max_shots=MAX_SHOTS, continue_on_miss=True,
        constrain_aim=False, extra_features=False,
    )
    env = RandomStartInningEnv(base) if random_start else base
    scores, fouls = [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed_base + ep)
        fouled = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            obs, _, terminated, truncated, info = env.step(action)
            if bool(info.get("fouled", False)):
                fouled = True
            if terminated or truncated:
                break
        inner = env.unwrapped if random_start else env
        scores.append(int(inner.cumulative_score))
        fouls.append(fouled)
    scores = np.asarray(scores)
    return {
        "mean": float(scores.mean()),
        "max": int(scores.max()),
        "p_ge1": 100.0 * float((scores >= 1).mean()),
        "p_ge3": 100.0 * float((scores >= 3).mean()),
        "foul_rate": 100.0 * float(np.mean(fouls)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for method in METHODS:
        results[method] = {}
        for start_name, rand in (("random", True), ("canonical", False)):
            per_seed = []
            for s in SEEDS:
                pol = RUN_ROOT / f"{method}_s{s}" / "policy.zip"
                if not pol.exists():
                    print(f"[eval] MISSING {pol}")
                    continue
                model = SAC.load(str(pol), device="cpu")
                stats = eval_policy(model, rand, EVAL_N, SEED_BASE)
                stats["seed"] = s
                per_seed.append(stats)
            if per_seed:
                means = [d["mean"] for d in per_seed]
                results[method][start_name] = {
                    "per_seed": per_seed,
                    "mean": float(np.mean(means)),
                    "std": float(np.std(means)),
                    "foul_rate": float(np.mean([d["foul_rate"] for d in per_seed])),
                }

    with (OUT_DIR / "eval_unified.json").open("w", encoding="utf-8") as f:
        json.dump({
            "protocol": {"continue_on_miss": True, "max_shots": MAX_SHOTS,
                         "eval_n": EVAL_N, "seed_base": SEED_BASE,
                         "constrain_aim": False, "extra_features": False},
            "results": results,
        }, f, indent=2)

    # ---- table ----
    print(f"\nUnified eval: continue_on_miss=True, max_shots={MAX_SHOTS}, n={EVAL_N} (score out of 10 shots)\n")
    for start_name in ("random", "canonical"):
        print(f"=== {start_name} start ===")
        print(f"{'method':14s} | mean±std        | per-seed              | foul%")
        print("-" * 64)
        for method in METHODS:
            r = results.get(method, {}).get(start_name)
            if not r:
                print(f"{method:14s} | (no policies)")
                continue
            ps = "/".join(f"{d['mean']:.2f}" for d in r["per_seed"])
            print(f"{method:14s} | {r['mean']:.3f}±{r['std']:.3f}    | {ps:21s} | {r['foul_rate']:.0f}")
        print()


if __name__ == "__main__":
    main()
