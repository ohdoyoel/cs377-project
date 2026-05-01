"""Collect single-shot rollouts under a baseline policy and dump to parquet.

Usage:
    uv run python scripts/collect_rollouts.py \
        --n_episodes 10000 --policy random --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `uv run python scripts/collect_rollouts.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from billiards.env import Billiards4BallEnv  # noqa: E402
from policies.geometric_aim import GeometricAimPolicy  # noqa: E402
from policies.random_policy import RandomPolicy  # noqa: E402

POLICY_CHOICES = ("random", "geometric")


def _make_policy(name: str, seed: int):
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "geometric":
        return GeometricAimPolicy()
    raise ValueError(f"unknown policy: {name}")


def _count_cue_hits_red(events: list[dict]) -> int:
    return sum(1 for ev in events if ev.get("type") == "cue_hit_red")


def _run_one(policy_name: str, seed: int, episode_idx: int, base_seed: int) -> dict:
    env = Billiards4BallEnv()
    policy = _make_policy(policy_name, seed=seed)
    obs, _ = env.reset(seed=base_seed + episode_idx)
    action = policy.act(obs)
    _, _, _, _, info = env.step(action)
    events = info.get("event_log", [])
    return {
        "ep": int(episode_idx),
        "theta": float(action[0]),
        "power": float(action[1]),
        "a": float(action[2]),
        "b": float(action[3]),
        "score": int(info["score"]),
        "fouled": bool(info["fouled"]),
        "cushion_hits": int(info["cushion_hits"]),
        "duration": float(info["duration"]),
        "n_events": int(len(events)),
        "n_cue_hits_red": int(_count_cue_hits_red(events)),
    }


def _run_chunk(
    policy_name: str,
    base_policy_seed: int,
    base_seed: int,
    start: int,
    end: int,
) -> list[dict]:
    env = Billiards4BallEnv()
    # One policy instance per chunk. Use a per-chunk RNG seed for random policy
    # so multiprocessing doesn't duplicate streams.
    policy = (
        RandomPolicy(seed=base_policy_seed + start)
        if policy_name == "random"
        else GeometricAimPolicy()
    )
    rows: list[dict] = []
    for ep in range(start, end):
        obs, _ = env.reset(seed=base_seed + ep)
        action = policy.act(obs)
        _, _, _, _, info = env.step(action)
        events = info.get("event_log", [])
        rows.append(
            {
                "ep": int(ep),
                "theta": float(action[0]),
                "power": float(action[1]),
                "a": float(action[2]),
                "b": float(action[3]),
                "score": int(info["score"]),
                "fouled": bool(info["fouled"]),
                "cushion_hits": int(info["cushion_hits"]),
                "duration": float(info["duration"]),
                "n_events": int(len(events)),
                "n_cue_hits_red": int(_count_cue_hits_red(events)),
            }
        )
    return rows


def _summary_row(rows: list[dict]) -> str:
    if not rows:
        return ""
    n = len(rows)
    score_rate = 100.0 * sum(r["score"] for r in rows) / n
    foul_rate = 100.0 * sum(1 for r in rows if r["fouled"]) / n
    mean_cush = sum(r["cushion_hits"] for r in rows) / n
    mean_dur = sum(r["duration"] for r in rows) / n
    return (
        f"ep={n} score_rate={score_rate:.1f}% foul={foul_rate:.1f}% "
        f"mean_cush={mean_cush:.1f} mean_dur={mean_dur:.1f}"
    )


def _benchmark_serial(policy_name: str, n: int, base_seed: int) -> tuple[float, list[dict]]:
    t0 = time.perf_counter()
    rows = _run_chunk(policy_name, base_seed, base_seed, 0, n)
    return time.perf_counter() - t0, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect rollouts under a baseline policy.")
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="random")
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 = auto (multiprocessing only if mini-benchmark > 9s).",
    )
    parser.add_argument("--bench_n", type=int, default=1000)
    args = parser.parse_args()

    out_path = (
        Path(args.out_path)
        if args.out_path
        else Path("data") / f"rollouts_{args.policy}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[collect_rollouts] policy={args.policy} n={args.n_episodes} "
        f"seed={args.seed} out={out_path}"
    )

    rows: list[dict] = []
    print(f"[collect_rollouts] mini-benchmark: {args.bench_n} eps serial...")
    bench_t, bench_rows = _benchmark_serial(args.policy, args.bench_n, args.seed)
    print(
        f"[collect_rollouts] benchmark={bench_t:.2f}s for {args.bench_n} eps "
        f"({args.bench_n / bench_t:.1f} eps/s)"
    )
    print(f"[collect_rollouts] {_summary_row(bench_rows)}")

    use_mp = args.workers > 1 or (args.workers == 0 and bench_t > 9.0)

    if not use_mp:
        if args.bench_n >= args.n_episodes:
            rows = bench_rows[: args.n_episodes]
        else:
            rows.extend(bench_rows)
            cursor = args.bench_n
            chunk = 1000
            while cursor < args.n_episodes:
                end = min(cursor + chunk, args.n_episodes)
                rows.extend(_run_chunk(args.policy, args.seed, args.seed, cursor, end))
                cursor = end
                if cursor % 1000 == 0 or cursor == args.n_episodes:
                    print(f"[collect_rollouts] {_summary_row(rows)}")
    else:
        n_workers = args.workers if args.workers > 1 else max(2, (os.cpu_count() or 4) - 1)
        chunk = max(200, args.n_episodes // (n_workers * 4))
        boundaries = list(range(0, args.n_episodes, chunk))
        if boundaries[-1] != args.n_episodes:
            boundaries.append(args.n_episodes)
        ranges = list(zip(boundaries[:-1], boundaries[1:]))
        print(f"[collect_rollouts] using {n_workers} workers, {len(ranges)} chunks")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_run_chunk, args.policy, args.seed, args.seed, s, e)
                for s, e in ranges
            ]
            for fut in as_completed(futures):
                rows.extend(fut.result())
                if len(rows) // 1000 != (len(rows) - 1) // 1000:
                    print(f"[collect_rollouts] {_summary_row(rows)}")
        rows.sort(key=lambda r: r["ep"])

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"[collect_rollouts] wrote {len(df)} rows → {out_path}")
    print(f"[collect_rollouts] FINAL {_summary_row(rows)}")


if __name__ == "__main__":
    main()
