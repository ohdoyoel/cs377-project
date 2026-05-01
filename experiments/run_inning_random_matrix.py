"""Sequential 3-seed random-start SAC matrix.

Each run:
    algo=sac, train_env=random, seed ∈ {0, 1, 2}.
After every run we cross-evaluate on both ``canonical`` and ``random``,
and append two rows (one per eval env) to
``experiments/results/inning_random_summary.parquet``.

Subprocesses are foreground; do not run in background. Each run is
~5–6 min wall on cpu.

Usage:
    uv run python experiments/run_inning_random_matrix.py
    uv run python experiments/run_inning_random_matrix.py --skip_existing
    uv run python experiments/run_inning_random_matrix.py --start 0 --end 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SEEDS = [0, 1, 2]
TRAIN_ENV = "random"
ALGO = "sac"


def get_configs() -> list[dict]:
    return [
        {
            "algo": ALGO,
            "seed": s,
            "train_env": TRAIN_ENV,
            "run_id": f"{ALGO}_{TRAIN_ENV}_s{s}",
        }
        for s in SEEDS
    ]


def _row_from_summary(cfg: dict, summary: dict, eval_env: str) -> dict:
    block = summary.get(f"eval_{eval_env}", {}) or {}
    return {
        "train_env": str(cfg["train_env"]),
        "eval_env": str(eval_env),
        "algo": str(cfg["algo"]),
        "seed": int(cfg["seed"]),
        "run_id": str(cfg["run_id"]),
        "mean_inning_score": float(block.get("mean_inning_score", float("nan"))),
        "max_inning": int(block.get("max_inning_score", 0)),
        "p_score_ge1": float(block.get("p_score_ge1", float("nan"))),
        "p_score_ge3": float(block.get("p_score_ge3", float("nan"))),
        "p_score_ge5": float(block.get("p_score_ge5", float("nan"))),
        "mean_shots": float(block.get("mean_shots", float("nan"))),
        "foul_rate": float(block.get("foul_rate", float("nan"))),
        "mean_cushions": float(block.get("mean_cushions", float("nan"))),
        "total_steps": int(summary.get("total_steps", 0))
            if summary.get("total_steps") is not None else 0,
        "train_wall_s": float(summary.get("train_wall_s", float("nan"))),
        "wall_s": float(summary.get("wall_s", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-start inning matrix.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=3)
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--max_shots", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--out_dir", type=str, default="experiments/runs_inning_random",
    )
    parser.add_argument(
        "--summary_path", type=str,
        default="experiments/results/inning_random_summary.parquet",
    )
    parser.add_argument(
        "--eval_envs", type=str, default="canonical,random",
    )
    args = parser.parse_args()

    runs_dir = Path(args.out_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    eval_envs = [e.strip() for e in args.eval_envs.split(",") if e.strip()]

    full_configs = get_configs()
    total_full = len(full_configs)
    end_idx = max(args.start, min(args.end, total_full))
    configs = full_configs[args.start:end_idx]
    total = len(configs)
    print(
        f"[run_inning_random_matrix] running {total}/{total_full} "
        f"(idx {args.start}:{end_idx}). runs_dir={runs_dir} "
        f"summary={summary_path}"
    )

    rows: list[dict] = []
    if summary_path.exists():
        try:
            existing = pd.read_parquet(summary_path).to_dict("records")
            rows.extend(existing)
            print(f"[run_inning_random_matrix] resumed with {len(rows)} prior rows")
        except Exception:  # noqa: BLE001
            pass

    t_matrix0 = time.perf_counter()
    for i, cfg in enumerate(configs, start=1):
        run_id = cfg["run_id"]
        out_dir = runs_dir / run_id
        summary_json = out_dir / "summary.json"

        if args.skip_existing and summary_json.exists():
            with summary_json.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            summary["total_steps"] = summary.get("total_steps", args.total_steps)
            rows = [
                r for r in rows
                if not (
                    r.get("run_id") == run_id and r.get("eval_env") in eval_envs
                )
            ]
            for ev in eval_envs:
                rows.append(_row_from_summary(cfg, summary, ev))
            print(f"[{i}/{total}] {run_id} SKIPPED (exists)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "experiments/run_inning_random.py",
            "--algo", str(cfg["algo"]),
            "--seed", str(cfg["seed"]),
            "--total_steps", str(args.total_steps),
            "--max_shots", str(args.max_shots),
            "--eval_episodes", str(args.eval_episodes),
            "--train_env", str(cfg["train_env"]),
            "--eval_envs", ",".join(eval_envs),
            "--out_dir", str(out_dir),
        ]
        t0 = time.perf_counter()
        elapsed_so_far = time.perf_counter() - t_matrix0
        avg = elapsed_so_far / max(1, i - 1) if i > 1 else float("nan")
        eta = avg * (total - i + 1) if i > 1 else float("nan")
        eta_str = f"{eta/60:.1f}min" if eta == eta else "?"
        print(f"[{i}/{total}] {run_id} starting (eta_remaining={eta_str})...",
              flush=True)

        try:
            proc = subprocess.run(cmd, check=False)
            wall = time.perf_counter() - t0
            ok = (proc.returncode == 0) and summary_json.exists()
        except Exception as e:  # noqa: BLE001
            wall = time.perf_counter() - t0
            print(f"[{i}/{total}] subprocess error: {e}")
            ok = False

        if not ok:
            print(f"[{i}/{total}] {run_id} FAILED, wall={wall:.1f}s")
            continue

        with summary_json.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["total_steps"] = args.total_steps
        rows = [
            r for r in rows
            if not (
                r.get("run_id") == run_id and r.get("eval_env") in eval_envs
            )
        ]
        for ev in eval_envs:
            rows.append(_row_from_summary(cfg, summary, ev))
        df = pd.DataFrame(rows)
        df.to_parquet(summary_path, engine="pyarrow", index=False)
        rep = {
            ev: summary.get(f"eval_{ev}", {}).get("mean_inning_score", float("nan"))
            for ev in eval_envs
        }
        rep_str = " ".join(f"{ev}={v:.3f}" for ev, v in rep.items())
        print(
            f"[{i}/{total}] {run_id} done. mean_inning {rep_str} "
            f"wall={wall:.1f}s",
            flush=True,
        )

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(summary_path, engine="pyarrow", index=False)

    elapsed = time.perf_counter() - t_matrix0
    print(
        f"[run_inning_random_matrix] {total} runs complete in "
        f"{elapsed/60:.1f} min"
    )
    print(f"[run_inning_random_matrix] summary -> {summary_path}")


if __name__ == "__main__":
    main()
