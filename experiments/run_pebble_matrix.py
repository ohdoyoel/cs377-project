"""Sequential PEBBLE 4×3 matrix runner.

12 configs = 4 kinds × 3 seeds. After each successful run, appends a row to
``experiments/results/pebble_summary.parquet``.

Usage:
    uv run python experiments/run_pebble_matrix.py
    uv run python experiments/run_pebble_matrix.py --start 0 --end 4 --skip_existing
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


CONFIG_KINDS = ["pebble_full", "sac_rm_frozen", "sac_env", "pebble_disagree"]
SEEDS = [0, 1, 2]


def get_configs() -> list[dict]:
    """4 kinds × 3 seeds = 12 configs (kind outer, seed inner)."""
    return [
        {"kind": k, "seed": s, "run_id": f"{k}_s{s}"}
        for k in CONFIG_KINDS for s in SEEDS
    ]


_TERM_NAMES = (
    "score_term", "foul_term", "cushion_term", "red_contact_term",
    "opp_contact_term", "position_term", "duration_term",
)


def _row_from_summary(cfg: dict, summary: dict) -> dict:
    attr = summary.get("attribution", {}) or {}
    row: dict = {
        "method": str(cfg["kind"]),
        "config_kind": str(cfg["kind"]),
        "seed": int(cfg["seed"]),
        "run_id": str(cfg["run_id"]),
        "true_score_rate": float(summary.get("true_score_rate", float("nan"))),
        "rm_score_mean": float(summary.get("rm_score_mean", float("nan"))),
        "foul_rate": float(summary.get("foul_rate", float("nan"))),
        "mean_cushions": float(summary.get("mean_cushions", float("nan"))),
        "queries_used": int(summary.get("queries_used", 0)),
        "n_phases": int(summary.get("n_phases", 0)),
        "last_rm_train_loss": float(summary.get("last_rm_train_loss", float("nan"))),
        "last_relabel_delta_mean": float(summary.get("last_relabel_delta_mean", float("nan"))),
        "last_relabel_delta_std": float(summary.get("last_relabel_delta_std", float("nan"))),
        "train_wall_s": float(summary.get("train_wall_s", float("nan"))),
        "wall_s": float(summary.get("wall_s", float("nan"))),
    }
    for name in _TERM_NAMES:
        row[name] = float(attr.get(name, float("nan")))
        row[f"{name}_std"] = float(attr.get(f"{name}_std", float("nan")))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PEBBLE 4×3 matrix.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=12)
    parser.add_argument("--total_steps", type=int, default=30_000)
    parser.add_argument("--query_phase_steps", type=int, default=5_000)
    parser.add_argument("--queries_per_phase", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--out_dir", type=str, default="experiments/runs_pebble",
    )
    parser.add_argument(
        "--summary_path", type=str,
        default="experiments/results/pebble_summary.parquet",
    )
    args = parser.parse_args()

    runs_dir = Path(args.out_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    full_configs = get_configs()
    total_full = len(full_configs)
    end_idx = max(args.start, min(args.end, total_full))
    configs = full_configs[args.start:end_idx]
    total = len(configs)
    print(f"[run_pebble_matrix] running {total}/{total_full} (idx {args.start}:{end_idx}). "
          f"runs_dir={runs_dir} summary={summary_path}")

    rows: list[dict] = []
    if summary_path.exists():
        try:
            existing = pd.read_parquet(summary_path).to_dict("records")
            rows.extend(existing)
            print(f"[run_pebble_matrix] resumed with {len(rows)} prior rows")
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
            rows = [r for r in rows if r.get("run_id") != run_id]
            rows.append(_row_from_summary(cfg, summary))
            print(f"[{i}/{total}] {run_id} SKIPPED (exists)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "experiments/run_pebble.py",
            "--config_kind", str(cfg["kind"]),
            "--seed", str(cfg["seed"]),
            "--total_steps", str(args.total_steps),
            "--query_phase_steps", str(args.query_phase_steps),
            "--queries_per_phase", str(args.queries_per_phase),
            "--eval_episodes", str(args.eval_episodes),
            "--out_dir", str(out_dir),
        ]
        t0 = time.perf_counter()
        elapsed_so_far = time.perf_counter() - t_matrix0
        avg_per_run = elapsed_so_far / max(1, i - 1) if i > 1 else float("nan")
        eta = avg_per_run * (total - i + 1) if i > 1 else float("nan")
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
        rows = [r for r in rows if r.get("run_id") != run_id]
        rows.append(_row_from_summary(cfg, summary))
        df = pd.DataFrame(rows)
        df.to_parquet(summary_path, engine="pyarrow", index=False)
        print(
            f"[{i}/{total}] {run_id} done. "
            f"true%={summary['true_score_rate']:.2f} "
            f"rm={summary['rm_score_mean']:.3f} "
            f"foul%={summary['foul_rate']:.2f} wall={wall:.1f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - t_matrix0
    print(f"[run_pebble_matrix] {total} runs complete in {elapsed/60:.1f} min")
    print(f"[run_pebble_matrix] summary -> {summary_path}")


if __name__ == "__main__":
    main()
