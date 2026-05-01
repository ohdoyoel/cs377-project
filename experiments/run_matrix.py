"""Sequential alpha-sweep matrix runner.

Spawns one ``run_one.py`` subprocess per (alpha, seed) config. After each
successful run, appends a row to ``experiments/results/summary.parquet``
in the analyst-aligned schema. Aborts after 5 consecutive failures with
``STATUS_FAILED.txt``.

Usage:
    uv run python experiments/run_matrix.py
    uv run python experiments/run_matrix.py --total_steps 50000 --eval_episodes 500
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

from experiments.configs import get_configs  # noqa: E402


_TERM_NAMES = (
    "score_term", "foul_term", "cushion_term", "red_contact_term",
    "opp_contact_term", "position_term", "duration_term",
)


def _row_from_summary(cfg: dict, summary: dict) -> dict:
    """Build one summary.parquet row in the analyst-aligned schema.

    ``hacking_gap`` = ``rm_score_norm - true_score_rate / 100`` (analyst can
    recompute if they prefer different normalization).
    """
    attr = summary.get("attribution", {}) or {}
    true_rate = float(summary["true_score_rate"])
    rm_norm = float(summary["rm_score_norm"])
    row: dict = {
        "alpha": float(cfg["alpha"]),
        "seed": int(cfg["seed"]),
        "run_id": str(cfg["run_id"]),
        "true_score_rate": true_rate,
        "rm_score_norm": rm_norm,
        "hacking_gap": rm_norm - (true_rate / 100.0),
        "foul_rate": float(summary["foul_rate"]),
        "mean_cushions": float(summary["mean_cushions"]),
        "train_wall_s": float(summary["train_wall_s"]),
        "wall_s": float(summary.get("wall_s", float("nan"))),
    }
    for name in _TERM_NAMES:
        row[name] = float(attr.get(name, float("nan")))
        row[f"{name}_std"] = float(attr.get(f"{name}_std", float("nan")))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the alpha-sweep matrix.")
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument(
        "--runs_dir", type=str, default="experiments/runs",
        help="Per-run output directory: <runs_dir>/<run_id>/",
    )
    parser.add_argument(
        "--summary_path", type=str, default="experiments/results/summary.parquet",
    )
    parser.add_argument(
        "--status_dir", type=str, default="experiments/results",
    )
    parser.add_argument(
        "--max_failures", type=int, default=5,
        help="Abort after this many consecutive failures.",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip a config if its summary.json already exists.",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Index of the first config in get_configs() to run (inclusive).",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Index of the last config (exclusive). Default: run to the end.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    status_dir = Path(args.status_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    full_configs = get_configs()
    total_full = len(full_configs)
    end_idx = args.end if args.end is not None else total_full
    end_idx = max(args.start, min(end_idx, total_full))
    configs = full_configs[args.start:end_idx]
    total = len(configs)
    print(f"[run_matrix] running {total}/{total_full} (idx {args.start}:{end_idx}). "
          f"runs_dir={runs_dir} summary={summary_path}")

    rows: list[dict] = []
    if summary_path.exists():
        try:
            existing = pd.read_parquet(summary_path).to_dict("records")
            rows.extend(existing)
            print(f"[run_matrix] resumed with {len(rows)} prior rows")
        except Exception:  # noqa: BLE001
            pass

    consecutive_failures = 0
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
            print(f"[{i}/{total}] alpha={cfg['alpha']} seed={cfg['seed']} "
                  f"SKIPPED (exists)")
            consecutive_failures = 0
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "experiments/run_one.py",
            "--alpha", f"{cfg['alpha']}",
            "--seed", f"{cfg['seed']}",
            "--total_steps", f"{args.total_steps}",
            "--eval_episodes", f"{args.eval_episodes}",
            "--out_dir", str(out_dir),
        ]
        t0 = time.perf_counter()
        print(f"[{i}/{total}] alpha={cfg['alpha']} seed={cfg['seed']} starting "
              f"({run_id})...", flush=True)
        try:
            proc = subprocess.run(cmd, check=False)
            wall = time.perf_counter() - t0
            ok = (proc.returncode == 0) and summary_json.exists()
        except Exception as e:  # noqa: BLE001
            wall = time.perf_counter() - t0
            print(f"[{i}/{total}] subprocess error: {e}")
            ok = False

        if not ok:
            consecutive_failures += 1
            print(f"[{i}/{total}] FAILED ({consecutive_failures} consecutive), "
                  f"wall={wall:.1f}s")
            if consecutive_failures >= args.max_failures:
                status_path = status_dir / "STATUS_FAILED.txt"
                with status_path.open("w", encoding="utf-8") as f:
                    f.write(
                        f"Aborted after {consecutive_failures} consecutive "
                        f"failures. Last failed run: {run_id}\n"
                    )
                print(f"[run_matrix] ABORT: {status_path}")
                sys.exit(1)
            continue

        consecutive_failures = 0
        with summary_json.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        rows = [r for r in rows if r.get("run_id") != run_id]
        rows.append(_row_from_summary(cfg, summary))
        df = pd.DataFrame(rows)
        df.to_parquet(summary_path, engine="pyarrow", index=False)
        print(
            f"[{i}/{total}] alpha={cfg['alpha']} seed={cfg['seed']} done. "
            f"true_score={summary['true_score_rate']:.2f}% "
            f"rm_norm={summary['rm_score_norm']:.3f} "
            f"foul={summary['foul_rate']:.2f}% wall={wall:.1f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - t_matrix0
    print(f"[run_matrix] all {len(configs)} runs complete in {elapsed/60:.1f} min")
    print(f"[run_matrix] summary -> {summary_path}")


if __name__ == "__main__":
    main()
