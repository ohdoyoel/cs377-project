"""Compare SAC training curves across the 4 learning methods (VALIDATION §1.2).

Reads training_curve.csv for each <method>_s<seed> run, interpolates
ep_return_mean onto a common timestep grid, averages over seeds (mean +/- std
band), writes a PNG and prints a checkpoint table.

Usage:
    uv run python experiments/plot_learnmethod_curves.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
RUN_ROOT = _REPO / "experiments" / "runs_inning_v2" / "valid_learnmethod"
OUT_DIR = _REPO / "experiments" / "artifacts" / "valid_learnmethod"

METHODS = {
    "canon_reset": ("canonical start, reset on miss", "#888888"),
    "canon_cont":  ("canonical start, continue on miss", "#1f77b4"),
    "rand_reset":  ("random start, reset on miss", "#ff7f0e"),
    "rand_cont":   ("random start, continue on miss", "#2ca02c"),
}
SEEDS = (0, 1, 2)
GRID = np.arange(0, 400_001, 4000)
CHECKPOINTS = (100_000, 200_000, 300_000, 400_000)


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    ts, rs = [], []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts.append(int(row["timesteps"]))
            rs.append(float(row["ep_return_mean"]))
    if len(ts) < 2:
        return None
    return np.asarray(ts), np.asarray(rs)


def _method_band(method: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Interpolate each seed onto GRID, return mean and std across seeds."""
    cols = []
    for s in SEEDS:
        cur = _load_curve(RUN_ROOT / f"{method}_s{s}" / "training_curve.csv")
        if cur is None:
            continue
        ts, rs = cur
        cols.append(np.interp(GRID, ts, rs, left=rs[0], right=rs[-1]))
    if not cols:
        return np.zeros_like(GRID, float), np.zeros_like(GRID, float), 0
    arr = np.vstack(cols)
    return arr.mean(0), arr.std(0), arr.shape[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bands = {m: _method_band(m) for m in METHODS}

    # ---- checkpoint table ----
    print(f"{'method':14s} | " + " | ".join(f"{c//1000}k" for c in CHECKPOINTS) + "  (n_seeds)")
    print("-" * 60)
    for m, (label, _c) in METHODS.items():
        mean, _std, n = bands[m]
        vals = [f"{np.interp(c, GRID, mean):.3f}" for c in CHECKPOINTS]
        print(f"{m:14s} | " + " | ".join(f"{v:>5s}" for v in vals) + f"   (n={n})")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); table only.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for m, (label, color) in METHODS.items():
        mean, std, n = bands[m]
        if n == 0:
            continue
        ax.plot(GRID, mean, color=color, label=f"{label} (n={n})", lw=1.8)
        ax.fill_between(GRID, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("env steps")
    ax.set_ylabel("ep_return_mean (inning score)")
    ax.set_title("SAC training curves by learning method (plain env, 400k, 3 seeds)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT_DIR / "learnmethod_curves.png"
    fig.savefig(out_png, dpi=130)
    print(f"\n[plot] saved -> {out_png}")


if __name__ == "__main__":
    main()
