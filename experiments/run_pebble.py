"""Train + evaluate one PEBBLE config.

Maps ``--config_kind`` to PEBBLEAgent kwargs:

    pebble_full     reward_mode=rm_only,  relabel=True,  query=uniform
    sac_rm_frozen   reward_mode=rm_only,  relabel=False, query=uniform
    sac_env         reward_mode=env_only, relabel=False, query=uniform   (RM unused)
    pebble_disagree reward_mode=rm_only,  relabel=True,  query=disagreement

Outputs into ``{out_dir}/<run_id>/``:
    training_curve.csv, eval.parquet, attribution.json, summary.json,
    config.json, run.log.

Usage:
    uv run python experiments/run_pebble.py \\
        --config_kind pebble_full --seed 0 \\
        --total_steps 30000 --query_phase_steps 5000 \\
        --queries_per_phase 50 --eval_episodes 500 \\
        --out_dir experiments/runs_pebble/pebble_full_s0
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3.common.utils import set_random_seed  # noqa: E402

from billiards.env import Billiards4BallEnv  # noqa: E402
from billiards.pebble import PEBBLEAgent  # noqa: E402
from billiards.preference.labeler_heuristic import position_bonus  # noqa: E402


T_MAX = 12.0


CONFIG_KINDS: dict[str, dict] = {
    "pebble_full":      {"reward_mode": "rm_only",  "relabel_after_query": True,  "query_strategy": "uniform"},
    "sac_rm_frozen":    {"reward_mode": "rm_only",  "relabel_after_query": False, "query_strategy": "uniform"},
    "sac_env":          {"reward_mode": "env_only", "relabel_after_query": False, "query_strategy": "uniform"},
    "pebble_disagree":  {"reward_mode": "rm_only",  "relabel_after_query": True,  "query_strategy": "disagreement"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
                st.flush()
            except Exception:  # noqa: BLE001
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass


def _attribution(eval_rows: list[dict], cue_id: int = 0) -> dict:
    """Per-episode breakdown of heuristic_score (mirrors run_one._attribution)."""
    cushion_term: list[float] = []
    score_term: list[float] = []
    foul_term: list[float] = []
    red_contact_term: list[float] = []
    opp_contact_term: list[float] = []
    position_term: list[float] = []
    duration_term: list[float] = []
    total: list[float] = []

    spec = {"cue_id": int(cue_id)}
    for row in eval_rows:
        cnt = Counter(row["event_types"])
        sc = 5.0 * float(row["score"])
        fl = -5.0 if bool(row["fouled"]) else 0.0
        cu = 0.4 * min(int(row["cushion_hits"]), 5)
        rc = 0.6 * int(cnt.get("cue_hit_red", 0))
        oc = -0.3 * int(cnt.get("cue_hit_opp", 0))
        pos = float(position_bonus(row["final_state"], spec))
        du = -0.05 * float(row["duration"])
        score_term.append(sc); foul_term.append(fl); cushion_term.append(cu)
        red_contact_term.append(rc); opp_contact_term.append(oc)
        position_term.append(pos); duration_term.append(du)
        total.append(sc + fl + cu + rc + oc + pos + du)

    out: dict = {"n": int(len(eval_rows))}

    def _ms(name: str, v: list[float]) -> None:
        a = np.asarray(v, dtype=np.float64)
        out[name] = float(a.mean()) if a.size else float("nan")
        out[f"{name}_std"] = float(a.std(ddof=0)) if a.size else float("nan")

    _ms("score_term", score_term)
    _ms("foul_term", foul_term)
    _ms("cushion_term", cushion_term)
    _ms("red_contact_term", red_contact_term)
    _ms("opp_contact_term", opp_contact_term)
    _ms("position_term", position_term)
    _ms("duration_term", duration_term)
    _ms("total", total)
    return out


_CURVE_COLS = [
    "timesteps", "ep_env_score_mean", "ep_rm_score_mean",
    "ep_cushion_mean", "ep_foul_rate", "queries_used_so_far",
    "rm_train_loss", "relabel_delta_mean", "relabel_delta_std",
]


def _write_curve(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CURVE_COLS)
        for r in rows:
            w.writerow([r.get(c, "") for c in _CURVE_COLS])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="One PEBBLE config run.")
    parser.add_argument("--config_kind", type=str, required=True, choices=list(CONFIG_KINDS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_steps", type=int, default=30_000)
    parser.add_argument("--query_phase_steps", type=int, default=5_000)
    parser.add_argument("--queries_per_phase", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_f = log_path.open("w", encoding="utf-8")
    tee = _Tee(sys.__stdout__, log_f)
    err_tee = _Tee(sys.__stderr__, log_f)

    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err_tee):
        t0 = time.perf_counter()
        kind_kwargs = CONFIG_KINDS[args.config_kind]
        config = {
            "config_kind": args.config_kind,
            "seed": int(args.seed),
            "total_steps": int(args.total_steps),
            "query_phase_steps": int(args.query_phase_steps),
            "queries_per_phase": int(args.queries_per_phase),
            "eval_episodes": int(args.eval_episodes),
            "t_max": T_MAX,
            **kind_kwargs,
        }
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"[run_pebble] kind={args.config_kind} seed={args.seed} "
              f"total_steps={args.total_steps} out_dir={out_dir}")

        set_random_seed(int(args.seed))

        def _env_factory():
            return Billiards4BallEnv(t_max=T_MAX)

        agent = PEBBLEAgent(
            env_factory=_env_factory,
            total_steps=int(args.total_steps),
            query_phase_steps=int(args.query_phase_steps),
            queries_per_phase=int(args.queries_per_phase),
            query_strategy=str(kind_kwargs["query_strategy"]),
            relabel_after_query=bool(kind_kwargs["relabel_after_query"]),
            ensemble_size=2,
            reward_mode=str(kind_kwargs["reward_mode"]),
            alpha=0.0,
            seed=int(args.seed),
            device="cpu",
        )

        try:
            t_train0 = time.perf_counter()
            train_summary = agent.learn()
            train_wall = time.perf_counter() - t_train0
            print(f"[run_pebble] train done in {train_wall:.1f}s "
                  f"queries={train_summary['queries_used']}")

            _write_curve(train_summary["curve_rows"], out_dir / "training_curve.csv")

            t_eval0 = time.perf_counter()
            eval_rows = agent.evaluate(
                n_episodes=int(args.eval_episodes),
                seed_base=int(args.seed) + 10_000,
            )
            eval_wall = time.perf_counter() - t_eval0
            eval_df = pd.DataFrame(eval_rows)
            eval_path = out_dir / "eval.parquet"
            eval_df.to_parquet(eval_path, engine="pyarrow", index=False)
            print(f"[run_pebble] eval n={len(eval_df)} done in {eval_wall:.1f}s "
                  f"-> {eval_path}")

            attribution = _attribution(eval_rows, cue_id=0)
            with (out_dir / "attribution.json").open("w", encoding="utf-8") as f:
                json.dump(attribution, f, indent=2)

            true_score_rate = 100.0 * float(eval_df["score"].mean()) if len(eval_df) else 0.0
            foul_rate = 100.0 * float(eval_df["fouled"].mean()) if len(eval_df) else 0.0
            mean_cushions = float(eval_df["cushion_hits"].mean()) if len(eval_df) else 0.0
            rm_mean = float(eval_df["rm_reward"].mean()) if len(eval_df) and "rm_reward" in eval_df else float("nan")
            wall = time.perf_counter() - t0
            summary = {
                "config_kind": args.config_kind,
                "seed": int(args.seed),
                "train_wall_s": float(train_wall),
                "eval_wall_s": float(eval_wall),
                "wall_s": float(wall),
                "true_score_rate": float(true_score_rate),
                "rm_score_mean": float(rm_mean),
                "foul_rate": float(foul_rate),
                "mean_cushions": float(mean_cushions),
                "queries_used": int(train_summary["queries_used"]),
                "n_phases": int(train_summary["n_phases"]),
                "last_rm_train_loss": float(train_summary["last_rm_train_loss"]),
                "last_relabel_delta_mean": float(train_summary["last_relabel_delta_mean"]),
                "last_relabel_delta_std": float(train_summary["last_relabel_delta_std"]),
                "attribution": attribution,
            }
            with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(
                f"[run_pebble] DONE kind={args.config_kind} seed={args.seed} "
                f"true_score%={true_score_rate:.2f} rm_mean={rm_mean:.3f} "
                f"foul%={foul_rate:.2f} cush={mean_cushions:.2f} wall={wall:.1f}s"
            )
        finally:
            log_f.close()


if __name__ == "__main__":
    main()
