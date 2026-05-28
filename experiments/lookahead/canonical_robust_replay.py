"""Side-by-side behavioral comparison: baseline lookahead vs +robust ranking.

Both variants share the SAME baseline proposer (``fast_long_fp02_s{1,4,6}``)
and the same h=2 lookahead, identical seeds — the only difference is whether
the candidate ranking adds the ``+ robust_beta * robustness(a1)`` term on
scoring candidates. Starts from the **canonical** rack (no random_start) and
forces ``continue_on_miss=True`` so both runs play exactly ``--max_shots``
shots, no early termination. This way per-shot behavior is paired across
variants for the same shot index.

Outputs (under ``--out`` dir):
  variant_true_baseline.html        full inning replay (50 shots merged)
  variant_baseline_robust.html
  per_shot.csv                      one row per (variant, shot_idx)
  summary.json                      aggregate score / robustness / cushions

Usage:
    python experiments/lookahead/canonical_robust_replay.py \\
        --max_shots 50 --k1 100 --k2 3 \\
        --robust_beta 0.2 --robust_eps 0.05 --robust_n 8 \\
        --out experiments/artifacts/canonical_robust_replay
"""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.render.replay import render_inning_html  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

GAMMA = 0.99

PROPOSER_PATHS = [
    "experiments/runs_inning_v2/fast_long_fp02_s1/policy.zip",
    "experiments/runs_inning_v2/fast_long_fp02_s4/policy.zip",
    "experiments/runs_inning_v2/fast_long_fp02_s6/policy.zip",
]


def K_schedule(shot_idx: int, k1: int) -> int:
    if shot_idx == 0:
        return k1
    if shot_idx <= 3:
        return max(k1 // 2, 1)
    if shot_idx <= 9:
        return max(k1 // 4, 1)
    return max(k1 // 6, 1)


def cands_multi(models, obs, k_per):
    out = []
    for m in models:
        det, _ = m.predict(obs, deterministic=True)
        out.append(np.asarray(det, np.float32).reshape(-1))
        for _ in range(k_per - 1):
            sto, _ = m.predict(obs, deterministic=False)
            out.append(np.asarray(sto, np.float32).reshape(-1))
    return out


def cands_single(model, obs, k):
    out = []
    det, _ = model.predict(obs, deterministic=True)
    out.append(np.asarray(det, np.float32).reshape(-1))
    for _ in range(k - 1):
        sto, _ = model.predict(obs, deterministic=False)
        out.append(np.asarray(sto, np.float32).reshape(-1))
    return out


def _snap(b):
    return (copy.deepcopy(b._state), b._shot_index, b._cumulative_score,
            b._cumulative_t, list(b._shot_trajectories),
            list(b._shot_offsets), list(b._inning_log_records))


def _rest(b, s):
    b._state = copy.deepcopy(s[0]); b._shot_index = s[1]
    b._cumulative_score = s[2]; b._cumulative_t = s[3]
    b._shot_trajectories = list(s[4]); b._shot_offsets = list(s[5])
    b._inning_log_records = list(s[6])


def _try(b, a):
    try:
        return b.step(a)
    except Exception:
        return None, -1e9, True, True, {}


def _estimate_robustness(base, s0, a1, robust_eps, robust_n, rng):
    succ = 0
    raw = np.asarray(a1, dtype=np.float64)
    for _ in range(robust_n):
        _rest(base, s0)
        delta = rng.normal(0.0, robust_eps, size=4)
        pert = (raw + delta).astype(np.float32)
        _, _, _, _, info_pert = _try(base, pert)
        if info_pert and (int(info_pert.get("score", 0)) > 0
                          and not bool(info_pert.get("fouled", False))):
            succ += 1
    return succ / max(1, robust_n)


def play_inning(models, robust_rank, *,
                max_shots, k1, k2, robust_beta, robust_eps, robust_n,
                rng_seed=0):
    """Play a single ``max_shots``-long canonical inning. continue_on_miss=True
    so both variants play exactly max_shots, regardless of misses.

    Returns (shot_rows, shot_trajectories, spec)."""
    base = Billiards4BallInningEnv(
        t_max=12.0, max_shots=max_shots,
        # Force the inning to last for max_shots regardless of miss/foul so
        # behavior is paired shot-by-shot across variants.
        continue_on_miss=True,
        constrain_aim=True, extra_features=True,
        foul_penalty=0.2, gentle_shot=True,
        setup_shaping=True, setup_alpha=0.05, setup_scale=0.3,
        robust_reward=False,  # ranking-only, no env-side bonus
    )
    # Canonical start: no RandomStartInningEnv wrapper.
    obs, _ = base.reset(seed=rng_seed)
    rng = np.random.default_rng(rng_seed)
    main_model = models[1]

    rows: list[dict] = []
    while True:
        shot_idx = base._shot_index  # 0-based: index of the shot we're about to take
        k_per = K_schedule(shot_idx, k1)
        c1 = cands_multi(models, obs, k_per)
        s0 = _snap(base)
        bv, ba, b_robust = -1e9, c1[0], float("nan")
        for a1 in c1:
            _rest(base, s0)
            obs1, r1, t1, tr1, info1 = _try(base, a1)
            r1_scored = (
                info1 is not None
                and int(info1.get("score", 0)) > 0
                and not bool(info1.get("fouled", False))
            )
            if t1 or tr1 or obs1 is None:
                v = r1
            else:
                s1_snap = _snap(base)
                br2 = -1e9
                for a2 in cands_single(main_model, obs1, k2):
                    _rest(base, s1_snap)
                    _, r2, _, _, _ = _try(base, a2)
                    if r2 > br2:
                        br2 = r2
                v = r1 + GAMMA * br2

            this_robust = float("nan")
            if robust_rank and r1_scored:
                this_robust = _estimate_robustness(
                    base, s0, a1, robust_eps, robust_n, rng,
                )
                v += robust_beta * this_robust

            if v > bv:
                bv, ba, b_robust = v, a1, this_robust

        # Execute the chosen action for real.
        _rest(base, s0)
        obs, _, term, trunc, info = base.step(ba)
        chose_scored = (
            int(info.get("score", 0)) > 0
            and not bool(info.get("fouled", False))
        )
        # Always measure chosen-action robustness for reporting (so the
        # column is comparable across variants).
        if chose_scored:
            if robust_rank and not np.isnan(b_robust):
                chosen_robust = float(b_robust)
            else:
                s_post = _snap(base)
                chosen_robust = _estimate_robustness(
                    base, s0, ba, robust_eps, robust_n, rng,
                )
                _rest(base, s_post)
        else:
            chosen_robust = float("nan")

        a = np.asarray(ba, dtype=float).reshape(-1)
        rows.append({
            "shot_idx": int(shot_idx),
            "theta": float(a[0]),
            "power": float(a[1]),
            "spin_a": float(a[2]),
            "spin_b": float(a[3]),
            "score": int(info.get("score", 0)),
            "fouled": bool(info.get("fouled", False)),
            "cushion_hits": int(info.get("cushion_hits", 0)),
            "duration": float(info.get("duration", 0.0)),
            "cumulative_score": int(base._cumulative_score),
            "chosen_robustness": (None if np.isnan(chosen_robust)
                                   else float(chosen_robust)),
        })

        if term or trunc:
            break

    return rows, list(base._shot_trajectories), base._spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_shots", type=int, default=50)
    parser.add_argument("--k1", type=int, default=100)
    parser.add_argument("--k2", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0,
                        help="Inning RNG seed (canonical layout is fixed; this "
                             "only seeds perturbation noise so the comparison "
                             "is paired across variants).")
    parser.add_argument("--robust_beta", type=float, default=0.2)
    parser.add_argument("--robust_eps", type=float, default=0.05)
    parser.add_argument("--robust_n", type=int, default=8)
    parser.add_argument("--out", type=str,
                        default="experiments/artifacts/canonical_robust_replay")
    args = parser.parse_args()

    out_dir = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[canon_replay] loading {len(PROPOSER_PATHS)} proposer models...",
          flush=True)
    models = [SAC.load(str(REPO / p), device="cpu") for p in PROPOSER_PATHS]

    variants = [
        ("true_baseline", False),
        ("baseline_robust", True),
    ]
    all_rows: list[dict] = []
    summary: dict[str, dict] = {}
    for label, robust_rank in variants:
        print(f"\n=== variant={label}  robust_rank={robust_rank} "
              f"max_shots={args.max_shots} k1={args.k1} k2={args.k2} "
              f"seed={args.seed} ===", flush=True)
        t0 = time.perf_counter()
        rows, trajs, spec = play_inning(
            models, robust_rank=robust_rank,
            max_shots=args.max_shots, k1=args.k1, k2=args.k2,
            robust_beta=args.robust_beta, robust_eps=args.robust_eps,
            robust_n=args.robust_n,
            rng_seed=args.seed,
        )
        wall = time.perf_counter() - t0
        for r in rows:
            r2 = dict(r); r2["variant"] = label
            all_rows.append(r2)

        # Render HTML.
        html_path = out_dir / f"variant_{label}.html"
        render_inning_html(trajs, spec=spec, save_path=html_path)

        # Aggregate stats.
        scores = [r["score"] for r in rows]
        chosen_rb_vals = [r["chosen_robustness"] for r in rows
                          if r["chosen_robustness"] is not None]
        cushions = [r["cushion_hits"] for r in rows]
        durations = [r["duration"] for r in rows]
        summary[label] = {
            "robust_rank": robust_rank,
            "n_shots": len(rows),
            "total_score": int(sum(scores)),
            "n_scoring_shots": int(sum(1 for s in scores if s > 0)),
            "n_fouls": int(sum(1 for r in rows if r["fouled"])),
            "mean_cushion_hits": float(np.mean(cushions)) if cushions else 0.0,
            "total_sim_time_s": float(sum(durations)),
            "mean_chosen_robustness": (
                float(np.mean(chosen_rb_vals)) if chosen_rb_vals else None
            ),
            "std_chosen_robustness": (
                float(np.std(chosen_rb_vals)) if chosen_rb_vals else None
            ),
            "wall_s": float(wall),
            "html": str(html_path.relative_to(REPO)),
        }
        print(f"--- {label}: total_score={summary[label]['total_score']} "
              f"scoring_shots={summary[label]['n_scoring_shots']}/{len(rows)} "
              f"fouls={summary[label]['n_fouls']} "
              f"mean_cushion={summary[label]['mean_cushion_hits']:.2f} "
              f"chosen_robust={(summary[label]['mean_chosen_robustness'] or 0):.3f} "
              f"sim_t={summary[label]['total_sim_time_s']:.1f}s "
              f"wall={wall:.0f}s -> {html_path.name}", flush=True)

    # Per-shot CSV (long format: one row per variant per shot).
    csv_path = out_dir / "per_shot.csv"
    fields = ["variant", "shot_idx", "score", "fouled", "cushion_hits",
              "duration", "cumulative_score", "chosen_robustness",
              "theta", "power", "spin_a", "spin_b"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k) for k in fields})

    # JSON summary.
    config = {
        "max_shots": args.max_shots, "k1": args.k1, "k2": args.k2,
        "seed": args.seed,
        "robust_beta": args.robust_beta, "robust_eps": args.robust_eps,
        "robust_n": args.robust_n,
        "proposers": PROPOSER_PATHS,
        "canonical_start": True,
        "continue_on_miss": True,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config": config, "results": summary}, f, indent=2)

    # Quick pairwise diff print: action delta per shot, score diff per shot.
    print("\n=== per-shot pairwise diff (true_baseline vs baseline_robust) ===")
    tb_rows = [r for r in all_rows if r["variant"] == "true_baseline"]
    br_rows = [r for r in all_rows if r["variant"] == "baseline_robust"]
    same_action = 0
    diff_score = 0
    for a, b in zip(tb_rows, br_rows):
        # An action is "the same" if all 4 components match within 1e-3.
        a_vec = np.array([a["theta"], a["power"], a["spin_a"], a["spin_b"]])
        b_vec = np.array([b["theta"], b["power"], b["spin_a"], b["spin_b"]])
        if np.allclose(a_vec, b_vec, atol=1e-3):
            same_action += 1
        if a["score"] != b["score"]:
            diff_score += 1
    n = len(tb_rows)
    print(f"  shots compared: {n}")
    print(f"  identical chosen action (≤1e-3): {same_action}/{n} "
          f"({same_action/n:.1%})")
    print(f"  diverged score: {diff_score}/{n}")
    print(f"\n[canon_replay] -> {out_dir}")


if __name__ == "__main__":
    main()
