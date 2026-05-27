"""A/B: lookahead with time-reward-fine-tuned proposers vs non-time proposers.

Both variants run the *same* h=2 multi-seed (s1, s4, s6) lookahead with the
SAME ``TIME_REWARD`` candidate ranking; only the proposer policy set differs:

  baseline : fast_long_fp02_s{1,4,6}   (non-time)
  time     : fast_time_fp02_s{1,4,6}   (time-reward fine-tuned, 2026-05-27)

Paired by seed (variant uses identical episode seeds), so a difference is
attributable to the proposer distribution, not the layout. Lookahead does not
*learn*; this measures whether time-fine-tuned candidates plan better/faster.

Reports per variant: mean score, mean shots, mean simulated table time, and
score per simulated second (the time-efficiency the time bonus targets).

    python experiments/lookahead/compare_time_proposers.py \\
        --n_eps 8 --max_shots 40 --k1 60 --k2 3 \\
        --out experiments/artifacts/lookahead_time_ab
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

GAMMA = 0.99
TIME_ALPHA = 0.2
TIME_SCALE = 3.0

PROPOSER_SETS = {
    "baseline": [
        "experiments/runs_inning_v2/fast_long_fp02_s1/policy.zip",
        "experiments/runs_inning_v2/fast_long_fp02_s4/policy.zip",
        "experiments/runs_inning_v2/fast_long_fp02_s6/policy.zip",
    ],
    "time": [
        "experiments/runs_inning_v2/fast_time_fp02_s1/policy.zip",
        "experiments/runs_inning_v2/fast_time_fp02_s4/policy.zip",
        "experiments/runs_inning_v2/fast_time_fp02_s6/policy.zip",
    ],
}

# Each condition = (proposer set, whether the lookahead RANKING applies the
# time bonus). Decomposes the time effect into ranking vs proposer:
#   true_baseline : non-time proposer, no time bonus anywhere (original lookahead)
#   baseline      : non-time proposer, time bonus only at candidate ranking
#   time          : time proposer + time bonus at ranking
# (true_baseline -> baseline) = pure ranking effect;
# (baseline -> time)          = pure proposer effect.
VARIANTS = {
    "true_baseline": {"proposers": "baseline", "time_rank": False},
    "baseline":      {"proposers": "baseline", "time_rank": True},
    "time":          {"proposers": "time",     "time_rank": True},
}


def K_schedule(shot_idx: int, k1: int) -> int:
    """Per-policy candidate count; total = K * n_policies. Lighter than the
    production multi_seed_h2 schedule so an A/B finishes in bounded time."""
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


def run_variant(models, n_eps, k1, k2, max_shots, seed_base, time_rank):
    scores, shots, sim_times = [], [], []
    t_start = time.time()
    for ep in range(n_eps):
        base = Billiards4BallInningEnv(
            t_max=12.0, max_shots=max_shots, continue_on_miss=False,
            constrain_aim=True, extra_features=True,
            foul_penalty=0.2, gentle_shot=True,
            setup_shaping=True, setup_alpha=0.05, setup_scale=0.3,
            time_reward=time_rank, time_alpha=TIME_ALPHA, time_scale=TIME_SCALE,
        )
        env = RandomStartInningEnv(base)
        obs, _ = env.reset(seed=seed_base + ep)
        main_model = models[1]  # s4 for second-step candidates
        ep_t = time.time()
        while True:
            k_per = K_schedule(base._shot_index, k1)
            c1 = cands_multi(models, obs, k_per)
            s0 = _snap(base)
            bv, ba = -1e9, c1[0]
            for a1 in c1:
                _rest(base, s0)
                obs1, r1, t1, tr1, _ = _try(base, a1)
                if t1 or tr1 or obs1 is None:
                    v = r1
                else:
                    s1 = _snap(base)
                    br2 = -1e9
                    for a2 in cands_single(main_model, obs1, k2):
                        _rest(base, s1)
                        _, r2, _, _, _ = _try(base, a2)
                        if r2 > br2:
                            br2 = r2
                    v = r1 + GAMMA * br2
                if v > bv:
                    bv, ba = v, a1
            _rest(base, s0)
            obs, _, term, trunc, _ = env.step(ba)
            if term or trunc:
                break
        scores.append(int(base.cumulative_score))
        shots.append(int(base.shot_index))
        sim_times.append(float(base._cumulative_t))
        print(f"  ep {ep}: score={scores[-1]} shots={shots[-1]} "
              f"sim_t={sim_times[-1]:.1f}s ep_wall={time.time()-ep_t:.0f}s "
              f"total={time.time()-t_start:.0f}s", flush=True)
    sc = np.array(scores, float)
    st = np.array(sim_times, float)
    return {
        "mean_score": float(sc.mean()),
        "std_score": float(sc.std()),
        "max_score": int(sc.max()),
        "mean_shots": float(np.mean(shots)),
        "mean_sim_time_s": float(st.mean()),
        "score_per_sim_s": float(sc.sum() / st.sum()) if st.sum() > 0 else 0.0,
        "scores": scores,
        "sim_times": sim_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_eps", type=int, default=8)
    parser.add_argument("--max_shots", type=int, default=40)
    parser.add_argument("--k1", type=int, default=60, help="Per-policy K for shot 0.")
    parser.add_argument("--k2", type=int, default=3, help="Second-step candidates.")
    parser.add_argument("--seed_base", type=int, default=99000)
    parser.add_argument("--variants", nargs="+",
                        default=["true_baseline", "baseline", "time"],
                        choices=list(VARIANTS))
    parser.add_argument("--out", type=str, default="experiments/artifacts/lookahead_time_ab")
    args = parser.parse_args()

    out_dir = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    model_cache: dict[str, list] = {}
    for variant in args.variants:
        spec = VARIANTS[variant]
        pset, time_rank = spec["proposers"], spec["time_rank"]
        paths = [REPO / p for p in PROPOSER_SETS[pset]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[ab] SKIP {variant}: missing {[str(m) for m in missing]}")
            continue
        if pset not in model_cache:
            model_cache[pset] = [SAC.load(str(p), device="cpu") for p in paths]
        models = model_cache[pset]
        print(f"\n=== variant={variant}  proposers={pset}  time_rank={time_rank}  "
              f"k1={args.k1} k2={args.k2} max_shots={args.max_shots} n={args.n_eps} ===")
        stats = run_variant(models, n_eps=args.n_eps, k1=args.k1, k2=args.k2,
                            max_shots=args.max_shots, seed_base=args.seed_base,
                            time_rank=time_rank)
        stats["proposers"] = pset
        stats["time_rank"] = time_rank
        results[variant] = stats
        print(f"--- {variant}: mean_score={stats['mean_score']:.2f}±{stats['std_score']:.2f} "
              f"max={stats['max_score']} mean_shots={stats['mean_shots']:.1f} "
              f"mean_sim_t={stats['mean_sim_time_s']:.1f}s "
              f"score/sim_s={stats['score_per_sim_s']:.3f}")

    config = {"n_eps": args.n_eps, "max_shots": args.max_shots, "k1": args.k1,
              "k2": args.k2, "seed_base": args.seed_base}
    with (out_dir / "ab_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config": config, "results": results}, f, indent=2)

    if all(v in results for v in ("true_baseline", "baseline", "time")):
        tb, b, t = (results["true_baseline"], results["baseline"], results["time"])
        print("\n=== effect decomposition (paired seeds, score/sim_s) ===")
        print(f"  true_baseline   = {tb['score_per_sim_s']:.3f}   (no time anywhere)")
        print(f"  +ranking time   = {b['score_per_sim_s']:.3f}   "
              f"(ranking effect {b['score_per_sim_s']-tb['score_per_sim_s']:+.3f})")
        print(f"  +time proposer  = {t['score_per_sim_s']:.3f}   "
              f"(proposer effect {t['score_per_sim_s']-b['score_per_sim_s']:+.3f})")
        print("  --- mean_sim_t (lower=faster to cap) ---")
        print(f"  true_baseline={tb['mean_sim_time_s']:.1f}s  "
              f"baseline={b['mean_sim_time_s']:.1f}s  time={t['mean_sim_time_s']:.1f}s")
    print(f"[ab] -> {out_dir / 'ab_summary.json'}")


if __name__ == "__main__":
    main()
