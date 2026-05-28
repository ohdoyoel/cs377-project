"""A/B/C: lookahead with action-space robustness ranking + robust proposer.

Three paired conditions decompose the robust effect (analogue of
``compare_time_proposers.py`` for the ``robust_reward`` line):

  true_baseline    : fast_long_fp02 proposer, no robust term at ranking
  baseline + robust: fast_long_fp02 proposer, robust bonus at ranking
  robust + robust  : robust_s* proposer, robust bonus at ranking

Each candidate ``a1`` is ranked by ``r1 + gamma * br2`` with an extra
``+ robust_beta * robustness(a1)`` term when ``robust_rank=True`` AND the
candidate's first-step result scored (matching the training-time bonus
condition). robustness(a1) is the fraction of N perturbed actions
``a1 + N(0, robust_eps)`` that still score from the pre-shot state.

For consistent across-variant reporting, we also measure robustness of the
*chosen* action on scoring shots in every variant (not just the ones with
robust_rank=True). So mean_robustness is comparable across all three.

    python experiments/lookahead/compare_robust_proposers.py \\
        --n_eps 12 --max_shots 25 --k1 100 --k2 3 \\
        --robust_beta 0.2 --robust_eps 0.05 --robust_n 8 \\
        --out experiments/artifacts/lookahead_robust_abc
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

PROPOSER_SETS = {
    "baseline": [
        "experiments/runs_inning_v2/fast_long_fp02_s1/policy.zip",
        "experiments/runs_inning_v2/fast_long_fp02_s4/policy.zip",
        "experiments/runs_inning_v2/fast_long_fp02_s6/policy.zip",
    ],
    "robust": [
        "experiments/runs_inning_v2/robust_s1/policy.zip",
        "experiments/runs_inning_v2/robust_s4/policy.zip",
        "experiments/runs_inning_v2/robust_s6/policy.zip",
    ],
}

VARIANTS = {
    "true_baseline":   {"proposers": "baseline", "robust_rank": False},
    "baseline_robust": {"proposers": "baseline", "robust_rank": True},
    "robust_robust":   {"proposers": "robust",   "robust_rank": True},
}


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


def _estimate_robustness_from_snap(base, s0, a1, robust_eps, robust_n, rng):
    """Replay ``robust_n`` perturbations of ``a1`` from pre-shot snapshot
    ``s0``. Returns fraction that scored (non-foul). Leaves base in a
    post-perturbation state — caller must _rest() if it needs s0 again."""
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


def run_variant(models, n_eps, k1, k2, max_shots, seed_base,
                robust_rank, robust_beta, robust_eps, robust_n):
    scores, shots, sim_times = [], [], []
    # Per-episode mean robustness of CHOSEN actions on scoring shots.
    # NaN if the episode had no scoring shots.
    ep_chosen_robustness: list[float] = []
    t_start = time.time()
    for ep in range(n_eps):
        base = Billiards4BallInningEnv(
            t_max=12.0, max_shots=max_shots, continue_on_miss=False,
            constrain_aim=True, extra_features=True,
            foul_penalty=0.2, gentle_shot=True,
            setup_shaping=True, setup_alpha=0.05, setup_scale=0.3,
            # robust_reward must stay OFF in the env: we compute it manually
            # for ranking + reporting (env path would double-count and slow
            # every candidate sim).
            robust_reward=False,
        )
        env = RandomStartInningEnv(base)
        obs, _ = env.reset(seed=seed_base + ep)
        main_model = models[1]  # s4-equivalent for second-step candidates
        # Deterministic per-episode RNG so perturbation noise is reproducible
        # and paired across variants (same seed_base + ep → same noise).
        rng = np.random.default_rng(seed_base + ep)
        chosen_robust_log: list[float] = []
        ep_t = time.time()
        while True:
            k_per = K_schedule(base._shot_index, k1)
            c1 = cands_multi(models, obs, k_per)
            s0 = _snap(base)
            bv, ba, b_robust = -1e9, c1[0], float("nan")
            for a1 in c1:
                _rest(base, s0)
                obs1, r1, t1, tr1, info1 = _try(base, a1)
                # Did this candidate score on its first shot?
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

                # Optional robust ranking bonus — only on scoring shots.
                this_robust = float("nan")
                if robust_rank and r1_scored:
                    this_robust = _estimate_robustness_from_snap(
                        base, s0, a1, robust_eps, robust_n, rng,
                    )
                    v += robust_beta * this_robust

                if v > bv:
                    bv, ba, b_robust = v, a1, this_robust

            # Apply the chosen action for real (from pre-shot snapshot).
            _rest(base, s0)
            obs, _, term, trunc, info = env.step(ba)
            chose_scored = (
                int(info.get("score", 0)) > 0
                and not bool(info.get("fouled", False))
            )
            if chose_scored:
                # Robustness measurement of chosen action, for consistent
                # cross-variant reporting. When robust_rank=True we already
                # computed b_robust during ranking; reuse it. When False,
                # measure now (preserve the real post-shot state).
                if robust_rank and not np.isnan(b_robust):
                    chosen_robust_log.append(float(b_robust))
                else:
                    s_post = _snap(base)
                    rb = _estimate_robustness_from_snap(
                        base, s0, ba, robust_eps, robust_n, rng,
                    )
                    _rest(base, s_post)
                    chosen_robust_log.append(float(rb))

            if term or trunc:
                break
        scores.append(int(base.cumulative_score))
        shots.append(int(base.shot_index))
        sim_times.append(float(base._cumulative_t))
        ep_chosen_robustness.append(
            float(np.mean(chosen_robust_log)) if chosen_robust_log
            else float("nan")
        )
        print(f"  ep {ep}: score={scores[-1]} shots={shots[-1]} "
              f"sim_t={sim_times[-1]:.1f}s "
              f"chosen_robust={ep_chosen_robustness[-1]:.3f} "
              f"ep_wall={time.time()-ep_t:.0f}s "
              f"total={time.time()-t_start:.0f}s", flush=True)
    sc = np.array(scores, float)
    st = np.array(sim_times, float)
    chosen_rb = np.array(ep_chosen_robustness, float)
    chosen_rb_clean = chosen_rb[~np.isnan(chosen_rb)]
    return {
        "mean_score": float(sc.mean()),
        "std_score": float(sc.std()),
        "max_score": int(sc.max()),
        "mean_shots": float(np.mean(shots)),
        "mean_sim_time_s": float(st.mean()),
        "score_per_sim_s": float(sc.sum() / st.sum()) if st.sum() > 0 else 0.0,
        "mean_chosen_robustness": (
            float(chosen_rb_clean.mean()) if chosen_rb_clean.size > 0 else None
        ),
        "std_chosen_robustness": (
            float(chosen_rb_clean.std()) if chosen_rb_clean.size > 0 else None
        ),
        "n_eps_with_score": int(chosen_rb_clean.size),
        "scores": scores,
        "sim_times": sim_times,
        "chosen_robustness_per_ep": [
            None if np.isnan(x) else float(x) for x in chosen_rb
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_eps", type=int, default=12)
    parser.add_argument("--max_shots", type=int, default=25)
    parser.add_argument("--k1", type=int, default=100,
                        help="Per-policy K for shot 0.")
    parser.add_argument("--k2", type=int, default=3,
                        help="Second-step candidates.")
    parser.add_argument("--seed_base", type=int, default=99000)
    parser.add_argument("--robust_beta", type=float, default=0.2)
    parser.add_argument("--robust_eps", type=float, default=0.05)
    parser.add_argument("--robust_n", type=int, default=8)
    parser.add_argument("--variants", nargs="+",
                        default=["true_baseline", "baseline_robust", "robust_robust"],
                        choices=list(VARIANTS))
    parser.add_argument("--out", type=str,
                        default="experiments/artifacts/lookahead_robust_abc")
    args = parser.parse_args()

    out_dir = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    model_cache: dict[str, list] = {}
    for variant in args.variants:
        spec = VARIANTS[variant]
        pset = spec["proposers"]
        robust_rank = spec["robust_rank"]
        paths = [REPO / p for p in PROPOSER_SETS[pset]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[abc] SKIP {variant}: missing {[str(m) for m in missing]}")
            continue
        if pset not in model_cache:
            model_cache[pset] = [SAC.load(str(p), device="cpu") for p in paths]
        models = model_cache[pset]
        print(f"\n=== variant={variant}  proposers={pset}  "
              f"robust_rank={robust_rank}  beta={args.robust_beta} "
              f"eps={args.robust_eps} N={args.robust_n} "
              f"k1={args.k1} k2={args.k2} max_shots={args.max_shots} "
              f"n={args.n_eps} ===")
        stats = run_variant(
            models, n_eps=args.n_eps, k1=args.k1, k2=args.k2,
            max_shots=args.max_shots, seed_base=args.seed_base,
            robust_rank=robust_rank, robust_beta=args.robust_beta,
            robust_eps=args.robust_eps, robust_n=args.robust_n,
        )
        stats["proposers"] = pset
        stats["robust_rank"] = robust_rank
        results[variant] = stats
        rb = stats.get("mean_chosen_robustness")
        rb_str = "n/a" if rb is None else f"{rb:.3f}"
        print(f"--- {variant}: mean_score={stats['mean_score']:.2f}±"
              f"{stats['std_score']:.2f} max={stats['max_score']} "
              f"mean_shots={stats['mean_shots']:.1f} "
              f"mean_sim_t={stats['mean_sim_time_s']:.1f}s "
              f"score/sim_s={stats['score_per_sim_s']:.3f} "
              f"mean_chosen_robust={rb_str}")

    config = {
        "n_eps": args.n_eps, "max_shots": args.max_shots,
        "k1": args.k1, "k2": args.k2, "seed_base": args.seed_base,
        "robust_beta": args.robust_beta, "robust_eps": args.robust_eps,
        "robust_n": args.robust_n,
    }
    with (out_dir / "abc_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config": config, "results": results}, f, indent=2)

    if all(v in results for v in ("true_baseline", "baseline_robust", "robust_robust")):
        tb = results["true_baseline"]
        br = results["baseline_robust"]
        rr = results["robust_robust"]
        print("\n=== decomposition (mean_score, score/sim_s, chosen_robustness) ===")
        for k, label in (("mean_score", "mean_score"),
                         ("score_per_sim_s", "score/sim_s"),
                         ("mean_chosen_robustness", "chosen_robust")):
            tb_v = tb.get(k); br_v = br.get(k); rr_v = rr.get(k)
            def _fmt(x):
                return "n/a" if x is None else f"{x:.3f}"
            d_rank = (br_v - tb_v) if (tb_v is not None and br_v is not None) else None
            d_prop = (rr_v - br_v) if (br_v is not None and rr_v is not None) else None
            print(f"  {label:16s}: tb={_fmt(tb_v)} → br={_fmt(br_v)} "
                  f"(rank effect {('n/a' if d_rank is None else f'{d_rank:+.3f}')}) "
                  f"→ rr={_fmt(rr_v)} "
                  f"(prop effect {('n/a' if d_prop is None else f'{d_prop:+.3f}')})")
    print(f"\n[abc] -> {out_dir / 'abc_summary.json'}")


if __name__ == "__main__":
    main()
