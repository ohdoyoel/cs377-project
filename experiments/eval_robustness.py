"""Direct measurement of action-space robustness for one or more policies.

Validates whether ``--robust_reward`` fine-tune actually produces policies
that pick higher-margin actions. For each random rack, we take the policy's
deterministic action ``a*``, perturb it with N independent samples
``a* + N(0, eps)``, replay each from the same pre-shot state, and report the
fraction that still score (no foul). Paired across policies (same random
seeds → same racks), so any difference is attributable to the policy.

Metric definitions (per policy, per eps):
  det_success_rate          fraction of states where a* itself scores
  robustness_overall        mean perturbation success rate, all states
  robustness_conditional    mean perturbation success rate, only on states
                            where a* scored. This is the metric the
                            ``robust_reward`` bonus directly maximizes:
                            "given the policy chose to score here, how
                            much motor-noise margin does the choice have?"

Usage:
    python experiments/eval_robustness.py \\
        --policy baseline_s1=experiments/runs_inning_v2/fast_long_fp02_s1/policy.zip \\
        --policy robust_s1=experiments/runs_inning_v2/robust_s1/policy.zip \\
        --policy baseline_s4=experiments/runs_inning_v2/fast_long_fp02_s4/policy.zip \\
        --policy robust_s4=experiments/runs_inning_v2/robust_s4/policy.zip \\
        --n_states 100 --n_perturb 32 \\
        --eps_list 0.02,0.05,0.10,0.20 \\
        --out experiments/artifacts/robustness_eval
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3 import SAC  # noqa: E402

from billiards.inning_env import Billiards4BallInningEnv, _project_action  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402
from billiards.physics import simulate_shot  # noqa: E402

T_MAX = 12.0


def _measure_policy(
    model,
    n_states: int,
    seed_base: int,
    eps_list: list[float],
    n_perturb: int,
    constrain_aim: bool,
    extra_features: bool,
) -> pd.DataFrame:
    """One row per (state, eps). The deterministic action is taken once per
    state; the same RNG seed is reused across policies (set per state from
    ``seed_base + state_idx``) so perturbation noise is paired too.
    """
    base = Billiards4BallInningEnv(
        t_max=T_MAX, max_shots=10, continue_on_miss=False,
        constrain_aim=constrain_aim, extra_features=extra_features,
    )
    env = RandomStartInningEnv(base)
    rows: list[dict] = []
    for s in range(n_states):
        obs, _ = env.reset(seed=seed_base + s)
        # Deterministic action under the policy.
        a, _ = model.predict(obs, deterministic=True)
        raw_action = np.asarray(a, dtype=np.float64).reshape(-1)

        # Snapshot the pre-shot state. RandomStartInningEnv mutates the inner
        # state on reset, so we grab base._state after reset.
        pre_state = copy.deepcopy(base._state)

        # Replay the deterministic action to see if it scores.
        base._state = copy.deepcopy(pre_state)
        cue_action = _project_action(raw_action)
        if constrain_aim:
            cue_action = base._apply_aim_constraint(cue_action)
        try:
            result = simulate_shot(base._state, cue_action, t_max=T_MAX)
            det_score = int(result["score"])
            det_fouled = bool(result["fouled"])
        except Exception:  # noqa: BLE001
            det_score, det_fouled = 0, True
        det_success = det_score > 0 and not det_fouled

        # Paired perturbation noise: same RNG state per state across policies.
        for eps in eps_list:
            rng = np.random.default_rng(seed_base * 10_000 + s * 100
                                        + int(eps * 1000))
            successes = 0
            for _ in range(n_perturb):
                base._state = copy.deepcopy(pre_state)
                delta = rng.normal(0.0, eps, size=4)
                pert_raw = raw_action + delta
                pert_action = _project_action(pert_raw)
                if constrain_aim:
                    pert_action = base._apply_aim_constraint(pert_action)
                try:
                    pert_result = simulate_shot(
                        base._state, pert_action, t_max=T_MAX,
                    )
                    if (int(pert_result["score"]) > 0
                            and not bool(pert_result["fouled"])):
                        successes += 1
                except Exception:  # noqa: BLE001
                    pass
            rows.append({
                "state_idx": s,
                "seed": seed_base + s,
                "eps": float(eps),
                "robustness": successes / max(1, n_perturb),
                "det_success": bool(det_success),
            })
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, eps_list: list[float]) -> dict:
    """Per-eps mean robustness, overall and conditional on det_success."""
    out: dict[str, dict] = {}
    n_states = int(df["state_idx"].nunique())
    out["n_states"] = n_states
    out["det_success_rate"] = float(
        df[df["eps"] == eps_list[0]]["det_success"].mean()
    )
    out["per_eps"] = {}
    for eps in eps_list:
        sub = df[df["eps"] == eps]
        cond = sub[sub["det_success"]]
        out["per_eps"][f"{eps:.3f}"] = {
            "robustness_overall_mean": float(sub["robustness"].mean()),
            "robustness_overall_std": float(sub["robustness"].std()),
            "robustness_conditional_mean": (
                float(cond["robustness"].mean()) if len(cond) > 0 else None
            ),
            "robustness_conditional_std": (
                float(cond["robustness"].std()) if len(cond) > 0 else None
            ),
            "n_conditional": int(len(cond)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", action="append", required=True,
                   help="LABEL=path/to/policy.zip; can be repeated.")
    p.add_argument("--n_states", type=int, default=100)
    p.add_argument("--seed_base", type=int, default=10_000)
    p.add_argument("--n_perturb", type=int, default=32)
    p.add_argument("--eps_list", type=str, default="0.02,0.05,0.10,0.20")
    p.add_argument("--no_constrain_aim", action="store_true",
                   help="Disable constrain_aim (default ON to match training).")
    p.add_argument("--no_extra_features", action="store_true",
                   help="Disable extra_features (default ON to match training).")
    p.add_argument("--out", type=str,
                   default="experiments/artifacts/robustness_eval")
    args = p.parse_args()

    eps_list = [float(x) for x in args.eps_list.split(",") if x.strip()]
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    constrain_aim = not args.no_constrain_aim
    extra_features = not args.no_extra_features

    summary_all: dict[str, dict] = {}
    dfs: list[pd.DataFrame] = []
    t_start = time.perf_counter()
    for spec in args.policy:
        if "=" not in spec:
            raise SystemExit(f"--policy expects LABEL=path, got: {spec}")
        label, path = spec.split("=", 1)
        path_p = (_REPO_ROOT / path) if not Path(path).is_absolute() else Path(path)
        if not path_p.exists():
            raise SystemExit(f"policy not found: {path_p}")
        print(f"[robust_eval] loading {label} <- {path_p}", flush=True)
        model = SAC.load(str(path_p), device="cpu")
        t0 = time.perf_counter()
        df = _measure_policy(
            model,
            n_states=int(args.n_states),
            seed_base=int(args.seed_base),
            eps_list=eps_list,
            n_perturb=int(args.n_perturb),
            constrain_aim=constrain_aim,
            extra_features=extra_features,
        )
        wall = time.perf_counter() - t0
        df["label"] = label
        dfs.append(df)
        stats = _summarize(df, eps_list)
        stats["wall_s"] = float(wall)
        summary_all[label] = stats
        per_eps = stats["per_eps"]
        line = (f"[robust_eval] {label}: det_success={stats['det_success_rate']:.2%} "
                f"wall={wall:.1f}s | "
                + " ".join(
                    f"ε={eps:.2f}→{per_eps[f'{eps:.3f}']['robustness_overall_mean']:.3f}"
                    f"(cond {per_eps[f'{eps:.3f}']['robustness_conditional_mean'] or 0.0:.3f})"
                    for eps in eps_list
                ))
        print(line, flush=True)

    full = pd.concat(dfs, ignore_index=True)
    full.to_parquet(out_dir / "per_state_robustness.parquet",
                    engine="pyarrow", index=False)
    config = {
        "n_states": int(args.n_states),
        "seed_base": int(args.seed_base),
        "n_perturb": int(args.n_perturb),
        "eps_list": eps_list,
        "constrain_aim": bool(constrain_aim),
        "extra_features": bool(extra_features),
        "policies": [s.split("=", 1) for s in args.policy],
        "total_wall_s": float(time.perf_counter() - t_start),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config": config, "results": summary_all}, f, indent=2)

    # Pretty-print a paired comparison if labels follow baseline_s* / robust_s*
    pairs: dict[str, dict[str, str]] = {}
    for label in summary_all:
        if label.startswith("baseline_s"):
            pairs.setdefault(label.split("_", 1)[1], {})["baseline"] = label
        elif label.startswith("robust_s"):
            pairs.setdefault(label.split("_", 1)[1], {})["robust"] = label
    if pairs:
        print("\n=== paired comparison (robustness_conditional) ===", flush=True)
        header = "seed | det% (B→R) | " + " | ".join(
            f"ε={eps:.2f}" for eps in eps_list
        )
        print(header)
        for seed_tag, kinds in sorted(pairs.items()):
            if "baseline" not in kinds or "robust" not in kinds:
                continue
            b = summary_all[kinds["baseline"]]
            r = summary_all[kinds["robust"]]
            cells = []
            for eps in eps_list:
                bb = b["per_eps"][f"{eps:.3f}"]["robustness_conditional_mean"] or 0.0
                rr = r["per_eps"][f"{eps:.3f}"]["robustness_conditional_mean"] or 0.0
                cells.append(f"{bb:.3f}→{rr:.3f} ({rr-bb:+.3f})")
            print(f"{seed_tag} | {b['det_success_rate']:.2f}→{r['det_success_rate']:.2f} | "
                  + " | ".join(cells))

    print(f"\n[robust_eval] -> {out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
