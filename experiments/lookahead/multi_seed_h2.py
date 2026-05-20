"""Multi-seed ensemble proposal + h=2 lookahead.

For each shot, candidates drawn from 3 different seed policies
(s1, s4, s6). Each contributes K_per_policy candidates. The
combined pool is more diverse than a single policy can produce.
"""
import copy, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/Users/ohdoyoel/work/side/cs377/project/.claude/worktrees/origin-main-eval")
sys.path.insert(0, str(REPO))

from billiards.inning_env import Billiards4BallInningEnv
from billiards.wrappers.random_start_env import RandomStartInningEnv
from billiards.render.replay import render_inning_html
from stable_baselines3 import SAC

POLICIES = [
    REPO / "experiments/runs_inning_v2/fast_long_fp02_s1/policy.zip",
    REPO / "experiments/runs_inning_v2/fast_long_fp02_s4/policy.zip",
    REPO / "experiments/runs_inning_v2/fast_long_fp02_s6/policy.zip",
]
OUT_HTML = REPO / "artifacts/best_inning"
GAMMA = 0.99


def K_schedule(shot_idx):
    """Per-policy K. Total = K * len(POLICIES)."""
    if shot_idx == 0: return 200  # 600 total candidates for first shot
    if shot_idx <= 3: return 100  # 300 total
    if shot_idx <= 9: return 50   # 150 total
    return 30                       # 90 total


def cands_multi(models, obs, K_per):
    out = []
    for m in models:
        det, _ = m.predict(obs, deterministic=True)
        out.append(np.asarray(det, np.float32).reshape(-1))
        for _ in range(K_per - 1):
            sto, _ = m.predict(obs, deterministic=False)
            out.append(np.asarray(sto, np.float32).reshape(-1))
    return out


def cands_single(model, obs, K):
    out = []
    det, _ = model.predict(obs, deterministic=True)
    out.append(np.asarray(det, np.float32).reshape(-1))
    for _ in range(K - 1):
        sto, _ = model.predict(obs, deterministic=False)
        out.append(np.asarray(sto, np.float32).reshape(-1))
    return out


def _snap(b):
    return (copy.deepcopy(b._state), b._shot_index, b._cumulative_score, b._cumulative_t,
            list(b._shot_trajectories), list(b._shot_offsets), list(b._inning_log_records))

def _rest(b, s):
    b._state = copy.deepcopy(s[0]); b._shot_index = s[1]
    b._cumulative_score = s[2]; b._cumulative_t = s[3]
    b._shot_trajectories = list(s[4]); b._shot_offsets = list(s[5]); b._inning_log_records = list(s[6])

def _try(b, a):
    try: return b.step(a)
    except Exception: return None, -1e9, True, True, {}


def multi_seed_eval(models, n_eps, K2=5, max_shots=2000):
    rows = []
    best = None
    t_start = time.time()
    for ep in range(n_eps):
        base = Billiards4BallInningEnv(
            t_max=12.0, max_shots=max_shots, continue_on_miss=False,
            constrain_aim=True, extra_features=True,
            foul_penalty=0.2, gentle_shot=True,
            setup_shaping=True, setup_alpha=0.05, setup_scale=0.3,
        )
        env = RandomStartInningEnv(base)
        obs, _ = env.reset(seed=99000 + ep)
        ep_t = time.time()
        # Use main model (s4) for second-step candidates (single policy)
        main_model = models[1]
        while True:
            K_per = K_schedule(base._shot_index)
            c1 = cands_multi(models, obs, K_per)
            s0 = _snap(base)
            bv, ba = -1e9, c1[0]
            for a1 in c1:
                _rest(base, s0)
                obs1, r1, t1, tr1, _ = _try(base, a1)
                if t1 or tr1 or obs1 is None:
                    v = r1
                else:
                    s1 = _snap(base)
                    c2 = cands_single(main_model, obs1, K2)
                    br2 = -1e9
                    for a2 in c2:
                        _rest(base, s1)
                        _, r2, _, _, _ = _try(base, a2)
                        if r2 > br2: br2 = r2
                    v = r1 + GAMMA * br2
                if v > bv: bv, ba = v, a1
            _rest(base, s0)
            obs, _, term, trunc, info = env.step(ba)
            if term or trunc: break
        score = int(base.cumulative_score)
        rows.append(score)
        if best is None or score > best[0]:
            best = (score, list(base.shot_trajectories), base._spec, ep, base.shot_index)
        print(f"  ep {ep}: score={score} shots={base.shot_index} ep_wall={time.time()-ep_t:.0f}s "
              f"total={time.time()-t_start:.0f}s mean_so_far={np.mean(rows):.1f}", flush=True)
    s = np.array(rows)
    return dict(mean=float(s.mean()), std=float(s.std()), max=int(s.max()),
                p100=float((s>=100).mean()), p200=float((s>=200).mean()),
                p500=float((s>=500).mean()), p1000=float((s>=1000).mean())), best


if __name__ == "__main__":
    print(f"Loading {len(POLICIES)} models...")
    models = [SAC.load(str(p), device="cpu") for p in POLICIES]

    print(f"\n=== Multi-seed (s1, s4, s6) ensemble K_per_pol={{200,100,50,30}}, K2=5 h=2, n=10 ===")
    stats, best = multi_seed_eval(models, n_eps=10, K2=5, max_shots=2000)
    print(f"\nFINAL: mean={stats['mean']:.1f}±{stats['std']:.1f} max={stats['max']} "
          f"P100={stats['p100']*100:.0f}% P200={stats['p200']*100:.0f}% "
          f"P500={stats['p500']*100:.0f}% P1000={stats['p1000']*100:.0f}%")

    sc, traj, spec, off, nshots = best
    name = f"INFINITY_multiseed_score{sc}_shots{nshots}_seed{99000+off}.html"
    out = OUT_HTML / name
    print(f"Rendering best: {name}")
    render_inning_html(traj, spec=spec, save_path=out)
