"""PUCT (Predictor + UCT) Monte Carlo Tree Search for continuous-action billiards.

Continuous action MCTS via sampled candidates:
  - At each new leaf, sample K candidates from the policy(s) → branching = K
  - PUCT selection: argmax_child [Q(c) + c_puct · π(c) · sqrt(N_parent) / (1+N_c)]
  - Backup: discounted return propagated up

Modes (set via candidate_fn):
  - 'uniform'   : K candidates from uniform action distribution (= UCT baseline)
  - 'policy'    : K candidates from single policy
  - 'multi_seed': K_per_policy from each of N policies → K_total = N · K_per_policy

Budget control:
  - n_sim_target: total number of `base.step()` calls per shot
  - Each leaf expansion costs K (children) simulations
  - Tree iteration: select_leaf + expand_and_eval, costing K sims per iter
  - n_iter = n_sim_target / K
"""

from __future__ import annotations
import copy
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from billiards.inning_env import Billiards4BallInningEnv
from billiards.wrappers.random_start_env import RandomStartInningEnv
from stable_baselines3 import SAC


# ---- Node ----------------------------------------------------------------

@dataclass
class Node:
    # env state snapshot to restore via _rest
    env_snap: tuple | None
    # observation at this node (32-dim float32, or None if terminal/synthetic)
    obs: np.ndarray | None = None
    parent: "Node | None" = None
    # action taken from parent to reach here, and immediate reward received
    action: np.ndarray | None = None
    reward: float = 0.0
    prior: float = 1.0
    visit: int = 0
    value_sum: float = 0.0
    children: list["Node"] = field(default_factory=list)
    is_terminal: bool = False
    expanded: bool = False

    def Q(self) -> float:
        return self.value_sum / max(1, self.visit)

    def is_leaf(self) -> bool:
        return not self.expanded


# ---- Env state snapshot/restore -------------------------------------------

def _snap(base):
    return (copy.deepcopy(base._state), base._shot_index, base._cumulative_score,
            base._cumulative_t, list(base._shot_trajectories),
            list(base._shot_offsets), list(base._inning_log_records))


def _rest(base, s):
    base._state = copy.deepcopy(s[0])
    base._shot_index = s[1]
    base._cumulative_score = s[2]
    base._cumulative_t = s[3]
    base._shot_trajectories = list(s[4])
    base._shot_offsets = list(s[5])
    base._inning_log_records = list(s[6])


def _try_step(base, action):
    try:
        return base.step(action)
    except Exception:
        return None, -1e9, True, True, {}


# ---- Candidate samplers (the mode) ----------------------------------------

def make_candidate_fn(mode: str, models: Sequence | None, K_per_policy: int,
                      action_low: np.ndarray, action_high: np.ndarray):
    """Returns a function: obs -> list of (action, prior). priors sum to 1."""

    if mode == "uniform":
        K = K_per_policy  # total candidates
        def fn(obs):
            acts = []
            for _ in range(K):
                a = action_low + np.random.rand(*action_low.shape).astype(np.float32) * (action_high - action_low)
                acts.append((a, 1.0 / K))
            return acts
        return fn

    if mode == "policy":
        assert models is not None and len(models) == 1
        model = models[0]
        K = K_per_policy
        def fn(obs):
            acts = []
            # 1 deterministic + K-1 stochastic, all uniform prior
            det, _ = model.predict(obs, deterministic=True)
            acts.append((np.asarray(det, np.float32).reshape(-1), 1.0 / K))
            for _ in range(K - 1):
                sto, _ = model.predict(obs, deterministic=False)
                acts.append((np.asarray(sto, np.float32).reshape(-1), 1.0 / K))
            return acts
        return fn

    if mode == "multi_seed":
        assert models is not None and len(models) >= 1
        K_total = K_per_policy * len(models)
        def fn(obs):
            acts = []
            for m in models:
                det, _ = m.predict(obs, deterministic=True)
                acts.append((np.asarray(det, np.float32).reshape(-1), 1.0 / K_total))
                for _ in range(K_per_policy - 1):
                    sto, _ = m.predict(obs, deterministic=False)
                    acts.append((np.asarray(sto, np.float32).reshape(-1), 1.0 / K_total))
            return acts
        return fn

    raise ValueError(f"unknown mode {mode}")


# ---- PUCT search ----------------------------------------------------------

def puct_score(child: Node, c_puct: float) -> float:
    parent_visit = child.parent.visit if child.parent else 1
    return child.Q() + c_puct * child.prior * math.sqrt(max(1, parent_visit)) / (1 + child.visit)


def select_leaf(root: Node, c_puct: float) -> Node:
    node = root
    while node.expanded and not node.is_terminal and node.children:
        node = max(node.children, key=lambda c: puct_score(c, c_puct))
    return node


def expand_and_evaluate(node: Node, base, candidate_fn, gamma: float) -> float:
    """Expand node by sampling K candidates, simulating each, attach as children.
    Returns rollout value = max immediate reward (leaf evaluation heuristic)."""
    if node.is_terminal:
        return 0.0
    # Restore env to node's state to get obs
    _rest(base, node.env_snap)
    obs = base._obs()
    # Sample candidates + priors
    cands = candidate_fn(obs)
    # Simulate each
    best_immediate = -1e9
    for action, prior in cands:
        _rest(base, node.env_snap)
        obs1, r1, term, trunc, _ = _try_step(base, action)
        if obs1 is None:
            # crashed; skip
            child = Node(env_snap=node.env_snap, obs=None, parent=node,
                         action=action, reward=-1e9, prior=prior, is_terminal=True)
        else:
            new_snap = _snap(base)
            child = Node(env_snap=new_snap, obs=obs1.copy(), parent=node,
                         action=action, reward=float(r1), prior=prior,
                         is_terminal=bool(term or trunc))
        node.children.append(child)
        if child.reward > best_immediate:
            best_immediate = child.reward
    node.expanded = True
    return best_immediate


def backup(leaf: Node, value: float, gamma: float):
    """Backup: propagate discounted value up. leaf.value already includes leaf reward."""
    node = leaf
    accum = value
    while node is not None:
        node.visit += 1
        node.value_sum += accum
        if node.parent is not None:
            accum = node.reward + gamma * accum
        node = node.parent


def mcts_select_action(base, candidate_fn, n_sim_budget: int,
                        K_per_expand: int, c_puct: float, gamma: float) -> np.ndarray:
    """Run MCTS from current env state, return best action (by visit count)."""
    root_snap = _snap(base)
    root = Node(env_snap=root_snap, obs=base._obs())

    # Bootstrap: expand root once
    val = expand_and_evaluate(root, base, candidate_fn, gamma)
    backup(root, val, gamma)
    sims_used = K_per_expand

    while sims_used < n_sim_budget:
        leaf = select_leaf(root, c_puct)
        if leaf.is_terminal:
            backup(leaf, 0.0, gamma)
            sims_used += 1
            continue
        val = expand_and_evaluate(leaf, base, candidate_fn, gamma)
        backup(leaf, val, gamma)
        sims_used += K_per_expand

    # Restore root state so caller can step authentically
    _rest(base, root_snap)

    # Pick action at root with most visits (AlphaZero standard)
    best_child = max(root.children, key=lambda c: c.visit)
    return best_child.action


# ---- Evaluation harness ---------------------------------------------------

def mcts_episode(base, env, candidate_fn, n_sim_budget, K_per_expand, c_puct, gamma):
    obs, _ = env.reset(seed=base._eval_seed)  # caller sets _eval_seed
    fouled = False
    while True:
        a = mcts_select_action(base, candidate_fn, n_sim_budget, K_per_expand, c_puct, gamma)
        obs, _, term, trunc, info = env.step(a)
        if info.get("fouled"):
            fouled = True
        if term or trunc:
            break
    return int(base.cumulative_score), fouled


def make_env(max_shots: int):
    base = Billiards4BallInningEnv(
        t_max=12.0, max_shots=max_shots, continue_on_miss=False,
        constrain_aim=True, extra_features=True,
        foul_penalty=0.2, gentle_shot=True,
        setup_shaping=True, setup_alpha=0.05, setup_scale=0.3,
    )
    env = RandomStartInningEnv(base)
    return base, env


def evaluate(mode: str, models: Sequence | None, K_per_policy: int,
             n_sim_budget: int, n_eps: int, max_shots: int,
             c_puct: float = 1.5, gamma: float = 0.99, seed_base: int = 99000):
    # Probe env for action bounds
    probe_base, _ = make_env(max_shots=10)
    candidate_fn = make_candidate_fn(mode, models, K_per_policy,
                                      probe_base.action_space.low.astype(np.float32),
                                      probe_base.action_space.high.astype(np.float32))
    # K_per_expand is the candidate count per expansion event
    K_per_expand = K_per_policy * (len(models) if (models and mode == "multi_seed") else 1)
    print(f"  [config] mode={mode}, K_per_expand={K_per_expand}, n_sim={n_sim_budget}, n_eps={n_eps}, max_shots={max_shots}")

    rows = []
    t0 = time.time()
    for ep in range(n_eps):
        base, env = make_env(max_shots=max_shots)
        base._eval_seed = seed_base + ep
        ep_t = time.time()
        score, fouled = mcts_episode(base, env, candidate_fn,
                                      n_sim_budget=n_sim_budget,
                                      K_per_expand=K_per_expand,
                                      c_puct=c_puct, gamma=gamma)
        rows.append((score, fouled))
        print(f"    ep {ep}: score={score} shots={base.shot_index} ep_wall={time.time()-ep_t:.0f}s mean={np.mean([r[0] for r in rows]):.1f}", flush=True)

    s = np.array([r[0] for r in rows])
    return dict(mode=mode, n_sim=n_sim_budget, K_per_pol=K_per_policy,
                mean=float(s.mean()), std=float(s.std()), max=int(s.max()),
                p100=float((s>=100).mean()), p200=float((s>=200).mean()),
                p500=float((s>=500).mean()), p1000=float((s>=1000).mean()),
                wall=time.time()-t0)


if __name__ == "__main__":
    POLICIES = [
        REPO / "experiments/runs_inning_v2/fast_long_fp02_s1/policy.zip",
        REPO / "experiments/runs_inning_v2/fast_long_fp02_s4/policy.zip",
        REPO / "experiments/runs_inning_v2/fast_long_fp02_s6/policy.zip",
    ]
    print(f"Loading {len(POLICIES)} policies...")
    models = [SAC.load(str(p), device="cpu") for p in POLICIES]
    print("Loaded.")

    # Fair comparison: same sim budget ≈ 600 per shot (matches greedy K1=100 K2=5)
    # n_eps small for first run (validate)
    SIM_BUDGET = 600
    N_EPS = 5
    MAX_SHOTS = 500

    results = []
    for mode, mods, K_per in [
        ("uniform", None, 60),                        # K=60 → 10 expansions to reach 600 sims
        ("policy", [models[1]], 60),                  # single policy s4
        ("multi_seed", models, 20),                   # 3 × 20 = 60 per expansion
    ]:
        print(f"\n=== {mode.upper()} (K_per={K_per}, sim_budget={SIM_BUDGET}, n_eps={N_EPS}) ===")
        r = evaluate(mode=mode, models=mods, K_per_policy=K_per,
                     n_sim_budget=SIM_BUDGET, n_eps=N_EPS, max_shots=MAX_SHOTS)
        results.append(r)
        print(f"  FINAL: mean={r['mean']:.1f}±{r['std']:.1f} max={r['max']} "
              f"P100={r['p100']*100:.0f}% P200={r['p200']*100:.0f}% "
              f"P500={r['p500']*100:.0f}% P1000={r['p1000']*100:.0f}% wall={r['wall']:.0f}s")

    print("\n\n=== SUMMARY (n_eps={}) ===".format(N_EPS))
    print(f"{'mode':<15} {'mean':>8} {'std':>8} {'max':>6} {'P100':>6} {'P500':>6} {'wall':>8}")
    for r in results:
        print(f"{r['mode']:<15} {r['mean']:>8.1f} {r['std']:>8.1f} {r['max']:>6} "
              f"{r['p100']*100:>5.0f}% {r['p500']*100:>5.0f}% {r['wall']:>7.0f}s")
