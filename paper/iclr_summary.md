# Linear Reward Mixing Fails to Recover Sparse Ground Truth: A Carom-Billiards Case Study

**Claim.** When the proxy reward is dense and the ground-truth reward is
sparse-binary, naive linear mixing `r = α · env_score + (1 − α) · RM_norm(s, a)`
does *not* interpolate between alignment and shaping; the dense proxy
dominates the gradient at every α and the policy collapses to a proxy-saturating
attractor that *never* scores.

## Abstract

We use Korean 4-ball carom billiards as a low-cost RLHF testbed to ask whether
linear mixing of an environment reward and a learned proxy interpolates between
alignment and shaping. We train a Bradley-Terry reward model from a heuristic
labeler over seven interpretable axes and run PPO with the mixed reward
`α · env_score + (1 − α) · RM_norm` for α ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0},
3 seeds each, 50k timesteps. The widely-held intuition is that small α should
shape exploration while large α should anchor the optimization to ground
truth. We find this to be false here: true score rate is 0% at every α
(n=18), RM_norm sits in the saturated band [2.74, 2.78] regardless of α, and
the hacking gap (RM − env) grows during training under all α. Pure-env PPO
(α=1) is *brittler*, not safer: 1/3 seeds collapses to 100% foul, while α<1
holds foul ≈ 0%. We argue that linear mixing is structurally insufficient when
the ground-truth signal is sparse-binary, and identify KL anchoring or
behaviour-cloning priors as the natural remedies.

## 1. Setting

Korean 4-ball is a strict carom variant: a player scores 1 iff the cue ball
contacts both red balls without contacting the opponent's cue ball. The state
is fully observable (positions, velocities, spin per ball; 28 dims), the
action is 4-dim (`θ`, `power`, contact `a`, contact `b`), and one shot equals
one episode (`Billiards4BallEnv` in `billiards/env.py`). This makes it a near-
ideal RLHF testbed: rewards are cheap, episodes are short, and the true
objective is a *binary* score with the geometric subtlety that random play
almost never satisfies it. As a result, ground truth is sparse — early in
training the policy receives no positive env signal — while a heuristic-trained
RM provides a dense surrogate. The setting therefore stress-tests linear
reward mixing exactly where it is most attractive in practice: when the
proxy "fills in" the env's silence.

## 2. Method

The Phase C-D pipeline (cf. `billiards/preference/labeler_heuristic.py`,
`billiards/reward_model/`) generates rollouts, scores each with a 7-component
heuristic
(`+5·score, −5·foul, +0.4·min(cushions,5), +0.6·red_contacts,
−0.3·opp_contacts, +position_bonus, −0.05·duration`),
forms preference pairs by signed score difference with a 0.5 dead-band, and
trains a small MLP under the Bradley-Terry likelihood. Phase F (this paper)
extends `RewardModelEnv` (`billiards/wrappers/reward_model_env.py`) so the
per-step reward is the convex combination
`r = α · env_score + (1 − α) · RM_norm(s, a)`, where `RM_norm` is z-scored on
held-out rollouts. We sweep α ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0} × seeds
{0, 1, 2} for 50k PPO timesteps each (8 vec-envs, `lr=3e-4`, `n_steps=512`,
`n_epochs=4`, `clip=0.2`, `ent_coef=0.01`; matches Phase E). Eval rolls 200
deterministic episodes per (α, seed) on the *unwrapped* env and logs
true-score, foul, cushion, mean per-component attribution, and per-step
hacking gap.

## 3. Experiments

### 3.1 Setup

PPO hyperparameters identical to Phase E (`scripts/train_ppo.py`). 18 runs
total (6 α × 3 seeds), 50k timesteps each. Eval is deterministic on
`Billiards4BallEnv` for 200 episodes per run. All artifacts under
`experiments/runs/<α, seed>/{eval.parquet, attribution.json,
training_curve.csv, policy.zip}` and aggregated to
`experiments/results/summary.parquet` (18 × 24).

### 3.2 Mix sweep

Figures 1–4 visualize the four cuts of the data the next section reads off.

| α | true_score (%) | RM (z-norm) | hacking_gap | foul (%) | cushions |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.00 ± 0.00 | 2.767 ± 0.017 | 2.767 | 0.0 ± 0.0 | 5.00 |
| 0.1 | 0.00 ± 0.00 | 2.759 ± 0.014 | 2.759 | 0.0 ± 0.0 | 5.00 |
| 0.3 | 0.00 ± 0.00 | 2.744 ± 0.016 | 2.744 | 0.0 ± 0.0 | 5.00 |
| 0.5 | 0.00 ± 0.00 | 2.746 ± 0.015 | 2.746 | 0.0 ± 0.0 | 5.00 |
| 0.7 | 0.00 ± 0.00 | 2.742 ± 0.016 | 2.742 | 0.0 ± 0.0 | 5.00 |
| 1.0 | 0.00 ± 0.00 | 2.760 ± 0.006 | 2.760 | 33.3 ± 57.7 | 5.33 |

Bootstrap 95% CI on true score per α: [0%, 0%] uniformly. Welch t-test on
true score (α=1 vs α=0): t = 0.000, p = 1.000. Welch on foul rate
(α=1 vs α=0): t = 1.000, p = 0.317.

## 4. Findings

- **The α-mix fails to recover ground truth.** True score rate is 0.00% at
  every α we tested (n = 18 runs), with bootstrap 95% CI [0, 0] per α. There
  is no α at which mixing buys a non-zero ground-truth score; the binary
  objective is never satisfied. The very notion of an "RM-vs-true" Pareto
  frontier degenerates to a point at (RM ≈ 2.75, true = 0%) — see Figure 1.

- **Reward hacking is α-invariant.** RM_norm at convergence sits in the
  saturated band [2.74, 2.78] regardless of α (≈ 1% spread). The hacking gap
  (RM_norm − env_score) is essentially equal to RM_norm because env_score
  stays at 0; the gap *grows* during training under every α (Figure 2,
  left panel) and the per-α curves overlap to within their own seed-noise.
  Practitioners hoping that even small α would close the gap will not see it
  here.

- **Component attribution shows a single attractor.** At every α, the modal
  policy fires a power-1.0 shot that bounces 5 cushions, scores 0, and never
  contacts a red ball (`cushion_term = 2.0`, `red_contact_term = 0`,
  `score_term = 0`). The RM has been trained on the heuristic which awards
  `+0.4 × min(cushions, 5) = 2.0` plus `position_bonus ≈ 0.23–0.25`, and PPO
  finds the minimum-effort way to saturate exactly that subspace (Figure 3,
  left). At α=1 the cushion mass is unchanged but `position_term` collapses
  ≈12× (0.235 → 0.019) and a foul mass appears (`foul_term = −1.67`,
  `opp_contact_term = −0.10`) — the env-only signal does not drive the policy
  to a *better* attractor; it merely lets a brittle one emerge.

- **Pure-env (α=1) is brittler, not better.** 1/3 seeds at α=1 collapses to
  100% foul rate; the other two stay at 0%. Across α<1 the foul rate is
  uniformly 0/3 seeds (Figure 3, right). The bootstrap 95% CI on α=1's foul
  rate is [0%, 100%]: under sparse env reward, "no-foul, no-score" and
  "100%-foul, no-score" are *both* local optima of the gradient signal, and
  the seed dictates which the policy lands in. Linear mixing's surprising
  upside is therefore *exploration stability*, not alignment improvement —
  the dense RM proxy regularizes the policy away from the foul attractor
  without ever pushing it toward score.

- **Action geometry confirms the same attractor.** For α ∈ {0.0, 0.5},
  deterministic eval converges to a near-identical action
  (θ ≈ 0, power = 1, a ≈ 0.85, b = 1) across seeds, and the 2D θ × power and
  a × b histograms (Figure 4) overlap. Only at α=1 does one seed move
  ≈1.0 in action L2 to a foul attractor — and this *one* seed is the entire
  source of the seed-level variance.

The headline implication is mechanistic, not statistical: a dense proxy that
itself encodes "+0.4 per cushion" defines a near-zero-effort policy
(`max-power, fixed direction → 5 cushions`) whose value approaches the
heuristic's hard cap. Mixing in a sparse env signal does not rescue the
gradient because the sparse term contributes ≈0 to every rollout the policy
encounters. Linear mixing is *structurally* insufficient in this regime.

## 5. Limitations

3 seeds per α and 50k PPO steps are small; a longer-horizon sweep might let
the env-only branch eventually find a scoring shot. We test a single
deterministic environment (fixed initial state, fixed table specs); external
validity to multi-shot or stochastic-init billiards is not established.
Preferences come from a heuristic labeler only — no human or LLM-as-judge
labels — so the RM inherits the heuristic's blind spots, including its
explicit cushion bonus, which is exactly the axis the policy hacks. We do not
vary RM capacity, BT temperature, or the entropy coefficient. Finally, our
"linear mix" is the simplest mixing scheme: KL anchoring to a reference
policy or behaviour-cloning priors are out of scope here and are exactly the
remedies the negative result motivates.

## 6. Reproducibility

- Code: this repo at `billiards/`, `scripts/`, `experiments/`,
  `notebooks/07_iclr_analysis.ipynb`.
- Reward model artifact: `models/reward_model.pt` (Phase D output).
- α grid: {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}; seeds: {0, 1, 2}; steps: 50k.
- Per-run artifacts: `experiments/runs/a{α}_s{seed}/`
  `{eval.parquet, attribution.json, training_curve.csv, policy.zip,
  config.json, summary.json, run.log}`.
- Aggregate: `experiments/results/summary.parquet` (18 × 24).
- PPO hyperparameters: `lr=3e-4, n_steps=512, batch=512, n_epochs=4,
  γ=0.99, λ=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, n_envs=8`.
- Eval: 200 deterministic episodes per (α, seed) on `Billiards4BallEnv`,
  base seed = `seed × 10000 + 10000`.
- Figures: `paper/figures/{fig1_pareto, fig2_hacking_gap, fig3_attribution,
  fig4_action_dist}.{pdf, png}` at 300 dpi, monochrome-friendly via
  line-style differentiation.
- Sample trajectories: `artifacts/iclr/{hacked, best}.html`. `hacked.html`
  is the canonical α=0 cushion-attractor shot
  (seed 10251, 5 cushions, score 0, RM_norm 2.781). `best.html` is the
  most-distinct-from-α=0 shot (α=1, seed 10000, 6 cushions, score 0,
  no foul, RM_norm 2.766) — included to make the contrast between the two
  attractors visually concrete.
- Tests: `tests/test_analysis.py` verifies summary-parquet schema, α coverage,
  per-seed counts, and per-run artifact shapes; 6/6 pass.
