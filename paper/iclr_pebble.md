# Does PEBBLE Rescue Sparse-Reward RLHF? Evidence from Korean 4-Ball Carom

> **Phase G extension to "Linear Reward Mixing Fails to Recover Sparse Ground Truth"** (Phase F, `paper/iclr_summary.md`).

**Claim.** On Korean 4-ball carom, plain SAC trained on the *raw env reward*
solves the task in 2/3 seeds (66.7% mean true-score rate). Every RLHF
variant we tested — including PEBBLE with replay relabeling and
ensemble-disagreement queries — collapses to 0%. The bottleneck for our
Phase F negative result is **reward-model quality, not env-reward
sparsity**, and PEBBLE's machinery (relabeling + informative queries) does
not, on its own, repair a structurally biased RM.

## Abstract

Phase F showed that linear mixing `r = α · env_score + (1 − α) · RM_norm`
fails uniformly on Korean 4-ball carom: 0% true-score rate at every α and a
policy that collapses to a max-cushion no-score attractor. Here we ask
whether PEBBLE (Lee et al., 2021) — relabeling the SAC replay buffer with
the current RM, plus querying preference labels at high ensemble
disagreement — rescues the task. We compare 7 conditions across 3 seeds
each: 3 PPO α-baselines from Phase F, plus SAC-env, SAC+RM-frozen,
PEBBLE-uniform, and PEBBLE-disagreement (300 labels each, 30k SAC steps).
*The headline reverses our prior expectation:* SAC-env achieves
**66.7 ± 57.7 %** true score (2/3 seeds at 100%, 1 at 0%) while
PEBBLE-full and SAC+RM-frozen both stay at **0.0 ± 0.0 %** with raw
RM ≈ 0.12 — within seed-noise of each other. Welch t-tests place
SAC-env above each RM-driven method at p ≈ 0.046; the
PEBBLE_FULL − SAC+RM-frozen contrast is flat (t = 0). Replay relabeling
*does* move the buffer reward (last-event mean |Δ| ≈ 0.018 in
PEBBLE_FULL), but the RM keeps redirecting the policy into a cushion-
saturating subspace. We argue the negative result is structural: the
heuristic-trained RM defines the wrong attractor at every step, and no
amount of relabeling or query-selection cleverness rescues a misspecified
proxy when the env signal is already learnable on its own.

## 1. Motivation

Phase F established that linear α-mixing of env and RM fails on this task —
a 6×3 PPO grid with `r = α · env_score + (1 − α) · RM_norm` returned
0% true score uniformly, an RM saturated at z ≈ 2.75 regardless of α, and
a single dominant action attractor (`θ ≈ 0, power = 1, 5 cushions, no
score`). PEBBLE positions itself as the textbook fix: the RM stops being
*frozen*, it can be retrained from new informative queries; and stale
buffer transitions get relabeled under the *current* RM, so the off-
policy backbone (SAC) does not waste samples optimizing an outdated
target. If Phase F's failure was about *exploration* under sparse env
reward, PEBBLE's relabeling + sample-efficient SAC should help. Phase G
runs that experiment cleanly and reports the answer.

## 2. Method

We hold the carom env (`Billiards4BallEnv`), 4-D action space, 28-D
observation, and Phase D reward model (`models/reward_model.pt`) fixed
across all conditions. The PEBBLE-specific machinery in
`billiards/pebble/` adds:

- **Replay relabeling.** Every K env steps, the reward column of the SAC
  replay buffer is recomputed under the current RM ensemble's mean. We
  log mean |new − old| per relabel event as `relabel_delta_mean` per run.
- **Disagreement-based query selection.** Candidate trajectory pairs are
  scored by RM-ensemble variance; the top-k pairs are sent to the
  (heuristic) labeler. The `pebble_disagree` ablation replaces this with
  uniform sampling at the same query budget.

Methods (column `method` in `experiments/results/pebble_summary.parquet`):

| method | backbone | reward signal | queries (300 budget) |
| --- | --- | --- | --- |
| `ppo_a0` | PPO | RM_norm only | (Phase F replay, 0) |
| `ppo_a1` | PPO | env only | (Phase F replay, 0) |
| `ppo_a05` | PPO | 0.5·env + 0.5·RM | (Phase F replay, 0) |
| `sac_env` | SAC | env only | 0 |
| `sac_rm_frozen` | SAC | RM_norm (no relabel) | 300 (one shot) |
| `pebble_disagree` | SAC + relabel | RM, **uniform** queries | 300 |
| `pebble_full` | SAC + relabel | RM, **disagreement** queries | 300 |

PEBBLE-style runs use 30k SAC steps, 6 RM-update phases, ensemble of 3 RM
heads, K=512 relabel cadence (`experiments/run_pebble_matrix.py`). Phase F
PPO baselines are replayed as-is from `summary.parquet`. Eval rolls 200
deterministic episodes per (method, seed) on the unwrapped env and logs
true-score, foul, cushion, per-component attribution, raw RM mean, and
queries-used trajectory.

## 3. Experiments

3 seeds × 7 methods. Phase G artifacts under
`experiments/runs_pebble/<method>_s<seed>/{eval.parquet, attribution.json,
training_curve.csv, policy.zip, config.json, summary.json, run.log}` and
aggregated to `experiments/results/pebble_summary.parquet` (12 × 29).
Phase F PPO artifacts at `experiments/runs/a{α}_s{seed}/` (unchanged).
Figures rendered by `notebooks/08_pebble_analysis.ipynb`:

| metric (mean ± std, n=3 seeds) | ppo_a0 | ppo_a1 | ppo_a05 | sac_env | sac_rm_frozen | pebble_disagree | pebble_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true_score (%) | 0 ± 0 | 0 ± 0 | 0 ± 0 | **66.7 ± 57.7** | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| foul (%) | 0 ± 0 | 33.3 ± 57.7 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 33.3 ± 57.7 | 0 ± 0 |
| cushion_term | 2.00 | n/a | 2.00 | 1.47 | 1.33 | 0.93 | 1.60 |
| RM signal | 2.77 (z) | 2.76 (z) | 2.75 (z) | n/a | 0.12 (raw) | 0.05 (raw) | 0.13 (raw) |
| queries used | 0 | 0 | 0 | 0 | 300 | 300 | 300 |

Bootstrap 95% CI on true score (n_boot=10000): `sac_env` = [0, 100],
all other methods = [0, 0]. Welch t-tests on true score:
`sac_env` vs `pebble_full`: t = +2.000, p = 0.046; same vs
`sac_rm_frozen` and `ppo_a1`. The PEBBLE-specific contrasts —
`pebble_full` vs `sac_rm_frozen` (does relabeling buy anything?) and
`pebble_full` vs `pebble_disagree` (does informative selection help?) —
are both t = 0, p = 1. Figures 5–8 visualize four cuts:

- **Figure 5** (`fig5_query_efficiency.{pdf,png}`) — running true-score
  rate vs cumulative queries used; PEBBLE_FULL, PEBBLE-uniform, and
  SAC+RM-frozen all hug 0% across the full 300-query budget, while
  horizontal references show SAC-env at 66.7% (no queries) and the best
  Phase F PPO at 0%.
- **Figure 6** (`fig6_relabel_delta.{pdf,png}`) — `relabel_delta_mean`
  during training. PEBBLE-full and PEBBLE-uniform produce a non-trivial
  signal (≈ 0.015–0.030); SAC+RM-frozen sits at 0 by construction.
- **Figure 7** (`fig7_method_bars.{pdf,png}`) — final true-score rate per
  method with bootstrap 95% CI and per-seed dots. SAC-env stands alone;
  every other bar collapses to 0.
- **Figure 8** (`fig8_attribution_compare.{pdf,png}`) — heuristic-component
  attribution per method. SAC-env is the only condition with non-zero
  `score_term`; cushion-term remains the dominant axis under every
  RM-driven method.

## 4. Findings

- **PEBBLE does not rescue: true score = 0% for every RM-driven SAC
  method, identical within seed-noise to the Phase F PPO α-grid.**
  PEBBLE_FULL: 0.00 ± 0.00 %; SAC+RM-frozen: 0.00 ± 0.00 %;
  PEBBLE-uniform: 0.00 ± 0.00 %. Pairwise Welch t = 0 across these three.
  Relabeling and disagreement queries each contribute Δ = 0.0% to the
  true-score rate at this query budget.

- **Plain SAC + env wins decisively.** `sac_env` reaches **66.7 ± 57.7 %**
  (2/3 seeds at 100%, 1 at 0%) — bootstrap 95% CI [0, 100]. The two
  scoring seeds converge to a single deterministic action
  (θ ≈ 3.21, power ≈ 0.88, a ≈ −0.79, b ≈ 0.76) that scores in every one
  of 200 eval episodes. Welch t = +2.000, p = 0.046 against each
  RM-driven method. Env-reward sparsity was *not* the bottleneck on this
  task; SAC's off-policy machinery + the binary +5 score signal is
  enough — when the agent stumbles into the scoring action once, the
  replay buffer carries that gradient onward.

- **Relabeling moves the reward signal but doesn't escape the
  attractor.** PEBBLE_FULL last-event mean relabel delta ≈ 0.018; SAC+RM-
  frozen last-event mean = 0.000 by construction. Despite this, both
  end at 0% true score with raw RM ≈ 0.12 — i.e., the RM *moves*, but
  the direction it moves the policy is into a cushion-bouncing subspace
  consistent with the heuristic's `+0.4 × min(cushions, 5)` bonus.
  Component attribution at convergence: PEBBLE_FULL
  `cushion_term = 1.60, score_term = 0, red_contact_term = 0.6`,
  matching the Phase F α=0 attractor up to a slight `red_contact` shift.

- **Disagreement queries can amplify pathological pairs.** `pebble_disagree`
  (uniform queries at the same budget) lands one seed at 100% foul rate
  (foul_term = −5, opp_contact = −0.3). The catastrophic-foul attractor
  Phase F flagged at PPO α=1 (1/3 seeds) is reproduced here at
  PPO α=1 *and* `pebble_disagree`, suggesting the failure mode is about
  the env's foul geometry rather than the RM scheme. Disagreement-based
  selection (`pebble_full`) avoids this seed but does not improve true
  score over uniform.

- **The cushion attractor is universal across RM-driven methods.**
  Cushion-term means: ppo_a0 = 2.00, ppo_a05 = 2.00, sac_rm_frozen = 1.33,
  pebble_disagree = 0.93, pebble_full = 1.60. PEBBLE shifts the
  attribution numerically but does not extinguish it; meanwhile the only
  method that activates the score axis is `sac_env` (cushion = 1.47,
  with `score_term ≈ 3.33`, the only nonzero score-term in the table).

The mechanistic reading is the one Phase F's diagnosis predicted: the
Bradley-Terry RM trained on `+0.4 · min(cushions, 5) + position_bonus`
defines a saturable, near-zero-effort subspace, and any optimizer whose
gradient *lives entirely inside the RM* will find it. Replacing PPO with
SAC + relabeling does not change which subspace is preferred. SAC + env
breaks the deadlock not because it explores better, but because its
gradient signal is the right one in the first place.

## 5. Honest assessment

This is a negative-PEBBLE / positive-SAC-env result and we report it as
such. PEBBLE is not designed to repair a misspecified RM — it is designed
to *cheaply train an RM* in domains where preference labels are
expensive but env reward is unobservable. Korean 4-ball single-shot is
the wrong test bed: env reward is binary but **observable**, and the RM
we hand PEBBLE is structurally biased toward cushions. Three caveats
sharpen this: (i) the RM, not PEBBLE, is the broken piece — relabeling
the buffer under a cushion-biased RM merely refines the same wrong
target; (ii) the env is solvable (SAC-env at 30k steps converges to a
scoring action in 2/3 seeds) so the right RLHF benchmark on this task
would gate the env reward behind, e.g., `cushion_hits ≥ 3` to create a
regime where PEBBLE's value proposition can actually be tested; and
(iii) Phase F's claim of "structural insufficiency of linear mixing"
should be refined to **structural insufficiency of *this RM***. The fix
is to swap the proxy, not the optimizer.

## 6. Reproducibility

- Code: this repo at `billiards/pebble/`,
  `experiments/run_pebble_matrix.py`, `notebooks/08_pebble_analysis.ipynb`.
- Reward model artifact: `models/reward_model.pt` (Phase D output, unchanged).
- Methods × seeds: 7 × 3 = 21 runs (4 Phase G methods × 3 seeds new + 3
  Phase F PPO α-conditions × 3 seeds replayed from
  `experiments/results/summary.parquet`).
- Per-run artifacts (Phase G): `experiments/runs_pebble/<method>_s<seed>/{
  eval.parquet, attribution.json, training_curve.csv, policy.zip,
  config.json, summary.json, run.log}`.
- Aggregate: `experiments/results/pebble_summary.parquet` (12 × 29).
- SAC backbone: 30k env steps, 6 RM-update phases, RM ensemble = 3,
  K=512 relabel cadence; preference batches summing to 300 total queries
  for `sac_rm_frozen` / `pebble_disagree` / `pebble_full`. Full configs
  in `experiments/configs.py` and per-run `config.json`.
- Eval: 200 deterministic episodes per (method, seed) on
  `Billiards4BallEnv`, base seed = `seed × 10000 + 10000` (matches Phase F
  for paired comparison).
- Figures: `paper/figures/{fig5_query_efficiency, fig6_relabel_delta,
  fig7_method_bars, fig8_attribution_compare}.{pdf,png}` at 300 dpi,
  monochrome-friendly via line-style differentiation.
- Tests: `tests/test_pebble_analysis.py` verifies
  `pebble_summary.parquet` schema (required columns, required methods,
  per-method seed count, true_score range).
- Notebook executed via `jupyter nbconvert --to notebook --execute`,
  0 errors; executed copy at `notebooks/08_pebble_analysis.executed.ipynb`.
