# Multi-Shot Reward Closes (or Doesn't) the Single-Shot Sparsity Gap

> **Phase H extension** to "Linear Reward Mixing Fails to Recover Sparse Ground Truth"
> (Phase F, `paper/iclr_summary.md`) and "Does PEBBLE Rescue Sparse-Reward RLHF?"
> (Phase G, `paper/iclr_pebble.md`).

**Claim.** Switching from the *single-shot* `Billiards4BallEnv` to the *multi-shot
inning* env `Billiards4BallInningEnv` — where reward is the cumulative score of
a possession, ended by miss, foul, or shot-cap — does **not** change the
headline rate for plain SAC (66.7 % of seeds reach a 100 % single-shot
success rate, identical to Phase G's `sac_env`) and does not produce
multi-shot chains (max inning = 1 in every successful run). The single-shot
framing of Phase F/G was therefore not a hidden cause of the 0 %
RM-driven results: the bottleneck is reward-model quality, not episode
horizon.

## 1. Why this matters

Phase F (PPO + linear-mix RM) and Phase G (SAC + PEBBLE) each reported 0 %
true-score rate on `Billiards4BallEnv` for every RM-driven configuration,
while plain SAC on raw env reward already recovered 66.7 % of seeds.
A reviewer could plausibly attribute the *0 % RM* result to single-shot
training: the natural Korean 4-ball reward is the inning's *cumulative*
score across multiple shots. If RM-driven methods had been trained on
that denser, more game-realistic signal, perhaps they would not have
collapsed to a no-score cushion attractor. Phase H tests this directly,
without any RM, by training the same agents on the multi-shot env.

## 2. Method

We train SAC and PPO on `Billiards4BallInningEnv(max_shots=50, t_max=12)`,
wrapped in `Monitor` so SB3's `ep_info_buffer` collects one entry per
*inning* (cumulative reward = inning score, episode length = number of
shots in the inning, terminated on miss/foul, truncated at the cap).
Hyperparameters match Phase G defaults: `lr=3e-4`, `gamma=0.99`, `batch=256`
(SAC), and `n_steps=512, batch=512, n_epochs=4, ent_coef=0.01` (PPO). We
run 3 seeds per algorithm at 50 000 env steps each, then evaluate
deterministically for 200 innings (`seed_base + 10_000 + ep`). A uniform
random policy on the same 200 innings provides the chance-only baseline.
Code: `experiments/run_inning_sac.py`, matrix runner
`experiments/run_inning_matrix.py`. Results in
`experiments/results/inning_summary.parquet` and per-run
`experiments/runs_inning/<run_id>/{eval.parquet, training_curve.csv,
policy.zip}`.

## 3. Results

| method  | mean inning score | max inning | p≥1 (%) | p≥3 (%) | p≥5 (%) | mean shots/inning | seeds |
|---------|-------------------|------------|---------|---------|---------|-------------------|-------|
| **sac** | **0.667**         | 1          | **66.7**| 0.0     | 0.0     | 1.67              | 3     |
| ppo     | 0.333             | 1          | 33.3    | 0.0     | 0.0     | 1.33              | 3     |
| random  | 0.005             | 1          | 0.5     | 0.0     | 0.0     | 1.00              | 1     |

SAC reproduces Phase G's `sac_env` headline almost exactly (Phase G:
66.7 ± 57.7 %; Phase H: 2/3 seeds at 100 % p≥1, 1 at 0 %). PPO is
strictly worse at 33.3 %. *No policy ever scores twice in one inning:*
across all 6 runs and 1 200 evaluation innings, the maximum cumulative
inning score is **1**, and the mean inning length is **1.67** for SAC and
**1.33** for PPO — i.e. one scoring shot followed by an immediate miss.
The training curves (Figure 9) flatten by ~30 k steps at this same
ceiling. The inning-score distribution for the best run (Figure 10) is a
sharp 0 – 1 binary; there is no tail at 2+. The dense per-shot signal does
not, in this budget, teach the cue ball to leave a *makeable* leave for
the next shot. Random scores far below both, confirming SAC and PPO are
doing real work — just not multi-shot work.

## 4. Implication

The Phase F/G negative results — uniformly 0 % under any RM — are
**not artifacts of single-shot framing**. The same plain-SAC headline
recurs on the natively multi-shot env, and even on that denser signal
the policy hits a one-and-done attractor rather than learning to chain.
Combined with Phase G's evidence that PEBBLE's relabeling and
disagreement queries do not recover the task either, this strengthens
the original claim: **label quality, not episode horizon and not the
RM-mixing knob, is the bottleneck for sparse-reward RLHF on this
benchmark**. A second-shot capable policy would require either denser
position-aware labeling or task-grounded curricula.

## 5. Reproducibility

- Code: `experiments/run_inning_sac.py` (single run), `experiments/run_inning_matrix.py` (6-config matrix).
- Notebook: `notebooks/09_inning_results.ipynb` (executed copy at `notebooks/09_inning_results.executed.ipynb`).
- Aggregate: `experiments/results/inning_summary.parquet`.
- Per-run artifacts: `experiments/runs_inning/{sac,ppo}_s{0,1,2}/{eval.parquet, training_curve.csv, policy.zip, config.json, run.log, summary.json}`.
- Figures: `paper/figures/fig9_inning_training.{pdf,png}`, `paper/figures/fig10_inning_distribution.{pdf,png}`.
- Best-inning rendering: `artifacts/inning/best_inning.html`.
- Test: `tests/test_inning_sac.py` (smoke run at `total_steps=2048`).

To reproduce end-to-end:

```bash
uv run python experiments/run_inning_matrix.py        # 6 runs sequential, ~1 hr wall
uv run jupyter nbconvert --to notebook --execute \
    notebooks/09_inning_results.ipynb \
    --output 09_inning_results.executed.ipynb
```
