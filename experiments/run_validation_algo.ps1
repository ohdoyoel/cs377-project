<#
  VALIDATION §1.1 — SAC vs PPO vs TD3 algorithm comparison.

  3 algos x 5 seeds x 400k steps (near-plateau; see VALIDATION_EXPS.md §1.1).
  All three share the SAME env config so only the algorithm varies. Per-run
  outputs (config.json / summary.json / training_curve.csv / eval.parquet /
  policy.zip) land under experiments/runs_inning_v2/valid_algo/<algo>_s<seed>/.

  Resumable: a run whose summary.json already exists is skipped.

  Usage (from repo root):
    powershell -File experiments/run_validation_algo.ps1
    powershell -File experiments/run_validation_algo.ps1 -Seeds 0,1 -Algos sac,ppo
    powershell -File experiments/run_validation_algo.ps1 -Steps 200000   # cheaper screen
#>

param(
  [int[]]    $Seeds   = @(0, 1, 2, 3, 4),
  [string[]] $Algos   = @('ppo', 'td3', 'sac'),   # ppo first (cheap), sac last
  [int]      $Steps   = 400000,
  [int]      $MaxShots = 10,
  [int]      $EvalEpisodes = 100,
  [int]      $NEnvs   = 8,
  [string]   $OutRoot = 'experiments/runs_inning_v2/valid_algo'
)

# --- Fixed env config (held identical across algorithms). ---------------------
# PLAIN setup: NO domain knowledge (no constrain_aim / extra_features /
# gentle_shot / setup_shaping). Reward = pure {0,1} carom score (foul_penalty=0).
# Only the env structure stays: random_start + continue_on_miss + max_shots.
# Goal of §1.1 = compare RAW algorithm performance on the bare sparse problem.
# (Domain knowledge is studied separately in §3.) Change here = change for all.
$CommonFlags = @(
  '--random_start',
  '--continue_on_miss',
  '--foul_penalty',  '0.0',
  '--max_shots',     "$MaxShots",
  '--eval_episodes', "$EvalEpisodes",
  '--n_envs',        "$NEnvs",
  '--total_steps',   "$Steps"
)

$total = $Algos.Count * $Seeds.Count
$i = 0
$t0 = Get-Date
Write-Host "[valid_algo] $($Algos -join ',') x seeds=$($Seeds -join ',') x ${Steps} steps -> $OutRoot"

foreach ($algo in $Algos) {
  foreach ($seed in $Seeds) {
    $i++
    $outDir = Join-Path $OutRoot "${algo}_s${seed}"
    $summary = Join-Path $outDir 'summary.json'
    if (Test-Path $summary) {
      Write-Host "[$i/$total] SKIP $algo s$seed (summary.json exists)"
      continue
    }

    $algoFlags = @('--algo', $algo, '--seed', "$seed", '--out_dir', $outDir)
    # Off-policy extras only apply to SAC/TD3 (PPO ignores them).
    if ($algo -ne 'ppo') {
      $algoFlags += @('--gradient_steps', '2', '--buffer_size', '200000')
    }

    $runStart = Get-Date
    Write-Host "[$i/$total] RUN  $algo s$seed -> $outDir  ($($runStart.ToString('HH:mm:ss')))"
    uv run python experiments/run_inning_sac.py @algoFlags @CommonFlags
    if ($LASTEXITCODE -ne 0) {
      Write-Host "[$i/$total] FAIL $algo s$seed (exit $LASTEXITCODE) — continuing" -ForegroundColor Red
    } else {
      $dur = ((Get-Date) - $runStart).TotalMinutes
      Write-Host ("[$i/$total] DONE $algo s$seed in {0:N1} min" -f $dur)
    }
  }
}

$elapsed = ((Get-Date) - $t0).TotalMinutes
Write-Host ("[valid_algo] ALL DONE in {0:N1} min. Results under $OutRoot" -f $elapsed)
