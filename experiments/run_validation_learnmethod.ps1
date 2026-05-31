<#
  VALIDATION §1.2 — learning-paradigm comparison (start x on-miss).

  SAC only, 3 seeds, 400k steps, PLAIN env (no domain knowledge, reward = pure
  {0,1} score, foul_penalty=0) — same backdrop as §1.1. The ONLY thing that
  varies is the 2x2 learning method:
    canonical vs random start   (random_start off/on)
    reset vs continue on miss   (continue_on_miss off/on)

  Outputs -> experiments/runs_inning_v2/valid_learnmethod/<method>_s<seed>/.
  Resumable: a run whose summary.json exists is skipped.

  Usage:
    powershell -File experiments/run_validation_learnmethod.ps1
    powershell -File experiments/run_validation_learnmethod.ps1 -Seeds 0,1
#>

param(
  [int[]]  $Seeds   = @(0, 1, 2),
  [int]    $Steps   = 400000,
  [int]    $MaxShots = 10,
  [int]    $EvalEpisodes = 100,
  [int]    $NEnvs   = 8,
  [string] $OutRoot = 'experiments/runs_inning_v2/valid_learnmethod'
)

# Run from the repo root regardless of the caller's CWD (script lives in experiments/).
Set-Location (Split-Path -Parent $PSScriptRoot)

# Shared PLAIN flags (identical across the 4 methods).
$CommonFlags = @(
  '--algo', 'sac',
  '--foul_penalty',  '0.0',
  '--max_shots',     "$MaxShots",
  '--eval_episodes', "$EvalEpisodes",
  '--n_envs',        "$NEnvs",
  '--gradient_steps', '2',
  '--buffer_size',   '200000',
  '--total_steps',   "$Steps"
)

# The 2x2 method matrix: name -> extra flags toggling start / on-miss.
$Methods = [ordered]@{
  'canon_reset' = @()                                    # canonical start, reset on miss
  'canon_cont'  = @('--continue_on_miss')                # canonical start, continue on miss
  'rand_reset'  = @('--random_start')                    # random start,    reset on miss
  'rand_cont'   = @('--random_start', '--continue_on_miss')  # random start, continue on miss
}

$total = $Methods.Count * $Seeds.Count
$i = 0
$t0 = Get-Date
Write-Host "[valid_learnmethod] SAC x $($Methods.Keys -join ',') x seeds=$($Seeds -join ',') x ${Steps} steps"

foreach ($method in $Methods.Keys) {
  foreach ($seed in $Seeds) {
    $i++
    $outDir = Join-Path $OutRoot "${method}_s${seed}"
    if (Test-Path (Join-Path $outDir 'summary.json')) {
      Write-Host "[$i/$total] SKIP $method s$seed (summary.json exists)"
      continue
    }
    $methodFlags = @('--seed', "$seed", '--out_dir', $outDir) + $Methods[$method]
    $runStart = Get-Date
    Write-Host "[$i/$total] RUN  $method s$seed -> $outDir  ($($runStart.ToString('HH:mm:ss')))"
    uv run python experiments/run_inning_sac.py @CommonFlags @methodFlags
    if ($LASTEXITCODE -ne 0) {
      Write-Host "[$i/$total] FAIL $method s$seed (exit $LASTEXITCODE) — continuing" -ForegroundColor Red
    } else {
      Write-Host ("[$i/$total] DONE $method s$seed in {0:N1} min" -f ((Get-Date) - $runStart).TotalMinutes)
    }
  }
}
Write-Host ("[valid_learnmethod] ALL DONE in {0:N1} min -> $OutRoot" -f ((Get-Date) - $t0).TotalMinutes)
