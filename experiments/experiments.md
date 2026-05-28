# 실험 기록 (Experiments Log)

CS377 한국식 4구 당구 RL 프로젝트의 학습/평가 실험 정리.

## 읽는 법 / 주의사항
- **날짜**: run 디렉터리의 파일 mtime은 전부 clone 시각(2026-05-24)으로 동일해 신뢰할 수 없음.
  타임라인은 **git 커밋 히스토리** 기준이다.
- **policy 파일**: 각 run의 학습된 정책은 `{run_dir}/policy.zip`. `.gitignore`로 제외되어
  (저장소엔 config.json / summary.json / training_curve.csv 메타데이터만 있음) 이 저장소
  체크아웃에는 실제 `.zip`이 없다. 재현하려면 해당 머신에서 재학습하거나 따로 동기화 필요.
- **평가 지표**: `mean` = inning당 평균 득점, `max` = 최고 득점, `p1/p3` = ≥1/≥3점 비율(%),
  `foul` = 파울 비율(%), `wall` = 학습 시간(초). 모두 `summary.json` 기준.
- **eval 패러다임(2026-05-19~)**: 학습·평가 모두 random start. seed는 큰 의미 없음(공 위치
  랜덤화). `max_shots`가 다르면 득점 상한이 달라지므로 mean 직접 비교 주의.

---

## 타임라인 (git 기준)

| 날짜 | 추가/시도한 것 |
|---|---|
| 2026-05-01 | Initial commit. 물리 시뮬레이터 + 단발 샷 env (`Billiards4BallEnv`) |
| (초기) | **Preference/RLHF 라인**: reward model + PEBBLE (`runs/`, `runs_pebble/`) → 대부분 0점, 실패 |
| 2026-05-18 | TD3 추가 / `continue_on_miss` / `ignore_opponent`(커리큘럼, 상대공 추가 시 붕괴) / `constrain_aim`(최근접 red 조준 → max_inning>1 첫 달성) |
| 2026-05-18 | `extra_features`(red까지 거리 + 3구 각도) + **VecEnv**(학습 가속) |
| 2026-05-19 | **Random Paradigm**(학습/평가 전부 random start) / 파울 보상 버그픽스 + `gentle_shot` |
| 2026-05-20 | env·sim 개선(lean info dict, dt_max 0.05→0.1, `setup_shaping`, snap-to-rest) / 학습 바이너리 untrack / **v2 run 메타데이터** 추가 |
| 2026-05-21 | **multi-seed h=2 lookahead** — infinite billiards mean 500+ (mean 741.8) 달성 |
| 2026-05-22 | **PUCT vs greedy** 정량 비교 (n=10) |
| 2026-05-26 | `time_reward` 추가 (빠른 득점 샷에 보너스) — 코드/테스트 완료, 학습은 미실행 |
| 2026-05-27 | `time_reward` 앙상블 fine-tune(s1/s4/s6) 실행 + 시간예산 eval(`eval_time_budget.py`) 추가 |
| 2026-05-28 | **TimeBudgetGameEnv game_solo** (s1/s4/s6, 150k fine-tune) — game mean↑·foul↓·std↓ 전 seed 확인 |
| 2026-05-28 | **lookahead game_solo proposer eval** — inning lookahead에선 game_solo proposer가 baseline보다 나쁨(miss-tolerance 불일치) |

---

## 1. 메인 SAC 라인 (`experiments/runs_inning_v2/`)

현재 주력. 전부 SAC, `n_envs=8`, `gradient_steps=2`, `buffer_size=200000`, `gamma=0.99`,
`constrain_aim + extra_features + random_start + continue_on_miss + gentle_shot` 공통.
아래 표는 그 외 **차이 나는 flag**와 결과만 표기.

### 1.1 현재 SOTA 정책
**`fast_long_fp02_s4`** — `mean=5.69, max=9, p1=100%, p3=99%, foul=51%` (800k steps, ~30분)
- flags: `foul_penalty=0.2`, `setup_shaping`(α=0.05, scale=0.3), `max_shots=10`
- policy: `experiments/runs_inning_v2/fast_long_fp02_s4/policy.zip`
- lookahead 후보 제안 정책(`experiments/lookahead/*.py`)으로도 이 정책(+s1,s6)을 사용.

### 1.2 주요 실험군 (max_shots=10, eval 100 innings)

| run | steps | 차이 flag | mean | max | p3 | foul | wall |
|---|---|---|---|---|---|---|---|
| fast_baseline_s1 | 200k | setup_shaping **없음** | 2.37 | 7 | 46% | 70% | 236s |
| fast_g2_shape_s0/1/2 | 200k | + setup_shaping | 3.6~3.7 | 7~8 | ~80% | ~57% | 0.5~21min |
| fast_fp02_s1 | 400k | foul_penalty=0.2 | 4.69 | 9 | 91% | 60% | 988s |
| fast_sharp_s1 | 400k | setup_scale=**0.15** | 5.26 | 9 | 95% | 53% | 1286s |
| fast_long_fp01_s1/s4 | 800k | foul_penalty=**0.1** | 5.05~5.12 | 8~9 | 95% | 58~61% | ~35min |
| fast_long_fp02_s0~s6 | 800k | foul_penalty=**0.2** | 4.50~**5.69** | 8~10 | 90~99% | 46~70% | ~30~35min |
| fast_long_s0/s1/s2 | 800k | foul_penalty=**0.5** | 5.05~5.50 | 8~10 | 95~99% | 47~52% | 25~35min |
| fast_big400_s4 | 800k | net_arch=**400,300** | 4.91 | 8 | 94% | 58% | 2519s |
| fast_big512_s4 | 800k | net_arch=**512,256,128** | 4.95 | 9 | 95% | 56% | 2824s |
| fast_bigbuf_s4 | 1.2M | (긴 학습) | 5.23 | 9 | 97% | 51% | 3426s |
| fast_long2_s1 | 1.6M | (더 긴 학습) | 5.00 | 9 | 92% | **23%** | 3365s |
| fast_huge_fp02_s1 | 2M | foul_penalty=0.2 | 4.86 | 10 | 89% | **29%** | 4206s |
| fast_g995_s4 | 800k | (gamma 변형 시도) | 4.79 | 9 | 88% | 49% | 2481s |
| fast_finetune_s1 | 200k | load_policy(g2_shape_s1) | — | — | — | — | — |
| fast_load_noshape_s1 | 200k | load(long_fp02_s1), shaping 없음 | 4.68 | 9 | 91% | 50% | 512s |

### 1.3 max_shots=20 (득점 상한 ↑)
| run | steps | mean | max | 비고 |
|---|---|---|---|---|
| fast_chain_s1 | 400k | **9.45** | **15** | max_shots=20. 체이닝 보상 ↑ |
| fast_ms20shape_s1 | 400k | **9.45** | **15** | max_shots=20 + setup_shaping |

→ mean이 ~2배지만 max_shots 상한이 2배라 1.2의 max_shots=10 run과 **직접 비교 불가**.

### 1.4 특이 케이스
- **fast_canon_s1**: `random_start` 없음(canonical 고정 위치만). mean=2.00, max=2, foul=0% —
  고정 위치 2점 이후 막힘. random 일반화와의 trade-off 확인용.

### 관찰 요약
- `setup_shaping`이 baseline 대비 가장 큰 점프(2.37→3.7), 이후 학습량(steps) 늘리면 ~5점대 수렴.
- `foul_penalty`는 0.1/0.2/0.5 모두 비슷한 mean. **파울률**은 0.5 + 장기학습(1.6~2M steps)에서
  23~29%로 크게 감소 → 파울 억제는 penalty 크기보다 학습량이 더 중요.
- 큰 네트워크(400,300 / 512,256,128)는 256,256 기본보다 나을 것 없었음.
- seed 의존성 존재(long_fp02: s0 4.50 ~ s4 5.69).

### 1.5 time_reward fine-tune (앙상블 s1/s4/s6, 2026-05-27)

`fast_long_fp02_s{1,4,6}` 800k 정책을 각각 `--load_policy`로 warm-start 후
`--time_reward`(time_alpha=0.2, time_scale=3.0)만 추가해 150k step fine-tune
(나머지 flag 동일, 각 ~5분). 결과는 `experiments/runs_inning_v2/fast_time_fp02_s{1,4,6}/`.
목적은 lookahead 앙상블(s1+s4+s6) 전체를 동일 shaping으로 옮겨 inference 비교를 공정히 하기 위함.

**표준 eval** (time_reward 적용 전→후, max_shots=10, 100 innings):

| seed | mean | foul% | mean_cushions |
|---|---|---|---|
| s1 | 5.10 → **5.61** | 52 → **48** | 1.182 → **1.099** |
| s4 | 5.69 → **5.82** | 51 → **44** | 1.085 → **0.863** |
| s6 | 5.21 → **5.49** | 54 → **67** ⚠️ | 1.544 → **1.259** |

→ 세 seed 모두 mean↑, mean_cushions↓(샷이 더 직접적·짧아짐 = time_reward 의도대로).
foul은 s1/s4 개선, **s6만 악화**(seed 의존 이상치).

### 1.6 시간예산 eval (`eval_time_budget.py`, 신규)

기존 eval은 *이닝당 점수*(miss 시 종료, max_shots 캡)라 "빠른 득점"의 시간 효율을 못 잡음.
신규 eval은 **고정 시뮬시간 예산** 안에서 이닝을 연속 플레이(miss/foul→이닝 종료→랜덤 위치 재시작)하며
점수를 누적, **누적점수 vs 시뮬시간 곡선**을 n_repeats 평균으로 기록. 단위 시간당 득점을 직접 측정.

결과(budget=120s, n_repeats=30, score@120s, ±std): `experiments/artifacts/time_budget/ensemble_compare/`
(곡선 CSV + `comparison.png`).

| seed | non-time | time |
|---|---|---|
| s1 | 12.63 ± 3.25 | **12.77 ± 4.59** |
| s4 | 14.23 ± 4.75 | **14.50 ± 3.13** |
| s6 | 12.97 ± 3.69 | **13.47 ± 2.84** |

→ 세 seed 모두 time 정책이 시간당 득점 소폭 우위(+0.14/+0.27/+0.50)나 **std 밴드 내** —
n=30에선 약한 효과. 표준 eval의 cushions↓와 방향은 일치(직접적 샷이 시간 예산을 덜 소모).

### 1.7 TimeBudgetGameEnv game_solo (2026-05-28)

`fast_long_fp02_s{1,4,6}` (비-time SOTA) warm-start 후 `TimeBudgetGameEnv`(budget_s=120)로
150k step fine-tune. `foul_penalty=1.0`(inning의 0.2보다 강함 — 이닝 종료+위치 손실 구조와 결합),
나머지 flag 동일(`constrain_aim + extra_features + gentle_shot + setup_shaping`).
결과: `experiments/runs_game_solo/game_s{1,4,6}/`. eval은 shaping 없는 clean env, 50 games.

**game_solo eval 결과 (budget=120s, n=50 games):**

| seed | mean_game_score | std | max | mean_innings | score/sim_s | foul_rate | cushions |
|---|---|---|---|---|---|---|---|
| s1 | **13.80** | 2.46 | 19 | 13.86 | 0.113 | **9.3%** | 1.234 |
| s4 | **15.24** | 3.03 | 22 | 13.22 | 0.124 | **7.9%** | 1.097 |
| s6 | **13.46** | 2.53 | 19 | 14.10 | 0.110 | **9.6%** | 1.185 |

**§1.6 eval_time_budget 기준선(비-time 정책, n=30) 대비 game_solo 개선:**

| seed | 기준선 mean±std | game_solo mean±std | Δmean | Δstd |
|---|---|---|---|---|
| s1 | 12.63 ± 3.25 | **13.80 ± 2.46** | **+1.17** | −0.79 |
| s4 | 14.23 ± 4.75 | **15.24 ± 3.03** | **+1.01** | −1.72 |
| s6 | 12.97 ± 3.69 | **13.46 ± 2.53** | **+0.49** | −1.16 |

**관찰:**
- 세 seed 모두 mean↑·std↓: game 목적 fine-tune이 scoring throughput + 안정성 동시 향상.
- foul_rate 극적 하락: inning env 46~70% → game_solo 7.9~9.6%. `foul_penalty=1.0`(강한 패널티)
  + 게임 구조(foul 시 이닝 종료, 위치 손실) 결합 효과. time_reward fine-tune(§1.5)의 foul 개선보다
  훨씬 큰 폭.
- cushion_hits↓: inning eval의 fast_long 1.1~1.5에서 game_solo 1.1~1.2로 소폭 감소 — 더 직접적인 샷.
- s4 SOTA 유지(15.24/22): inning SOTA(5.69/10 shots)와 일관된 seed 우위.

---

## 2. Lookahead (추론 시점 탐색, `experiments/lookahead/`)

학습된 SAC 정책을 **후보 제안기**로 쓰고, 각 후보를 env에서 시뮬레이션해 `r1 + γ·r2`로
순위를 매기는 greedy/MCTS 탐색. 재학습 없이 점수를 크게 끌어올림. `artifacts/best_inning/`에
대표 리플레이 HTML 저장(파일명에 score 기록).

| 스크립트 | 방식 | 대표 best score | 비고 |
|---|---|---|---|
| `multi_seed_h2.py` | s1+s4+s6 앙상블 제안 + h=2 | **2000** (shots 2000) | mean 741.8 (2026-05-21) |
| `adaptive_K.py` | shot 초반 K1 대량(1000→300→150→80) | 1000~1303 | front-load 탐색 |
| `multistep_uncap.py` | K1=100 K2=5, max_shots 큰 cap | 742 | true ceiling 탐색 |
| `puct.py` | PUCT MCTS (uniform/policy/multi_seed) | — | greedy와 정량 비교(2026-05-22) |

공통 env 설정: `constrain_aim, extra_features, foul_penalty=0.2, gentle_shot, setup_shaping(0.05/0.3)`.
*(주의: 위 4개 스크립트 상단 `REPO` 경로가 다른 머신 절대경로로 하드코딩 — 이 머신에서 돌리려면 수정 필요.
신규 `compare_time_proposers.py`는 `Path(__file__).parents[2]`로 REPO 자동탐지.)*

### 2.1 time effect 분해: ranking vs proposer (`compare_time_proposers.py`, 2026-05-27)

질문: lookahead에서 time_reward가 시간 효율에 기여한다면 그게 (a) 후보 **랭킹**의 time bonus
때문인가, (b) §1.5 time-fine-tune 정책을 **proposer**로 써서인가? 3조건 paired 비교로 분해.
셋업: h=2 multi-seed(s1,s4,s6), paired seeds, n=12, max_shots=25, k1=100(per-policy), k2=3.

| variant | proposer | 랭킹 time | score/sim_s | mean_sim_t | mean_score |
|---|---|---|---|---|---|
| true_baseline | 비-time | off | 0.255 | 88.6s | 22.92 |
| baseline | 비-time | **on** | **0.282** | 81.2s | 22.92 |
| time | **time** | on | 0.284 | 80.7s | 22.92 |

**분해 (score/sim_s):** 랭킹 효과 = **+0.027** (true_baseline→baseline, ~+11%) ·
proposer 효과 = **+0.002** (baseline→time, noise 내).
→ lookahead에서 time_reward의 시간 효율 이득은 **거의 전부 랭킹 `TIME_REWARD` 토글**에서 나오고,
proposer를 time 정책으로 바꾸는 건 무의미. 랭킹 bonus를 켜면 cap 도달 sim_time이 88.6→81.2s로
~7.4s 단축(점당 더 빠른 샷 선택). 즉 **lookahead엔 time 정책 불필요, 랭킹만 켜면 됨.**
(§1.6 단독 정책 eval에선 fine-tune이 효과 있었으나, lookahead는 search가 그 역할을 대신.)

*Caveat:* max_shots=25 cap 탓에 raw mean_score는 세 조건 모두 22.92로 saturate — 구분은
sim_time/score-per-sim-s에서만(11/12 ep가 cap 도달, seed 99000만 첫 샷 miss, paired 동일).
체이닝 길이 차이까지 보려면 cap↑(훨씬 느림). 결과: `experiments/artifacts/lookahead_time_ab/ab_summary.json`.

### 2.2 game_solo proposer eval (`compare_time_proposers.py`, 2026-05-28)

질문: game_solo fine-tune 정책(`runs_game_solo/game_s{1,4,6}`)을 lookahead proposer로 쓰면
`time_rank=False` 기준으로 true_baseline(비-time, time_rank=False)보다 나은가?
셋업: true_baseline과 동일 조건(paired seeds, n=12, max_shots=25, k1=100, k2=3, time_rank=False).

| variant | proposer | mean_score | std | score/sim_s | mean_sim_t | 비고 |
|---|---|---|---|---|---|---|
| true_baseline | fast_long_fp02 | **22.58** | 6.90 | **0.255** | 88.6s | 기준선(§2.1 재현) |
| game_solo | runs_game_solo | 20.08 | 8.02 | 0.243 | **82.6s** | game 목적 fine-tune |

**결과: game_solo proposer가 true_baseline보다 나쁨** (mean_score −2.5, score/sim_s −0.012).

**원인 분석:**
- ep 2 (score=13, sim_t=59.7s), ep 8 (score=9, sim_t=43.7s): game_solo 정책이 inning 중간에
  miss를 더 자주 냄(게임 종료가 아닌 이닝 종료이므로 miss-tolerant하게 학습됨).
- game_solo 학습 환경에서 miss/foul은 이닝 reset 후 게임 계속 → miss를 "허용"하는 리스크 모델.
  lookahead inning env에서는 miss = 에피소드 종료 → miss-tolerance가 점수 손실.
- sim_time↓(82.6s < 88.6s): 더 짧은 샷 제안(faster travel)은 유지되지만, 점수를 내기 전에
  miss해서 이닝이 짧아진 것도 포함.
- std↑: game_solo proposer가 고점은 동일(max=25)이지만 실패 케이스가 늘어 분산 증가.

**결론:** lookahead proposer로는 inning env에서 직접 학습한 fast_long_fp02가 우수. game_solo 정책은
TimeBudgetGameEnv 맥락(miss 후 계속 플레이)에서만 이점을 보임. proposer와 eval env의 종료 조건
일치가 중요함. 결과: `experiments/artifacts/lookahead_game_solo/ab_summary.json`.

---

## 3. 초기 Phase II 탐색 (`experiments/runs_inning/`, 50k steps 위주)

v2 이전의 단일-env(non-vec) 탐색. 대부분 50k steps, max_shots 작음. 핵심 발견만:
- **PPO 전멸**: ppo_s0/1/2 모두 mean 0 → 이후 SAC/TD3로 전환.
- **constrain_aim 효과**: sac_aim_s0 mean 4.0(p3 100%) — 조준 제약이 첫 득점의 열쇠.
- **TD3**: td3_aim/feat/novec 등 — 득점은 하나 **foul 100%**가 잦아 SAC 대비 불안정.
- ignore_opponent 커리큘럼(stage1/2): 상대공 추가 시 정책 붕괴(0점) — 커리큘럼 실패.
- sac_s0_1M: 1M steps인데 mean 0 — 당시 설정에선 장기학습이 도움 안 됨(이후 VecEnv+shaping으로 해결).

---

## 4. Preference/RLHF 라인 (`experiments/runs/`, `experiments/runs_pebble/`) — 대체로 실패

단발 샷 env + reward model(선호 학습)/PEBBLE 라인. **scoring 정책 학습 실패**.
- `runs/` alpha sweep(α=0.0~1.0, env reward vs reward-model 혼합, 각 3 seed):
  **모든 α에서 true_score_rate = 0.0**.
- `runs_pebble/` (pebble_full / pebble_disagree / sac_rm_frozen, 각 3 seed): 거의 0점,
  파울 잦음. 유일하게 `sac_env`(순수 env reward) seed 0/1만 true_score_rate=100% →
  **순수 env reward가 reward-model보다 나음**을 보여줌. 이 결론이 이후 직접 RL(inning) 전환의 근거.

---

## 5. Flag 레퍼런스 (`run_inning_sac.py` / `inning_env.py`)

| flag | 의미 | 사용된 값 |
|---|---|---|
| `--constrain_aim` | theta를 ±arcsin(2r/d) 윈도로 제한, 최근접 red 첫 접촉 보장 | on (전 v2) |
| `--extra_features` | obs +4dim: d(cue,red1), d(cue,red2), sin φ, cos φ | on (전 v2) |
| `--random_start` | 매 reset 공 위치 랜덤화 (canonical overfit 방지) | on (canon 제외) |
| `--continue_on_miss` | miss/foul 후에도 max_shots까지 계속(다양한 상태 노출) | on (전 v2) |
| `--foul_penalty` | 파울 시 reward −값 (점수와 무관) | 0.1 / 0.2 / 0.5 |
| `--gentle_shot` | 득점 후 수구~2번째 목적구 거리 Gaussian 보너스(α=0.2, d_target=0.2, σ=0.1) | on (전 v2) |
| `--setup_shaping` | 매 non-foul 샷마다 `setup_alpha·exp(−d_min/setup_scale)` | on (baseline/g2 제외) |
| `--setup_alpha` | setup_shaping 크기 | 0.05 / 0.1 |
| `--setup_scale` | setup_shaping 거리 스케일(m) | 0.3 / 0.15(sharp) |
| `--time_reward` | 득점 샷에 `time_alpha·exp(−duration/time_scale)` (빠를수록 ↑) | on (fast_time_fp02_s{1,4,6}) |
| `--time_alpha` / `--time_scale` | time_reward 크기 / 시간 스케일 | 기본 0.2 / 3.0 |
| `--n_envs` | 병렬 env(SubprocVecEnv) | 8 |
| `--gradient_steps` | env step당 그라디언트 스텝 | 2 |
| `--net_arch` | MLP 은닉층 | 기본256,256 / 400,300 / 512,256,128 |
| `--load_policy` | warm-start할 policy.zip 경로 | (finetune 계열) |
| `--max_shots` | inning당 최대 샷 수 (득점 상한) | 10 / 20 |
| `--total_steps` | 학습 step 수 | 50k ~ 2M |

---

## 6. 진행 중 / 다음 (2026-05-28~)
- **time_reward 앙상블 fine-tune** ✅ (§1.5): s1/s4/s6 모두 150k step fine-tune 완료.
  표준 eval에서 mean↑·cushions↓ 확인. 시간예산 eval(§1.6)에선 time 정책이 소폭 우위지만
  n=30 std 내 → 효과 약함.
- **s6 foul 악화**: time_reward 후 s6만 foul 54→67%. seed 단위 이상치.
- **lookahead time effect 분해** ✅ (§2.1): 랭킹 `TIME_REWARD` 토글이 시간 효율 이득의 거의 전부.
  proposer 교체 효과 미미(+0.002). lookahead엔 time 정책 불필요, 랭킹만 켜면 됨.
- **game_solo fine-tune** ✅ (§1.7): s1/s4/s6 150k 완료. 기준선 대비 mean+0.5~+1.2, std −0.8~−1.7,
  foul_rate 극적 하락(~70% → ~9%). 가장 강력한 단일 foul 억제 기법으로 확인.
- **lookahead + game_solo proposer** ✅ (§2.2): game_solo proposer + time_rank=False가 true_baseline
  대비 mean_score −2.5(20.08 vs 22.58), score/sim_s −0.012. miss-tolerance 불일치가 원인.
  → lookahead proposer는 inning env 직접 학습 정책(fast_long_fp02)이 우수. game_solo는 게임 맥락 전용.
