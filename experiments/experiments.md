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
| 2026-05-26 | `time_reward` 추가 (빠른 득점 샷에 보너스) — 코드/테스트 완료, 학습은 미실행 *(현재 작업)* |

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
*(주의: 스크립트 상단 `REPO` 경로가 다른 머신 절대경로로 하드코딩 — 이 머신에서 돌리려면 수정 필요.)*

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
| `--time_reward` *(신규)* | 득점 샷에 `time_alpha·exp(−duration/time_scale)` (빠를수록 ↑) | (미실행) |
| `--time_alpha` / `--time_scale` | time_reward 크기 / 시간 스케일 | 기본 0.2 / 3.0 |
| `--n_envs` | 병렬 env(SubprocVecEnv) | 8 |
| `--gradient_steps` | env step당 그라디언트 스텝 | 2 |
| `--net_arch` | MLP 은닉층 | 기본256,256 / 400,300 / 512,256,128 |
| `--load_policy` | warm-start할 policy.zip 경로 | (finetune 계열) |
| `--max_shots` | inning당 최대 샷 수 (득점 상한) | 10 / 20 |
| `--total_steps` | 학습 step 수 | 50k ~ 2M |

---

## 6. 진행 중 / 다음 (2026-05-26~)
- **time_reward**: 빠르게 끝나는 득점 샷에 보너스. SAC는 현재 best(`fast_long_fp02_s4`)에서
  `--load_policy`로 fine-tune 예정(~100~150k steps, ~5분). Lookahead는 env reward 공유 →
  추론 시점 즉시 반영(스크립트의 `TIME_REWARD` 상수 toggle). policy.zip 확보가 선행 조건.
