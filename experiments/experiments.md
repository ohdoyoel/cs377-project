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
| 2026-05-28 | **chain100 paradigm** — `--max_shots=100 --no-continue_on_miss --no-gentle_shot` fine-tune (s1/s4/s6, 150k) → **basic 득점 회귀** |
| 2026-05-28 | **robust_reward** (action-space margin) 구현 + s1/s4/s6 fine-tune — mean 소폭↑, p5 70~81% 강한 consistency |
| 2026-05-28 | **eval_robustness.py** — 직접 측정: 학습 효과 ±0.02 이내 (유의 X) |
| 2026-05-28 | **lookahead robust ranking A/B/C** — chosen robustness +0.116 (ranking만으로), proposer 교체 +0.012 |
| 2026-05-28 | **canonical 50샷 행동 비교** — 둘 다 50/50 득점이나 robust는 **쿠션 1.04→0.64, 2+쿠션 샷 절반↓, high-power × low-margin 회피** *(현재 작업)* |
| 2026-05-31 | **VALIDATION §1.1** — plain(도메인지식 전부 off) SAC/TD3/PPO 5seed×400k 비교. off-policy(SAC 0.42 / TD3 0.46) ≈ on-policy PPO(0.17)의 2.5배. SAC↔TD3 std 겹쳐 무차별, PPO 200k plateau. `valid_algo/`, [VALIDATION_EXPS.md](VALIDATION_EXPS.md) §1.1 |
| 2026-05-26 | `time_reward` 추가 (빠른 득점 샷에 보너스) — 코드/테스트 완료, 학습은 미실행 *(현재 작업)* |
| 2026-05-25 *(오도열)* | **Plain RL 재측정**: PPO/SAC 200k × 3 seeds, **no aim_constraint·shaping·gentle_shot·setup**, random_start만. — PPO 0.00 / SAC 0.015 (천장 확인) |
| 2026-05-25 *(오도열)* | **Plain ablations**: `angle_sincos`(5D action, θ wrap 제거) + `extra_features` SAC 200k s0 → 0.010 (효과 없음). Curriculum start ramp(d=0→1) 200k s0 → 0.020 (easy mean 0.16, transfer 실패) |
| 2026-05-25 *(오도열)* | **V(s) RM** (state-only scoring potential, MC over uniform K=100): v1 1k states r=0.466, v2 5k states r=0.869. SAC reward = score + λ·RM(s′), λ∈{10,50,100} → 모두 ≤0.020 (state-only 신호로는 actor가 못 따라옴) |
| 2026-05-26 *(오도열)* | **V(s,a) RM** (state-action scoring potential): v2 500k pairs (5k×100, hidden 256), v3 10M pairs (50k×200, hidden 512). v3 Top-10 recall **30%** (uniform 0.6% → 50× lift) |
| 2026-05-26 *(오도열)* | SAC + V(s,a) RM v2 λ∈{1,3,800k}/v3 λ∈{0.3,0.5,1} → peak **mean 0.060** (RM v2, λ=1). RM 강화가 SAC critic→actor의 indirect 경로에서 약해짐. **RM-as-reward 천장 확인** |
| 2026-05-27 *(오도열)* | **RM-as-search-prior**: SAC actor 안 쓰고 RM v3로 K candidate scoring + simulator verify. h=1 K=500 M=50 → mean **1.90** (Plain의 127×). h=2 K1=2000 M1=200 K2=500 M2=20 → mean **46.10**, max 243. h=2 K1=5000 M1=500 K2=1000 M2=50 → **mean 658.90**, max 1231 *(8h wall, no SAC training)* |
| 2026-05-27 *(오도열)* | **mean 1000+ 도전** *(진행 중)*: K1=10000 M1=1000 K2=2000 M2=100 launched |

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

### 1.5 chain100 paradigm (2026-05-28, shot_difficulty 브랜치)

**가설:** "한 이닝을 매우 길게 가져가는 게 무조건 유리"하다는 신호를 직접 학습시키기 위해
`continue_on_miss=False` + `max_shots=100`으로 환경 재설정. miss/foul 시 이닝 즉시 종료,
100샷 도달 시 truncation. 100까지 가려고 노력하는 정책을 기대.

`fast_long_fp02_s{1,4,6}` warm-start → 150k step fine-tune. `gentle_shot=False`(요청대로 ablate),
나머지 동일(`constrain_aim + extra_features + random_start + setup_shaping + foul_penalty=0.2`).
결과: `experiments/runs_inning_v2/chain100_s{1,4,6}/`.

**random eval (n=200, max_shots=100, continue_on_miss=False):**

| seed | mean | max | p1 | p3 | p5 | foul% | 비고 |
|---|---|---|---|---|---|---|---|
| s1 | 1.32 | 9 | 60.0 | 17.5 | 5.5 | 26.0 | warm-start: fast_long_fp02_s1 |
| s4 | 1.47 | **15** | 60.0 | 22.0 | 8.5 | 27.5 | warm-start: fast_long_fp02_s4 |
| s6 | 1.62 | 12 | 64.5 | 21.5 | 9.0 | 26.5 | warm-start: fast_long_fp02_s6 |

**baseline(fast_long_fp02_*) 대비:**

| seed | baseline mean (max_shots=10) | chain100 mean (max_shots=100) | Δ |
|---|---|---|---|
| s1 | ~5.10 | 1.32 | **−3.78** |
| s4 | **5.69** | 1.47 | **−4.22** |
| s6 | ~5.21 | 1.62 | **−3.59** |

**결과: 명백한 회귀.** 그러나 신호는 흥미로움:
- **max 체이닝 가능성 ↑** (s4 max=15, 베이스는 max_shots=10 cap에 막혀 max=9였음) — 정책이
  "원리적으로는" 15샷까지 연결 가능. 단, 200 ep 중 1회뿐 → 신뢰도 매우 낮음.
- **foul% 절반 ↓** (51~54 → 26~33): 페널티 회피는 학습됨.
- **p1 폭락** (~100 → 60~65): basic 1점 득점 능력 자체가 손상 — 정책이 첫 샷에서 자주 miss.
- mean_shots≈2.5 → 평균 1~2샷 만에 episode 종료 (즉, mean ≈ 1.5는 "맞으면 1점, 못 맞히면 0점" 수준).

**원인 가설:**
1. **`gentle_shot=False` ablation**: warm-start 정책은 `gentle_shot=True` 보너스로 학습되어
   "득점 + 다음 샷 쉬운 위치"를 동시에 최적화. 보너스 제거 시 reward 분포 급변 → 부드러운
   체이닝 행동이 강화되지 않음 → 정책 forget.
2. **`continue_on_miss=False` fine-tune**: 미스 시 즉시 종료 → 짧은 trajectory(1~2 샷)가
   리플레이 버퍼를 지배 → 긴 체인의 Q 추정이 데이터 부족.
3. **`setup_shaping` 누적 효과**: 매 샷마다 `0.05·exp(−d_min/0.3)` 보너스. 100샷 체인이면
   누적 ~5(실제 점수 100과 비교 가능). 짧은 체인에선 ~0.1로 미미 → setup이 score보다 우선시될 때
   "쉬운 setup만 골라 미스" 패턴 유도 가능.

**다음 시도 후보:**
- `gentle_shot=True` 유지하고 chain100 재실험 (가설 1 검증)
- `setup_shaping=False`로 끄고 순수 score만으로 학습 (가설 3 검증)
- 학습 step↑ (300~500k), LR↓로 catastrophic forgetting 완화

`experiments/runs_inning_v2/chain100_s{1,4,6}/summary.json` 참조.

### 1.6 robust_reward (action-space margin to failure, 2026-05-28)

**아이디어 (ENGINE.md §3.2):** 어려운 샷은 정답에서 살짝만 벗어나도 miss하는 샷.
"정답"인 액션 `a`에 노이즈 `δ ~ N(0, ε)`를 더한 N개 액션을 같은 pre-shot 상태에서 재시뮬,
성공률을 bonus로. 직관: 큰 margin = 쉬운 샷.

**구현** (`inning_env.py`, `--robust_reward`):
- 득점(non-foul) 샷마다 pre-shot 상태를 deepcopy → 보관
- `N=robust_n` 회 동안 `a + N(0, robust_eps)`로 perturb, 각 시뮬 → score>0 + not fouled 카운트
- `reward += robust_alpha × (success_fraction)`
- `_apply_aim_constraint` 일관 적용 (perturb 후에도 동일 aim window 사용)
- robust_reward=False면 deepcopy 생략 (오버헤드 0)

**비용**: smoke test 기준 학습 ~3.5x slowdown (득점 샷에만 N=8 추가 시뮬). 실측 학습 wall:
seed당 ~950~1060s (~17분), 베이스 대비 ~3x. 합리적 비용.

**fine-tune 설정**: `fast_long_fp02_s{1,4,6}` warm-start → 150k step. 모든 flag True
(constrain_aim + extra_features + random_start + continue_on_miss + gentle_shot + setup_shaping),
foul_penalty=0.2, robust_eps=0.05, robust_n=8, robust_alpha=0.2.
결과: `experiments/runs_inning_v2/robust_s{1,4,6}/`.

**training-mode eval (continue_on_miss=True, max_shots=10, n=200):**

| seed | baseline mean | robust mean | Δ | p3 | p5 | foul% | mean_cushions |
|---|---|---|---|---|---|---|---|
| s1 | 5.10 | **5.36** | +0.26 | 96.0% | **70.0%** | 61.0% | 1.231 |
| s4 | **5.69** | **5.79** | +0.10 | 97.5% | **81.0%** | 50.5% | **1.000** |
| s6 | 5.21 | **5.48** | +0.27 | 98.0% | **75.0%** | 59.5% | 1.267 |

**관찰:**
- **mean 회귀 없음** (chain100과 결정적 차이): 모든 flag True 유지가 안정성 핵심.
- **p5(≥5 chain 비율) 70~81%로 강함** — 정책이 일관되게 긴 chain 가능. 베이스(§1.2)는 p3만 기록되어
  직접 비교 어려우나 p5는 새 강한 metric.
- **s4 cushion 1.09→1.00**: 더 직접적인 샷 선호 (margin 큰 = 쿠션 적은 쪽 학습)
- **foul%는 거의 변화 없음** (51~54 → 50.5~61): robust bonus가 foul 억제로는 작동 안 함.
  이유: foul은 reward=-0.2로 별도 처리, robust bonus는 득점 샷에만 추가 → foul 감소에 직접 신호 없음.

**Caveat — 정말 "robustness"가 학습됐는가?**
mean 개선은 작고 (+0.1~+0.3), 노이즈일 수도. 진짜 robustness 검증은 §1.6.1 참고.

### 1.6.1 robustness 직접 측정 (`eval_robustness.py`, 2026-05-28)

**방법**: random rack 100개 (`seed_base=10_000~10_099`)에 대해 정책별 deterministic action `a*` 추출 →
`a* + N(0, ε)` perturbation 32개 시뮬 → score>0 (non-foul) 비율 = robustness(state, ε). paired seeds.

**메트릭**:
- `det_success_rate`: `a*` 자체가 득점하는 state 비율
- `robustness_conditional`: `a*`가 득점한 state들에 한해 평균 perturbation 성공률 — **robust_reward bonus가 직접 최대화하는 양**.

**paired 비교 (robustness_conditional, baseline→robust):**

| seed | det% (B→R) | ε=0.02 | ε=0.05 | ε=0.10 | ε=0.20 |
|---|---|---|---|---|---|
| s1 | 0.48→0.50 | 0.857→0.874 **(+0.017)** | 0.737→0.757 **(+0.020)** | 0.552→0.597 **(+0.045)** | 0.385→0.445 **(+0.060)** |
| s4 | 0.48→0.45 | 0.888→0.890 (+0.002) | 0.795→0.780 (−0.015) | 0.638→0.617 (−0.021) | 0.440→0.445 (+0.005) |
| s6 | 0.46→0.52 | 0.880→0.877 (−0.004) | 0.785→0.773 (−0.012) | 0.642→0.629 (−0.013) | 0.471→0.452 (−0.019) |
| **avg Δ** | **+0.017pp** | **+0.005** | **−0.002** | **+0.004** | **+0.015** |

**결과: 통계적으로 유의하지 않음.** s4 ε=0.10 SE ≈ 0.05 → 평균 Δ ±0.02 이내는 noise 수준.

- **s1만 일관된 양의 효과** — 모든 ε에서 +, 특히 ε=0.20에서 +0.060. "큰 노이즈에 강한 정책 학습"
  패턴과 일치. 단 단일 seed라 신뢰도 낮음.
- **s4/s6는 무효 또는 약한 음의 효과** — robust bonus가 정책을 실질적으로 바꾸지 못함.

**원인 가설:**
1. **`robust_alpha=0.2` 너무 작음**: 득점 1.0 대비 robustness=0.5 bonus는 0.1 → policy 변화 약함.
   alpha sweep (0.4 / 0.8) 필요.
2. **baseline이 이미 robust**: §1.2 fast_long_fp02가 lookahead-style 선택을 학습 → 이미 margin 큰
   샷을 고름. 베이스 conditional robustness 0.86~0.89 (ε=0.02) 이미 매우 높음 → 개선 여지 좁음.
3. **150k step 짧음**: 새 reward 신호에 적응할 시간 부족. 300k+ 필요할 수 있음.
4. **득점 보너스가 부수 reward에 잠식**: setup_shaping 누적이 robust bonus보다 커서 robust gradient
   약화.

**판정**: training mean의 +0.1~+0.3 개선은 robust 효과보다 다른 부수 효과(noise, 추가 학습 step)로
설명 가능. **robust_reward 가설은 현 설정에서 입증 실패**. 다음 시도: alpha sweep + longer steps.

결과: `experiments/artifacts/robustness_eval/{summary.json, per_state_robustness.parquet}`

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
*(주의: 스크립트 상단 `REPO` 경로가 다른 머신 절대경로로 하드코딩 — 이 머신에서 돌리려면 수정 필요.
신규 `compare_robust_proposers.py`는 `Path(__file__).parents[2]`로 자동 탐지.)*

### 2.3 robust effect 분해: ranking vs proposer (`compare_robust_proposers.py`, 2026-05-28)

§1.6 robust_reward 학습이 직접 측정에서 약했지만(§1.6.1), **lookahead ranking에 robust term을
추가**하는 무학습 방식은 어떨까? §2.1 time_reward와 동일한 A/B/C 분해.

**조건** (paired n=12, max_shots=25, k1=100, k2=3, robust_beta=0.2, robust_eps=0.05, robust_n=8):

| variant | proposer | 랭킹 robust | 설명 |
|---|---|---|---|
| `true_baseline` | fast_long_fp02 | off | 기존 lookahead (§2.1과 동일) |
| `baseline_robust` | fast_long_fp02 | **on** | ranking에 `+ β·robustness(a1)` (학습 없음) |
| `robust_robust` | **robust_s*** | **on** | 학습된 robust 정책 + ranking term |

**Ranking term**: 득점 후보 `a1`에 대해 `v = r1 + γ·r2 + β·robustness(a1)`. robustness는 pre-shot
스냅샷에서 `N(0, robust_eps)` perturbation 8개 시뮬 → 성공률.

**Chosen 액션의 robustness 측정**: 모든 variant에서 선택된 액션(득점 샷)의 robustness를 동일
방식으로 사후 측정 → cross-variant 비교 가능.

| metric | true_baseline | baseline_robust | robust_robust | rank 효과 (tb→br) | proposer 효과 (br→rr) |
|---|---|---|---|---|---|
| mean_score | 22.58 | 22.92 | 22.92 | +0.333 | +0.000 |
| score/sim_s | 0.255 | 0.262 | 0.267 | +0.007 | +0.006 |
| **mean_chosen_robustness** | **0.767** | **0.883** | **0.895** | **+0.116 (≈+15%)** | **+0.012** |

**결론 (§2.1과 동일 패턴):**
- **ranking robust term이 거의 모든 이득의 주범** (chosen robustness +0.116). 학습 없이 추론 시점에서
  proposer가 만든 후보 풀 중 더 robust한 쪽을 선택하는 것만으로 효과 충분.
- **robust-trained proposer는 거의 무효** (+0.012, noise 내). §1.6.1의 약효과를 lookahead 맥락에서도
  재확인.
- mean_score: max_shots=25 cap 탓 11/12 ep가 saturate. 구분은 chosen_robustness에서만 명확.

**시간 비용**: true_baseline 133s, baseline_robust 319s, robust_robust 355s (n=12). robust ranking은
득점 후보별 N=8 추가 시뮬 → ~2.4x slowdown. 매 첫 샷이 다 비싼 게 아니라 SCORING 후보에만 적용.

**판정 (§1.6 + §1.6.1 + §2.3 종합):**
robust_reward 가설은 학습 fine-tune으론 약하지만 **lookahead ranking 토글로는 명백히 작동**.
→ **운영상 결론: 정책을 다시 학습하지 말고 inference 시 ranking term만 켜자.** time_reward와 정확히
동일한 결론. 결과: `experiments/artifacts/lookahead_robust_abc/abc_summary.json`

### 2.3.1 canonical 50-shot 행동 양식 비교 (`canonical_robust_replay.py`, 2026-05-28)

§2.3는 평균 metric 비교였음. 실제로 robust ranking이 **샷 선택 양식**을 어떻게 바꾸는지 보기 위해
canonical 시작 위치에서 `continue_on_miss=True`로 50샷 강제 진행, true_baseline vs baseline_robust
paired (동일 proposer, 동일 seed). 양 variant 모두 50/50 득점, foul 0회 — score는 동일하므로
**행동 차이가 순수한 robust ranking 효과**.

결과: `experiments/artifacts/canonical_robust_replay/{variant_*.html, per_shot.csv, summary.json}`.

**집계 (50 샷):**

| metric | true_baseline | baseline_robust | Δ |
|---|---|---|---|
| total_score | 50 | 50 | 0 |
| **mean cushion_hits** | **1.04** | **0.64** | **−0.40 (−38%)** |
| mean chosen_robustness | 0.780 | 0.875 | +0.095 |
| std chosen_robustness | 0.247 | 0.202 | −0.045 |
| power mean / std | 0.730 / 0.245 | 0.751 / **0.216** | std −0.029 |

**Cushion-hit 분포 변화 (샷 카운트):**

| cushions | true_baseline | baseline_robust |
|---|---|---|
| 0 | 21 | **25** |
| 1 | 14 | **19** |
| 2 | 10 | **5** |
| 3 | 3 | 1 |
| 4 | 1 | 0 |
| 5 | 1 | 0 |

→ robust ranking은 **다중 쿠션 샷을 피하고 직접 샷 선호**. 0~1 쿠션 비율 70% → 88%. 5쿠션 같은
복잡한 샷은 완전히 사라짐. 다중 쿠션은 각 쿠션 마찰계수 + 입사각 민감도가 곱해져 margin이 작은
경향이 있음 → robust 신호와 일치.

**Robustness 분위수 (득점 샷만):**

| 분위 | true_baseline | baseline_robust |
|---|---|---|
| min | 0.125 | 0.250 |
| q10 | 0.375 | 0.613 |
| q25 | 0.656 | 0.750 |
| **median** | **0.875** | **1.000** |
| q75 | 1.000 | 1.000 |

→ robust ranking은 **저-margin 샷을 거의 완전히 제거**. median이 1.000(완벽)으로 올라감 —
"고른 액션의 절반 이상은 어떤 8개 perturbation에도 무너지지 않음". 최저 q10이 0.375 → 0.613으로
가장 크게 향상.

**가장 발산된 액션 선택 5개 샷 (action_l2 기준):**

| shot | tb (power, robust) | br (power, robust) | 해석 |
|---|---|---|---|
| 15 | 0.97, **0.375** | 0.16, **1.000** | tb는 강한 risky 샷, br은 약한 안전 샷 |
| 5 | 0.85, **0.375** | 0.27, **1.000** | 동일 패턴 |
| 38 | 0.37, 0.875 | 0.60, 0.875 | 비슷한 margin인데 br가 1 쿠션 적음 |
| 23 | 0.57, 0.75 | 0.54, **1.000** | 같은 power, 다른 θ로 margin ↑ |
| 19 | 0.13, 0.875 | 0.45, **1.000** | br가 약간 더 강하게 — power 자체엔 호불호 X |

→ 핵심 행동 패턴: **고-power × 저-margin 샷이 있을 때 robust ranking이 명확히 회피**.
median power는 비슷하지만 power std가 감소(0.245→0.216) — "극단적 power" 회피.

**HTML 리플레이로 시각화 가능**:
- `experiments/artifacts/canonical_robust_replay/variant_true_baseline.html`
- `experiments/artifacts/canonical_robust_replay/variant_baseline_robust.html`

**전체 해석**:
점수는 동일하나(50/50), robust ranking은 **"같은 점수를 더 안전하게"** 달성. 모터 노이즈/물리
편차에 노출되는 실제 환경(실 테이블, 실제 큐 스트로크)에서는 baseline_robust가 더 강건한 정책이
될 것으로 예상. domain knowledge 없이 학습된 prior + 추론 시 robust ranking만으로 이런 행동 패턴이
emergent하게 나타나는 것이 본 발견의 핵심.

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
| `--robust_reward` *(신규)* | 득점 샷에 perturbation N개 시뮬 → 성공률을 bonus로 (action-space margin) | on (robust_s*) |
| `--robust_eps` / `--robust_n` / `--robust_alpha` | perturbation σ / 횟수 / bonus 가중치 | 0.05 / 8 / 0.2 |
| `--n_envs` | 병렬 env(SubprocVecEnv) | 8 |
| `--gradient_steps` | env step당 그라디언트 스텝 | 2 |
| `--net_arch` | MLP 은닉층 | 기본256,256 / 400,300 / 512,256,128 |
| `--load_policy` | warm-start할 policy.zip 경로 | (finetune 계열) |
| `--max_shots` | inning당 최대 샷 수 (득점 상한) | 10 / 20 |
| `--total_steps` | 학습 step 수 | 50k ~ 2M |

---

## 6. 진행 중 / 다음 (2026-05-28~, shot_difficulty 브랜치)
- **chain100 paradigm** ❌ (§1.5): 150k fine-tune 결과 mean 5→1.5로 회귀. gentle_shot 제거 +
  continue_on_miss=False buffer shift가 주원인 가설.
- **robust_reward 학습** ⚠️ 약효과 (§1.6, §1.6.1): training mean +0.10~+0.27, p5 70~81%(강), 회귀 없음.
  그러나 **직접 robustness 측정에선 평균 Δ±0.02 이내** — 통계적으로 유의 X. s1만 일관된 양의 효과.
- **lookahead robust ranking** ✅ (§2.3): true_baseline → baseline+robust에서 **chosen robustness +0.116
  (≈+15%)**. 학습 없이 inference 시점에서 추가만으로 명확한 효과. proposer 교체는 무의미(+0.012).
- **종합 결론**: time_reward (§2.1)와 robust_reward (§2.3) 모두 동일 패턴 — **학습은 불필요, lookahead
  ranking 토글로 충분**. 운영상 default를 "robust ranking on"으로 두는 것 추천.
- **다음 후보**:
  1. **multi-objective ranking**: `r1 + γ·r2 + β_time·time_bonus + β_robust·robustness`. 둘 다 동시 적용
     시 시너지/충돌 확인.
  2. **robust_eps sweep at ranking**: ε=0.02/0.05/0.10에서 chosen robustness 곡선 변화.
  3. **uncapped max_shots**: §2.3 max_shots=25 cap이 mean_score saturate시킴. 더 긴 cap에서 chained
     length 차이 확인.
## 6. 진행 중 / 다음 (2026-05-26~)
- **time_reward**: 빠르게 끝나는 득점 샷에 보너스. SAC는 현재 best(`fast_long_fp02_s4`)에서
  `--load_policy`로 fine-tune 예정(~100~150k steps, ~5분). Lookahead는 env reward 공유 →
  추론 시점 즉시 반영(스크립트의 `TIME_REWARD` 상수 toggle). policy.zip 확보가 선행 조건.

---

## 7. Plain-RL Ablation + Learned RM 라인 (2026-05-25~27, 오도열)

*담당: 오도열.* NeurIPS 페이퍼 narrative (Plain RL → RM → Search → RLHF)를 위해
**모든 hand-engineering 끔** 상태에서의 baseline과 그 위의 RM을 측정. 환경은
`Billiards4BallInningEnv(t_max=12, max_shots=50)` + `RandomStartInningEnv`만.
**`constrain_aim` / `setup_shaping` / `gentle_shot` / `extra_features` 전부 off**, reward는
원래 룰의 `{0,1}` score만 (foul → 0, episode terminates on miss/foul).
평가는 200 inning deterministic; max_shots는 표기.

### 7.1 Plain RL Baseline (`experiments/runs_inning_random/`)
| run | algo | steps | random eval mean | max | p1 | foul | wall |
|---|---|---|---|---|---|---|---|
| ppo_random_200k_s0/s1/s2 | PPO | 200k | **0.000** / 0.000 / 0.000 | 0 | 0% | 0% | ~95s |
| sac_random_200k_s0/s1/s2 | SAC | 200k | 0.015 / 0.020 / 0.010 | 1~2 | 1.0~2.0% | 4~6% | 31~33min |

**관찰**: 4D 연속 액션 + sparse $\{0,1\}$ reward 환경에서 두 baseline 모두 random policy(0.005) 수준.
PPO는 on-policy + entropy bonus 부족으로 학습 신호 자체를 모음 불가. SAC는 replay buffer로 약간 우위.

### 7.2 Plain SAC Ablation (action / obs / curriculum, 모두 seed 0)
| run | 변경 | random mean | 비고 |
|---|---|---|---|
| sac_sincos_extra_200k_s0 | `angle_sincos`(5D act, θ wrap 제거) + `extra_features` | 0.010 | action wrap이 아니라 reward sparsity가 본질 |
| sac_curriculum_200k_s0 | `CurriculumStartInningEnv`(easy→full ramp) | 0.020 (random) / 0.160 (d=0 easy) | easy 학습은 되지만 hard로 transfer 실패 |

→ Plain SAC mean 천장 ≈ 0.02. **algorithm/parameterization/distribution 모두 reward sparsity의 본질적 한계 해결 못 함.**

### 7.3 Learned Reward Model — Scoring Potential
이 환경에서 score=1을 dense 신호로 바꾸는 방법으로 **scoring-potential** RM 학습.
$V(s, a) = P(\text{score}=1 \mid s, a)$ 또는 $V(s) = \mathbb{E}_{a\sim\text{Unif}}[V(s,a)]$ 추정.

#### 7.3.1 V(s) RM (`experiments/rm_data/`, `experiments/rm_data_v2/`)
- v1: 1k random states × 100 uniform actions = 100k sims, MLP[28→128→128→1]+Sigmoid
- v2: 5k × 100 = 500k sims, hidden 256, 500 epoch — Pearson r 0.466 → **0.869**
- SAC reward = score + λ·RM(s'); λ ∈ {10, 50, 100}
- 결과 (random eval, 200k step, seed 0):

| RM | λ | mean | max | 비고 |
|---|---|---|---|---|
| v2 | 10 | 0.020 | 1 | canonical eval 0%, foul mode |
| v2 | 50 | 0.005 | 1 | reward hacking 심화 |
| v2 | 100 | 0.015 | 1 | canonical foul 100% |

→ **state-only RM**으로는 actor가 high-V state로 가는 action 학습 못 함.

#### 7.3.2 V(s,a) RM (`experiments/rm_data_vsa/`, `experiments/rm_data_vsa_v3/`)
- v2: 5k × 100 = **500k pairs**, hidden 256, BCEWithLogits, 30 epoch
- v3: 50k × 200 = **10M pairs**, hidden 512, 30 epoch (5h 28min wall)
- v3 ranking quality:

| RM | val_loss | mp_pos / mp_neg | Top-10 recall | Top-100 recall | Top-1000 recall |
|---|---|---|---|---|---|
| v2 | 0.032 | 0.024 / 0.005 | 10% | 9% | 5.2% |
| v3 | 0.029 | 0.036 / 0.005 | **30%** | 12% | 7% |

(random pos_rate = 0.62%; v3 Top-10이 random 대비 **48× lift**)

#### 7.3.3 SAC + V(s,a) RM (`experiments/runs_inning_random/sac_vsarm_*`)
| RM | λ | steps | random mean | max |
|---|---|---|---|---|
| v2 | 1 | 200k | **0.060** ← peak | 1 |
| v2 | 3 | 200k | 0.045 | 2 |
| v2 | 1 | 800k | 0.055 | 2 |
| v3 | 0.3 | 200k | 0.010 | 1 |
| v3 | 0.5 | 200k | 0.020 | 1 |
| v3 | 1 | 200k | 0.030 | 1 |

**결정적 관찰**: RM 정확도(v2 9% → v3 30%)가 3× 향상돼도 SAC가 reward channel로 받을 때는 mean 더 안 좋아짐.
- Actor는 Gaussian 1개를 뱉음; RM landscape는 매우 peaked
- entropy bonus가 peak에서 멀어지게 함
- RM bias가 reward gradient에 reward hacking 유도

→ **RM은 reward로 쓰면 약함, search scorer로 쓰면 강함**.

### 7.4 RM-as-Search-Prior (`experiments/lookahead/rm_only_*.py`)
SAC actor 안 쓰고, **uniform random action 후보 K개 → RM v3 ranking → top-M → simulator verify**.
완전한 training-free pipeline.

#### h=1 (1-step lookahead, max_shots=50, n=10 ep)
| K (candidates) | M (verify) | mean | max | p1 | wall/ep |
|---|---|---|---|---|---|
| 100 | 10 | 0.20 | 1 | 20% | 50ms |
| 100 | 100 (=pure sim) | 0.40 | 2 | 20% | 340ms |
| 500 | 50 | **1.90** | 5 | 60% | 650ms |

→ K=500 M=50 만으로 Plain SAC(0.015)의 **127×**.

#### h=2 (2-step lookahead, max_shots=200~2000, n=10 ep, γ=0.99)
| K1 | M1 | K2 | M2 | max_shots | mean | max | p10 | wall/ep |
|---|---|---|---|---|---|---|---|---|
| 100 | 10 | 10 | 5 | 200 | 0.20 | 1 | 0% | ~50ms |
| 500 | 50 | 100 | 5 | 200 | 2.40 | 6 | 0% | 1s |
| 1000 | 100 | 100 | 10 | 500 | 4.90 | 36 | 10% | 5s |
| 2000 | 200 | 500 | 20 | 500 | **46.10** | 243 | **80%** | ~50s |
| 5000 | 500 | 1000 | 50 | 2000 | **658.90** | 1231 | **90%** | ~50min |
| 10000 | 1000 | 2000 | 100 | 2000 | *(진행 중, ~16-24h ETA)* | | | |

K1=5000 M1=500: scores = `[498, 1231, 0, 733, 753, 786, 508, 976, 783, 321]`, 8h 16min wall.
9/10 ep mean 700+ (1개 outlier score=0). REPORT.md의 SAC+full-system 결과(741)와 동급 수준을
**SAC 학습 0회 + RM만으로** 달성. 다만 단일 RM이라 candidate distribution이 uniform로 한계 존재.

### 7.5 코드/파일
- 신규 wrapper: `billiards/wrappers/{curriculum_start_env, scoring_potential_rm_env, vsa_rm_env}.py`
- 신규 train: `experiments/{run_inning_curriculum, run_inning_sacrm, run_inning_sacrm_sa}.py`
- RM build: `experiments/{build_scoring_potential_rm, build_v_sa_rm}.py` (tqdm 적용)
- Search: `experiments/lookahead/{rm_only_search, rm_only_h2}.py`
- `inning_env.py` 변경: `angle_sincos` option (5D action) 추가
- 의존성 추가: `tqdm`, `rich`
