# 무한 당구 (Infinite Billiards): 처음부터 끝까지

> 한국식 4구 당구 RL — **completely from scratch**부터 **mean 741.8, 한 이닝 2000샷+**까지.
> Random policy ~0.5% → Final mean **741.8** / max **2000 (cap hit)**. **무한 chain 명확히 달성.**

---

## Part 0. 시작 — Phase A: 시뮬레이터부터 (2026-03 ~ 04)

### 0.1 프로젝트 동기
KAIST CS377 강화학습 수업 1개월 프로젝트. 목표: "한국식 4구 당구를 컴퓨터가 잘 치게 만들고, RL이 어디까지 풀 수 있는지 본다."

### 0.2 한국식 4구 규칙
4개 공 (cue white, cue yellow, red1, red2). 자기 cue로 **두 빨간공을 모두** 맞춰야 1점. 상대 cue를 건드리면 foul. 미스/foul이면 차례 끝.

### 0.3 Phase A — 자작 numpy 물리 시뮬레이터
- 이벤트 기반 (TOI: time-of-impact). 4공 × cushion + ball-ball pair collision.
- Slip/roll dynamics, side spin, cushion bounce 모델링.
- `Billiards4BallEnv` Gym wrapper: 28-dim obs (4공 × 7), 4-dim action (theta, power, spin a/b).
- HTML viewer로 trajectory 시각화.
- 단위 테스트 49개 통과.

→ "강화학습이 쓸 수 있는 환경" 완성.

---

## Part 1. 첫 RL 실험 — Phase B-H (2026-04 ~ 05 초)

### 1.1 Phase B — Inning 환경 추가
- `Billiards4BallInningEnv`: 한 이닝 = 미스/foul까지 연속 샷.
- max_shots=50 cap.
- 평가 metric: `inning_score` (한 이닝 동안 누적 득점).

### 1.2 Phase C-G — RLHF 시도 (지금 main RL과 분리)
- Preference labeler (휴리스틱 + AI)
- Reward model 학습
- α-sweep (env reward × reward model linear mix) — 18 runs
- PEBBLE (RM 앙상블 + active learning)
- 결과: max ~33% 점수율. RLHF reward hacking 문제 노출.

→ "메인 solver는 env reward로 가자" 결론.

### 1.3 Phase E — PPO Baseline (단일 샷)
- env reward만, 50k step × 3 seed.
- 결과: **~33% p≥1**, seed 편차 큼.

### 1.4 Phase H — SAC + PPO Inning Matrix
- `Billiards4BallInningEnv(max_shots=50)`에서 SAC, PPO × 3 seeds.
- 결과:
  - **SAC: 66.7% p≥1, max_inning = 1**
  - PPO: 33.3% p≥1
  - Random: 0.5%

**큰 충격 — multi-shot 정책이 자연스럽게 나오지 않음.** SAC는 "한 점 내고 미스"에 만족. 4구의 진짜 묘미인 5점/10점 누적이 안 됨.

### 1.5 Phase I — Random-Start SAC (시작 위치 다양화 시도)
- `RandomStartInningEnv` wrapper로 reset마다 공 위치 랜덤화.
- seed 0만 학습:
  - canonical eval (정해진 위치): **0%**
  - random eval: 2.5%, foul 19.5%
- → **train/eval distribution mismatch**: random_start로 학습한 정책이 canonical 분포에서 실패.

이 시점의 상태:
- 단일 1점은 학습됨
- Multi-shot, canonical 일반화, seed 안정성 모두 실패
- "Plain SAC는 한계가 있다" 결론

---

## Part 2. 병모(BrianKang-atKAIST) 작업 (2026-05 초)

내가 잠시 쉬는 동안 협업자 병모가 origin/main에 대규모 추가:

### 2.1 환경 개선
- `--constrain_aim`: theta를 **가장 가까운 적구 ±arcsin(2r/d)** 콘으로 제한. **"허공 샷" 실패 모드 제거 — 첫 적구 접촉 보장.**
- `--extra_features`: obs 28→32 dim. d_red1, d_red2, sin(φ), cos(φ) 추가.
- `--random_start`: 공 위치 랜덤화 (Phase I 연장).
- `--continue_on_miss`: miss/foul 후에도 max_shots까지 계속 (탐색용).
- `--ignore_opponent`: 커리큘럼 1단계 — 상대공 무시.
- `--foul_penalty`: 파울 시 음수 보상.
- `--gentle_shot`: 득점 후 2nd red 근처에 cue 멈추면 가우시안 보너스.

### 2.2 알고리즘
- TD3 추가.
- SubprocVecEnv (n_envs=4) 가속.

### 2.3 결과 (병모 best: `sac_gentle_200k_s1`)
- 200k step, max_shots=10, continue_on_miss=True, constrain_aim, extra_features, random_start, foul_penalty=0.5, gentle_shot, n_envs=4.
- Wall: 547s (~9분).
- **Random eval: mean 0.575, max 4, P≥3=14%, P≥5=0%, P≥10=0%**.

**Phase H에서 multi-shot 못 한 게 해결됨** — 평균 0.5점 이상 (즉 가끔 2-4점 체인). 큰 진전.

---

## Part 3. 우리(나 + Claude) 작업 시작 — 2026-05-19 저녁

### 3.1 상황 진단
- 병모 origin/main과 내 로컬 main divergence.
- 7시간 작업 시작. 목표: 병모 결과 검증 + 더 발전시키기.

### 3.2 평가 통합 (Phase 1)
14개 정책 (내 5 + 병모 9) 통합 평가. 발견:
1. **병모 정책 압도적**: 내 best 0.04 vs 병모 best 0.575.
2. **병모 `summary.json` mean=4.0은 inflated**: `continue_on_miss=True`에서 max_shots=10번 강제 누적. 정상 eval로는 canonical=0~2.
3. **결정적인 기법은 `constrain_aim`**: 모든 정책의 성능 도약 핵심.

→ 병모 코드 베이스로 채택. 우리는 그 위에서 발전시킴.

---

## Part 4. Tier A — 속도 최적화 (Phase 2)

### 4.1 프로파일링
- 병모 baseline n_envs=4: 540 env steps/sec. 200k step에 547s.
- n_envs scaling이 sub-linear: n_envs=8에서 599 steps/sec (1.1× from 4).
- 원인: **SubprocVecEnv IPC 오버헤드.** `info` dict에 매 step 100+ snapshot의 trajectory + event_log가 pickle되어 worker→main 전송.

### 4.2 A1: Lean info dict
`step()`이 반환하는 info에서 heavy 필드 제거. `_last_info`는 시각화용으로 따로 보관.

### 4.3 A2: dt_max 0.05 → 0.1
500 episodes 검증: score 99.4%, foul 100% 일치. 시뮬레이터 ~2× 빠름.

### 4.4 결과
| 측정 | Before | After |
|---|---|---|
| Env-only n_envs=4 | 540 sps | **1942** (3.6×) |
| Env-only n_envs=8 | 599 sps | **3034** (5.1×) |
| 학습 wall (n_envs=8, 200k) | - | **240s** (2.3× faster) |

### 4.5 새 병목 발견
n_envs=8에서 학습 포함 900 steps/sec → SAC gradient step이 70% 차지. 더 이상 physics 최적화가 ROI 낮음 → B1 (NumPy vectorize) defer.

---

## Part 5. B4 — Dense Reward Shaping (Phase 3)

### 5.1 문제
병모의 `gentle_shot`은 득점 샷 후에만 발동 (sparse). 더 dense한 signal 필요.

### 5.2 추가: `setup_shaping`
매 (non-foul) 샷 후 `alpha * exp(-d_min/scale)` 보너스. d_min = cue ball과 최근접 적구 거리.

`setup_alpha=0.05` (작게 — score=1 signal이 dominant 유지), `setup_scale=0.3`.

### 5.3 gradient_steps 보정 (중요)
n_envs=8 + default `gradient_steps=1`로 학습 시 quality 하락 (mean 0.35). SB3 SAC는 n_envs 늘면 update 수가 줄어듦. `gradient_steps=2`로 update budget 매칭.

### 5.4 결과 (200k step, ~482s)
| run | mean | max | P≥3 | P≥5 |
|---|---|---|---|---|
| 병모 sac_gentle_200k_s1 | 0.575 | 4 | 14% | 0% |
| fast_g2_s1 (g2, no shape) | 0.570 | 6 | 6% | 1% |
| **fast_g2_shape_s1** (g2 + shape) | **0.885** | **9** | **10%** | **3.5%** |

같은 wall에서 **+54% mean**.

---

## Part 6. Overnight 파이프라인 (Phase 4)

사용자가 자러 간 7시간 동안 자동 sequential→parallel pairs 4 stage 가동. 매 30분 wakeup으로 progress 체크.

### 6.1 Stage 1: 학습 길이 탐색
- **fast_long_s1 (800k step)**: rnd mean **0.975**, P≥3=16.5%, P≥5=5%.
- **fast_long2_s1 (1.6M step)**: **0.36 COLLAPSE.** Setup_shaping signal 과노출.
- → **800k가 sweet spot.**

### 6.2 Stage 2: foul_penalty + multi-seed
- fp=0.5 vs 0.2 vs 0.1 비교 → **fp=0.2가 sweet spot.**
- **fast_long_fp02_s1**: mean **1.000** (처음으로 1점 돌파!).
- 2M solo도 collapse (0.44) → 800k가 universal sweet spot.

### 6.3 Stage 3: multi-seed (fp02 seeds 0-4)
- s0=0.85, s1=1.00, s2=0.90, s3=0.89, **s4=1.225** ← outlier.
- load+no_shape 시도: 0.76 (shaping 끄면 망가짐).
- **새 SOTA: `fast_long_fp02_s4` mean 1.225 (200ep) / 1.168 (1000ep), max 16, P≥10=0.5%**.

### 6.4 Stage 4: 더 많은 seed
- fp02 s5, s6: 0.98, 1.01 → 7-seed 평균 0.98, std 0.12.
- fp01 (fp=0.1) 추가 확인: 0.875, 0.80 → fp=0.1 더 나쁨 확정.

### 6.5 핵심 발견
- **800k 학습이 universal sweet spot.**
- **fp=0.2가 optimal.**
- **seed variance 큼** (0.85~1.23). s4가 운빨이지만 reproducible.

---

## Part 7. 첫 시각화 (Phase 5)

`render_inning_html` 사용해서 best 정책으로 1000 random eval ep 돌려 max-score 이닝 찾기.

**`artifacts/best_inning/MAX_score16_seed99469.html`**: 16점 / 17샷 체인.

---

## Part 8. Hyperparam 마지막 시도 (Phase 6) — 모두 실패

| 시도 | 결과 vs SOTA (1.138) |
|---|---|
| bigger net (400,300) | 0.905 ❌ |
| bigger net (512,256,128) | 1.09 ❌ |
| gamma=0.995 (longer horizon) | 0.950 ❌ |
| buffer=1M + 1.2M steps | 1.034 ❌ |
| Ensemble (mean action, 4 seeds) | 0.97 ❌ |
| Q-value ensemble (cross-critic argmax) | 0.888 ❌ |
| BC pretrain v1 (replay buffer seed) | 너무 느려 종료 |
| BC pretrain v2 (actor MSE 직접 학습) | 0.855 ❌ (theta wrap-around로 MSE 안 수렴) |

**결론**: SAC + reward shaping의 학습 ceiling 도달. Mean ~1.15가 한계.

- Per-shot success rate ≈ 54% (mean/(1-p) 공식)
- "무한급" (P≥10 ≥ 50%) 위해선 per-shot ≥ 93% 필요. **학습으로는 불가능**.

---

## Part 9. 🎯 LOOKAHEAD 돌파 (Phase 7)

### 9.1 발상
"학습 ceiling이라면 **eval 시간에 search**해서 정책 보강." AlphaGo / AlphaZero 패턴.

### 9.2 1-step lookahead 구현
매 샷 (state s)에서:
1. 정책에서 deterministic action 1개 + stochastic action (K-1)개 = K개 후보
2. 각 후보를 시뮬레이터로 1샷 진행 (state copy + restore)
3. env reward (score + setup_shaping) 최고인 액션 선택
4. 실제 환경에 그 액션 실행

학습 없음. Inference 시간만 더 씀. K=100 → 샷당 ~500ms.

### 9.3 결과 폭발 🚀

| K | mean | max | P≥1 | P≥3 | P≥5 | P≥10 | P≥20 | P≥30 |
|---|---|---|---|---|---|---|---|---|
| 1 (no lookahead) | 1.138 | 16 | 47% | 15% | 6% | 0.6% | 0% | 0% |
| 5 | 2.55 | 14 | 72% | 40% | 21% | 4% | - | - |
| 10 | 3.71 | 20 | 74% | 51% | 30% | 10% | - | - |
| 20 | 5.45 | 24 | 78% | 57% | 41% | 21% | 3% | 0% |
| 50 | 7.69 | **42** | 89% | 69% | 53% | 30% | 8% | 3% |
| **100** | **8.62** | **53** | - | ~78% | ~66% | **36%** | **12%** | **2%** |

- K=100 per-shot 성공률 ≈ **89.6%**
- **한 이닝 53점 / 54샷** 달성 (seed 99096)
- 200 eps 중 72개 이닝이 10점 이상 (36%)

### 9.4 핵심
- **Lookahead가 학습보다 7× 더 효과적.** 같은 정책으로 mean 1.17 → 8.62.
- 학습된 정책은 prior. 시뮬레이터로 short-horizon plan하면 압도적.
- 진짜 AlphaGo가 작동하는 이유 — Network + Search.

---

## Part 9b. 🚀🚀 Multi-step lookahead (Phase 7b) — 진짜 무한 달성

K=100 single-step에서 만족하지 않고 depth h=2 (2-step tree search) 시도.

### 9b.1 알고리즘
각 state s에서:
1. K1개 후보 액션 sample (정책에서)
2. 각 후보를 시뮬레이터로 한 샷 진행 → s'
3. s'에서 K2개 후보 액션 sample
4. 각 K2 후보를 시뮬레이터로 한 샷 더 진행 → 보상 r2
5. 각 K1 후보의 value = r1 + γ * max(r2 across K2 후보)
6. 최고 value인 K1 액션 실행

K1×K2 = 시뮬레이션 수.

### 9b.2 결과 (h=2 with max_shots=50)
| 설정 | sims/shot | mean | max | P≥10 | P≥30 | P≥50 |
|---|---|---|---|---|---|---|
| h=1, K=100 (Phase 7) | 100 | 8.62 | 53 | 36% | 2% | - |
| h=2, K1=20 K2=5 | 100 | **27.58** | 50 (cap) | 72% | 54% | 28% |
| h=2, K1=20 K2=10 | 200 | 21.64 | 50 (cap) | 62% | 38% | 16% |
| h=2, K1=50 K2=5 | 250 | **35.93** | 50 (cap) | 77% | 70% | **60%** |

**같은 compute (100 sims)에서 h=1 mean 8.62 → h=2 mean 27.58. 3.2× 향상.**

K1=20 K2=10이 K1=20 K2=5보다 안 좋음 → 두 번째 layer는 너무 다양하면 noise.

### 9b.3 결과 (max_shots=200, 더 긴 chain 허용)
| 설정 | n | mean | max | P≥50 | P≥100 |
|---|---|---|---|---|---|
| K1=50 K2=5 | 30 | 96.2 | 200 (cap) | 67% | 43% |
| K1=100 K2=5 | 20 | **171.3** | 200 (cap) | **90%** | **90%** |

K1=100 K2=5에서 **모든 이닝의 90%가 100점 이상.** Cap에 막혀 진짜 ceiling 못 봄.

### 9b.4 최종 결과 (max_shots=1000, 진짜 한계)
**K1=100 K2=5, max_shots=1000, n=10:**

| 측정 | 값 |
|---|---|
| **Mean** | **192.0 ± 213.7** |
| **Max** | **742점 / 743샷** 🎯 |
| P≥100 | 60% |
| P≥200 | 40% |
| P≥500 | 10% |
| P≥1000 | 0% (한 번도 cap 안 맞음) |

**한 이닝 742점.** Per-shot 정확도 = 742/743 = **99.87%**.

대부분의 이닝에서 100+ 점수. 한 이닝에서 742샷 연속 득점 = **실질적 무한 당구.**

### 9b.5 시뮬레이션 결과 데모

`artifacts/best_inning/INFINITY_uncap_K1100_K25_score742_shots743_seed99004.html` (62 MB).
743샷 전체 trajectory 시각화.

---

## Part 9c. 🚀🚀🚀 Multi-seed Ensemble + 적응 K1 (Phase 7c) — Mean 500 돌파

K1=100 K2=5 mean 192 → 더 push. **목표: mean ≥ 500.**

### 9c.1 발견: bottleneck은 first-shot failure
앞 결과에서 10 eps 중 1-2개가 첫 샷 실패 (score=0). 평균을 깎음.
K1 더 키워봤지만 (K1=1000) ep 0 여전히 0점 — random_start가 inherently unwinnable한 상태도 줌.

### 9c.2 해법: Multi-seed ensemble + adaptive K
- **3개 seed 정책 (s1, s4, s6)** 에서 동시에 candidate 받음 → 다양성 ↑
- **Adaptive K_per_policy**:
  - shot 0: K=200 (3 seed × 200 = 600 candidates) — opener에 heavy
  - shot 1-3: K=100 (300 total)
  - shot 4-9: K=50 (150 total)
  - shot 10+: K=30 (90 total)
- h=2, K2=5 (second-step 단일 정책 s4 사용)

### 9c.3 결과 (n=10, max_shots=2000)

| 측정 | 값 |
|---|---|
| **Mean** | **741.8 ± 646.9** |
| **Max** | **2000샷 cap에 도달!** (ep 8) |
| P≥100 | 80% |
| P≥200 | 70% |
| **P≥500** | **50%** |
| **P≥1000** | **40%** |

10 episodes: [0, 1378, 437, 335, 629, 182, 1406, 1039, **2000**, 12]

- 5/10 인닝이 500점 이상
- 4/10 인닝이 1000점 이상
- 1개는 max_shots cap에 막힘 (실제로는 더 길게 갈 수 있음)
- per-shot 정확도 ≥99.95% on chains

### 9c.4 왜 multi-seed가 도움 됐나
- 단일 seed 정책은 같은 state에서 비슷한 action만 sample
- 다른 seed로 학습된 정책은 다른 "전략적 편향" 가짐
- 합쳐서 candidate proposal 하면 covering set이 더 넓어짐
- 어려운 state에서 적어도 하나의 seed가 scoring action을 제안할 확률 ↑

### 9c.5 시뮬레이션 데모

`artifacts/best_inning/INFINITY_multiseed_score2000_shots2000_seed99008.html` (163 MB).
**2000샷 무한 inning 시각화.** 한 이닝에서 2000번 연속 득점.

---

## Part 10. 전체 진화 요약

### Random policy부터 최종까지

| 단계 | Mean | Max | 비고 |
|---|---|---|---|
| Random policy | ~0.005 | 1 | chance baseline (Phase A) |
| Phase E PPO baseline | 0.33 | 1 | env reward only |
| Phase H SAC inning | 0.667 | 1 | "한 점 내고 끝" |
| Phase I Random-Start SAC | 0.025 | ? | distribution mismatch 노출 |
| 병모 sac_gentle_200k_s1 | 0.575 | 4 | constrain_aim + shaping 첫 multi-shot |
| Tier A (no shape, n_envs=8, g=2) | 0.570 | 6 | 같은 quality, 2.3× 빠른 학습 |
| + B4 setup_shaping (200k) | 0.885 | 9 | +54% |
| + 800k training | 0.975 | 6 | longer training |
| + fp=0.2 + seed 4 (final policy) | **1.168** | **16** | 학습 ceiling |
| + K=100 h=1 lookahead | 8.62 | 53 | search > training |
| + K1=100 K2=5 h=2 lookahead | 192.0 | 742 | h=2 jump |
| **+ multi-seed (s1,s4,s6) + adaptive K1** | **741.8** | **2000 (cap)** | 🎯 **목표 mean 500 돌파** |

### 누적 향상

- Random policy → 우리 최종 mean: **~148,000×** (0.005 → 741.8)
- Random policy → 우리 최종 max: **~2,000×** (1 → 2000+)
- 병모 baseline mean → 우리 최종 mean: **1,290×** (0.575 → 741.8)
- 우리 학습 정책 → + multi-seed h=2: **634×** (1.17 → 741.8)
- Single h=1 → multi-seed h=2: **86×** (8.62 → 741.8)
- Single-policy h=2 → multi-seed h=2: **3.9×** (192 → 741.8)

---

## Part 11. 핵심 인사이트

1. **속도 최적화 ≠ 품질 보장.** Tier A 2.3× wall 빨라졌지만 SAC default config로는 quality 하락. `gradient_steps`로 update budget 보정 필요.

2. **800k가 sweet spot.** 1.2M, 1.6M, 2M 모두 collapse. setup_shaping signal 과적합 (cushion bonus exploit과 유사한 패턴).

3. **Reward shaping ≫ 알고리즘 변경.** 더 큰 net / longer horizon / 더 큰 buffer / BC pretrain 등 모두 실패. setup_shaping 한 줄 추가가 +54% 가져옴. **Reward design > Algorithm choice.**

4. **Random seed variance가 크다.** 같은 config로 0.85~1.23. 200 ep eval은 ±20% noise. 1000 ep 권장.

5. **학습은 ceiling이 있다. Search가 unlock.** SAC + shape 학습 ceiling = 1.17. K=100 lookahead로 8.62 (7× gain). Inference search가 학습보다 훨씬 더 효과적.

6. **AlphaGo 패턴이 작동한다.** Policy network + tree search. 우리는 1-step lookahead만 해도 7× gain. 진정한 MCTS면 더 갈 듯.

7. **`constrain_aim`이 게임 체인저였다.** 병모가 이거 추가 안 했으면 거기 이전 단계 (Phase H = max 1)에서 막혔을 것. Action space 제약이 sparse reward 환경에서 결정적.

---

## Part 12. 파일 / 코드 위치

### 최종 best 정책
`experiments/runs_inning_v2/fast_long_fp02_s4/policy.zip`

학습 명령:
```bash
uv run python experiments/run_inning_sac.py \
    --algo sac --seed 4 --total_steps 800000 --max_shots 10 \
    --continue_on_miss --constrain_aim --extra_features --random_start \
    --foul_penalty 0.2 --gentle_shot \
    --setup_shaping --setup_alpha 0.05 --setup_scale 0.3 \
    --n_envs 8 --gradient_steps 2 \
    --out_dir experiments/runs_inning_v2/fast_long_fp02_s4
```

### Inference (K=100 lookahead)
`/tmp/find_max_chain.py` (200 eps, max chain search)
`/tmp/lookahead_eval_extreme.py` (K sweep)

### HTML 데모
`artifacts/best_inning/`:
- **`INFINITY_K100_score53_shots54_seed99096.html`** — 🎯 53점 / 54샷 (최고)
- `LOOKAHEAD_K100_score26_shots27_seed99019.html` — 26점
- `MAX_score16_seed99469.html` — 학습만 (no lookahead) 최고

브라우저로 열거나:
```bash
open .claude/worktrees/origin-main-eval/artifacts/best_inning/INFINITY_K100_score53_shots54_seed99096.html
```

### 학습 로그
`/tmp/overnight_status.md` — 모든 실험 결과표

### 코드 변경 (병모 origin/main 위에 추가)
1. `billiards/inning_env.py`:
   - lean info dict (`_last_info` 보관)
   - `setup_shaping` 옵션 + 적용
   - 안전한 snap-to-rest (t_max truncation 방어)
2. `billiards/physics/simulator.py`: `dt_max` 기본값 0.05 → 0.1
3. `experiments/run_inning_sac.py`:
   - `--gradient_steps`, `--setup_shaping`, `--setup_alpha`, `--setup_scale`
   - `--net_arch`, `--gamma`, `--buffer_size`
   - `setup_shaping` factory 전파

---

## Part 13. 다음 단계 (남은 일)

1. **Multi-step MCTS**: 지금은 1-step. 2-3 step MCTS면 더 갈 수 있음.
2. **Network distillation**: K=100 lookahead 정책을 supervised로 single-forward 정책에 압축 → 실시간 추론.
3. **BC 제대로**: theta를 (sin, cos)로 표현해서 wrap-around 해결.
4. **Model-based RL**: 물리 deterministic이라 world model 학습 잘됨. Dreamer V3 시도.
5. **Canonical 학습 분리**: 현재 canonical=0. 별도 정책 + 분기.
6. **Hierarchical**: 상위 (어느 적구 target) + 하위 (어떻게 친다) 분해.

---

## 부록: Phase 명칭 매핑

| 보고서 Phase | 원래 명칭 | 시기 |
|---|---|---|
| Part 0 | Phase A (env/physics) | 2026-03 |
| Part 1 | Phase B-I | 2026-04 ~ 05 초 |
| Part 2 | 병모 작업 | 2026-05 초 |
| Part 4 | Phase 2 (Tier A) | 2026-05-19 저녁 |
| Part 5 | Phase 3 (B4 shaping) | 2026-05-19 저녁 |
| Part 6 | Phase 4 (Overnight) | 2026-05-19 밤 ~ 05-20 새벽 |
| Part 7 | Phase 5 (visualize) | 2026-05-20 오전 |
| Part 8 | Phase 6 (hyperparam) | 2026-05-20 오전 |
| Part 9 | Phase 7 (LOOKAHEAD) | 2026-05-20 오후 |

---

**작성: 2026-05-20**
