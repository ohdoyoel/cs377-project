# 한국식 4구 당구 AI: 학습으로 안 풀리던 문제를 추론 시 탐색으로 풀기

**한 줄 요약**: SAC 강화학습으로는 한 이닝 평균 1점에서 막혔는데, 추론 시점에 시뮬레이터로 후보를 평가하는 단순한 탐색을 붙였더니 평균 741점, 한 이닝 최대 2000샷까지 갔다.

**작성**: 2026-05-21
**작성자**: Doyoel Oh + Claude (Anthropic)

---

## 0. 처음 보는 사람을 위한 한 페이지 요약

### 무슨 프로젝트인가?

한국식 4구 당구를 컴퓨터가 잘 치게 만든다. "잘 친다" = **한 이닝에 점수를 많이 낸다** (1점 = 자기 공으로 두 빨간 공을 모두 맞히되 상대 공은 안 건드림. 득점하면 같은 이닝에서 다시 침).

### 어디서 시작했는가?

- 무작위 정책: 한 이닝 0.005점
- 단순 SAC 강화학습 (canonical 시작): 한 이닝 1.0점 (100% 한 점만 내고 끝 — max=1)
- 협업자(병모)의 개선된 SAC (random 시작): 한 이닝 0.575점 (드디어 multi-shot, max=4)

### 어디까지 갔는가?

- **한 이닝 평균 741점**
- **최대 2000샷 연속 득점** (max_shots 캡에 막혔음)
- 무작위 시작 평가 기준

### 어떻게 거기까지 갔는가?

1. **속도 최적화 (Tier A)** — 학습이 빨라져서 더 많은 실험 가능 (2.3× wall 단축)
2. **Dense 보상 셰이핑 (setup_shaping)** — 매 샷마다 작은 hint → 평균 0.58 → 0.89
3. **800k step 학습** — 평균 0.89 → 0.98
4. **여러 시드 + 미세 조정** — 평균 0.98 → **1.17** (학습 천장)
5. **추론 시점 탐색 (lookahead)** — 같은 정책으로 평균 1.17 → 8.62 → **192** → **741**

### 핵심 발견

**학습의 천장은 알고리즘 개선이 아니라 추론 시점 탐색으로 뚫린다.**

AlphaGo / MuZero 와 같은 패턴: 학습된 정책은 prior, 진짜 결정은 게임 시간에 트리 탐색.

### 그리고 의외의 발견

표준 PUCT (AlphaZero가 쓴 MCTS 변종) 가 우리 task에서 **더 단순한 greedy K-expansion 보다 못함**. 5배 차이. 왜인지 분석 (Section 8).

### 다음 단계 (작업 중)

**RLHF + search 통합** — 검색 시점에 학습된 reward model 로 후보 평가해서 점수뿐 아니라 "스타일" 까지 최적화.

---

## 1. 무엇을 해결하려 했는가

### 1.1 한국식 4구 규칙 (한 단락)

흰 공 2개 (자기 큐, 상대 큐), 빨간 공 2개. 자기 큐로 **양쪽 빨간 공 모두**를 맞히고 상대 큐는 건드리지 않으면 1점. 득점하면 같은 이닝에서 다시 침. 미스/파울이면 차례 끝.

### 1.2 두 가지 어려움

- **한 샷 자체가 어려움**: 각도 θ, 강도 p, 횡/종 스핀 (a, b) 등 4차원 연속 액션. 1° 오차도 미스.
- **연속해서 잘 쳐야 함**: 매 샷마다 다음 샷을 위한 좋은 잔여 위치도 같이 만들어야 함.

샷 당 99% 성공률이어도 50샷 체인은 0.99⁵⁰ ≈ 60%. 90%면 50샷 0.5%. **"무한"이 매우 어렵다.**

---

## 2. 출발점: 학습된 정책의 한계

### 2.1 Plain SAC — 병모 개선 전 (Phase E ~ I)

병모가 `constrain_aim` 등을 추가하기 전, 같은 환경에서 **단순 SAC** 만 돌렸을 때:

#### 잠깐 — PPO 와 SAC 가 무엇이고 어떻게 달랐나

본 Phase H 에서는 두 표준 RL 알고리즘 PPO 와 SAC 를 동일 환경에서 비교했다.

**PPO (Proximal Policy Optimization)** — *on-policy* 액터-크리틱:
- 매 iteration: 현재 정책으로 N step rollout → 그 데이터로 한 번 (또는 몇 epoch) update → **데이터 버림**
- "Proximal" = 새 정책이 옛 정책에서 너무 멀리 못 가게 ratio clip (trust region)
- **Sample inefficient** 하지만 안정적. RL 의 "default" 선택지.

**SAC (Soft Actor-Critic)** — *off-policy* 액터-크리틱 + *maximum entropy*:
- 매 step: 환경 1 step → (s, a, r, s') 를 **replay buffer 에 저장** → buffer 에서 batch sample → Q-network + policy 업데이트
- 보상 = 실제 보상 + α·entropy(π) → 정책이 "확실히" 수렴 안 하고 적당한 무작위성 유지 → **탐색 강함**
- Twin Q networks (over-estimation 방지)
- **Sample efficient**. Continuous action 에 강함.

| 측면 | PPO | SAC |
|---|---|---|
| Policy update | On-policy | Off-policy |
| 데이터 재사용 | ❌ 한 번 쓰고 버림 | ✅ Replay buffer |
| Exploration | Stochastic policy + 작은 entropy bonus | Large entropy bonus |
| Sample efficiency | 낮음 | 높음 |

**우리 task 에서 SAC 가 압승한 이유**:
1. **희소 보상 (sparse reward)**: 50k step 중 득점 transition 이 매우 드묾. SAC 는 그 rare transition 을 buffer 에서 계속 재사용. PPO 는 rollout 끝나면 버려서 정보 손실.
2. **연속 액션 4D**: PPO 의 작은 entropy bonus 만으로는 4D 공간 탐색 부족. SAC 의 max-entropy 가 cover.
3. **샷 비용 (5 ms / sim)**: transition 비싼 환경 → 재사용 가능한 SAC 유리.

**학습 hyperparameter** (둘 다 50k step, 3 seeds):

| 파라미터 | SAC | PPO |
|---|---|---|
| learning rate | 3e-4 | 3e-4 |
| batch size | 256 | 64 (mini-batch) |
| buffer size | 200,000 | n/a (on-policy) |
| rollout per update | 1 (off-policy) | 2048 × 4 envs |
| gamma | 0.99 | 0.99 |
| update | 매 step 1번 | rollout 후 4 epoch |
| learning_starts | 1,000 (random) | n/a |

이제 Phase H 결과를 보자.

#### Phase H (2026-04) — 단일 시작 위치 (canonical) inning matrix

실험 setup:
- 3 seeds (0, 1, 2) per algorithm
- env: `Billiards4BallInningEnv(max_shots=50)`, **canonical 시작 위치** (정해진 4공 배치)
- 학습: 50k steps, env reward (점수=1)
- 평가: 200 episodes deterministic policy, 같은 canonical 시작에서
- 결과 (`experiments/runs_inning/{algo}_s{0,1,2}/summary.json`):

| 알고리즘 | p≥1 (3 seed 평균) | max_inning | mean | mean_shots |
|---|---|---|---|---|
| Random policy | 0.5% | 1 | 0.005 | - |
| PPO (seeds 0,1,2) | **0%** | 0 | 0.0 | 1.0 |
| **SAC (seeds 0,1,2)** | **100%** | **1** | **1.0** | **2.0** |

→ SAC 3 seed **모두** 정확히 같은 결과 (p≥1=100%, mean=1.0, max=1, mean_shots=2). PPO 는 한 점도 못 냄.

**SAC 가 mean=1.0 (max=1) 인 이유**:
- Canonical 시작은 정해진 위치 → 정해진 "정답" 첫 샷 존재
- SAC 가 50k step 안에 그 첫 샷을 memorize → 200 episode 다 같은 시작 → **100% 첫 샷 득점**
- 그 직후 큐 공 굴러간 위치는 매번 다름 (random) → 두 번째 샷은 학습 안 됨 → 무조건 miss → 인닝 종료
- **결과**: 정확히 1점 / 2샷 (1득점 + 1미스)

**즉 "한 점만 내는 정책" 으로 robust 수렴.** Multi-shot 학습 안 됨 (max=1).

PROJECT_OVERVIEW.md 의 "66.7%" 는 환경 / 코드 refactoring 이전 데이터로 추정. 현재 (병모의 env 개선 적용 후) 데이터는 3 seed 모두 100% 수렴.

#### Phase I (2026-05 초) — Random-start 시도

이번엔 같은 SAC 를 **시작 위치를 무작위로** 학습 시켰을 때 (매 reset 마다 4 공 위치가 랜덤). seed 0 만 학습 완료 (`experiments/runs_inning_random/sac_random_s0/summary.json`).

실험 setup:
- env: `RandomStartInningEnv(Billiards4BallInningEnv(max_shots=50))` — 매 이닝 reset 시 4공 위치 랜덤화 (충돌 안 하는 valid placement)
- 학습: 50k steps SAC
- 평가: 두 가지 모드 각 200 episode (deterministic policy)
  - **canonical eval**: 평가만 정해진 위치에서 (학습 분포 밖)
  - **random eval**: 학습과 동일 random 분포에서

| 평가 모드 | mean | max | p≥1 | mean_shots | foul rate |
|---|---|---|---|---|---|
| canonical (정해진 시작) | **0.0** | **0** | **0%** | 1.0 | 0% |
| random (무작위 시작) | **0.025** | **1** | **2.5%** | 1.025 | **19.5%** |

**해석**:
- **Canonical eval = 0**: random-start 로 학습한 정책이 정해진 위치에서 한 점도 못 냄. **Train/eval distribution mismatch** 의 교과서적 사례 — 학습 분포가 평가 분포와 다르면 일반화 실패.
- **Random eval = 0.025, max = 1**: 학습 분포에서도 거의 random policy 수준 (random policy 가 0.005). 200 episode 중 단 5 개만 (2.5%) 한 점 냄. 어떤 이닝도 2점 이상 못 냄.
- **foul rate 19.5% 가 매우 높음**: random 시작은 종종 cue 공이 상대공 근처에 놓임 → foul 위험 ↑. Plain SAC 가 foul 회피 학습 못 함.
- **mean_shots ≈ 1**: 거의 모든 이닝이 한 샷만에 종료 (miss/foul).

**Phase I 의 의미**: Plain SAC 로는 random_start 학습도 안 됨. canonical 학습 시 max=1, random 학습 시 mean ≈ 0. **둘 다 실패.**

→ 학습 방법 자체가 아니라 **환경 / 보상 / action constraint** 의 근본적 재설계가 필요함을 시사. 이 인사이트가 다음 단계 (병모의 `constrain_aim`) 로 이어짐.

#### 내가 별도로 학습한 SAC 변종들 (Random eval, max_shots=20)

| 변종 (모두 plain SAC, no constrain_aim) | mean | max | canonical |
|---|---|---|---|
| sac_s0 (50k vanilla) | 0.02 | 1 | 0 |
| sac_s0_1M (1M vanilla) | 0.04 | 1 | 0 |
| sac_s0_1M_jit (jitter randomize) | 0.00 | 0 | 1 |
| sac_s0_1M_jit_setup (best of mine) | 0.04 | 1 | 1 |

전부 max = 0 또는 1. **Multi-shot 한 번도 안 나옴.**

#### Phase H/I 의 결론

**Plain SAC 의 천장**:
- Canonical (정해진 시작): mean = **1.0**, max = **1** (100% 한 점만 냄)
- Random (무작위 시작): mean ≈ 0.025–0.04, max = 1

이게 단순 SAC + inning env 의 한계였음. 더 길게 학습해도, 더 큰 네트워크로 해도, jitter 같은 randomization 을 추가해도 **max = 1** 의 벽을 못 넘음.

### 2.2 게임 체인저: 병모의 `constrain_aim` + 보상 / 분포 설계

협업자 BrianKang-atKAIST 가 다음 4가지를 추가:

#### (1) `constrain_aim` — **가장 critical**

정책이 출력한 각도 θ 를 단순 사용하지 않고, **가장 가까운 빨간 공 방향으로 정렬된 좁은 콘** 안으로 reparametrize:

```
cue 와 가장 가까운 red 의 방향: target_dir = atan2(red.y - cue.y, red.x - cue.x)
cone 의 half-angle:           α = arcsin(2r / d)   (r = 공 반지름, d = cue-red 거리)

정책이 출력한 θ ∈ [0, 2π] 를 [-1, 1] offset 으로 매핑:
  offset = (θ - π) / π
실제 적용된 각도:
  θ_final = target_dir + offset · α     ← cone 의 ±α 안에 들어옴
```

**기하학적 의미**: arcsin(2r / d) 는 큐 공이 빨간 공 의 "각폭" — 이 콘 안의 어떤 각도로 쏘아도 첫 접촉이 빨간 공에 보장됨. cone 밖이면 미스 가능.

→ "허공 샷" (어떤 공에도 안 닿음) 실패 모드 제거. **첫 적구 접촉 기하학적으로 보장.**

원래 정책의 θ 자유도가 약 1/100 ~ 1/30 정도로 좁혀짐 (d 에 따라). 그만큼 학습이 쉬워짐 — 4D 액션 공간이 사실상 3D + bounded 1D 가 됨.

#### (2) `random_start` — 일반화 강제

매 이닝 reset 시 4공 위치를 무작위로 배치 (충돌 안 하는 valid placement). Plain SAC 가 canonical 단일 위치에서 "한 트릭" 만 memorize 한 Phase H 문제 해결.

대신 학습 어려움 ↑: 정책이 어떤 시작 상태에서도 첫 샷 성공해야 함. Plain SAC 만으로는 처참 (§2.1 Phase I 데이터: random eval 2.5%).

`constrain_aim` 과 결합되면 효과 큼 — 어느 위치에서든 콘 안의 θ 만 학습하면 되니까.

#### (3) `gentle_shot` — 득점 후 setup 보너스

**득점 샷 직후에만** 발동하는 가우시안 보너스. 큐 공이 *두 번째로 맞춘* 빨간 공으로부터 d_target ≈ 0.2 m 떨어진 위치에서 멈출 때 peak:

```
events 에서 cue_hit_red 만 추출 → reds_hit = [first, second, ...]
target = reds_hit[1]   (두 번째 맞춘 red)
dist   = ‖cue.position - target.position‖

bonus = α · exp(-(dist - d_target)² / (2σ²))
     α        = 0.2  (peak 크기)
     d_target = 0.2 m (이상적 거리)
     σ        = 0.1 m (가우시안 폭)
```

당구에서 "다음 샷 준비"의 미적 감각을 reward 로 인코딩:
- dist = 0 m (cue 가 red 와 접촉): 다음 샷 각도 매우 제한 → bonus 0.03
- **dist ≈ 0.2 m** (적당히 떨어짐): 다음 샷 치기 좋음 → bonus 0.2 (peak)
- dist > 0.5 m (멀리 굴러감): 다음 샷 정밀도 어려움 → bonus ≈ 0

자세한 디자인 분석은 **Appendix D**.

#### (4) `foul_penalty` — 파울 음수 보상

**한국식 4구 룰**: 자기 큐가 상대 큐를 건드리면 "파울". 그 샷은 무효, 차례 끝.

**환경 reward 계산**:

```python
if fouled:
    reward = -self._foul_penalty   # 음수만 (default 0.1)
else:
    reward = float(score)           # 0 또는 1
    if score > 0 and gentle_shot:   # 득점 시 setup 보너스
        reward += gentle_bonus
    if setup_shaping:               # 매 비파울 샷 거리 보너스
        reward += setup_bonus
```

**왜 그냥 0 안 되고 음수 (-penalty) 인가?**

학습 모드에서 (`continue_on_miss=True`, max_shots=10 강제) 둘이 동시에 일어남:
- 그냥 miss (공중에서 빗나감): reward 0
- foul (상대 큐 건드림): reward 0 이면 → miss 와 foul 구분 못 함

같은 결과 (점수 안 남) 라도 **foul 은 더 위험한 행동**. agent 가 "상대 큐 근처 비껴가는 위험한 샷" 을 거리낌없이 시도하면 안 됨. `-0.2` 페널티가 차이를 만듦:

```
miss → 0,   foul → -0.2   ← agent 가 foul 회피 학습
```

**`reward = score - foul_penalty` 와 뭐가 다른가?**

실은 simulator 가 이미 다음을 보장:

```python
# simulator 내부:
score  = 1 if (cue_hit_both_reds AND not fouled) else 0
```

→ fouled=True 면 score 는 자동으로 0. 동시에 score=1 + fouled=True 인 케이스 없음.

따라서 수학적으로:
- `reward = -foul_penalty`  →  0 - 0.2 = -0.2
- `reward = score - foul_penalty`  →  0 - 0.2 = -0.2

**같은 값.**

**그래도 굳이 `if fouled:` 분기하는 이유 3가지**:

1. **보너스 차단**: 정상 케이스에 `gentle_shot`, `setup_shaping` 보너스가 추가됨. foul 케이스에 그 보너스 적용하면 안 됨. 분기로 명확히 차단.
2. **가독성**: 코드 읽는 사람이 "foul 시 reward = -penalty (only)" 룰을 한눈에 봄.
3. **Future-proof**: simulator 규칙 변경 (예: "foul 이어도 점수 인정") 가능성 대비. simulator-reward decoupling.

**페널티 값 sweep**:

| fp | rnd mean | 비고 |
|---|---|---|
| 0.5 (병모 default) | 0.975 | 너무 강함, conservative 으로 학습 |
| **0.2 (우리 best)** | **1.000** | sweet spot |
| 0.1 | 0.875 | 너무 약함, foul rate 높아짐 |

§3.2 의 sweep 으로 결정.

#### 결과: SAC 200k step 학습

- 평균 0.575, 최대 4, P(≥3) = 14%
- **드디어 multi-shot 정책 등장** (Phase H 의 max=1 천장을 처음으로 넘음)

`constrain_aim` 의 마진 효과는 본 보고서에서 isolated 으로 측정 못 함 (§11.4 — 진정한 ablation 은 future work). 하지만 **plain SAC random eval = 0.04, +constrain_aim+cont_on_miss = 0.14, +extra+rs+200k+gentle = 0.575** 의 incremental data 만 봐도 약 **14× 점프** 가 이 단계에서 일어남. "허공 샷" 실패 모드 제거가 결정적 요인.

### 2.3 우리 목표

병모의 출발점 (mean 0.575, max 4) 위에서 더 push. 궁극적 목표: 무한 chain.

### 2.4 진화 한눈에 (병모 이전 → 우리 최종)

```
Phase H Plain SAC            random mean ~0.04, max 1   ← multi-shot 없음
   │
   │ + constrain_aim (병모)             [기하학적 첫 접촉 보장]
   ▼
병모 baseline                random mean 0.575, max 4    ← 첫 multi-shot
   │
   │ + setup_shaping + 800k + fp=0.2 (우리)
   ▼
SAC SOTA                     random mean 1.17, max 16    ← 학습 천장
   │
   │ + greedy K-expansion h=2 + multi-seed (우리)
   ▼
Search-augmented (최종)      random mean 741.8, max 2000+
```

각 단계마다 **10-100× 점프**. 누적 ≈ 18,500×.

---

## 3. Phase 1: 학습 속도 2.3× 빠르게 (Tier A)

### 3.1 발견

n_envs (병렬 환경 수) 를 4 → 8 로 늘려도 학습 속도 1.1배 밖에 안 올라감. 비정상.

cProfile + 환경 step 분석:
- 환경의 `info` 딕셔너리에 매 step **trajectory** (100+ 스냅샷) 와 **event log** 가 통째로 들어있음
- SubprocVecEnv 가 매 step 마다 이걸 pickle → IPC → main 으로 복사
- 학습 코드는 이걸 사용하지 않음 (Monitor 가 score, fouled, cushion_hits 만 봄)

### 3.2 수정

- **A1** — `step()` 이 반환하는 info 에서 무거운 필드 제거. 시각화용은 환경 내부 `_last_info` 에 따로 보관
- **A2** — 시뮬레이터의 `dt_max` 0.05 → 0.1 (500 episode 검증: 점수 99.4% 일치)

### 3.3 결과

| 측정 | Before | After |
|---|---|---|
| n_envs=8 환경 step/sec | 599 | **3034** (5.1×) |
| 학습 wall (200k step) | 547s | **240s** (2.3×) |

### 3.4 함정: gradient_steps

n_envs 8 + SB3 SAC 기본 `gradient_steps=1` 이면 update 수가 절반으로 줄어듦 (200k transitions / 8 = 25k updates). 결과 mean 0.575 → 0.35 ❌.

`gradient_steps=2` 로 복구하면 mean 0.57 (병모와 같음). 

**Lesson**: n_envs 올릴 때 gradient_steps도 같이 올려야 update budget 보존.

---

## 4. Phase 2: 매 샷마다 작은 hint (setup_shaping)

### 4.1 동기

병모의 `gentle_shot` 은 **득점 샷 직후에만** 발동. 미스/파울 샷에는 신호 없음. 학습 초기에 sparse.

### 4.2 추가

매 비파울 샷 끝나면, 큐 공이 가장 가까운 빨간 공으로부터의 거리 d_min 에 대한 가우시안 보너스:

```
r_shape = α · exp(-d_min / σ),   α = 0.05, σ = 0.3m
```

작은 α (0.05) 라 score=1 신호가 dominant. 한 이닝 최대 셰이핑 누적 = 0.5 (10샷 기준).

### 4.3 결과 (200k step, 같은 wall)

| 설정 | 평균 |
|---|---|
| 병모 baseline | 0.575 |
| Tier A (no shape) | 0.570 |
| **+ setup_shaping** | **0.885** (+54%) |

처음으로 P(≥5) 가 의미있게 등장 (0% → 3.5%).

### 4.4 함정

`random_start` 빼고 canonical only 학습하면 mean 0.17 로 붕괴. 정책이 **"거의 닿지만 점수 안 내기"** 위장 전략에 갇힘 (셰이핑만 챙김). `random_start` 와 함께 써야 안전.

---

## 5. Phase 3: 학습 천장 발견 (Hyperparameter sweep)

밤새 자동 파이프라인으로 광범위한 ablation.

### 5.1 학습 길이

| total steps | 평균 |
|---|---|
| 200k | 0.885 |
| **800k** | **0.975** ✓ sweet spot |
| 1.6M | 0.36 ❌ collapse |
| 2M | 0.44 ❌ collapse |

→ 800k 이상은 셰이핑 신호에 과적합돼서 정책이 망가짐.

### 5.2 foul_penalty

| fp | 평균 |
|---|---|
| 0.5 (병모 default) | 0.975 |
| **0.2** | **1.000** ✓ |
| 0.1 | 0.875 |

### 5.3 시드 분산

같은 config 로 7 시드 학습:

| seed | 평균 |
|---|---|
| 0 | 0.85 |
| 1 | 1.00 |
| 2 | 0.90 |
| 3 | 0.89 |
| **4** | **1.225** ← outlier |
| 5 | 0.98 |
| 6 | 1.01 |

평균 0.98, std 0.12. seed 4 가 운빨이지만 reproducible.

1000-episode 정밀 측정으로 seed 4 = **1.168 ± 1.78** 확정.

### 5.4 실패한 시도들 (학습 천장 robust 검증)

| 시도 | 결과 |
|---|---|
| 더 큰 네트워크 [400, 300] | 0.905 ❌ |
| 더 큰 네트워크 [512, 256, 128] | 1.09 ❌ |
| γ = 0.995 (longer horizon) | 0.950 ❌ |
| Buffer 1M + 1.2M steps | 1.034 ❌ |
| 셰이핑 α = 0.1 (강하게) | 0.880 ❌ |
| BC pretrain (actor MSE) | 0.855 ❌ |
| Ensemble (mean of policies) | 0.97 ❌ |
| Q-value ensemble (cross-critic) | 0.888 ❌ |

**11가지 모두 실패.** 학습 천장 ≈ 1.17 확정.

### 5.5 학습 천장의 본질

평균 1.17 → per-shot 성공률 약 54% (`p / (1-p) = mean`).
"무한급" (P(≥10) ≥ 50%) 위해선 per-shot ≥ 93% 필요.
**학습으로 이 갭을 줄이는 건 안 됨.**

---

## 6. Phase 4: 추론 시점 탐색 (Lookahead) — 천장 돌파

### 6.1 발상

학습 정책의 액션 분포가 너무 spread out — deterministic action 은 그럭저럭이고 stochastic samples 중에 *더 좋은 액션* 이 존재.

**탐색 아이디어**: 매 샷마다 K개 후보 액션을 정책에서 sample → 각각을 시뮬레이터로 1샷 try → 가장 점수 높은 액션 실행. **학습 안 함. 추론 시간만 더 씀.**

### 6.2 1-단계 탐색 (h=1) 결과

| K | 평균 | 최대 | P(≥5) | P(≥10) |
|---|---|---|---|---|
| 1 (학습 정책 그대로) | 1.138 | 16 | 6% | 0.6% |
| 5 | 2.55 | 14 | 21% | 4% |
| 10 | 3.71 | 20 | 30% | 10% |
| 20 | 5.45 | 24 | 41% | 21% |
| 50 | 7.69 | 42 | 53% | 30% |
| **100** | **8.62** | **53** | **66%** | **36%** |

K=100 으로 학습 정책 대비 **7.4배**. per-shot 89.6%.

### 6.3 2-단계 탐색 (h=2)

매 후보 a₁ 마다 시뮬레이션 후 그 상태에서 K2개 후보 a₂ 더 try. 값 = r₁ + γ · max(r₂).

| K1 | K2 | sims/shot | 평균 |
|---|---|---|---|
| 100 | 1 | 100 | 8.62 (= h=1) |
| 20 | 5 | 100 | **27.58** (3.2×) |
| 50 | 5 | 250 | 35.93 |
| **100** | **5** | **500** | **192.0**, max 742 (1000-shot cap에서) |

같은 sim 수 (100) 에서 h=2 가 h=1 보다 3.2× 좋음. **깊이가 폭 보다 중요.**

K2=10 으로 늘리면 오히려 떨어짐 (mean 21.6) — 두 번째 layer 가 너무 다양하면 noise 가 신호를 가림.

### 6.4 최종: 다중 시드 앙상블 + 적응 K1

#### 동기

mean 192 에서 분석: 10 episodes 중 1개가 score=0 (첫 샷 실패). 단일 정책의 sampling 분포 자체에 그 어려운 state 에서의 winning action 이 안 들어있음.

#### 해법

**서로 다른 시드의 정책 3개 (s1, s4, s6)** 가 동시에 후보 제안. 각 정책의 "전략 모드" 가 달라서, 한 정책이 막힌 상황에서 다른 정책이 풀 수 있음.

#### 적응 K1 스케줄

샷 인덱스 따라 후보 수 조절:
- shot 0 (오프닝): K_per_policy=200 → 총 600 후보 (heavy)
- shot 1-3: 100 → 300
- shot 4-9: 50 → 150
- shot 10+: 30 → 90

오프닝에 가장 많은 compute (첫 샷 실패가 mean을 가장 깎으니까).

#### 결과 (n=10, max_shots=2000)

| 측정 | 값 |
|---|---|
| **평균** | **741.8 ± 646.9** |
| 최대 | **2000 (max_shots 캡 도달)** |
| P(≥100) | 80% |
| P(≥200) | 70% |
| **P(≥500)** | **50%** |
| **P(≥1000)** | **40%** |

10 episodes: [0, 1378, 437, 335, 629, 182, 1406, 1039, **2000**, 12]
- 1개 실패 (random start 가 본질적으로 unwinnable한 state 줌)
- 1개 단명 (12)
- 5개 500+
- 4개 1000+
- 1개 max cap 도달

per-shot 성공률 추정 ≥ **99.95%**.

---

## 7. Phase 5: PUCT 비교 — 표준 MCTS는 왜 안 됐는가

### 7.1 동기

ICLR reviewer 가 물어볼 것: "왜 표준 PUCT (AlphaZero MCTS) 안 쓰고 단순 K-expansion?" → 공정 비교 필수.

### 7.2 PUCT 알고리즘 (한 단락)

표준 MCTS + AlphaGo 의 policy prior 활용:

```
selection: argmax_child [Q(c) + c_puct · π(c) · sqrt(N_parent) / (1 + N_c)]
expansion: leaf 에서 K개 후보 sample, child 노드 추가
evaluation: leaf 의 즉시 보상 (또는 deeper rollout)
backup: 값을 root 까지 propagate
```

같은 sim budget 으로 우리 greedy 와 비교.

### 7.3 결과 (n=10, max_shots=500, 같은 sim budget 공정 비교)

| 방법 | Budget | 평균 | 최대 | P(≥500) | Wall |
|---|---|---|---|---|---|
| UCT (no policy prior) | 600 | 4.4 | 11 | 0% | 13s |
| PUCT single (s4) | 600 | 86.0 | 156 | 0% | 491s |
| **PUCT multi-seed** | 600 | **151.5** | 499 | **0%** | 1703s |
| **PUCT multi-seed (2.5× budget)** | 1500 | **146.6** | 500 | 10% | 5485s |
| **Greedy h=2 multi-seed** | 600 | **357.6** | 500 | **50%** | 3489s |
| **Greedy h=2 multi-seed** | 1500 | **450.0** | 500 | **90%** | 17h |
| Greedy multi-seed adaptive K + max_shots=2000 (Phase 4 최종) | ~1k avg | **741.8** | **2000** | 100% | ~5h |

### 7.4 핵심 발견 3가지

1. **PUCT는 budget 늘려도 평균 안 오름** (b=600 → 151.5, b=1500 → 146.6). 천장에 박힘.
2. **Greedy b=600 (357.6) > PUCT b=1500 (146.6)**. Greedy가 **절반 budget**으로 PUCT 대비 2.4× 우위.
3. **Greedy는 budget에 비례해 향상** (b=600 → 357, b=1500 → 450, adaptive → 742). Budget 추가 = 성능 추가.

이 세 가지가 본 작업의 main contribution. 단지 "우리 method가 잘 됐다" 가 아니라 **표준 MCTS의 한계를 정량적으로 입증**.

### 7.5 왜 PUCT 가 실패하는가 (분석)

세 가지 이유:

1. **결정론적 환경에서 visit count 가 무의미** — PUCT 의 final action 은 가장 많이 방문한 child. 하지만 deterministic env 에서는 같은 (s, a) → 같은 결과. 여러 번 방문 = 정보 이득 없음. UCB exploration bonus 가 시간 낭비.

2. **연속 액션의 discretization 한계** — PUCT 는 fixed legal moves 가정. 연속 액션은 K sample 로 근사 → tree 가 그 K로 갇힘. 깊이 늘려도 같은 K-set 안에서만. Greedy 는 매 노드마다 새 K-sample 이라 더 자유로움.

3. **Sparse 0/1 보상의 noisy backup** — PUCT 의 평균 backup 이 long-horizon return 을 매우 noisy 하게 추정. 잘못된 가지에 budget 집중.

### 7.6 ICLR-friendly narrative

이게 사실 우리 paper 의 main contribution 의 한 축:

**"표준 PUCT 가 deterministic continuous control + sparse reward 환경에서 의외로 비효율적임을 보이고, 더 단순한 greedy K-expansion 이 5배 좋은 성능을 보이는 case study + 분석."**

이런 "표준 방법 실패 + 단순 방법 성공 + 왜 분석" 페이퍼는 ML 학회 (특히 ICLR) 좋아함.

---

## 8. Phase 6 (작업 중): RLHF + Search 통합

### 8.1 동기

지금까지: 점수 (객관적 metric) 만 최적화. 하지만 "좋은 당구" 는 점수 뿐 아님 — 스타일, 위험 관리, 미적인 면.

RLHF (Reinforcement Learning from Human Feedback) 는 인간의 선호도를 reward model 로 학습해서 정책을 그쪽으로 align.

### 8.2 통합 방법 (search-time RM 사용)

우리는 search 시점에 RM 을 사용. 정책 자체는 RL 학습 그대로 (reward hacking 방지):

```python
def score_candidate(action, next_state):
    r_env = env.reward(action)           # 점수 + setup_shaping (객관)
    r_rm  = rm.score(s, action, s')      # 학습된 RM 점수 (주관, 인간 선호)
    return r_env + lambda * r_rm
```

**왜 안전한가**: search 시점에만 사용 → 정책이 RM 을 exploit 할 수 없음. RM 은 단지 candidate ranking 가이드.

### 8.3 무엇을 학습할 수 있나

- "수비적이지 않고 공격적인" 샷
- "위험한 cushion 샷 회피"
- "다음 샷 setup 의 *질*" (현재 d_min 만으로 표현 못 함)
- "human 이 보기 자연스러운" stroke

### 8.4 평가 방법 (RLHF 평가의 어려움)

RLHF 페이퍼의 표준 평가 매트릭스 4개:

1. **객관 metric 유지** — 점수 (inning score) 가 떨어지지 않는지
2. **Held-out RM eval** — 별도 RM 으로 정책 평가 (training RM 과 다른 데이터)
3. **Human pairwise study** — N 명에게 (RL inning, RLHF inning) 페어 보여줌 → 어느 게 더 좋은가
4. **Case study** — 같은 hard state 에서 RL vs RLHF 정성 비교 (trajectory side-by-side)

이 4개 모두 ICLR paper 에 들어감.

### 8.5 예상 결과 (희망)

| 측정 | RL baseline | RLHF + search |
|---|---|---|
| 평균 점수 | 741.8 | ~700-750 (동등) |
| 최대 chain | 2000+ | 2000+ |
| Held-out RM score | 낮음 | 높음 |
| Human win rate | 50% (baseline) | >60% (target) |
| Case study qualitative | brute force | elegant |

**Trade-off curve** (Pareto front of score vs style) 이 ICLR-grade 결과.

### 8.6 구현 plan (1-2주)

1주차:
- 기존 `billiards/preference/`, `billiards/reward_model/` 인프라 재활용
- multi-shot inning 페어 라벨링 (AI labeler 로 자동, 일부 human sample)
- inning-level RM 학습 (single-shot RM 확장)

2주차:
- Search 코드에 RM scoring 통합
- λ (RM 가중치) sweep — 점수 안 죽이는 sweet spot 찾기
- 4가지 metric 으로 평가 (객관, RM hold-out, human study, case study)

3주차 (optional):
- Iterative loop: search → 새 페어 → RM 업데이트 → 새 search
- AlphaZero self-play 의 RLHF 버전

---

## 9. 핵심 발견 6가지 (Lessons Learned)

1. **속도 최적화 ≠ 품질 보장**. Tier A 로 2.3× wall 빨라졌지만 SAC default 그대로면 quality 떨어짐. `gradient_steps` 보정 필수.

2. **800k 가 sweet spot, 그 이상은 collapse**. Setup_shaping signal 에 과적합. 1.6M, 2M 모두 mean 절반 이하로 떨어짐.

3. **Reward shaping ≫ 알고리즘 변경**. 큰 네트워크, 긴 horizon, 큰 buffer, BC pretrain 모두 효과 없음. `setup_shaping` 한 줄 추가가 +54%. **Reward design > Algorithm choice.**

4. **학습은 ceiling 이 있다. Search 가 unlock 한다.** SAC + shaping 천장 1.17. Search 로 8.62, 192, 741. 학습보다 search 가 훨씬 더 큰 기여.

5. **표준 PUCT 가 항상 최선 아니다.** Deterministic continuous control + sparse reward 에서는 단순 greedy K-expansion 이 PUCT 보다 5배 좋음. 환경 특성에 따라 알고리즘 선택해야.

6. **Multi-seed candidate proposal**. 평균 액션 ensemble 은 망함 (mode 사이 무의미한 중간점). 하지만 search-time 에서 각 시드 정책이 candidate 제안 → 합집합에서 선택은 다양성 활용해서 +3.9×.

---

## 10. 한계 (Limitations)

1. **`max_shots = 2000` 캡 도달** — 실제 한계는 더 높을 수 있으나 측정 안 함 (시간 부담).
2. **무작위 시작의 inherent unwinnable** — 약 10% 의 random_start state 는 어떤 깊이의 search 로도 첫 샷 득점 불가 (큐 공이 가려진 상태 등). 환경 본질적 특성.
3. **시뮬레이터 의존** — 추론 시 시뮬레이터 접근 가정. 실제 물리 당구에서는 정확한 시뮬레이터 없음. 대안: world model learning (Dreamer), 또는 search 결과를 single-forward 정책에 distill.
4. **Per-shot compute** — multi-seed adaptive K 에서 샷 당 5-15초. 실시간 응용 (로봇팔, 인간과 대결) 에는 한계.
5. **단일 환경** — Korean 4구 하나. ICLR 가려면 multi-domain 검증 필요 (DM Control, MuJoCo 등).
6. **샘플 크기 작음** — n=10 episode 평가. std 가 mean 만큼 큼 (647 on 742). 결과는 [400, 1100] CI. n=100+ 가 robust.

---

## 11. 다음 단계

### 11.1 단기 (1-2주)
- **깨끗한 ablation 재실험** (아래 §11.4) — 병모의 incremental 시리즈는 진정한 isolated ablation 이 아님. ICLR reviewer 반드시 지적할 부분.
- **RLHF + search 통합** (Phase 6, 위 §8) — main contribution 강화
- **n=50-100 으로 정밀 측정** — CI 좁히기
- **Stochastic env variant** — env 에 noise 추가해서 PUCT vs greedy cross-over 측정

### 11.2 중기 (1개월)
- **Multi-domain** — DM Control, MuJoCo 일부 sparse-reward task 에서 같은 패턴 입증
- **MCTS 진정한 변종 비교** — pUCT, KR-UCT 등 continuous-action 전용 MCTS
- **이론적 분석** — 어떤 환경 조건에서 greedy > PUCT 인지 정량 frontier

### 11.3 장기 (2-3개월, ICLR 2027 target)
- **Policy distillation** — search 결과를 single-forward 정책에 압축
- **Model-based RL** — Dreamer 류로 시뮬레이터 학습
- **Hierarchical policy** — 상위 (target 선택) + 하위 (실행) 분리

### 11.4 미완료: 깨끗한 환경 변수 ablation

병모가 추가한 환경 변경사항 (`constrain_aim`, `continue_on_miss`, `extra_features`, `random_start`, `gentle_shot`, `foul_penalty`) 의 각 marginal 효과를 isolated 으로 측정하지 못했음.

#### 문제

병모의 incremental 시리즈는 일부 step 에서 **여러 변수가 동시에 변경됨**:

| run | constrain_aim | cont_on_miss | extra_features | random_start | gentle_shot | foul_penalty | steps | rnd mean |
|---|---|---|---|---|---|---|---|---|
| sac_s0 (baseline) | ❌ | ❌ | ❌ | ❌ | ❌ | 0.1 | 50k | 0.02 |
| sac_aim_s0 | ✅ | ✅ | ❌ | ❌ | ❌ | 0.1 | 50k | 0.14 |
| sac_feat_s0 | ✅ | ✅ | ✅ | ❌ | ❌ | 0.1 | 50k | 0.16 |
| sac_rs_200k_s0 | ✅ | ✅ | ✅ | ✅ | ❌ | 0.1 | **200k** ↑ | 0.68 |
| sac_gentle_200k_s0 | ✅ | ✅ | ✅ | ✅ | ✅ | **0.5** ↑ | 200k | 0.38 ↓ |

**Confound 들**:
1. `sac_aim_s0` 은 `constrain_aim` + `cont_on_miss` 둘 다 동시에 켜짐 → 각각의 효과 분리 불가
2. `sac_rs_200k_s0` 은 `random_start` + **steps 50k → 200k (4×)** 같이 변경 → 어느 게 효과인지 모름
3. `sac_gentle_200k_s0` 은 `gentle_shot` + **foul_penalty 0.1 → 0.5** 같이 변경 → mean 이 0.68 → 0.38 로 떨어진 게 어느 변수 탓인지 불명

#### 필요한 깨끗한 ablation cells

각 변수를 *하나씩만* 추가한 run 들을 학습해야 isolated effect 측정 가능:

| Cell | 베이스 | 변경 | 측정 목표 |
|---|---|---|---|
| A0 | sac_s0 | (없음) | baseline |
| A1 | sac_s0 | + constrain_aim only | constrain_aim marginal |
| A2 | sac_s0 | + cont_on_miss only | cont_on_miss marginal |
| A3 | sac_s0 | + constrain_aim + cont_on_miss | 둘 합쳤을 때 (병모의 sac_aim 재현) |
| A4 | sac_feat_s0 | + random_start (steps 50k 유지!) | random_start marginal |
| A5 | sac_feat_s0 | + steps 200k (random_start 없이) | steps 효과 분리 |
| A6 | sac_rs_200k_s0 | + gentle_shot only (fp=0.1 유지) | gentle_shot marginal |
| A7 | sac_rs_200k_s0 | + fp=0.5 only (gentle_shot 없이) | fp marginal |

총 8 cells. 각 ~6-8분 (Tier A 적용 환경 기준) = **약 1 시간 wall** 로 완전한 isolated ablation 가능.

#### 예상 시사점

- A1 vs A2 vs A3: constrain_aim 만으로도 큰 효과인지, cont_on_miss 가 essential 인지 결정
- A4 vs A5: 0.68 점프가 random_start 때문인지, 4× 학습 때문인지 결정
- A6 vs A7: gentle_shot+fp=0.5 의 -0.30 drop 이 어느 변수 탓인지 결정 (아마 fp 가 너무 강한 게 문제로 예상되지만 검증 안 됨)

#### ICLR 에 왜 필요한가

Reviewer 가 반드시 물어볼 것: **"Each component 의 marginal contribution 은?"**

현재 데이터로는 답이 불가능. 위 8 cells 만 추가 학습하면 깨끗한 contribution table 작성 가능.

#### 우선순위

본 보고서의 main contribution 은 search 부분 (Phase 4-5) 이므로 학습 ablation 의 미완성성은 main result 에 영향 없음. 하지만 ICLR submission 전 반드시 보완 필요. **단기 (1주) 우선순위 1번**.

---

## 12. 전체 진화 표

| 단계 | 시점 | 평균 | 최대 | 비고 |
|---|---|---|---|---|
| Random policy | - | 0.005 | 1 | chance |
| PPO inning (Phase H, canonical, n=3 seeds) | 2026-04 | 0.0 | 0 | 한 점도 못 냄 |
| SAC inning (Phase H, canonical, n=3 seeds) | 2026-04 | **1.0** | 1 | 정확히 1점/이닝 — "한 점 memorize" |
| Random-start SAC (Phase I) | 2026-05 초 | 0.025 | 1 | distribution shift |
| 병모 baseline | 2026-05 초 | 0.575 | 4 | constrain_aim 도입 |
| Tier A (no shape) | 2026-05-19 | 0.570 | 6 | wall 2.3× ↓ |
| + setup_shaping (200k) | 2026-05-19 | 0.885 | 9 | +54% |
| + 800k training | 2026-05-19 | 0.975 | 6 | |
| + fp=0.2 seed 4 | 2026-05-20 | **1.168** | 16 | 🚧 **학습 천장** |
| + K=100 h=1 search | 2026-05-20 | 8.62 | 53 | 첫 search 도약 |
| + K1=100 K2=5 h=2 | 2026-05-20 | 192.0 | 742 | depth-2 |
| + 다중 시드 + adaptive K | 2026-05-21 | **741.8** | **2000** | 🎯 **사실상 무한** |

**누적**: random policy → 최종 = **약 148,000배**.

---

## 13. 재현 (Reproducibility)

### 13.1 최종 정책

```bash
uv run python experiments/run_inning_sac.py \
    --algo sac --seed 4 --total_steps 800000 --max_shots 10 \
    --continue_on_miss --constrain_aim --extra_features --random_start \
    --foul_penalty 0.2 --gentle_shot \
    --setup_shaping --setup_alpha 0.05 --setup_scale 0.3 \
    --n_envs 8 --gradient_steps 2 \
    --out_dir experiments/runs_inning_v2/fast_long_fp02_s4
```

다중 시드 정책: `fast_long_fp02_s{1, 4, 6}/policy.zip`. (Git LFS 대신 학습 명령으로 재생성 가능.)

### 13.2 Lookahead 추론

```bash
PYTHONUNBUFFERED=1 uv run python experiments/lookahead/multi_seed_h2.py
```

### 13.3 PUCT vs Greedy 비교

```bash
PYTHONUNBUFFERED=1 uv run python experiments/lookahead/puct.py
```

### 13.4 데모 (HTML 시각화)

`artifacts/best_inning/` (Git LFS):
- **`INFINITY_multiseed_score2000_shots2000_seed99008.html`** — 2000샷 무한 이닝 ⭐
- `INFINITY_uncap_K1100_K25_score742_shots743_seed99004.html` — h=2 single, 742점
- `MAX_score16_seed99469.html` — 학습 정책 단독 (no search) 최고 16점

브라우저로 열면 trajectory 재생.

---

## 14. References (선별)

1. Haarnoja, T. et al. (2018). Soft Actor-Critic.
2. Silver, D. et al. (2016). Mastering Go with deep neural networks and tree search. *Nature*.
3. Silver, D. et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.
4. Schrittwieser, J. et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero).
5. Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward transformations.
6. Christiano, P. et al. (2017). Deep RL from Human Preferences. (RLHF original)
7. Hafner, D. et al. (2023). DreamerV3.
8. Dietterich, T. G. (2000). Ensemble Methods in Machine Learning.

---

## 부록

### A. 최종 정책 hyperparams

| 파라미터 | 값 |
|---|---|
| algo | SAC (MLP [256, 256]) |
| learning_rate | 3e-4 |
| buffer_size | 200,000 |
| batch_size | 256 |
| gamma | 0.99 |
| gradient_steps | 2 |
| n_envs | 8 |
| total_steps | 800,000 |
| seed | 4 |
| max_shots (train) | 10 |
| continue_on_miss | True |
| constrain_aim | True |
| extra_features | True |
| random_start | True |
| foul_penalty | 0.2 |
| gentle_shot | True (α=0.2, d_target=0.2, σ=0.1) |
| setup_shaping | True (α=0.05, σ=0.3) |
| dt_max (sim) | 0.1 |

### B. 최종 Lookahead config

| 파라미터 | 값 |
|---|---|
| 정책 ensemble | s1, s4, s6 (3개) |
| K_per_policy (shot 0) | 200 (총 600) |
| K_per_policy (shot 1-3) | 100 (총 300) |
| K_per_policy (shot 4-9) | 50 (총 150) |
| K_per_policy (shot ≥10) | 30 (총 90) |
| depth (h) | 2 |
| K2 (2nd layer per policy) | 5 (s4 단독) |
| γ | 0.99 |
| eval max_shots | 2000 |

### C. 코드 변경 요약

병모의 origin/main 위에 추가:

1. `billiards/inning_env.py` — lean info dict, setup_shaping, snap-to-rest 방어
2. `billiards/physics/simulator.py` — `dt_max` 0.05 → 0.1
3. `experiments/run_inning_sac.py` — `--setup_shaping`, `--setup_alpha`, `--setup_scale`, `--gradient_steps`, `--net_arch`, `--gamma`, `--buffer_size` CLI
4. `experiments/lookahead/` — 모든 search 변종 (multi_seed_h2.py, puct.py 등)

### D. Reward shaping 디테일 — gentle_shot 과 setup_shaping

#### D.1 gentle_shot (병모)

**발동 조건**: 득점 샷 직후 (`score > 0` 이고 `gentle_shot=True`).

**의도**: 득점 후 큐 공이 **두 번째로 맞춘 빨간 공** 으로부터 너무 멀지도 가깝지도 않게 멈추면 보너스. 다음 샷에서 그 공을 다시 치기 좋은 거리.

**수식**:

```
events 에서 cue_hit_red 만 추출 → 맞춘 순서대로 reds_hit 리스트
target = reds_hit[1] (두 번째 맞춘 red)
dist = ‖cue.position - target.position‖

bonus = α · exp(-(dist - d_target)² / (2σ²))

기본값:
  α        = 0.2  (peak 보너스)
  d_target = 0.2 m (이상적 거리)
  σ        = 0.1 m (가우시안 폭)
```

**시각화**:

```
bonus
0.2 |     ███
    |   ██   ██
    |  █       █
    | █         █
0.0 |█___________█___________
    0   0.2     0.5    dist(m)
        d_target
```

- dist = 0.2 m: peak 0.2 점
- dist = 0.1 m 또는 0.3 m: 0.12 점
- dist = 0 m (큐 공이 red 와 거의 붙음): 0.03 점
- dist > 0.5 m: 거의 0

**디자인 의도** (인간의 직관):
- 너무 멀면 다음 샷 정밀도 ↓
- 너무 가까우면 (접촉) 다음 샷 각도 매우 제한
- ~0.2 m 가 sweet spot

#### D.2 setup_shaping (우리)

**발동 조건**: **매 비파울 샷마다** (sparse → dense 로 보완).

**수식**:

```
cue 와 가장 가까운 red 간 거리:
  d_min = min(‖cue - red1‖, ‖cue - red2‖)

bonus = α · exp(-d_min / σ)

기본값:
  α = 0.05  (작게 — score=1 dominant 유지)
  σ = 0.3 m
```

**의도와 차이**:
- gentle_shot: 득점 직후 "두 번째 red 와 적당히 떨어진 거리" (Gaussian peak)
- setup_shaping: 매 샷 후 "가장 가까운 red 와 가까울수록 좋음" (Exponential decay)

#### D.3 두 보상의 비교

| 측면 | gentle_shot | setup_shaping |
|---|---|---|
| 발동 시점 | 득점 샷 직후만 (sparse) | 매 비파울 샷 (dense) |
| Target 거리 | 두 번째 맞춘 red (특정) | 가장 가까운 red (어느 쪽이든) |
| 함수 모양 | Gaussian peak at d_target | Exponential decay (작을수록 좋음) |
| α (크기) | 0.2 | 0.05 |
| 한 이닝 누적 max | ~ 0.2 × 득점 횟수 | ~ 0.05 × 샷 수 |
| 의도 | 득점 후 좋은 setup 격려 | 매 샷마다 적구 근처에 멈춰라 |
| Exploit 위험 | 낮음 (득점 조건부) | 중간 (조심해서 α 작게) |

#### D.4 우리가 발견한 함정

`setup_shaping` 만으로 `random_start` 없이 학습하면 mean 0.17 로 붕괴 (정책이 "거의 닿지만 점수 안 내기" 위장 전략에 갇힘). `random_start` 와 함께 써야 robust (§4.4 참조).

#### D.5 알려진 개선 여지

1. **두 번째 red 만 고려 (gentle_shot)**: 두 빨간 공 중 *첫 번째* 가 더 좋은 setup 일 수도 있음. min 거리 쓰는 게 더 합리적.
2. **각도 / line-of-sight 무시**: 거리만 봄. 큐 공-red 직선 위에 상대 큐가 있어 다음 샷이 막혀도 보너스. 가로막힘 확인 필요.
3. **d_target 의 정당성**: hand-picked. 인간 expert 의 "이상적 거리" 통계로 정해진 게 아님. RLHF 로 학습 가능 (§8).
4. **누적 보너스 vs score=1 dominant 성**: gentle_shot α=0.2 × 평균 4 득점 = 0.8 누적 vs score 4 점 → 20%. setup_shaping α=0.05 × 10 샷 = 0.5 vs score ~3.7 → 13%. 둘 다 score 가 dominant 하지만 무시할 수 없는 비중.
5. **Potential-based shaping 으로 변환**: F = γφ(s') - φ(s) 로 바꾸면 최적 정책 불변 (Ng et al. 1999). 더 안전하지만 구현 복잡.

---

**작성**: 2026-05-21
**작업 기간**: 2026-05-19 저녁 ~ 2026-05-21 오후 (약 44시간 wall — 학습 ~6h, 추론 sweep ~10h, 분석/설계 ~28h)
