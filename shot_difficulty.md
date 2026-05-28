# Shot Difficulty — Action-Space Margin to Failure

본 문서는 `shot_difficulty` 브랜치에서 수행한 연구의 정리.
"4구 당구에서 어려운 샷이란 무엇인가?"를 정의하고, 이를 학습/추론에 반영해
정책 행동이 어떻게 변하는지 측정한다.

핵심 결과(spoiler): **학습은 약효과, lookahead ranking 토글이 즉시·명확하게 작동**.
점수는 동일한 50샷에서 **쿠션 수 −38%, 저-margin 샷 거의 제거, 고-power × 저-margin
샷 회피**가 emergent하게 나타남.

---

## 1. Shot difficulty 정의 후보 (ENGINE.md §3)

| 정의 | 설명 | 장점 | 단점 |
|---|---|---|---|
| **A. Action-space robustness (margin)** | 정답 액션 `a`에 노이즈 `δ` 추가, perturb한 액션이 여전히 득점하는 비율 | 시뮬로 직접 측정 가능, domain-agnostic, ENGINE.md §3.2와 정확히 일치 | perturb N개 × 시뮬 비용 |
| B. Heuristic | 스핀·큐 파워·쿠션 수 등 직접 함수화 | 비용 0, 즉시 가용 | "Domain Knowledge Removal" 방향과 충돌, RL 창의성 억제 |
| C. 앙상블 분산 | K 개 정답 후보의 액션 분산이 작으면 쉬움 | 기존 multi_seed lookahead 인프라 재사용 | 학습 reward로 쓰려면 K 개 정책을 동시 유지해야 함 |

**선택: A (Action-space robustness)** — 학습 reward로 통합 가능하고, 어떤 휴리스틱보다 더 미묘한
"실수에 강한 샷" 신호를 포착할 수 있다.

수식:
```
robustness(a) = (1/N) Σ_{i=1..N} I[ score(a + δ_i) > 0 AND not fouled(a + δ_i) ]
                δ_i ~ N(0, ε)  (default ε = 0.05)

reward_bonus = β · robustness(a)  if a scored and not fouled
                                  (default β = 0.2)
```

---

## 2. 사전 작업: chain100 paradigm (실패한 우회)

robust_reward에 들어가기 전, "한 이닝을 길게 가져가는 게 무조건 유리"를 직접 학습시키는
시도를 했다. 환경: `--max_shots=100 --no-continue_on_miss`로 miss/foul 시 즉시 종료, 100샷
도달 시 truncation. 100까지 가려는 정책을 기대.

`fast_long_fp02_s{1,4,6}` warm-start, `gentle_shot=False` (의도된 ablation), 150k step.

| seed | baseline mean | chain100 mean | Δ |
|---|---|---|---|
| s1 | 5.10 | 1.32 | **−3.78** |
| s4 | 5.69 | 1.47 | **−4.22** |
| s6 | 5.21 | 1.62 | **−3.59** |

**결과: 명백한 회귀.** 부분적 긍정 신호로 max 체이닝(15샷)과 foul率↓이 있었으나
basic 득점 손실이 압도적. 원인 가설:
1. **gentle_shot 제거**: warm-start 정책이 gentle 보너스에 맞춰 학습돼 있어 reward 분포
   급변 → catastrophic forgetting
2. **continue_on_miss=False fine-tune**: 짧은 trajectory가 버퍼 지배 → 긴 chain의 Q 추정 데이터 부족
3. **setup_shaping 100샷 누적**: 누적 ~5로 실제 score와 동급 → "쉬운 setup만 노리다 miss" 패턴

→ 교훈: warm-start와 학습 환경 reward 구조가 일관해야 함. **모든 flag True 유지가 안정성 핵심**.
이후 모든 실험은 fast_long_fp02 환경과 reward 구조를 그대로 두고 추가 신호만 도입.

---

## 3. robust_reward 학습 (약효과)

`Billiards4BallInningEnv`에 4개 인자 추가:
- `robust_reward: bool` — 토글
- `robust_eps: float = 0.05` — perturbation σ
- `robust_n: int = 8` — perturbation 수
- `robust_alpha: float = 0.2` — bonus 가중치

구현 (`billiards/inning_env.py`):
- `step()` 진입 시 pre-shot state deepcopy (toggle off면 skip)
- 득점(non-foul) 샷에 한해 `_estimate_robustness(pre_shot_state, raw_action)` 호출
  - N개 perturbation을 pre-shot에서 재시뮬, success 비율 반환
  - `_apply_aim_constraint`를 일관 적용해 perturbation도 same aim window에서 동작
- `reward += robust_alpha * robustness`

비용: smoke test 기준 ~3.5x slowdown. 실측 학습 wall ~17분/seed (vs baseline ~5분).

**fine-tune 설정**: `fast_long_fp02_s{1,4,6}` warm-start, 모든 flag True, 150k step.

### 3.1 표준 training-mode eval (continue_on_miss=True, max_shots=10, n=200)

| seed | baseline mean | robust mean | Δ | p5 | foul% | cushion |
|---|---|---|---|---|---|---|
| s1 | 5.10 | **5.36** | +0.26 | **70.0%** | 61.0% | 1.23 |
| s4 | **5.69** | **5.79** | +0.10 | **81.0%** | 50.5% | **1.00** |
| s6 | 5.21 | **5.48** | +0.27 | **75.0%** | 59.5% | 1.27 |

mean 회귀 없음 (chain100과 결정적 차이). p5(≥5점 비율) 70~81%로 강한 consistency. 그러나 mean
개선은 +0.10~+0.27로 작아 noise 가능성 존재. 진짜 검증은 §3.2.

### 3.2 직접 robustness 측정 (`eval_robustness.py`)

**방법**: random rack 100개에서 정책별 deterministic action `a*` 추출 → `a* + N(0, ε)`
perturbation 32개 시뮬 → score>0 비율. paired (동일 rack), 4개 ε 스윕.

**메트릭**:
- `det_success_rate`: `a*` 자체가 득점한 state 비율
- `robustness_conditional`: `a*` 득점한 state들에 한해 평균 perturbation 성공률 — **robust_reward
  bonus가 직접 최대화하는 양**

**paired Δ (baseline → robust):**

| seed | det% (B→R) | ε=0.02 | ε=0.05 | ε=0.10 | ε=0.20 |
|---|---|---|---|---|---|
| s1 | 0.48→0.50 | **+0.017** | **+0.020** | **+0.045** | **+0.060** |
| s4 | 0.48→0.45 | +0.002 | −0.015 | −0.021 | +0.005 |
| s6 | 0.46→0.52 | −0.004 | −0.012 | −0.013 | −0.019 |
| **avg** | **+0.017pp** | **+0.005** | **−0.002** | **+0.004** | **+0.015** |

**결과: 통계적으로 유의하지 않음.** s4 ε=0.10 SE ≈ 0.05 → 평균 Δ ±0.02 이내는 noise 수준.

- s1만 모든 ε에서 일관된 양의 효과 (특히 ε=0.20에서 +0.060). 큰 노이즈에 강한 정책 학습 패턴.
- s4/s6는 무효 또는 약한 음의 효과.

**원인 가설:**
1. **`robust_alpha=0.2` 너무 작음**: 득점 1.0 대비 robustness=0.5 bonus는 0.1 → policy 변화 약함
2. **baseline이 이미 robust**: fast_long_fp02가 lookahead-style 선택을 학습 → 이미 margin 큰
   샷을 고름. baseline conditional robustness 0.86~0.89 (ε=0.02) 매우 높음 → 개선 여지 좁음
3. **150k step 짧음**: 새 reward 신호에 적응할 시간 부족
4. **setup_shaping 누적이 robust gradient 잠식**

**판정**: robust_reward 가설은 **학습 fine-tune으로는 입증 실패**. mean +0.10~+0.27도
다른 부수 효과로 설명 가능.

---

## 4. Lookahead robust ranking (성공)

학습이 약하다면 추론 시점은 어떨까? §2.1 time_reward 패턴과 동일하게,
**lookahead 후보 ranking에 robust term을 추가**하는 무학습 방식을 A/B/C로 분해.

### 4.1 3-variant 분해 (`compare_robust_proposers.py`)

```
ranking value:  v = r1 + γ·r2  +  β·robustness(a1)  (if scored, when robust_rank=True)
```

| variant | proposer | 랭킹 robust | 설명 |
|---|---|---|---|
| `true_baseline` | fast_long_fp02 | off | 기존 lookahead |
| `baseline_robust` | fast_long_fp02 | **on** | ranking term만 추가 (학습 없음) |
| `robust_robust` | **robust_s*** | **on** | 학습된 robust 정책 + ranking term |

**결과 (paired n=12, max_shots=25, k1=100, k2=3):**

| metric | true_baseline | baseline_robust | robust_robust | rank 효과 | proposer 효과 |
|---|---|---|---|---|---|
| mean_score | 22.58 | 22.92 | 22.92 | +0.333 | +0.000 |
| score/sim_s | 0.255 | 0.262 | 0.267 | +0.007 | +0.006 |
| **mean_chosen_robustness** | **0.767** | **0.883** | **0.895** | **+0.116 (+15%)** | **+0.012** |

**핵심 발견:**
- **Ranking 효과가 거의 모든 이득의 주범**: chosen robustness +0.116 (15%↑). 학습 없이 inference
  시점에서 robust 후보를 선택하는 것만으로 충분.
- **Robust-trained proposer 추가는 거의 무효** (+0.012, noise 내). §3.2의 약효과를 lookahead 맥락에서도 재확인.
- **time_reward §2.1과 정확히 동일한 패턴** — 운영상 결론: 학습보다 ranking 토글.

**비용**: true_baseline 133s, baseline_robust 319s (~2.4x). robust ranking은 SCORING 후보에만 N=8 추가
시뮬 → 합리적.

### 4.2 Canonical 50샷 행동 양식 비교 (`canonical_robust_replay.py`)

§4.1는 평균 metric이었음. 실제로 robust ranking이 **샷 선택 양식**을 어떻게 바꾸는지 보기 위해
canonical 시작, `continue_on_miss=True`로 50샷 강제 진행, true_baseline vs baseline_robust paired
(동일 proposer/seed). **양 variant 모두 50/50 득점, foul 0회 → 행동 차이가 순수한 robust ranking 효과.**

#### 4.2.1 집계 차이

| metric | true_baseline | baseline_robust | Δ |
|---|---|---|---|
| total_score | 50 | 50 | 0 |
| **mean cushion_hits** | **1.04** | **0.64** | **−0.40 (−38%)** |
| mean chosen_robustness | 0.780 | 0.875 | +0.095 |
| std chosen_robustness | 0.247 | 0.202 | −0.045 |
| power mean / std | 0.730 / 0.245 | 0.751 / **0.216** | std −0.029 |

#### 4.2.2 Cushion-hit 분포 변화

| cushions | true_baseline | baseline_robust |
|---|---|---|
| 0 | 21 | **25** |
| 1 | 14 | **19** |
| 2 | 10 | **5** |
| 3 | 3 | 1 |
| 4 | 1 | 0 |
| 5 | 1 | 0 |

→ robust ranking은 **다중 쿠션 샷을 거의 절반으로 줄임** (2+쿠션: 15회 → 6회). 0~1 쿠션 비율 70% → 88%.
직관: 다중 쿠션은 각 쿠션 마찰계수 + 입사각 민감도가 곱해져 margin이 작은 경향. robust 신호와 일치.

#### 4.2.3 Robustness 분위수 (득점 샷만)

| 분위 | true_baseline | baseline_robust |
|---|---|---|
| min | 0.125 | 0.250 |
| q10 | 0.375 | **0.613** |
| q25 | 0.656 | 0.750 |
| **median** | **0.875** | **1.000** |
| q75 | 1.000 | 1.000 |

→ **저-margin 샷을 거의 완전히 제거**. median이 1.000(완벽)으로 올라감 — "robust 정책 절반 이상의
샷은 8개 perturbation 어떤 것에도 무너지지 않음". 최저 q10이 0.375 → 0.613으로 가장 크게 향상.

#### 4.2.4 가장 발산된 액션 5개 (action_l2 기준)

| shot | true_baseline (power, robust) | baseline_robust (power, robust) | 해석 |
|---|---|---|---|
| 15 | 0.97, **0.375** | 0.16, **1.000** | tb는 강한 risky 샷, br은 약한 안전 샷 |
| 5 | 0.85, **0.375** | 0.27, **1.000** | 동일 패턴 |
| 38 | 0.37, 0.875 | 0.60, 0.875 | 비슷한 margin인데 br가 1 쿠션 적음 |
| 23 | 0.57, 0.75 | 0.54, **1.000** | 같은 power, 다른 θ로 margin ↑ |
| 19 | 0.13, 0.875 | 0.45, **1.000** | br가 약간 더 강하게 — power 자체엔 호불호 X |

→ 핵심 행동 패턴: **고-power × 저-margin 샷이 있을 때 robust ranking이 명확히 회피**.
median power는 비슷하지만 power std가 감소 — "극단적 power" 회피.

#### 4.2.5 산출물

- `experiments/artifacts/canonical_robust_replay/variant_true_baseline.html`
- `experiments/artifacts/canonical_robust_replay/variant_baseline_robust.html`
- `experiments/artifacts/canonical_robust_replay/per_shot.csv` — 50샷 × 2 variant per-shot raw
- `experiments/artifacts/canonical_robust_replay/summary.json`

브라우저에서 두 HTML을 나란히 띄우면 행동 차이가 시각적으로 확인 가능.

---

## 5. 종합 결론

1. **Shot difficulty는 "action-space margin"으로 정의 가능**하고 시뮬로 직접 측정 가능
   (ENGINE.md §3.2). 휴리스틱(스핀·쿠션 수 등) 없이도 정량화된다.

2. **학습 fine-tune은 약효과 (§3)**. baseline 정책이 이미 충분히 robust한 샷을 고르고
   있어 개선 여지가 좁음. `robust_alpha=0.2`도 score 1.0 대비 너무 작아 정책을 실질적으로
   못 바꿈. 더 큰 alpha, 더 긴 학습이 필요하지만 cost-effective 하지 않음.

3. **Lookahead ranking 토글은 명확하고 즉시 작동 (§4)**. 학습 없이도 chosen robustness
   0.767 → 0.883 (+15%). 비용은 ~2.4x slowdown만.

4. **Canonical 50샷 비교(§4.2)는 행동 양식 변화가 emergent하게 나타남을 보여준다**:
   - 쿠션 수 38% 감소 (1.04 → 0.64)
   - 2+ 쿠션 샷 절반 감소
   - 저-margin 샷 거의 제거 (median margin 0.875 → 1.000)
   - 고-power × 저-margin 샷 회피
   - "같은 점수를 더 안전하게"
   
   이는 domain knowledge 주입 없이도 **사람이 "프로의 샷"이라 부르는 직관과 일치하는
   행동 패턴**이 emergent하게 나타난다는 점에서 본 연구의 핵심 발견.

5. **Time_reward(§2.1)와 정확히 같은 패턴 — 운영 원칙 도출:**
   > 학습된 prior + 추론 시점 multi-objective ranking 이 가장 효율적.
   > shot-engine의 기본은 학습이 아니라 ranking. 새 신호(time, robustness, ...)는 reward 추가가
   > 아니라 ranking term 추가로 도입한다.

---

## 6. 다음 단계 후보

1. **Multi-objective ranking**: `r1 + γ·r2 + β_time·time + β_robust·robustness`. time과 robust 동시
   적용 시 시너지/충돌 확인.
2. **robust_eps sweep at ranking** (0.02 / 0.05 / 0.10): chosen robustness 곡선의 ε 의존성 측정.
3. **Uncapped max_shots**: §4.1은 max_shots=25 cap이 mean_score saturate시킴. 더 긴 cap에서
   chained length 차이 확인.
4. **robust_alpha sweep at training** (0.4 / 0.8): bonus 4~8배 강화 시 §3.2 학습 효과 회복 여부.

---

## 산출 파일

| 종류 | 경로 |
|---|---|
| 환경 구현 | `billiards/inning_env.py` (4개 인자, `_estimate_robustness`) |
| 학습 CLI | `experiments/run_inning_sac.py` (`--robust_reward` 등) |
| 학습 결과 | `experiments/runs_inning_v2/robust_s{1,4,6}/` |
| 직접 측정 | `experiments/eval_robustness.py`, `experiments/artifacts/robustness_eval/` |
| Lookahead A/B/C | `experiments/lookahead/compare_robust_proposers.py`, `experiments/artifacts/lookahead_robust_abc/` |
| Canonical 50샷 | `experiments/lookahead/canonical_robust_replay.py`, `experiments/artifacts/canonical_robust_replay/` |
| 실험 기록 | `experiments/experiments.md` §1.5, §1.6, §1.6.1, §2.3, §2.3.1 |
