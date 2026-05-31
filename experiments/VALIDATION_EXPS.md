# 검증 실험 결과 (Validation Experiments)

연구 보고서에 들어갈 **검증 실험의 결과**를 정리하는 문서.
설계는 [`../VALIDATION.md`](../VALIDATION.md), 탐색 로그는 [`experiments.md`](experiments.md).
이 문서는 그 둘 사이 — "보고서에 실을, 재현 가능한 검증 결과"만 담는다.

- **섹션 번호는 `VALIDATION.md`와 1:1**로 맞춘다 (0~5). 설계와 결과가 항상 정렬되도록.
- **원본은 옮기지 않는다.** `experiments/runs_*/.../summary.json`, `experiments/artifacts/.../`,
  CSV/HTML 등 기존 산출물은 제자리에 두고 **상대경로로 링크만** 한다.
- 학습 정책 `.zip`은 `.gitignore` 대상 — 저장소엔 없을 수 있음. 재현은 "재현 커맨드"로.

---

## 형식 규칙 (Format Convention)

> **format을 바꾸고 싶으면 여기 "항목 블록 템플릿"만 고치고 아래 항목들을 맞춰 수정한다.**
> 모든 검증 실험은 아래 골격을 그대로 따른다. 표 컬럼/필드를 늘리거나 줄일 때 이 정의가 단일 기준.

### 항목 블록 템플릿

각 검증 실험은 `### <섹션번호>.<n> <제목>` 헤더 + 다음 6개 필드로 구성:

```markdown
### 1.1 <제목>

- **status**: ⬜ 미실행 | 🔄 진행중 | ✅ 완료 | ❌ 기각
- **date**: YYYY-MM-DD  (git 커밋 기준, 미실행이면 비움)
- **가설**: 한 줄.
- **설정**: algo / steps / seeds / eval(innings, max_shots) / 핵심 flag. 베이스와 *다른 것만* 강조.
- **결과**: 아래 결과표(시드별 + 평균). 단위는 헤더 주석에 한 번만.
- **판정**: 한두 줄 결론 (가설 채택/기각 + 왜).
- **원본**: 결과 산출물 상대경로 (summary.json / csv / html ...).
- **재현**: 실행 커맨드 한 줄(또는 스크립트 경로).
```

### 공통 지표 정의 (한 번만)

| 약어 | 의미 |
|---|---|
| `mean` | inning당 평균 득점 |
| `max` | 최고 inning 득점 |
| `p1 / p3 / p5` | inning 득점 ≥1 / ≥3 / ≥5 비율(%) |
| `foul` | 파울 비율(%) |
| `wall` | 학습 wall-clock(초) |
| `seed` | 학습 시드 (random eval에선 공 위치 랜덤화라 큰 의미 X) |

> ⚠️ `max_shots`가 다르면 득점 상한이 달라 `mean` 직접 비교 불가 — 표마다 `max_shots` 명시.

### status 범례

- ⬜ **미실행** — 설계만 됨
- 🔄 **진행중** — 일부 시드/조건 돌아감
- ✅ **완료** — 보고서에 실을 수 있음
- ❌ **기각** — 가설이 데이터로 반증됨 (보고서에 "안 된 것"으로 실음)

---

## 0. 도메인 & 공통 학습 방법 (실험 없음)

> `VALIDATION.md §0`. 4구 당구 규칙(수구/목적구/상대구), `--total_steps` = 샷 횟수 의미 등
> 설명용. 검증 실험 아님 — 보고서 본문 서술로 처리.

---

## 1. Algorithm and Learning Methods

> `VALIDATION.md §1`.

### 1.1 SAC vs PPO vs TD3

- **status**: ⬜ 미실행
- **date**:
- **가설**: sparse {0,1} scoring 환경에서 off-policy(SAC/TD3)가 on-policy(PPO)보다 우세하다.
- **설정**: algo ∈ {SAC, PPO, TD3} / **steps=400k (near-plateau — SAC 400k는 800k의 ~85–90%,
  곡선이 이미 knee 구간이라 순위 역전 없음; `training_curve.csv`로 곡선 함께 보고)** / seeds=5 (s0~s4)
  / eval(innings=100, max_shots=10) / n_envs=8.
  - **공통 env flag (3 algo 동일)**: `constrain_aim + extra_features + random_start +
    continue_on_miss + gentle_shot + setup_shaping(α=0.05, scale=0.3) + foul_penalty=0.2`.
    프로젝트 표준 학습 설정(SOTA SAC 라인과 동일) — 각 알고리즘에 공정한 학습 가능 backdrop 제공.
  - SAC/TD3만: `gradient_steps=2, buffer_size=200000`. PPO: `n_steps=512, n_epochs=4`.
  - **budget 의존성 검증**: 단일 to-400k run의 곡선이 곧 모든 짧은 budget의 답(prefix). budget을
    따로 sweep하지 않고 곡선 교차 여부로 "순위 역전" 판단.
- **결과**: *(mean = inning당 평균 득점, eval 100 innings, max_shots=10)*

  | algo | s0 | s1 | s2 | s3 | s4 | **mean±std** | foul% | wall/run |
  |---|---|---|---|---|---|---|---|---|
  | SAC |  |  |  |  |  |  |  |  |
  | TD3 |  |  |  |  |  |  |  |  |
  | PPO |  |  |  |  |  |  |  |  |

- **판정**:
- **원본**: `experiments/runs_inning_v2/valid_algo/{sac,td3,ppo}_s{0..4}/summary.json`
  (곡선: 각 run `training_curve.csv`)
- **재현**: `powershell -File experiments/run_validation_algo.ps1`
  (cheaper screen: `-Steps 200000`; 일부만: `-Algos sac,ppo -Seeds 0,1`)

### 1.2 학습 패러다임 4종 (start × miss 처리)

- **status**: ⬜ 미실행
- **date**:
- **가설**: canonical start vs random start, continue-on-miss vs reset-on-miss 4조합 중
  일반화에 유리한 조합이 있다.
- **설정**: algo=SAC / steps=___ / seeds=5 / eval(innings=100, max_shots=10).
- **결과**:

  | start | on-miss | **mean** | max | p3 | foul% | 비고 |
  |---|---|---|---|---|---|---|
  | canonical | continue |  |  |  |  |  |
  | canonical | reset |  |  |  |  |  |
  | random | continue |  |  |  |  |  |
  | random | reset |  |  |  |  |  |

- **판정**:
- **원본**:
- **재현**:

---

## 2. Baseline Efforts

> `VALIDATION.md §2`. hand-engineering 없는 순수 RL 천장 측정.

### 2.1 Plain RL Baseline (no shaping/aim/feature)

- **status**: ⬜ 미실행
- **date**:
- **가설**: 모든 hand-engineering을 끄면 순수 RL은 random policy 수준(≈0.005)을 못 넘는다.
- **설정**: algo ∈ {PPO, SAC} / steps=___ / seeds=3 / 모든 보조 flag off / reward= {0,1} score만.
- **결과**:

  | algo | s0 | s1 | s2 | **mean** | max | foul% |
  |---|---|---|---|---|---|---|
  | PPO |  |  |  |  |  |  |
  | SAC |  |  |  |  |  |  |

- **판정**:
- **원본**:
- **재현**:

---

## 3. Introduction of Domain Knowledge

> `VALIDATION.md §3`. 도메인 지식(조준 제약, shaping, feature) 도입 효과의 단계별 분해.

### 3.1 도메인 지식 ablation (누적 추가)

- **status**: ⬜ 미실행
- **date**:
- **가설**: `constrain_aim → extra_features → setup_shaping → gentle_shot` 순으로 누적 추가할수록
  단조 개선되며, 첫 득점의 열쇠는 `constrain_aim`이다.
- **설정**: algo=SAC / steps=___ / seeds=___ / eval(innings=100, max_shots=10).
- **결과**:

  | 추가된 도메인 지식 | **mean** | max | p3 | foul% | Δ(직전 대비) |
  |---|---|---|---|---|---|
  | (none, baseline) |  |  |  |  | — |
  | +constrain_aim |  |  |  |  |  |
  | +extra_features |  |  |  |  |  |
  | +setup_shaping |  |  |  |  |  |
  | +gentle_shot |  |  |  |  |  |

- **판정**:
- **원본**:
- **재현**:

---

## 4. Search Engine

> `VALIDATION.md §4`. 학습 정책을 후보 제안기로 쓰는 추론 시점 lookahead(greedy/MCTS).

### 4.1 Lookahead vs greedy (탐색 깊이/폭 효과)

- **status**: ⬜ 미실행
- **date**:
- **가설**: 재학습 없이 추론 시점 lookahead만으로 mean을 크게(수십~수백 배) 끌어올릴 수 있다.
- **설정**: proposer=___ / depth h=___ / (K1,M1,K2,M2)=___ / max_shots=___ / n_ep=___.
- **결과**:

  | 방식 | depth | K/M | max_shots | **mean** | max | p(≥k) | wall/ep |
  |---|---|---|---|---|---|---|---|
  | greedy(h=1) |  |  |  |  |  |  |  |
  | lookahead(h=2) |  |  |  |  |  |  |  |

- **판정**:
- **원본**:
- **재현**:

---

## 5. Tuning the Engine

> `VALIDATION.md §5`. 엔진 ranking term(time / robust 등) 토글의 효과 분해.

### 5.1 Robust ranking term (A/B/C 분해)

- **status**: ⬜ 미실행
- **date**:
- **가설**: ranking에 robustness term을 더하면 같은 점수를 더 안전한(high-margin) 액션으로 달성한다.
- **설정**: paired n=___ / max_shots=___ / (k1,k2)=___ / robust(beta, eps, n)=___.
- **결과**:

  | variant | mean_score | mean_chosen_robustness | Δ(robustness) |
  |---|---|---|---|
  | true_baseline |  |  | — |
  | baseline_robust |  |  |  |
  | robust_robust |  |  |  |

- **판정**:
- **원본**:
- **재현**:

---

## 부록: 새 검증 실험 추가하는 법

1. `VALIDATION.md`에 해당 섹션(설계)이 있는지 확인 — 없으면 거기 먼저 추가.
2. 이 문서에서 같은 섹션 번호 아래 `### N.x <제목>` 으로 위 **항목 블록 템플릿**을 복붙.
3. 실험을 돌리고, 산출물은 **기존 위치 그대로** 두고 "원본" 필드에 상대경로만 링크.
4. status를 ⬜→🔄→(✅|❌)로 갱신. `experiments.md` 타임라인에도 한 줄 남긴다.
