# `random_start` only SAC 및 domain-free lookahead 실험 기록

작성일: 2026-05-25 KST

## 공통 목표

`ENGINE.md`의 domain knowledge removal 방향에 맞춰, 사람이 설계한 조준 보정/feature/shaping 없이 성능을 확인한다.

핵심 원칙은 다음과 같다.

| 항목 | 사용 여부 | 설명 |
|---|---:|---|
| `--random_start` | 사용 | 시작 공 배치를 랜덤화해서 일반화된 시작 상태에서 학습/평가 |
| `--constrain_aim` | 미사용 | theta를 목표 공 방향 cone으로 제한하는 조준 보정 제거 |
| `--extra_features` | 미사용 | hand-crafted observation feature 제거 |
| `--gentle_shot` | 미사용 | 다음 포지션을 좋게 만드는 setup 보상 제거 |
| `--setup_shaping` | 미사용 | 비득점 샷의 cue-red 근접 shaping 제거 |
| `--ignore_opponent` | 미사용 | 상대 공을 무시하는 쉬운 curriculum 제거 |
| `--continue_on_miss` | 미사용 | 미스/파울 후에도 계속 치는 보조 규칙 제거 |
| `--foul_penalty` | `0.0` | 파울은 이닝 종료로만 처리하고 추가 보상 조작은 제거 |

---

## 1. Baseline: `random_start` only SAC

### 실행 커맨드

```powershell
uv run python experiments/run_inning_sac.py `
  --algo sac `
  --seed 0 `
  --total_steps 800000 `
  --max_shots 50 `
  --eval_episodes 200 `
  --random_start `
  --foul_penalty 0.0 `
  --n_envs 8 `
  --gradient_steps 2 `
  --out_dir experiments/runs_inning_nodomain/sac_random_only_s0_800k
```

### 학습 설정

| 항목 | 값 |
|---|---:|
| 알고리즘 | SAC |
| seed | 0 |
| total steps | 800,000 |
| max shots | 50 |
| eval episodes | 200 |
| random start | true |
| foul penalty | 0.0 |
| n envs | 8 |
| gradient steps | 2 |
| gamma | 0.99 |
| learning rate | 0.0003 |
| batch size | 256 |
| buffer size | 200,000 |
| learning starts | 1,000 |

### 최종 평가 결과

| 평가 항목 | 값 |
|---|---:|
| train wall time | 6,555.9 sec |
| total wall time | 6,574.5 sec |
| 최종 training window mean | 0.0544 |
| eval mean inning score | 0.090 |
| eval max inning score | 3 |
| eval P(score >= 1) | 7.5% |
| eval P(score >= 3) | 0.5% |
| eval P(score >= 5) | 0.0% |
| eval mean shots | 1.09 |
| eval foul rate | 21.0% |
| eval mean cushions | 3.873 |

### Standard Eval

| 평가 환경 | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | foul rate |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.000 | 0 | 0.0% | 0.0% | 0.0% | 0.0% |
| random | 0.050 | 1 | 5.0% | 0.0% | 0.0% | 19.0% |

### 결론

순수 `random_start + SAC`만으로는 목표였던 평균 1점을 넘지 못했다. 최종 random-start eval 기준 평균은 `0.090`, standard random eval 기준 평균은 `0.050`이다.

---

## 2. Domain-free lookahead 적용

이번 실험은 위 baseline policy를 그대로 두고, 행동 선택 시점에만 lookahead search를 적용했다. `constrain_aim`, `extra_features`, `gentle_shot`, `setup_shaping`, `ignore_opponent`, `continue_on_miss`는 계속 사용하지 않았다.

중요한 해석:

| 구분 | 의미 |
|---|---|
| raw SAC policy | policy가 바로 action 1개를 출력 |
| lookahead policy | policy 주변 action 후보를 여러 개 샘플링하고, engine transition으로 즉시 결과를 본 뒤 best action 선택 |
| domain knowledge 여부 | 조준 cone, handcrafted feature, setup reward는 없음. 다만 lookahead는 simulator를 쓰는 planning/search compute임 |

### 2-1. 1-step lookahead 평가

실행 커맨드:

```powershell
uv run python experiments/lookahead_eval.py `
  --policy experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy.zip `
  --algo sac `
  --ks 20,50,100 `
  --depth 1 `
  --n_episodes 200 `
  --max_shots 50 `
  --mode random `
  --foul_penalty 0.0 `
  --no_setup_shaping `
  --out_dir experiments/runs_inning_nodomain/sac_random_only_s0_800k
```

| K 후보 수 | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | P(score >= 10) | foul rate | wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1.030 | 6 | 51.0% | 14.5% | 4.0% | 0.0% | 22.0% | 100.7 sec |
| 50 | 2.305 | 16 | 67.0% | 30.5% | 16.5% | 3.5% | 23.0% | 290.7 sec |
| 100 | 5.355 | 25 | 85.0% | 60.5% | 44.5% | 18.0% | 25.0% | 1,180.5 sec |

결론: domain knowledge를 켜지 않아도 lookahead를 inference-time search로 쓰면 평균 1점을 확실히 넘었다. 특히 K=100에서는 평균 `5.355`까지 올라갔다.

### 2-2. 2-step lookahead 평가

실행 커맨드:

```powershell
uv run python experiments/lookahead_eval.py `
  --policy experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy.zip `
  --algo sac `
  --ks 50 `
  --depth 2 `
  --k2 5 `
  --beam_width 8 `
  --future_weight 0.99 `
  --n_episodes 200 `
  --max_shots 50 `
  --mode random `
  --foul_penalty 0.0 `
  --no_setup_shaping `
  --out_dir experiments/runs_inning_nodomain/sac_random_only_s0_800k
```

| depth | K | K2 | beam | future weight | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | P(score >= 10) | foul rate | wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 50 | 5 | 8 | 0.99 | 2.930 | 16 | 70.5% | 37.0% | 21.5% | 7.5% | 16.5% | 543.4 sec |

결론: depth=2, K=50은 depth=1, K=50보다 좋았다. 하지만 depth=1, K=100보다는 낮았다. 현재 비용 대비 가장 강한 단일 결과는 depth=1, K=100이다.

---

## 3. Lookahead action distillation

lookahead가 고른 best action을 dataset으로 모아서 actor에 behavior cloning을 적용했다. 목표는 search 없이도 raw policy 자체가 lookahead 행동을 흉내 내게 만드는 것이다.

### 실행 커맨드

```powershell
uv run python experiments/lookahead_distill.py `
  --policy experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy.zip `
  --out_dir experiments/runs_inning_nodomain/sac_random_only_s0_800k `
  --k 100 `
  --n_episodes 200 `
  --max_shots 50 `
  --epochs 20 `
  --batch_size 256 `
  --lr 0.0001 `
  --eval_episodes 200
```

### Dataset 및 BC 결과

| 항목 | 값 |
|---|---:|
| lookahead K | 100 |
| episodes | 200 |
| collected samples | 1,365 |
| obs shape | `(1365, 28)` |
| action shape | `(1365, 4)` |
| epochs | 20 |
| batch size | 256 |
| learning rate | 0.0001 |
| train loss | 0.014049 -> 0.010396 |
| val loss | 0.013396 -> 0.014185 |

### Distilled raw policy 평가

| policy | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | mean shots | foul rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline SAC raw | 0.090 | 3 | 7.5% | 0.5% | 0.0% | 1.09 | 21.0% |
| BC from K=100 lookahead | 0.080 | 2 | 7.5% | 0.0% | 0.0% | 1.08 | 20.0% |

결론: 단순 behavior cloning은 lookahead 성능을 raw actor로 옮기지 못했다. train loss는 내려갔지만 validation loss는 약간 악화되었고, 최종 raw policy 평균도 baseline보다 낮았다.

---

## 4. BC warm-start 후 SAC 200k 추가 학습

BC policy를 초기 policy로 불러온 뒤, domain-free SAC를 200k step 더 학습했다.

### 실행 커맨드

```powershell
uv run python experiments/run_inning_sac.py `
  --algo sac `
  --seed 10 `
  --total_steps 200000 `
  --max_shots 50 `
  --eval_episodes 200 `
  --random_start `
  --foul_penalty 0.0 `
  --n_envs 8 `
  --gradient_steps 2 `
  --load_policy experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy_bc_k100_n200.zip `
  --out_dir experiments/runs_inning_nodomain/sac_random_only_s0_800k_bc_sac200k
```

### 최종 평가 결과

| 평가 항목 | 값 |
|---|---:|
| seed | 10 |
| total steps | 200,000 |
| train wall time | 1,486.0 sec |
| total wall time | 1,499.7 sec |
| eval mean inning score | 0.045 |
| eval max inning score | 2 |
| eval P(score >= 1) | 4.0% |
| eval P(score >= 3) | 0.0% |
| eval P(score >= 5) | 0.0% |
| eval mean shots | 1.045 |
| eval foul rate | 17.0% |
| eval mean cushions | 3.575 |

### Standard Eval

| 평가 환경 | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | foul rate |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.000 | 0 | 0.0% | 0.0% | 0.0% | 0.0% |
| random | 0.060 | 1 | 6.0% | 0.0% | 0.0% | 18.5% |

결론: BC warm-start 후 200k SAC 추가 학습도 raw policy 성능을 개선하지 못했다. main random-start eval 평균은 `0.045`로 baseline `0.090`보다 낮았다.

---

## 5. Lookahead rollout 기반 state value 실험

이번에는 K=100 lookahead teacher rollout에서 방문한 상태들로 별도 state-value model을 학습했다. 목표는 action 자체를 BC로 복제하는 대신, 다음 상태의 좋고 나쁨을 `V(s)`로 예측하게 만드는 것이다.

학습 target은 각 teacher trajectory에서 현재 state 이후 남은 총 득점이다.

```text
V(s_t) ~= score_t + score_{t+1} + ... + score_T
```

평가 때는 후보 action을 한 번씩 engine으로 미리 실행한 뒤 다음 점수로 선택했다.

```text
candidate_score = immediate_reward + beta * gamma * V(next_state)
```

### 실행 커맨드

```powershell
uv run python experiments/value_lookahead.py `
  --policy experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy.zip `
  --dataset experiments/runs_inning_nodomain/sac_random_only_s0_800k/lookahead_bc_k100_n200.npz `
  --meta experiments/runs_inning_nodomain/sac_random_only_s0_800k/lookahead_bc_k100_n200_meta.parquet `
  --out_dir experiments/runs_inning_nodomain/value_lookahead_k50_beta01_n200 `
  --ks 50 `
  --betas 0.1 `
  --n_episodes 200 `
  --seed_base 99000 `
  --epochs 300 `
  --max_shots 50 `
  --policy_seed 12345
```

### Value model 학습 데이터

| 항목 | 값 |
|---|---:|
| teacher | 1-step lookahead K=100 |
| source episodes | 200 |
| value samples | 1,365 |
| target | return-to-go, 남은 총 득점 |
| target mean | 5.421 |
| target std | 5.625 |
| target min / max | 0 / 35 |
| best validation loss | 1.890 |

주의할 점: validation R2는 좋지 않았다. 예측값이 실제 return range 전체를 잘 따라가기보다는 대략적인 평균 근처에 몰리는 경향이 있었다. 그래도 action selection에서는 `beta=0.1`로 약하게 섞었을 때 K=50 기준 성능이 올라갔다.

### K=20 beta sweep

| K | beta | episodes | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | P(score >= 10) | foul rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.00 | 100 | 1.010 | 5 | 48.0% | 16.0% | 3.0% | 0.0% | 23.0% |
| 20 | 0.05 | 100 | 1.010 | 8 | 48.0% | 12.0% | 3.0% | 0.0% | 21.0% |
| 20 | 0.10 | 100 | 1.010 | 8 | 48.0% | 12.0% | 3.0% | 0.0% | 21.0% |
| 20 | 0.20 | 100 | 1.010 | 8 | 48.0% | 12.0% | 3.0% | 0.0% | 21.0% |

K=20에서는 평균 개선은 없었다. value term이 long run max를 조금 늘렸지만 평균 점수에는 거의 영향을 주지 못했다.

### K=50 확인 실험

| K | beta | episodes | seed base | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | P(score >= 10) | foul rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.00 | 100 | 160000 | 2.490 | 13 | 69.0% | 38.0% | 22.0% | 3.0% | 34.0% |
| 50 | 0.10 | 100 | 160000 | 2.770 | 21 | 68.0% | 34.0% | 21.0% | 6.0% | 23.0% |
| 50 | 0.10 | 200 | 99000 | 2.875 | 20 | 75.0% | 39.0% | 23.5% | 5.0% | 21.5% |

같은 200 episode seed base에서 기존 immediate-only K=50은 mean `2.305`였다. value를 섞은 K=50, beta=0.1은 mean `2.875`로 올라갔다. 이는 depth=2 K=50의 mean `2.930`에 거의 근접하면서 wall time은 `407.3 sec`로 depth=2의 `543.4 sec`보다 낮았다.

### 결론

state-value 방식은 단순 BC보다 훨씬 가능성이 있어 보인다. 현재 V model 자체의 validation 품질은 낮지만, `reward + 0.1 * gamma * V(next_state)`로 약하게 섞으면 K=50 lookahead에서 평균 점수가 개선되었다.

다만 이 결과는 아직 “좋은 신호” 수준이다. V가 진짜로 좋은 배치를 잘 이해하고 있다기보다는, 일부 후보 선택에서 long-run 가능성을 건드린 정도로 보인다. 다음 개선은 value dataset을 더 키우거나, 한 state당 여러 teacher rollout을 돌려 MC return noise를 줄이는 쪽이 맞다.

### 5-2. 같은 state에서 teacher rollout 여러 번 돌리기

위 single-label value는 한 teacher trajectory의 return-to-go만 target으로 썼다. 그래서 특정 state의 value label이 우연히 너무 높거나 낮을 수 있다. 이를 줄이기 위해 일부 state를 정확히 복원한 뒤, 같은 state에서 lookahead teacher rollout을 여러 번 돌려 평균 return을 label로 사용했다.

구현 방식:

```text
1. 기존 lookahead dataset의 ep seed와 저장된 teacher action을 replay
2. 선택된 row의 내부 table state를 snapshot
3. 같은 snapshot에서 teacher lookahead rollout을 N번 실행
4. 남은 득점 평균을 MC value label로 저장
5. 기존 single-label dataset에 MC label을 섞어서 V(s) 재학습
```

추가한 스크립트:

| 스크립트 | 역할 |
|---|---|
| `experiments/value_mc_lookahead.py` | MC value label 수집, value model 학습, value-guided lookahead 평가 |
| `experiments/eval_value_lookahead_model.py` | 저장된 value model만 재사용해서 평가 |

### MC label 실험 결과

| 설정 | teacher K | MC rollouts/state | MC states | MC repeat | eval episodes | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | P(score >= 10) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weak teacher MC | 10 | 3 | 60 | 8 | 100 | 2.410 | 10 | 72.0% | 44.0% | 18.0% | 2.0% |
| strong teacher MC | 50 | 3 | 60 | 8 | 100 | 2.920 | 22 | 72.0% | 44.0% | 22.0% | 5.0% |
| strong teacher MC | 50 | 3 | 60 | 8 | 200 | 2.680 | 22 | 72.5% | 39.0% | 20.0% | 4.0% |
| strong teacher MC | 50 | 3 | 60 | 2 | 200 | 2.875 | 26 | 75.0% | 39.0% | 22.0% | 5.5% |
| larger MC dataset | 50 | 3 | 400 | 2 | 200 | 2.195 | 16 | 68.5% | 30.5% | 16.5% | 2.5% |

비교 기준:

| 방법 | eval episodes | mean | max | P(score >= 1) | P(score >= 5) | P(score >= 10) |
|---|---:|---:|---:|---:|---:|---:|
| immediate-only K=50 | 200 | 2.305 | 16 | 67.0% | 16.5% | 3.5% |
| single-label value K=50 beta=0.1 | 200 | 2.875 | 20 | 75.0% | 23.5% | 5.0% |
| MC value K=50 beta=0.1, repeat=2 | 200 | 2.875 | 26 | 75.0% | 22.0% | 5.5% |
| MC value K=50 beta=0.1, 400 states | 200 | 2.195 | 16 | 68.5% | 16.5% | 2.5% |

해석:

MC label 자체는 의미가 있었다. 선택된 60개 state에서 기존 single return 평균은 `4.967`이었지만, K=50 teacher를 3번씩 다시 굴린 MC 평균은 `2.244`였다. 즉 기존 single trajectory label이 일부 state를 꽤 과대평가하고 있었다.

다만 MC label을 너무 강하게 반복한 `mc_repeat=8`은 평균 성능을 `2.680`으로 낮췄다. 반대로 `mc_repeat=2`로 약하게 섞으면 평균은 single-label value와 같은 `2.875`를 유지하면서 max가 `20 -> 26`, P(score >= 10)이 `5.0% -> 5.5%`로 조금 좋아졌다.

결론적으로, 이번 작은 MC 실험은 평균 성능을 더 올리지는 못했지만 high-score tail은 개선했다. 더 큰 이득을 보려면 MC state 수를 60개보다 크게 늘리고, `mc_repeat`를 낮게 유지하는 쪽이 좋아 보인다.

추가로 `MC states=400`, `teacher K=50`, `MC rollouts/state=3`, `mc_repeat=2`도 실행했다. 이 경우 MC label 통계는 다음과 같았다.

| 항목 | 값 |
|---|---:|
| MC states | 400 |
| single return mean | 5.230 |
| MC return mean | 2.361 |
| MC return std mean | 1.676 |
| mean abs delta | 4.208 |
| obs replay max abs diff | 0.0 |
| augmented value samples | 1,765 |
| best validation loss | 1.760 |

겉으로는 value validation loss가 좋아졌지만, 최종 policy 성능은 `mean=2.195`로 떨어졌다. 즉 MC state를 무작정 늘리면 label은 더 보수적으로 안정되지만, value-guided action selection이 high-return 후보를 덜 고르게 되는 문제가 생겼다. 이번 결과 기준으로는 “MC state 수 확대” 자체보다, MC label을 평균값 하나로 회귀시키는 방식을 바꾸는 것이 더 중요해 보인다.

---

## 6. Single-policy adaptive h=2 domain-free

기존 `Search-augmented Engine SOTA`는 multi-seed policy proposal, h=2 greedy lookahead, adaptive K schedule을 함께 쓴 방식이었다. 이번에는 domain-free 조건을 유지해야 하므로, 현재 사용 가능한 domain-free SAC policy 하나만으로 그 구조를 최대한 가깝게 재현했다.

중요한 차이는 다음과 같다.

| 항목 | 기존 SOTA | 이번 domain-free 실험 |
|---|---|---|
| proposal policy | seed 1, 4, 6의 여러 policy | `sac_random_only_s0_800k/policy.zip` 단일 policy |
| domain knowledge | 사용 | 전부 사용하지 않음 |
| search depth | h=2 | h=2 |
| K schedule | shot 수에 따라 adaptive | `200,100,50,30` |
| second-step rollout | 사용 | `k2=5` |
| 시작 상태 | random start 평가 | random start 평가 |
| max shots | long-chain 허용 | 200 |

사용한 domain-free 환경 flag는 다음과 같다.

| flag | 값 |
|---|---:|
| `random_start` | true |
| `continue_on_miss` | false |
| `constrain_aim` | false |
| `extra_features` | false |
| `gentle_shot` | false |
| `setup_shaping` | false |
| `foul_penalty` | 0.0 |

실행 커맨드:

```powershell
uv run python experiments\adaptive_h2_nodomain.py `
  --policy experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy.zip `
  --out_dir experiments/runs_inning_nodomain/adaptive_h2_single_s0_n100_m200 `
  --n_episodes 100 `
  --seed_base 99000 `
  --max_shots 200 `
  --schedule 200,100,50,30 `
  --k2 5 `
  --beam_width 0 `
  --policy_seed 12345
```

`beam_width=0`은 1-step 후보를 줄이지 않고 모두 h=2로 확장한다는 뜻이다. 즉 계산량은 크지만, 단일 policy로 할 수 있는 h=2 search를 꽤 정직하게 돌린 설정이다.

### 결과

| 설정 | episodes | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | P(score >= 10) | foul rate | wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive h=2 single policy | 20 | 6.300 | 13 | 100.0% | 80.0% | 65.0% | 20.0% | 15.0% | 163.8 sec |
| adaptive h=2 single policy | 100 | 6.020 | 17 | 92.0% | 79.0% | 66.0% | 18.0% | 14.0% | 828.3 sec |

### 비교

| 방법 | episodes | mean | max | P(score >= 1) | P(score >= 3) | P(score >= 5) | P(score >= 10) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-step lookahead K=100 | 200 | 5.355 | 25 | 85.0% | 60.5% | 44.5% | 18.0% |
| 2-step lookahead K=50 | 200 | 2.930 | 16 | 70.5% | 37.0% | 21.5% | 7.5% |
| value lookahead K=50 beta=0.1 | 200 | 2.875 | 20 | 75.0% | 39.0% | 23.5% | 5.0% |
| single-policy adaptive h=2 | 100 | 6.020 | 17 | 92.0% | 79.0% | 66.0% | 18.0% |

해석: single-policy adaptive h=2는 이 시점까지의 domain-free 실험 중 평균 점수 기준 최고 결과였다. 특히 `P(score >= 1)`, `P(score >= 3)`, `P(score >= 5)`가 크게 좋아져서 안정성은 확실히 올라갔다. 다만 max는 `17`에서 멈췄고, `P(score >= 20)`은 `0.0%`였다. 즉 old SOTA처럼 긴 연속 득점이 폭발하지는 않았다.

현재 결과만 보면 old SOTA의 폭발력은 단순 h=2 search만으로 나온 것이 아니라, domain knowledge가 들어간 policy prior와 multi-seed proposal diversity가 함께 만든 효과였을 가능성이 크다. domain-free에서 SOTA 구조를 더 밀어보려면 다음 단계는 단일 policy search를 더 키우는 것보다 domain-free SAC seed를 여러 개 학습해서 multi-policy proposal을 복원하는 쪽이 더 타당해 보인다.

---

## 7. Compute-matched h=2와 multi-seed domain-free

이번에는 old SOTA의 search-augmented engine 구조를 domain-free 조건에 최대한 맞춰서 다시 실험했다. 핵심은 두 가지를 분리해서 보는 것이다.

1. 단일 policy라도 후보 수 K를 3배로 늘리면 좋아지는가?
2. 같은 총 후보 수에서 seed가 다른 policy 3개를 쓰면 더 좋아지는가?

모든 실험은 여전히 domain knowledge를 쓰지 않았다. 즉 `--constrain_aim`, `--extra_features`, `--gentle_shot`, `--setup_shaping`, `--ignore_opponent`, `--continue_on_miss`를 켜지 않았다. 환경은 random start만 켰고, foul penalty는 `0.0`이다.

### 추가 domain-free seed 학습

seed0과 같은 설정으로 seed1, seed2를 800k step 추가 학습했다.

```powershell
uv run python experiments/run_inning_sac.py `
  --algo sac `
  --seed 1 `
  --total_steps 800000 `
  --max_shots 50 `
  --eval_episodes 200 `
  --random_start `
  --foul_penalty 0.0 `
  --n_envs 8 `
  --gradient_steps 2 `
  --out_dir experiments/runs_inning_nodomain/sac_random_only_s1_800k

uv run python experiments/run_inning_sac.py `
  --algo sac `
  --seed 2 `
  --total_steps 800000 `
  --max_shots 50 `
  --eval_episodes 200 `
  --random_start `
  --foul_penalty 0.0 `
  --n_envs 8 `
  --gradient_steps 2 `
  --out_dir experiments/runs_inning_nodomain/sac_random_only_s2_800k
```

| seed | raw eval mean | raw max | raw P(score >= 1) | raw P(score >= 3) | raw foul rate | standard random mean | standard random max | standard random P(score >= 1) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.090 | 3 | 7.5% | 0.5% | 21.0% | 0.050 | 1 | 5.0% |
| 1 | 0.090 | 2 | 8.0% | 0.0% | 21.5% | 0.050 | 1 | 5.0% |
| 2 | 0.070 | 1 | 7.0% | 0.0% | 20.0% | 0.050 | 2 | 4.5% |

추가 seed의 raw policy 자체는 seed0보다 뚜렷하게 강해지지 않았다. 따라서 multi-seed 실험의 목적은 raw mean 개선이 아니라, search 단계에서 서로 다른 SAC seed가 다른 후보 action을 제안해 proposal diversity를 만드는지 확인하는 것이다.

### Single-policy compute-matched h=2

multi-seed의 총 후보 수와 맞추기 위해 단일 seed0 policy의 K를 3배로 늘렸다.

```powershell
uv run python experiments/adaptive_h2_nodomain.py `
  --policy experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy.zip `
  --out_dir experiments/runs_inning_nodomain/adaptive_h2_single_s0_compute_matched_n100_m200 `
  --n_episodes 100 `
  --seed_base 99000 `
  --max_shots 200 `
  --schedule 600,300,150,90 `
  --k2 5 `
  --beam_width 0 `
  --policy_seed 12345
```

이 설정은 첫 shot에서 K=600, shot 1-3에서 K=300, shot 4-9에서 K=150, 이후 K=90을 쓴다. 이전 single-policy adaptive h=2의 `200,100,50,30`보다 정확히 3배 큰 후보 수다.

| 방법 | policies | K schedule | total K1 candidates | episodes | mean | max | P(score >= 1) | P(score >= 5) | P(score >= 10) | P(score >= 20) | P(score >= 30) | P(score >= 50) | foul rate | wall time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-policy adaptive h=2 | seed0 | 200,100,50,30 | 200,100,50,30 | 100 | 6.020 | 17 | 92.0% | 66.0% | 18.0% | 0.0% | 0.0% | 0.0% | 14.0% | 828.3 sec |
| single-policy compute-matched h=2 | seed0 | 600,300,150,90 | 600,300,150,90 | 100 | 15.650 | 69 | 97.0% | 90.0% | 73.0% | 22.0% | 11.0% | 3.0% | 27.0% | 7,855.0 sec |

결과적으로 후보 수 자체가 매우 중요했다. 같은 seed0 policy만 써도 mean이 `6.020`에서 `15.650`으로 올랐고, max도 `17`에서 `69`까지 올라갔다. 즉 이전 adaptive h=2에서 long-chain이 약했던 이유 중 하나는 domain-free policy 품질만이 아니라 search compute도 부족했기 때문이다.

### Multi-seed adaptive h=2 domain-free

다음으로 old SOTA 엔진의 핵심 구조처럼 첫 번째 step 후보를 seed0/seed1/seed2 policy에서 모았다. 단, 총 후보 수는 compute-matched single과 같게 맞췄다.

```powershell
uv run python experiments/adaptive_h2_multiseed_nodomain.py `
  --policies `
    experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy.zip `
    experiments/runs_inning_nodomain/sac_random_only_s1_800k/policy.zip `
    experiments/runs_inning_nodomain/sac_random_only_s2_800k/policy.zip `
  --out_dir experiments/runs_inning_nodomain/adaptive_h2_multiseed_s0_s1_s2_n100_m200 `
  --n_episodes 100 `
  --seed_base 99000 `
  --max_shots 200 `
  --schedule 200,100,50,30 `
  --k2 5 `
  --beam_width 0 `
  --main_policy_index 0 `
  --policy_seed 12345
```

여기서 `schedule=200,100,50,30`은 policy 하나당 후보 수다. policy가 3개이므로 실제 총 K1 후보 수는 `600,300,150,90`이고, compute-matched single과 같다. h=2의 두 번째 step 후보는 seed0 policy를 main policy로 사용했다.

| 방법 | policies | per-policy K schedule | total K1 candidates | episodes | mean | max | P(score >= 1) | P(score >= 5) | P(score >= 10) | P(score >= 20) | P(score >= 30) | P(score >= 50) | foul rate | wall time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-policy compute-matched h=2 | seed0 | 600,300,150,90 | 600,300,150,90 | 100 | 15.650 | 69 | 97.0% | 90.0% | 73.0% | 22.0% | 11.0% | 3.0% | 27.0% | 7,855.0 sec |
| multi-seed adaptive h=2 | seed0/1/2 | 200,100,50,30 | 600,300,150,90 | 100 | 19.110 | 72 | 98.0% | 93.0% | 78.0% | 40.0% | 23.0% | 2.0% | 23.0% | 6,163.3 sec |

선택된 action이 어느 policy에서 왔는지도 확인했다.

| 선택된 첫-step 후보 policy | 평균 선택 비율 |
|---|---:|
| seed0 | 70.4% |
| seed1 | 19.5% |
| seed2 | 10.1% |

해석: seed0이 여전히 주력이지만, seed1/seed2 후보도 약 29.6% 선택됐다. 즉 multi-seed는 단순히 seed0을 반복한 것이 아니라 실제로 다른 seed의 action proposal을 활용했다. 같은 총 후보 수에서 mean은 `15.650 -> 19.110`, `P(score >= 20)`은 `22.0% -> 40.0%`, `P(score >= 30)`은 `11.0% -> 23.0%`로 올랐다. 다만 `P(score >= 50)`은 `3.0% -> 2.0%`로 표본상 약간 낮았고, old SOTA처럼 수백 점 이상으로 폭발하지는 않았다.

결론: domain-free에서도 old SOTA 엔진의 multi-seed proposal 구조는 효과가 있다. 하지만 old SOTA의 수백 점 scale은 search 구조만으로 나온 것이 아니라, domain knowledge가 들어간 policy prior와 함께 만들어진 결과였을 가능성이 높다.

---

## 8. Domain knowledge 추가 multi-seed adaptive h=2

이번에는 section 7의 multi-seed adaptive h=2와 같은 search 구조에 domain knowledge를 다시 넣었다. 목적은 domain-free 결과가 왜 old SOTA scale까지 가지 못했는지 확인하는 것이다.

차이점은 다음과 같다.

| 항목 | domain-free multi-seed | domain-knowledge multi-seed |
|---|---|---|
| proposal policies | seed0/1/2 domain-free SAC | seed1/4/6 domain-knowledge SAC |
| random_start | true | true |
| continue_on_miss 평가 | false | false |
| constrain_aim | false | true |
| extra_features | false | true |
| gentle_shot | false | true |
| setup_shaping | false | true |
| foul_penalty | 0.0 | 0.2 |
| per-policy K schedule | 200,100,50,30 | 200,100,50,30 |
| total K1 candidates | 600,300,150,90 | 600,300,150,90 |
| h=2 K2 | 5 | 5 |

주의: domain-knowledge 쪽은 계산량이 매우 커서 `n=10`, `max_shots=200`으로 별도 diagnostic 평가만 했다. section 7의 domain-free multi-seed는 `n=100`, `max_shots=200`이므로 표본 수는 다르다. 하지만 같은 `seed_base=99000`이고 같은 max cap을 쓰므로 스케일 차이는 확인할 수 있다.

### domain-trained seed 재학습

로컬에 old SOTA에서 쓰던 seed1/6 `policy.zip`이 없어서 같은 설정으로 재학습했다. seed4는 기존 `fast_long_fp02_s4_retrain/policy.zip`을 사용했다.

```powershell
uv run python experiments/run_inning_sac.py `
  --algo sac `
  --seed 1 `
  --total_steps 800000 `
  --max_shots 10 `
  --eval_episodes 200 `
  --continue_on_miss `
  --constrain_aim `
  --extra_features `
  --random_start `
  --foul_penalty 0.2 `
  --gentle_shot `
  --setup_shaping `
  --setup_alpha 0.05 `
  --setup_scale 0.3 `
  --n_envs 8 `
  --gradient_steps 2 `
  --out_dir experiments/runs_inning_v2/fast_long_fp02_s1_retrain

uv run python experiments/run_inning_sac.py `
  --algo sac `
  --seed 6 `
  --total_steps 800000 `
  --max_shots 10 `
  --eval_episodes 200 `
  --continue_on_miss `
  --constrain_aim `
  --extra_features `
  --random_start `
  --foul_penalty 0.2 `
  --gentle_shot `
  --setup_shaping `
  --setup_alpha 0.05 `
  --setup_scale 0.3 `
  --n_envs 8 `
  --gradient_steps 2 `
  --out_dir experiments/runs_inning_v2/fast_long_fp02_s6_retrain
```

| seed | train/eval run | train-mode mean | train-mode max | train-mode P(score >= 5) | standard random mean | standard random max | standard random P(score >= 1) | standard random P(score >= 5) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | fast_long_fp02_s1_retrain | 6.205 | 10 | 84.5% | 1.450 | 12 | 53.0% | 8.0% |
| 4 | fast_long_fp02_s4_retrain | 5.310 | 10 | 69.0% | 0.977 | 12 | 45.5% | 4.1% |
| 6 | fast_long_fp02_s6_retrain | 6.055 | 10 | 80.5% | 1.370 | 13 | 52.5% | 6.0% |

domain-free seed들의 standard random mean이 모두 `0.050` 근처였던 것과 비교하면, domain knowledge가 들어간 policy prior 자체가 이미 훨씬 강하다.

### 실행 커맨드

```powershell
uv run python experiments/adaptive_h2_multiseed_nodomain.py `
  --policies `
    experiments/runs_inning_v2/fast_long_fp02_s1_retrain/policy.zip `
    experiments/runs_inning_v2/fast_long_fp02_s4_retrain/policy.zip `
    experiments/runs_inning_v2/fast_long_fp02_s6_retrain/policy.zip `
  --out_dir experiments/runs_inning_v2/adaptive_h2_multiseed_domain_s1_s4_s6_n10_m200 `
  --n_episodes 10 `
  --seed_base 99000 `
  --max_shots 200 `
  --schedule 200,100,50,30 `
  --k2 5 `
  --beam_width 0 `
  --main_policy_index 1 `
  --policy_seed 12345 `
  --constrain_aim `
  --extra_features `
  --foul_penalty 0.2 `
  --setup_shaping `
  --setup_alpha 0.05 `
  --setup_scale 0.3 `
  --gentle_shot `
  --output_prefix adaptive_h2_multiseed_domain
```

### 결과

| 방법 | domain knowledge | policies | episodes | max_shots | mean | max | P(score >= 1) | P(score >= 50) | P(score >= 100) | P(score >= 200) | foul rate | wall time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi-seed adaptive h=2 | 없음 | seed0/1/2 | 100 | 200 | 19.110 | 72 | 98.0% | 2.0% | 0.0% | 0.0% | 23.0% | 6,163.3 sec |
| multi-seed adaptive h=2 | 있음 | seed1/4/6 | 10 | 200 | 180.000 | 200 | 90.0% | 90.0% | 90.0% | 90.0% | 10.0% | 7,220.5 sec |

episode별 점수는 다음과 같다.

| ep | seed | score | shots | note |
|---:|---:|---:|---:|---|
| 0 | 99000 | 0 | 1 | 첫 샷 foul |
| 1 | 99001 | 200 | 200 | cap 도달 |
| 2 | 99002 | 200 | 200 | cap 도달 |
| 3 | 99003 | 200 | 200 | cap 도달 |
| 4 | 99004 | 200 | 200 | cap 도달 |
| 5 | 99005 | 200 | 200 | cap 도달 |
| 6 | 99006 | 200 | 200 | cap 도달 |
| 7 | 99007 | 200 | 200 | cap 도달 |
| 8 | 99008 | 200 | 200 | cap 도달 |
| 9 | 99009 | 200 | 200 | cap 도달 |

선택된 첫-step 후보 policy 비율은 다음과 같다.

| policy | 평균 선택 비율 |
|---|---:|
| seed1 | 38.9% |
| seed4 | 35.3% |
| seed6 | 25.8% |

첫 episode는 seed1 후보 하나가 바로 foul로 끝난 케이스라 전체 평균에 조금 영향을 준다. cap에 도달한 9개 episode만 보면 세 policy가 더 고르게 쓰였다. 즉 domain-knowledge multi-seed는 특정 policy 하나가 독식한 것이 아니라, 세 seed의 proposal을 모두 활용했다.

해석: domain knowledge를 추가하면 같은 multi-seed adaptive h=2 구조가 완전히 다른 regime으로 들어간다. domain-free 최고가 mean `19.110`, max `72`였는데, domain-knowledge 버전은 10개 중 9개 episode가 `max_shots=200` cap에 도달했다. 따라서 이 설정에서는 실제 성능 천장이 200이 아니라 cap에 막혀 있고, 더 큰 `max_shots=500` 또는 `2000`으로 재평가해야 old SOTA scale을 제대로 측정할 수 있다.

---

## 전체 결론

| 방법 | domain knowledge | search 사용 | raw policy 학습 | mean | max | P(score >= 1) | 핵심 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline SAC | 없음 | 없음 | 800k SAC | 0.090 | 3 | 7.5% | 순수 SAC만으로 평균 1점 실패 |
| 1-step lookahead K=20 | 없음 | 있음 | baseline 그대로 | 1.030 | 6 | 51.0% | 목표 평균 1점 달성 |
| 1-step lookahead K=50 | 없음 | 있음 | baseline 그대로 | 2.305 | 16 | 67.0% | search compute 증가로 성능 상승 |
| 1-step lookahead K=100 | 없음 | 있음 | baseline 그대로 | 5.355 | 25 | 85.0% | 강한 단일-depth 기준선 |
| 2-step lookahead K=50 | 없음 | 있음 | baseline 그대로 | 2.930 | 16 | 70.5% | 같은 K=50에서는 depth=1보다 좋음 |
| Single-policy adaptive h=2 | 없음 | 있음 | baseline 그대로 | 6.020 | 17 | 92.0% | adaptive K+h=2의 초기 기준선 |
| Single-policy compute-matched h=2 | 없음 | 있음 | baseline 그대로 | 15.650 | 69 | 97.0% | 후보 수를 3배로 맞추자 long-chain이 크게 늘어남 |
| Multi-seed adaptive h=2 | 없음 | 있음 | seed0/1/2 800k SAC | 19.110 | 72 | 98.0% | 현재 domain-free 최고 평균, multi-seed proposal diversity 효과 확인 |
| Multi-seed adaptive h=2 | 있음 | 있음 | seed1/4/6 800k SAC | 180.000 | 200 | 90.0% | n=10, max_shots=200에서 9/10 cap 도달 |
| Value lookahead K=50 beta=0.1 | 없음 | 있음 | K=100 rollout으로 V 학습 | 2.875 | 20 | 75.0% | depth=2에 가까운 성능, 비용은 더 낮음 |
| MC value lookahead K=50 beta=0.1 | 없음 | 있음 | 60개 state에 MC value label 추가 | 2.875 | 26 | 75.0% | 평균은 동일, high-score tail 개선 |
| MC value lookahead K=50 beta=0.1 | 없음 | 있음 | 400개 state에 MC value label 추가 | 2.195 | 16 | 68.5% | label은 안정됐지만 평균 성능 하락 |
| K=100 BC raw | 없음 | 없음 | BC distill | 0.080 | 2 | 7.5% | search 행동이 actor로 잘 전달되지 않음 |
| K=100 BC + SAC 200k | 없음 | 없음 | BC 후 SAC | 0.045 | 2 | 4.0% | 추가 SAC도 개선 실패 |

최종적으로, “domain knowledge 없이 평균 1점을 넘기기”는 lookahead를 inference-time search로 사용할 때 성공했다. 가장 강한 domain-free 결과는 multi-seed adaptive h=2의 mean `19.110`, max `72`다. 단일 policy라도 compute를 크게 늘리면 mean `15.650`까지 올라가고, 같은 총 후보 수에서 seed0/1/2를 섞으면 mean `19.110`까지 더 올라갔다. 따라서 domain-free에서도 search compute와 multi-seed proposal diversity는 모두 효과가 있다.

다만 domain-free에서는 old SOTA처럼 수백 점 이상의 long-chain이 나오지 않았다. 반대로 domain knowledge를 넣은 multi-seed adaptive h=2는 `max_shots=200`에서 10개 중 9개가 cap에 도달했다. 이는 old SOTA의 폭발력이 search 구조만이 아니라 domain knowledge가 들어간 policy prior와 결합되어 나왔다는 해석을 강하게 지지한다. search-free actor가 필요하다면 단순 MSE BC보다 더 강한 policy improvement 방식이 필요하다. 예를 들면 더 큰 lookahead dataset, on-policy DAgger식 반복 수집, replay buffer에 lookahead action trajectory를 섞는 방식, 또는 offline RL/advantage-weighted regression 쪽이 후보이다.

---

## 산출물

| 파일 | 경로 |
|---|---|
| baseline config | `experiments/runs_inning_nodomain/sac_random_only_s0_800k/config.json` |
| baseline policy | `experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy.zip` |
| baseline training curve | `experiments/runs_inning_nodomain/sac_random_only_s0_800k/training_curve.csv` |
| baseline eval summary | `experiments/runs_inning_nodomain/sac_random_only_s0_800k/summary.json` |
| lookahead depth=1 summary | `experiments/runs_inning_nodomain/sac_random_only_s0_800k/lookahead_random_d1_m50_k20_50_100_summary.json` |
| lookahead depth=2 summary | `experiments/runs_inning_nodomain/sac_random_only_s0_800k/lookahead_random_d2_m50_k50_summary.json` |
| BC dataset | `experiments/runs_inning_nodomain/sac_random_only_s0_800k/lookahead_bc_k100_n200.npz` |
| BC policy | `experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy_bc_k100_n200.zip` |
| BC summary | `experiments/runs_inning_nodomain/sac_random_only_s0_800k/policy_bc_k100_n200_summary.json` |
| BC + SAC policy | `experiments/runs_inning_nodomain/sac_random_only_s0_800k_bc_sac200k/policy.zip` |
| BC + SAC summary | `experiments/runs_inning_nodomain/sac_random_only_s0_800k_bc_sac200k/summary.json` |
| BC + SAC standard eval | `experiments/runs_inning_nodomain/sac_random_only_s0_800k_bc_sac200k/eval_summary.json` |
| value lookahead script | `experiments/value_lookahead.py` |
| value K=20 sweep | `experiments/runs_inning_nodomain/value_lookahead_k20_sweep/value_lookahead_summary.json` |
| value K=50 check | `experiments/runs_inning_nodomain/value_lookahead_k50_check/value_lookahead_summary.json` |
| value K=50 beta=0.1 n=200 | `experiments/runs_inning_nodomain/value_lookahead_k50_beta01_n200/value_lookahead_summary.json` |
| MC value lookahead script | `experiments/value_mc_lookahead.py` |
| saved value-model eval script | `experiments/eval_value_lookahead_model.py` |
| MC value K=10 teacher | `experiments/runs_inning_nodomain/value_mc_k10_s60_r3/mc_value_lookahead_summary.json` |
| MC value K=50 teacher repeat=8 | `experiments/runs_inning_nodomain/value_mc_k50_s60_r3/mc_value_lookahead_summary.json` |
| MC value K=50 repeat=8 n=200 eval | `experiments/runs_inning_nodomain/value_mc_k50_s60_r3_eval_n200/value_model_eval_summary.json` |
| MC value K=50 repeat=2 n=200 | `experiments/runs_inning_nodomain/value_mc_k50_s60_r3_repeat2/mc_value_lookahead_summary.json` |
| MC value K=50 repeat=2 states=400 n=200 | `experiments/runs_inning_nodomain/value_mc_k50_s400_r3_repeat2/mc_value_lookahead_summary.json` |
| adaptive h=2 domain-free script | `experiments/adaptive_h2_nodomain.py` |
| adaptive h=2 n=20 summary | `experiments/runs_inning_nodomain/adaptive_h2_single_s0_n20_m200/adaptive_h2_nodomain_summary.json` |
| adaptive h=2 n=100 summary | `experiments/runs_inning_nodomain/adaptive_h2_single_s0_n100_m200/adaptive_h2_nodomain_summary.json` |
| domain-free seed1 policy | `experiments/runs_inning_nodomain/sac_random_only_s1_800k/policy.zip` |
| domain-free seed1 summary | `experiments/runs_inning_nodomain/sac_random_only_s1_800k/summary.json` |
| domain-free seed1 standard eval | `experiments/runs_inning_nodomain/sac_random_only_s1_800k/eval_summary.json` |
| domain-free seed2 policy | `experiments/runs_inning_nodomain/sac_random_only_s2_800k/policy.zip` |
| domain-free seed2 summary | `experiments/runs_inning_nodomain/sac_random_only_s2_800k/summary.json` |
| domain-free seed2 standard eval | `experiments/runs_inning_nodomain/sac_random_only_s2_800k/eval_summary.json` |
| adaptive h=2 compute-matched summary | `experiments/runs_inning_nodomain/adaptive_h2_single_s0_compute_matched_n100_m200/adaptive_h2_nodomain_summary.json` |
| adaptive h=2 compute-matched eval rows | `experiments/runs_inning_nodomain/adaptive_h2_single_s0_compute_matched_n100_m200/adaptive_h2_nodomain_eval.parquet` |
| multi-seed adaptive h=2 domain-free script | `experiments/adaptive_h2_multiseed_nodomain.py` |
| multi-seed adaptive h=2 summary | `experiments/runs_inning_nodomain/adaptive_h2_multiseed_s0_s1_s2_n100_m200/adaptive_h2_multiseed_nodomain_summary.json` |
| multi-seed adaptive h=2 eval rows | `experiments/runs_inning_nodomain/adaptive_h2_multiseed_s0_s1_s2_n100_m200/adaptive_h2_multiseed_nodomain_eval.parquet` |
| domain-knowledge seed1 retrain policy | `experiments/runs_inning_v2/fast_long_fp02_s1_retrain/policy.zip` |
| domain-knowledge seed1 retrain summary | `experiments/runs_inning_v2/fast_long_fp02_s1_retrain/summary.json` |
| domain-knowledge seed6 retrain policy | `experiments/runs_inning_v2/fast_long_fp02_s6_retrain/policy.zip` |
| domain-knowledge seed6 retrain summary | `experiments/runs_inning_v2/fast_long_fp02_s6_retrain/summary.json` |
| multi-seed adaptive h=2 domain-knowledge summary | `experiments/runs_inning_v2/adaptive_h2_multiseed_domain_s1_s4_s6_n10_m200/adaptive_h2_multiseed_domain_summary.json` |
| multi-seed adaptive h=2 domain-knowledge eval rows | `experiments/runs_inning_v2/adaptive_h2_multiseed_domain_s1_s4_s6_n10_m200/adaptive_h2_multiseed_domain_eval.parquet` |

---

## 9. Privileged domain-knowledge teacher distillation

목표: domain-knowledge search teacher는 사용하되, 최종 student는 lookahead 없이 domain-free actor로 평가한다. 핵심은 teacher policy의 raw action을 그대로 BC하지 않고, `constrain_aim` projection 이후 실제 실행된 물리 action을 label로 저장하는 것이다.

구현 파일: `experiments/privileged_teacher_distill.py`

### Label 정의

| 항목 | teacher | student |
|---|---|---|
| observation | 32차원, `extra_features=True` | 28차원 raw table state |
| action source | seed1/4/6 domain policy + adaptive h=2 search | actor 1회 `model.predict` |
| action label | raw policy action 아님 | projection 이후 executable action |
| eval env | domain knowledge 사용 | `constrain_aim=False`, `extra_features=False`, shaping 없음 |

projection label은 다음 흐름으로 만든다.

```text
teacher raw action
-> env _project_action
-> constrain_aim _apply_aim_constraint
-> executable physical action(theta, power, a, b)
-> domain-free student BC label
```

라벨 검증도 같이 넣었다. 같은 state에서 executable action을 domain-free env에 넣었을 때 teacher의 score/foul과 일치하는지 기록한다.

### 실험 결과

| 실험 | 데이터 구성 | samples | teacher label match | eval mean | max | P>=1 | foul |
|---|---|---:|---:|---:|---:|---:|---:|
| 기존 domain-free K=100 BC | one-step domain-free teacher | 1,365 | - | 0.080 | 2 | 7.5% | 20.0% |
| privileged teacher BC, long-chain biased | domain-knowledge multi-seed h=2, 긴 trajectory 위주 | 1,000 | 100.0% | 0.035 | 1 | 3.5% | 20.0% |
| privileged teacher BC, diverse starts | episode당 최대 5 samples, start 다양성 증가 | 1,000 | 100.0% | 0.070 | 2 | 6.0% | 14.0% |
| privileged teacher BC, angle_mse | 같은 diverse dataset, theta circular loss | 1,000 | 100.0% | 0.015 | 1 | 1.5% | 17.5% |
| privileged teacher BC, angle_nll_mse | 같은 diverse dataset, NLL + theta circular loss | 1,000 | 100.0% | 0.030 | 1 | 3.0% | 13.5% |
| privileged teacher DAgger r1 | diverse 1k + student rollout states 300, NLL+MSE 재학습 | 1,300 | 100.0% | 0.015 | 1 | 1.5% | 15.0% |
| AWR return-weighted | long-chain 1k + diverse 1k, return-to-go weight | 2,000 | 100.0% | 0.030 | 1 | 3.0% | 14.0% |
| AWR shot-baseline | 같은 2k, shot index별 baseline advantage weight | 2,000 | 100.0% | 0.030 | 1 | 3.0% | 14.0% |
| IQL-style weighted BC | 같은 2k, Q/V 기반 advantage weight | 2,000 | 100.0% | 0.020 | 1 | 2.0% | 17.5% |
| compiled domain actor s1 | policy 내부 extra features + aim projection, env는 domain-free | - | - | 1.364 | 13 | 53.6% | 21.2% |
| compiled domain actor s4 | 같은 방식 | - | - | 0.857 | 8 | 40.1% | 18.6% |
| compiled domain actor s6 | 같은 방식, best | - | - | 1.506 | 17 | 57.6% | 25.4% |
| clean domain-free lookahead K=20 | no constrain/extra/shaping/gentle/foul penalty, depth=1 | - | - | 0.875 | 7 | 51.0% | 25.0% |
| clean domain-free lookahead K=50 | 같은 strict no-domain, depth=1 | - | - | 2.590 | 14 | 74.5% | 24.0% |
| Q-guided actor, uniform probes | uniform simulator labels, learned critic -> actor | 90,000 | - | 0.010 | 1 | 1.0% | 17.0% |
| Q-guided actor, proposal positives | domain-free SAC proposal labels, strong BC | 120,000 | - | 0.050 | 1 | 5.0% | 22.0% |
| critic-ranked proposal policy | domain-free SAC proposals + learned rank critic, no env lookahead | 120,000 | - | 0.060 | 1 | 6.0% | 18.0% |

해석: 구현 자체는 의도대로 됐다. projection 이후 physical action label이 teacher 결과와 100% 일치했다. 하지만 1k BC만으로는 lookahead-free actor 성능이 올라가지 않았다. diverse-start 버전은 long-chain-biased 버전보다 낫지만, raw SAC baseline mean `0.090`에는 아직 못 미친다.

현재 실패 원인은 label mismatch가 아니라 supervised BC의 한계에 가깝다. `theta` circular loss도 바로 확인했지만, 이 1k dataset에서는 기존 `nll_mse`보다 나빠졌다. 따라서 단순 angle wrap이 주 병목은 아닌 것으로 보인다. 더 큰 문제는 같은 state 근처에서 여러 성공 shot mode가 섞이는 multimodality와, DAgger 없이 student가 만드는 off-distribution state를 보정하지 못하는 점이다.

DAgger 1차 실험도 같은 방향을 확인했다. student rollout state 300개를 teacher로 다시 라벨링해서 aggregate 1,300개로 재학습했지만 actor-only 평가는 mean `0.015`, max `1`, P>=1 `1.5%`로 오히려 나빠졌다. 수집 로그상 대부분이 `student_score=0`, `student_shots=1`인 첫 샷 실패 state였기 때문에, 단순 aggregate BC는 teacher의 좋은 long-chain 행동을 배우기보다 실패 주변 state에서 평균적인 보정 action을 따라가며 더 보수적으로 무너진 것으로 해석된다. 즉 DAgger 자체가 틀렸다기보다, 현재 actor 구조와 Gaussian BC 목적함수로는 teacher의 multimodal action 선택을 충분히 담지 못한다.

AWAC/IQL-style 후보도 1차로 확인했다. 구현 파일은 `experiments/advantage_weighted_distill.py`이고, teacher replay의 Monte-Carlo return-to-go를 계산한 뒤 높은 return action에 더 큰 weight를 주는 방식이다. `long-chain 1k`와 `diverse 1k`를 합친 2,000 samples의 return-to-go는 평균 `18.272`, max `63.397`이었다. 그러나 return-weighted AWR, shot-baseline AWR, Q/V 기반 IQL-style weighted BC 모두 actor-only mean이 `0.020~0.030`에 머물렀다. 순수 IQL weight는 현재 데이터가 거의 one state-one action이라 V가 Q를 따라가며 advantage가 작아지는 경향도 있었다.

해석: advantage weighting은 “좋은 trajectory action을 더 세게 보자”는 점에서는 맞지만, 결국 최종 actor update가 Gaussian BC라는 점은 그대로다. 즉 평균화/단일 mode 병목을 완전히 깨지는 못했다. 이 결과는 다음 단계가 단순 weighting보다 policy 표현력 자체, 예를 들면 mixture policy 또는 actor 후보 여러 개를 내부적으로 출력하는 구조 쪽이어야 함을 시사한다.

mean 1을 넘기는 search-free 방법은 `compiled domain actor`에서 달성했다. 구현 파일은 `experiments/compiled_domain_actor_eval.py`이다. 이 방법은 평가 env 자체는 `constrain_aim=False`, `extra_features=False`, shaping 없음으로 둔다. 대신 policy call 안에서 기존 domain-knowledge SAC actor가 기대하는 32차원 extra features를 계산하고, actor raw action에 aim projection을 적용한 뒤, 그 executable physical action 하나만 env에 실행한다. 즉 inference-time lookahead/search는 전혀 없고 shot마다 `model.predict` 1회만 사용한다.

결과는 1,000 episodes 기준 s1 mean `1.364`, s6 mean `1.506`으로 목표 mean 1을 넘겼다. 다만 이것은 “domain knowledge를 완전히 제거한 neural actor”는 아니다. domain knowledge를 env privilege가 아니라 policy 내부 deterministic transform으로 옮긴 형태다. 따라서 현재 결론은 다음과 같다. **lookahead 없이 mean > 1은 가능하다. 하지만 순수 domain-free neural actor로는 아직 실패했고, 성공한 방식은 aim projection/extra features라는 geometric prior를 policy에 compile한 방식이다.**

그 다음 strict no-domain을 다시 분리해서 확인했다. clean domain-free 조건은 `constrain_aim=False`, `extra_features=False`, `setup_shaping=False`, `gentle_shot=False`, `foul_penalty=0.0`, `continue_on_miss=False`이다. 이 조건에서 K=20 one-step lookahead는 mean `0.875`로 1을 넘지 못했지만, K=50 one-step lookahead는 mean `2.590`, max `14`, P>=1 `74.5%`로 mean 1을 확실히 넘겼다. 따라서 **domain knowledge를 전혀 쓰지 않는다는 조건만 보면 K=50 clean lookahead가 성공**이다.

다만 actor-only strict no-domain은 아직 성공하지 못했다. uniform random action probe로 학습한 Q-guided actor, domain-free SAC proposal label을 이용한 strong BC, 그리고 learned rank critic으로 proposal을 고르는 방식까지 확인했지만 모두 mean `0.01~0.06`에 머물렀다. 특히 learned critic은 train group에서는 positive action을 어느 정도 골랐지만 새 random-start state로 일반화하지 못했다. 현재 관찰상 search 없이 mean 1을 넘기려면 단순 BC/critic보다 훨씬 큰 state-action coverage 또는 다른 policy representation이 필요하다.

다음 후보:

1. 10k 이상 diverse-start privileged dataset을 만든다.
2. DAgger를 계속한다면 round 수만 늘리기보다 성공 trajectory와 실패-state correction 비율을 제어한다.
3. 단일 Gaussian actor 대신 mixture/CVAE/diffusion policy처럼 multimodal action을 표현할 수 있는 모델을 쓴다.
4. BC/AWR policy를 바로 평가하지 말고 teacher transition replay로 SAC/TD3 fine-tune한다.

추가 산출물:

| 파일 | 경로 |
|---|---|
| privileged distillation script | `experiments/privileged_teacher_distill.py` |
| long-chain biased summary | `experiments/runs_inning_nodomain/priv_teacher_exec_bc_s1s4s6_1k/priv_teacher_exec_bc_s160000_n40_m100_samples1000_summary.json` |
| long-chain biased policy | `experiments/runs_inning_nodomain/priv_teacher_exec_bc_s1s4s6_1k/priv_teacher_exec_bc_s160000_n40_m100_samples1000_policy.zip` |
| diverse-start summary | `experiments/runs_inning_nodomain/priv_teacher_exec_bc_s1s4s6_1k_diverse/priv_teacher_exec_bc_s161000_n300_m100_samples1000_summary.json` |
| diverse-start policy | `experiments/runs_inning_nodomain/priv_teacher_exec_bc_s1s4s6_1k_diverse/priv_teacher_exec_bc_s161000_n300_m100_samples1000_policy.zip` |
| angle_mse summary | `experiments/runs_inning_nodomain/priv_teacher_exec_bc_s1s4s6_1k_diverse_angle/priv_teacher_exec_bc_s161000_n300_m100_samples1000_angle_mse_e50_summary.json` |
| angle_nll_mse summary | `experiments/runs_inning_nodomain/priv_teacher_exec_bc_s1s4s6_1k_diverse_angle/priv_teacher_exec_bc_s161000_n300_m100_samples1000_angle_nll_mse_e25_summary.json` |
| DAgger r1 summary | `experiments/runs_inning_nodomain/priv_teacher_dagger_s1s4s6_r1_300/priv_teacher_exec_bc_s161000_1k_dagger_r1_300_summary.json` |
| DAgger r1 policy | `experiments/runs_inning_nodomain/priv_teacher_dagger_s1s4s6_r1_300/priv_teacher_exec_bc_s161000_1k_dagger_r1_300_policy.zip` |
| DAgger r1 new dataset | `experiments/runs_inning_nodomain/priv_teacher_dagger_s1s4s6_r1_300/priv_teacher_exec_bc_s161000_1k_dagger_r1_300_dagger_r1_samples300.npz` |
| DAgger r1 aggregate dataset | `experiments/runs_inning_nodomain/priv_teacher_dagger_s1s4s6_r1_300/priv_teacher_exec_bc_s161000_1k_dagger_r1_300_dagger_aggregate.npz` |
| advantage-weighted distill script | `experiments/advantage_weighted_distill.py` |
| AWR return summary | `experiments/runs_inning_nodomain/adv_weighted_long_diverse_return/awr_return_long_diverse_2k_summary.json` |
| AWR shot-baseline summary | `experiments/runs_inning_nodomain/adv_weighted_long_diverse_shot_baseline/awr_shot_baseline_long_diverse_2k_summary.json` |
| IQL-style weighted summary | `experiments/runs_inning_nodomain/adv_weighted_long_diverse_iql/awr_iql_long_diverse_2k_summary.json` |
| compiled domain actor eval script | `experiments/compiled_domain_actor_eval.py` |
| compiled domain actor s1 summary | `experiments/runs_inning_nodomain/compiled_domain_actor_s1/compiled_domain_actor_s1_n1000_summary.json` |
| compiled domain actor s4 summary | `experiments/runs_inning_nodomain/compiled_domain_actor_s4/compiled_domain_actor_s4_n1000_summary.json` |
| compiled domain actor s6 summary | `experiments/runs_inning_nodomain/compiled_domain_actor_s6/compiled_domain_actor_s6_n1000_summary.json` |
| clean domain-free K20 lookahead summary | `experiments/runs_inning_nodomain/domainfree_lookahead_k20_clean_confirm/lookahead_random_d1_m100_k20_summary.json` |
| clean domain-free K50 lookahead summary | `experiments/runs_inning_nodomain/domainfree_lookahead_k50_clean_confirm/lookahead_random_d1_m100_k50_summary.json` |
| Q-guided domain-free script | `experiments/q_guided_domain_free.py` |
| proposal probe collector | `experiments/collect_proposal_probe_dataset.py` |
| critic-ranked policy eval script | `experiments/critic_ranked_policy_eval.py` |
| Q-guided uniform summary | `experiments/runs_inning_nodomain/q_guided_uniform_s300_k300_fasttrain/qg_uniform_s300_k300_fasttrain_summary.json` |
| Q-guided proposal-BC summary | `experiments/runs_inning_nodomain/q_guided_proposal_s800_bc100/qg_proposal_s800_bc100_summary.json` |
| critic-ranked proposal summary | `experiments/runs_inning_nodomain/critic_ranked_proposal_s800_rank_k50x3/critic_ranked_proposal_s800_rank_k50x3_summary.json` |

### 2026-05-28 actor-only strict no-domain 추가 시도

목표 조건은 더 엄격하게 잡았다. evaluation과 policy 내부 모두에서 `constrain_aim=False`, `extra_features=False`, `setup_shaping=False`, `gentle_shot=False`, `continue_on_miss=False`, lookahead/search 없음, raw 28-dim observation만 사용한다. 학습 데이터 생성에는 domain-free simulator probe label만 사용했다.

이번에는 세 가지를 확인했다.

1. **success-only angle-aware BC/MDN**: 성공 action만 골라 theta를 sin/cos로 학습했다.
2. **robust success target**: 성공 action 주변을 simulator에서 jitter해서, 작은 오차에도 성공하는 action만 다시 모았다.
3. **initial-only probe data**: 이전 800-state proposal dataset은 trajectory state가 많고 initial random-start state가 76개뿐이라, random-start 첫 state 800개를 새로 probe했다.

| 실험 | 데이터 | policy/eval | eval mean | max | P>=1 | foul | 메모 |
|---|---|---|---:|---:|---:|---:|---|
| success BC s800 deterministic | proposal trajectory 800 states | deterministic actor, medoid target | 0.020 | 1 | 2.0% | 21.5% | train positive fixed success 27.0% |
| success BC s800 MDN | proposal trajectory 800 states | mixture mode | 0.070 | 1 | 7.0% | 18.5% | train positive fixed success 37.0% |
| robust success BC s800 MDN | jitter-robust trajectory targets 8,505 | mixture mode | 0.090 | 3 | 7.5% | 17.5% | robust target helps a little |
| robust initial s800 MDN | initial-only robust targets 8,182 | mixture mode | 0.140 | 2 | 12.0% | 21.0% | 이번 actor-only 최고 |
| robust initial s800 MDN sample | 같은 데이터 | one stochastic sample | 0.035 | 1 | 3.5% | 23.5% | stochastic sample은 mode보다 나쁨 |
| robust initial s800 deterministic h1024 best | 같은 데이터 | best checkpoint deterministic | 0.075 | 1 | 7.5% | 21.5% | train fixed success 79.1%, eval 일반화 실패 |
| robust initial+trajectory s1600 deterministic | initial robust + trajectory robust | best checkpoint deterministic | 0.075 | 1 | 7.5% | 24.0% | follow-up 데이터 추가만으로는 개선 없음 |
| KNN robust initial s800 | initial robust medoid targets | nearest raw-observation action | 0.030 | 1 | 3.0% | 27.0% | retrieval만으로는 coverage 부족 |
| KNN robust initial+trajectory s1600 | initial+trajectory robust medoid | nearest raw-observation action | 0.105 | 3 | 8.5% | 23.5% | neural보다 낫지 않음 |

해석: 이제 실패 원인은 더 분명하다. 모델이 학습 state 자체를 못 외우는 문제는 어느 정도 해결됐다. `robust initial s800 deterministic h1024 best`는 train fixed-positive state에서 success `79.1%`까지 올라갔다. 그런데 새 random-start eval에서는 mean `0.075`에 머문다. 즉 현재 병목은 **action precision보다 state coverage/generalization**이다. 800개 initial state와 800개 trajectory state 정도로는 raw observation에서 성공 action manifold를 일반화하지 못한다.

현재 strict actor-only no-domain 최고값은 `robust initial s800 MDN(mode)`의 mean `0.140`이다. 목표 mean `>1`은 아직 미달이다. 다음으로 의미 있는 방향은 initial random-start probe를 최소 수천~수만 state 규모로 늘리고, robust target 또는 다른 local policy representation을 붙이는 것이다.

추가 산출물:

| 파일 | 경로 |
|---|---|
| success BC/MDN actor script | `experiments/success_bc_actor.py` |
| robust target builder | `experiments/robust_success_targets.py` |
| KNN actor-only eval script | `experiments/knn_success_policy_eval.py` |
| initial-only proposal dataset | `experiments/runs_inning_nodomain/proposal_probe_initial_s0s1s2_800_combined/proposal_probe_initial_s0s1s2_800_dataset.npz` |
| initial-only robust target dataset | `experiments/runs_inning_nodomain/robust_targets_initial_s800_j8/robust_targets_initial_s800_j8_dataset.npz` |
| best actor-only summary so far | `experiments/runs_inning_nodomain/success_bc_robust_initial_s800_mdn_all/success_bc_robust_initial_s800_mdn_all_summary.json` |
| deterministic overfit diagnostic summary | `experiments/runs_inning_nodomain/success_bc_robust_initial_s800_det_h1024_best/success_bc_robust_initial_s800_det_h1024_best_summary.json` |

### 2026-05-28 continuation: larger initial coverage and token actor

이어서 strict actor-only no-domain 조건을 유지한 채 더 큰 initial-state coverage와 다른 policy architecture를 확인했다. 모든 평가는 `constrain_aim=False`, `extra_features=False`, `setup_shaping=False`, `gentle_shot=False`, `continue_on_miss=False`, no lookahead/search, one actor action per shot이다.

추가로 random-start 첫 state 1,600개를 더 probe했다. 기존 800개와 합치면 raw proposal dataset은 총 2,400 states, 360,000 action labels이고, success rate는 `5.03%`, state hit rate는 `93.6%`였다. 새 1,600 states에서 robust jitter target을 다시 만들었고, 1,491 states에서 15,511개 robust successful actions를 얻었다.

| 실험 | 데이터/정책 | eval episodes | eval mean | max | P>=1 | foul | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| proposal initial s800 MDN, no robust | 성공 proposal만 직접 BC | 200 | 0.050 | 1 | 5.0% | 20.0% | robust jitter 없이 약함 |
| token deterministic actor | raw 4 ball x 7 token transformer | 200 | 0.010 | 1 | 1.0% | 19.0% | MLP보다 나쁨 |
| robust initial s2400 MDN fast | s800 robust + s1600 robust, 220 epochs | 200 | 0.080 | 1 | 8.0% | 22.0% | underfit |
| robust initial s2400 MDN e800 | 같은 데이터, 800 epochs | 200 | 0.065 | 2 | 6.0% | 18.5% | 더 오래 학습해도 개선 없음 |
| robust initial+trajectory s1600 MDN | initial robust + trajectory robust | 200 | 0.080 | 2 | 7.5% | 17.5% | follow-up 포함도 개선 없음 |
| robust initial s800 MDN component-sample | component 하나 샘플, noise 없음 | 500 | 0.094 | 2 | 8.2% | 22.6% | mode보다 낫지 않음 |
| robust initial s2400 MDN component-sample | component 하나 샘플, noise 없음 | 500 | 0.072 | 2 | 7.0% | 19.0% | 큰 데이터도 개선 없음 |
| robust initial s800 MDN mode 재평가 | 기존 best 후보 500 episodes 재평가 | 500 | 0.092 | 2 | 8.2% | 22.4% | 200 episode mean 0.140은 noise였음 |

해석 업데이트: coverage를 800 initial states에서 2,400 initial states로 늘렸지만 actor-only 성능은 올라가지 않았다. 오히려 500 episode 재평가 기준으로 신뢰 가능한 최고값은 mean `0.09~0.10` 근처다. 즉 현재 BC/MDN류는 더 많은 같은 종류의 proposal target만으로 mean 1에 접근하지 못한다. token transformer도 raw observation 구조를 더 잘 쓰지 못했다.

현재까지의 결론은 더 강해졌다. strict domain-free actor-only를 달성하려면 단순 supervised distillation이 아니라, final actor 자체가 sparse simulator reward로 policy improvement를 하거나, 완전히 다른 action representation을 써야 한다. 다만 SAC stochastic/deterministic one-sample도 500 episodes에서 mean `0.05~0.07` 수준이라, pure sparse RL 역시 현재 설정으로는 부족하다.

추가 산출물:

| 파일 | 경로 |
|---|---|
| saved success-BC evaluator | `experiments/eval_success_bc_actor.py` |
| extra initial proposal dataset | `experiments/runs_inning_nodomain/proposal_probe_initial_s0s1s2_1600_more_combined/proposal_probe_initial_s0s1s2_1600_more_dataset.npz` |
| combined initial proposal dataset | `experiments/runs_inning_nodomain/proposal_probe_initial_s0s1s2_2400_combined/proposal_probe_initial_s0s1s2_2400_dataset.npz` |
| extra robust initial targets | `experiments/runs_inning_nodomain/robust_targets_initial_s1600_more_j8/robust_targets_initial_s1600_more_j8_dataset.npz` |
| s2400 MDN e800 summary | `experiments/runs_inning_nodomain/success_bc_robust_initial_s2400_mdn_e800/success_bc_robust_initial_s2400_mdn_e800_summary.json` |
| s800 MDN 500-episode re-eval summary | `experiments/runs_inning_nodomain/success_bc_robust_initial_s800_mdn_mode_eval500/success_bc_robust_initial_s800_mdn_mode_eval500_summary.json` |

### 2026-05-28 continuation: direct RL fine-tune and learned internal critic

BC/MDN 계열이 plateau에 걸렸기 때문에, 이번에는 final actor 자체를 sparse reward로 직접 개선하는 방향과, MDN의 여러 component 중 learned critic으로 하나를 고르는 방향을 확인했다. 여전히 evaluation은 strict no-domain이다.

| 실험 | 방식 | eval episodes | eval mean | max | P>=1 | foul | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| REINFORCE smoke | deterministic h1024 BC actor에서 stochastic policy fine-tune | 20 | det 0.050 / stoch 0.300 | - | - | - | 20ep라 noise 큼 |
| REINFORCE i60 | same init, 60 iters x 128 episodes, BC anchor | 200 | best det 0.105 / best stoch 0.060 | - | - | - | sparse PG도 개선 없음 |
| MDN component + BCE critic | s800 MDN의 8 component를 learned critic으로 선택 | 500 | 0.060 | 3 | 5.4% | 22.4% | mode보다 나쁨 |
| MDN component + rank critic | 같은 구조, rank loss critic | 500 | 0.054 | 2 | 4.8% | 22.4% | train top1은 올라가도 eval 일반화 실패 |

해석 업데이트: REINFORCE는 actor-only 조건에 맞는 순수 sparse-reward fine-tune이지만, BC anchor를 유지한 상태에서도 deterministic/stochastic eval 모두 mean `0.1` 근처를 넘지 못했다. MDN component를 learned critic으로 고르는 방식도 외부 simulator lookahead는 아니지만, critic이 새 random-start state에서 좋은 component를 고르지 못했다. 현재까지는 **raw observation -> single physical action** 문제에서 성공 manifold를 안정적으로 일반화하는 방법을 찾지 못했다.

추가 산출물:

| 파일 | 경로 |
|---|---|
| REINFORCE actor-only trainer | `experiments/reinforce_actor_only.py` |
| REINFORCE i60 summary | `experiments/runs_inning_nodomain/reinforce_det_h1024_s800_i60/reinforce_det_h1024_s800_i60_summary.json` |
| MDN internal critic evaluator | `experiments/mdn_critic_actor_eval.py` |
| MDN+BCE critic summary | `experiments/runs_inning_nodomain/mdn_critic_s800_policy_initial2400_bce/mdn_critic_s800_policy_initial2400_bce_summary.json` |
| MDN+rank critic summary | `experiments/runs_inning_nodomain/mdn_critic_s800_policy_initial2400_rank/mdn_critic_s800_policy_initial2400_rank_summary.json` |

### 2026-05-28 continuation: basin-center targets and local search distillation

이번에는 성공 action을 하나로 요약하는 방식 자체를 바꿔 보았다. 먼저 state별 robust successful actions를 action representation `(sin theta, cos theta, scaled power, a, b)`에서 greedy clustering했고, 가장 큰 성공 cluster의 중심을 target으로 삼았다. 그 다음에는 현재 actor action 주변을 local perturbation해서 새 random-start state에서 success candidate를 찾고, 그 local correction만 distill하는 방식도 확인했다. policy/eval은 계속 strict actor-only no-domain이다.

| 실험 | 방식 | eval episodes | eval mean | max | P>=1 | foul | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| cluster targets r=0.12 | 2,247 states, largest success cluster centroid | 500 | 0.046 | 2 | 4.4% | 19.8% | 실패 |
| cluster targets r=0.20 | 더 큰 cluster radius | 500 | 0.086 | 3 | 7.4% | 21.0% | 기존 best와 비슷 |
| local search target collection | saved actor 주변 perturb + uniform candidates | - | - | - | - | - | 300 states 중 238 states hit, candidate success 3.48% |
| cluster + local targets | cluster r=0.20 + local correction 238 states | 500 | 0.064 | 2 | 5.6% | 20.6% | local correction이 희석/일반화 실패 |
| local targets only | local correction 238 states만 BC | 500 | 0.048 | 2 | 4.4% | 20.8% | local teacher만으로도 일반화 실패 |

해석 업데이트: local search collection 자체는 흥미로운 신호를 보였다. 현재 deterministic actor의 action 주변을 perturb하면 새 random-start state의 `79.3%`에서 적어도 하나의 성공 후보가 발견된다. 즉 actor는 완전히 엉뚱한 방향이 아니라 성공 basin 근처까지는 가는 경우가 많다. 하지만 그 correction을 다시 supervised actor로 distill하면 mean `0.05~0.06`으로 돌아간다. 병목은 “성공 후보 존재 여부”가 아니라, **state별로 얇고 불연속적인 correction을 actor 함수 하나가 안정적으로 출력하지 못하는 것**으로 보인다.

추가 산출물:

| 파일 | 경로 |
|---|---|
| cluster target builder | `experiments/cluster_success_targets.py` |
| local search target collector | `experiments/local_search_targets.py` |
| cluster r=0.20 summary | `experiments/runs_inning_nodomain/success_bc_cluster_initial_s2400_r020_det/success_bc_cluster_initial_s2400_r020_det_summary.json` |
| local search dataset | `experiments/runs_inning_nodomain/local_search_det_h1024_s800_states300/local_search_det_h1024_s800_states300_dataset.npz` |
| cluster+local summary | `experiments/runs_inning_nodomain/success_bc_cluster_local_s2700_det/success_bc_cluster_local_s2700_det_summary.json` |
| local-only summary | `experiments/runs_inning_nodomain/success_bc_local_only_states300_det/success_bc_local_only_states300_det_summary.json` |

### 2026-05-28 continuation: residual codebook diagnostics

local search에서 성공 후보가 많이 발견되었기 때문에, 이번에는 actor가 직접 action 전체를 다시 예측하지 않고 `base actor action + residual correction` 형태로 바꿔 보았다. residual 후보는 cluster target과 local-search target에서 만든 뒤 K-means codebook으로 압축했다. 여전히 policy/eval은 strict actor-only no-domain이고, lookahead/search 없이 shot마다 action 하나만 실행한다.

| 실험 | 방식 | eval episodes | eval mean | max | P>=1 | foul | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| residual codebook K=16 | state -> residual code class | 500 | 0.068 | 2 | 6.0% | 19.4% | train acc 100%, eval 일반화 실패 |
| residual codebook K=64 | 더 세밀한 residual code | 500 | 0.058 | 2 | 5.6% | 19.4% | code 수를 늘려도 개선 없음 |
| residual critic codebook K=32 | learned critic으로 residual code 선택 | 500 | 0.054 | 1 | 5.4% | 21.2% | critic score는 높게 고르지만 실제 성공과 불일치 |
| global residual K=32 | validation 100ep에서 best global residual 선택 후 held-out eval | 500 | 0.060 | 2 | 5.8% | 22.0% | 고정 residual은 validation best 0.120도 held-out에서 무너짐 |

해석 업데이트: residual formulation도 핵심 병목을 깨지 못했다. local search는 base actor 주변에서 성공 후보를 찾지만, 그 correction이 state마다 매우 얇고 불연속적이라 raw observation만 보고 하나의 residual code를 안정적으로 고르지 못한다. 특히 residual codebook classifier는 train accuracy가 100%까지 올라갔는데 eval mean이 `0.068`에 머물렀다. 이는 optimization 실패라기보다 generalization 실패에 가깝다.

현재 strict actor-only no-domain의 신뢰 가능한 최고 구간은 여전히 mean `0.09~0.10` 근처다. clean domain-free lookahead K=50은 mean `2.590`으로 성공하지만, actor-only 조건에서는 아직 mean 1을 넘기지 못했다.

추가 산출물:

| 파일 | 경로 |
|---|---|
| residual codebook actor | `experiments/residual_codebook_actor.py` |
| residual critic codebook eval | `experiments/residual_critic_codebook_eval.py` |
| global residual eval | `experiments/global_residual_eval.py` |
| residual K=16 summary | `experiments/runs_inning_nodomain/res_code_cluster_local_k16/res_code_cluster_local_k16_summary.json` |
| residual K=64 summary | `experiments/runs_inning_nodomain/res_code_cluster_local_k64/res_code_cluster_local_k64_summary.json` |
| residual critic K=32 summary | `experiments/runs_inning_nodomain/res_critic_codebook_k32_s800/res_critic_codebook_k32_s800_summary.json` |
| global residual K=32 summary | `experiments/runs_inning_nodomain/global_residual_k32/global_residual_k32_summary.json` |

### 2026-05-28 continuation: learned ranker and ensemble attempts

다음으로는 simulator lookahead를 learned model로 대체할 수 있는지 확인했다. runtime에서는 simulator를 전혀 부르지 않고, domain-free SAC policy들이 제안한 후보 action을 neural critic, tree critic, kNN retrieval 등으로 고르는 방식이다. 또한 여러 actor의 action을 평균내는 search-free ensemble도 확인했다.

| 실험 | 방식 | eval episodes | eval mean | max | P>=1 | foul | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| angle-aware neural ranker h512 e30 | theta를 sin/cos로 넣은 rank critic, SAC 3개 x K50 후보 | 500 | 0.062 | 1 | 6.2% | 21.8% | linear theta 문제를 고쳐도 실패 |
| angle-aware neural ranker h256 e80 | vectorized rank loss, 더 오래 학습 | 500 | 0.056 | 1 | 5.6% | 17.8% | train top1도 9.6% 수준 |
| SAC3 + BC2 action ensemble | SAC 3개 평균과 BC actor 2개 평균을 alpha sweep | 500 | 0.052 | 1 | 5.2% | - | best alpha는 BC를 안 섞는 쪽 |
| tree critic ranker iter300 | gradient boosting classifier로 후보 ranking | 500 | 0.068 | 1 | 6.8% | 21.8% | train top1 50%, val top1 4.4%로 과적합 |
| tree critic shallow | 더 얕고 regularized한 tree | 500 | 0.060 | 2 | 5.4% | 23.6% | 과적합은 줄었지만 성능 개선 없음 |
| kNN candidate ranker | joint `(obs, action)`에서 성공 sample nearest 후보 선택 | 500 | 0.042 | 2 | 4.0% | - | retrieval도 새 state/action으로 일반화 실패 |

해석 업데이트: 이제 failure mode가 더 단단해졌다. domain-free SAC proposal 안에는 성공 후보가 자주 있지만, simulator로 직접 평가하지 않고 learned critic/retrieval로 고르면 거의 raw actor 수준으로 돌아간다. tree critic은 train group에서는 후보를 어느 정도 외우지만 validation group top1이 `4.4%`까지 떨어졌다. 즉 현재 데이터 규모와 raw feature만으로는 “어떤 후보가 실제로 맞는지”를 일반화하지 못한다.

따라서 현재까지 엄격한 actor-only no-domain 조건에서 mean 1을 넘기는 방법은 아직 없다. no-domain이지만 simulator lookahead를 허용하면 K=50 clean lookahead가 mean `2.590`으로 이미 성공했고, actor-only만 고집하면 최고 신뢰 구간은 여전히 mean `0.09~0.10`이다.

추가 산출물:

| 파일 | 경로 |
|---|---|
| angle-aware neural ranker | `experiments/angle_critic_ranked_policy_eval.py` |
| actor ensemble eval | `experiments/ensemble_actor_eval.py` |
| tree critic ranker | `experiments/tree_critic_ranked_policy_eval.py` |
| kNN candidate ranker | `experiments/knn_candidate_ranked_eval.py` |
| angle h512 summary | `experiments/runs_inning_nodomain/angle_rank_initial2400_h512_e30_k50x3/angle_rank_initial2400_h512_e30_k50x3_summary.json` |
| angle h256 summary | `experiments/runs_inning_nodomain/angle_rank_initial2400_h256_e80_k50x3/angle_rank_initial2400_h256_e80_k50x3_summary.json` |
| ensemble summary | `experiments/runs_inning_nodomain/ensemble_sac3_bc2_alpha/ensemble_sac3_bc2_alpha_summary.json` |
| tree iter300 summary | `experiments/runs_inning_nodomain/tree_rank_initial2400_iter300_k50x3/tree_rank_initial2400_iter300_k50x3_summary.json` |
| tree shallow summary | `experiments/runs_inning_nodomain/tree_rank_initial2400_shallow_k50x3/tree_rank_initial2400_shallow_k50x3_summary.json` |
| kNN ranker summary | `experiments/runs_inning_nodomain/knn_rank_initial2400_k50x3/knn_rank_initial2400_k50x3_summary.json` |
