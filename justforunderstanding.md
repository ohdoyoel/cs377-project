# 프로젝트 구조 가이드

## 📁 최상위 폴더

| 폴더 | 용도 |
|---|---|
| **`billiards/`** | 핵심 패키지: 물리 시뮬레이터 + Gym 환경 + 보상모델 + 렌더링 |
| **`policies/`** | 학습 안 된 베이스라인 정책 (random, geometric_aim) |
| **`scripts/`** | 데이터 수집·전처리 + 단일 학습 스크립트 (PPO·reward model 등) |
| **`experiments/`** | 본 실험 (matrix run, seed 비교, 결과 파케이) |
| **`notebooks/`** | EDA·시각화·결과 분석 노트북 (단계별 9개) |
| **`tests/`** | pytest 49개 — 환경·정책·wrapper·smoke 테스트 |
| **`data/`** | 수집된 rollout / preference 페어 (parquet, jsonl) |
| **`models/`** | 학습된 PPO 정책·reward model 가중치 |
| **`artifacts/`** | 렌더된 HTML, PNG (디버그·시각화 산출물) |
| **`paper/`** | ICLR 스타일 보고서 초안 (md) |

---

## 📦 `billiards/` (핵심 패키지)

| 하위 | 역할 |
|---|---|
| `physics/` | 이벤트 기반 4구 당구 시뮬레이터 (state, cue_impact, dynamics, collisions, simulator) |
| `env.py` | **단일 샷** Gym 환경 (obs 28-dim, action 4-dim, terminate=True) |
| `inning_env.py` | **이닝 환경** — 득점 시 같은 에피소드에서 연속 샷 (multi-shot) |
| `wrappers/` | `mixed_reward_env`(env+RM 혼합), `reward_model_env`(RM 단독), `random_start_env`(공 위치 랜덤화) |
| `reward_model/` | preference 기반 reward model 네트워크 + 학습 코드 |
| `preference/` | preference 페어 데이터셋 + 휴리스틱/AI 라벨러 |
| `pebble/` | PEBBLE 알고리즘 (RM 앙상블 + replay buffer + agent) |
| `render/` | HTML 리플레이 뷰어, 스프라이트, dual replay |

---

## 🧪 `experiments/` (실험)

| 파일 | 역할 |
|---|---|
| `run_one.py` | PPO 단일 학습 (env + RM mixed alpha) |
| `run_matrix.py` | 18-run α×seed grid (Phase F α-sweep) |
| `configs.py` | α=[0,0.1,0.3,0.5,0.7,1.0] × seed=[0,1,2] |
| `run_inning_sac.py` | **이닝 환경 SAC/TD3 학습** (PPO 주석 처리, 현재 메인) |
| `run_inning_matrix.py` | 이닝 환경에서 여러 seed 일괄 |
| `run_inning_random.py` / `_matrix.py` | 랜덤 시작 위치로 이닝 학습 |
| `run_pebble.py` / `_matrix.py` | PEBBLE 학습 |
| `normalize_rm.py` | RM 출력 정규화 통계 계산 |
| `runs_inning/` | **현재 결과 저장소** — sac_s{0,1,2}, ppo_s{0,1,2}, td3_s{0,1,2}, td3_smoke |
| `runs/` | Phase F α-sweep 18개 결과 |
| `runs_pebble/` | PEBBLE 비교 (env-only / RM frozen / disagree-query) |
| `results/` | 집계된 summary parquet |

---

## 📓 `notebooks/` (분석 노트북)

각각 `.ipynb`(원본)와 `.executed.ipynb`(실행 결과 포함) 페어로 존재.

| 번호 | 내용 |
|---|---|
| **01_env_smoketest** | 환경 sanity check — 한 샷 굴려서 obs/reward/done 확인 |
| **02_baseline_stats** | Random / GeometricAim 정책의 득점률·foul률 베이스라인 |
| **03_inning_demo** | 이닝 환경 시연 — 득점 시 multi-shot 진행 |
| **04_preference_demo** | preference 페어 생성·시각화 (휴리스틱 라벨러) |
| **05_reward_model** | RM 학습 결과 (loss curve, 분포, 정렬도) |
| **06_eval_results** | α-sweep 18-run 결과 분석 (Phase F) |
| **07_iclr_analysis** | 논문용 그림·표 생성 |
| **08_pebble_analysis** | PEBBLE 변형 비교 (env / frozen / disagree) |
| **09_inning_results** | **현재 진행 중인 SAC/TD3 이닝 결과 분석** |

---

## 🛠 `scripts/` (재현용 CLI)

- `collect_rollouts.py` — random/geometric 정책으로 rollout 데이터 수집 → `data/rollouts_*.parquet`
- `generate_preference_dataset.py` — rollout 페어에 선호 라벨 → `data/preference_pairs*.jsonl`
- `train_reward_model.py` — preference 데이터로 RM 학습 → `models/reward_model.pt`
- `train_ppo.py` — RM 기반 PPO 학습 (구 버전, 단일 샷)
- `eval_policies.py` — 학습 정책 평가 → `data/eval_results.parquet`

---

## 🎯 현재 위치

지금 **`experiments/run_inning_sac.py`**에 TD3가 추가된 상태이고, 결과는 **`experiments/runs_inning/{sac,ppo,td3}_s{0,1,2}/`**에 저장돼 있습니다. 분석은 **`notebooks/09_inning_results.ipynb`**에서 이어가면 자연스럽습니다.
