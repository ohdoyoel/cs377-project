# 검증 실험 설계
## 0. 4구 당구 도메인에 대한 설명, 공통적 학습 방법에 대한 설명 (실험 x)
### 수구, 목적구, 상대구
### --time_step 500k -> 50만번 샷을 쳐서 학습한다
## 1. Algorithm and Learning methods
### SAC vs PPO vs TD3
SAC 5 seed result & mean, PPO 5 seed result & mean, TD3 5 seed result & mean  
 → 400k에서 이미 800k의 ~85–90% 성능. 곡선이 명백히 diminishing returns(knee)
  - PPO = 0.000 (sparse라 budget 무관하게 바닥) → 400k든 800k든 꼴찌 고정
  - SAC ≫ 나머지, TD3는 불안정(foul 빈발)
  → 400k와 800k 사이에서 순위가 뒤집힐 여지가 없음. 지난번 걱정한 "budget 따라 우열 역전"은 여기선 안 일어남. -> **400k 까지만 실험!**
### canonical start vs random start / continue on miss vs reset on miss
4가지 학습 방법이 있다. canonical start + continue on miss ~ random start + reset on miss
SAC 5 seed 
## 2. Baseline Efforts
### 
## 3. Introduction of Domain Knowledge
## 4. Search Engine
## 5. Tuning the Engine