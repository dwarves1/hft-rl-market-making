# [Quant ML] Deep Reinforcement Learning 기반 HFT 마켓 메이킹 시스템

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gymnasium-0.29-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-6f42c1?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
</p>

---

## Table of Contents

1. [Introduction — 왜 RL 기반 마켓 메이킹인가](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [모듈별 기술 상세](#3-모듈별-기술-상세)
   - [hft_env.py — Gymnasium 환경 & Feature Engineering](#31-hft_envpy)
   - [feature_extractor.py — Dual-Stream Neural Network](#32-feature_extractorpy)
   - [train.py — PPO 학습 파이프라인](#33-trainpy)
   - [dashboard_server.py — 비동기 FastAPI 백엔드](#34-dashboard_serverpy)
   - [dashboard_ui.html — React 실시간 대시보드](#35-dashboard_uihtml)
4. [Business Impact — 퀀트 전략의 수익 구조](#4-business-impact)
5. [Technical Deep Dive — Avellaneda-Stoikov 모델과 RL의 접점](#5-technical-deep-dive)
6. [Getting Started](#6-getting-started)
7. [Project Structure](#7-project-structure)

---

## 1. Introduction

### 마켓 메이킹(Market Making)이란

마켓 메이커는 매수·매도 양방향 지정가 주문을 동시에 제출하여 시장에 유동성을 공급하고, 그 대가로 **스프레드(Spread) 수익**과 거래소 **유동성 공급 리베이트(Maker Rebate)** 를 수취합니다. 코인원·빗썸·바이낸스 등 암호화폐 거래소에서는 이 전략이 전체 거래량의 40~60%를 차지합니다.

### 전통적 방법의 한계

수십 년간 마켓 메이킹은 Avellaneda-Stoikov(2008) 모델 등 해석적 확률 미적분 기반 이론으로 지배되어 왔습니다. 이 모델은 가격이 **기하 브라운 운동(GBM)** 을 따른다는 가정 하에 최적 스프레드를 닫힌 형태(closed-form)로 유도합니다. 그러나 실제 시장은:

- 호가창의 **비대칭적 물량 집중** (스푸핑, 레이어링)
- 일목균형표·피벗 포인트 등 **기술적 레벨에서의 비선형 반응**
- 시장 국면 전환 (Regime Change) 시의 **급격한 파라미터 이동**

등으로 인해 정적 모델 파라미터의 실시간 적응이 근본적으로 어렵습니다.

### RL이 해결하는 문제

이 프로젝트는 **Deep Reinforcement Learning(PPO)** 이 위 한계를 어떻게 극복할 수 있는지를 보여줍니다.

```
전통적 방법                          RL 기반 방법
────────────────────────────────     ─────────────────────────────
수식으로 최적 스프레드 계산           환경과 상호작용하며 스스로 학습
GBM 가정 필요                        가정 없음, 데이터로부터 귀납
파라미터 수동 튜닝                   보상 함수만 설계하면 자동 최적화
호가창 패턴 활용 불가                CNN이 매물대·스푸핑 패턴 직접 탐지
```

---

## 2. System Architecture

### 전체 구성도

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PHASE                                   │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    HFTMarketMakerEnv  (hft_env.py)                  │  │
│  │                                                                     │  │
│  │  TickSimulator (GBM)                                                │  │
│  │     └─ 호가창 생성 (10-Level, 지수 감쇠 잔량)                        │  │
│  │          └─ Feature Engineering                                     │  │
│  │               ├─ OBI · 체결강도           ─┐                        │  │
│  │               ├─ VWAP                      │                        │  │
│  │               ├─ 일목균형표 전환/기준선    ├─► vector (22-D)        │  │
│  │               ├─ 피벗 포인트 P/R1~2/S1~2  │                        │  │
│  │               └─ 볼린저 + RSI(14)         ─┘                        │  │
│  │          └─ 호가창 히트맵 2D ──────────────► image (1×100×20)       │  │
│  │                                                                     │  │
│  │  Reward = spread_pnl + rebate − inventory² − adverse_selection     │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                  │  Dict Obs + MultiDiscrete Action        │
│  ┌───────────────────────────────▼──────────────────────────────────────┐  │
│  │               Dual-Stream Feature Extractor  (feature_extractor.py) │  │
│  │                                                                      │  │
│  │   image ──► OrderBookCNN                                             │  │
│  │              ├─ Temporal Stream  Conv(T×1): 물량 소멸 속도           │  │
│  │              ├─ Level   Stream   Conv(1×L): 레벨 간 불균형           │  │
│  │              └─ Mixed   Stream   Conv(T×L): 시공간 복합 패턴  ─┐    │  │
│  │                                                                 │    │  │
│  │   vector ─► IndicatorMLP                                        ├──► │  │
│  │              ├─ Embed → LayerNorm → GELU                        │    │  │
│  │              └─ GLU×2 (조건부 게이팅) ────────────────────────►─┘    │  │
│  │                                                                      │  │
│  │              Fusion [256-D] → LayerNorm → GELU                      │  │
│  └──────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                          │
│  ┌───────────────────────────────▼───────────────────────────────────────┐  │
│  │                  PPO  (train.py)                                      │  │
│  │   Actor  [256→128→MultiDiscrete(bid, ask)]                           │  │
│  │   Critic [256→128→V(s)]                                              │  │
│  │   LR Schedule: Cosine Warmup  │  VecNormalize  │  8× SubprocVecEnv  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE / DASHBOARD PHASE                      │
│                                                                            │
│  HFTMarketMakerEnv ──► AgentWrapper.act()                                  │
│          │                    │                                            │
│          │            PPO.predict() or Heuristic fallback                  │
│          │                    │                                            │
│          └──── SimState ◄─────┘                                            │
│                   │  update_bar() → 1분봉 집계                              │
│                   │  check_alert() → RSI≤25 + BB 하단이탈                  │
│                   │                                                        │
│          ConnectionManager.broadcast()   (asyncio, non-blocking)           │
│              │           │         │          │                            │
│           candle      orderbook  metrics    alert                          │
│              └───────────┴─────────┴──────────┘                            │
│                          WebSocket /ws                                     │
│                               │                                            │
│  ┌────────────────────────────▼───────────────────────────────────────┐    │
│  │                   React SPA  (dashboard_ui.html)                   │    │
│  │                                                                    │    │
│  │  ┌──────────────────────────────────┐  ┌─────────────────────┐   │    │
│  │  │  CandleChart (LightweightCharts) │  │  OrderBook (SVG)    │   │    │
│  │  │  VWAP · 일목 · 피벗 오버레이      │  │  AGENT BID/ASK 마킹 │   │    │
│  │  └──────────────────────────────────┘  └─────────────────────┘   │    │
│  │  ┌──────────────────────────────────────────────────────────────┐  │    │
│  │  │  BottomPanel: PnL 게이지 · 재고 게이지 · RSI · 보상 분해     │  │    │
│  │  └──────────────────────────────────────────────────────────────┘  │    │
│  │  AlertBanner (RSI≤25 + BB하단이탈 오버레이)                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 모듈별 기술 상세

### 3-1. `hft_env.py`

**Gymnasium 환경 + Pandas 기반 Feature Engineering**

#### 5가지 초단타 지표

| 지표 | 계산 함수 | 의미 |
|------|-----------|------|
| OBI + 체결강도 | `compute_order_book_features()` | 매수·매도 잔량 불균형, 실시간 매매 압력 |
| VWAP | `compute_vwap()` | 거래량 가중 평균가 — 기관 주문 기준선 |
| 일목 전환/기준선 | `compute_ichimoku()` | 9봉/26봉 중간값 — 단기·중기 추세 전환 신호 |
| 피벗 포인트 | `compute_pivot_points()` | 전일 고/저/종 기반 P·R1·R2·S1·S2 지지/저항 레벨 |
| 볼린저 + RSI | `compute_bollinger_rsi()` | BB 하단 이탈(과매도) + RSI(14) 모멘텀 |

#### 관측 공간 (Dict Space)

```python
observation_space = Dict({
    "image":  Box(0.0, 1.0, (1, 100, 20)),   # 100틱 × 20레벨 호가창 히트맵
    "vector": Box(-inf, inf, (22,)),           # 지표 18 + 계좌상태 4
})
```

**히트맵 설계 의도**: 좌측 10열=매수 잔량, 우측 10열=매도 잔량. 시간이 흐를수록 행(row)이 아래로 밀려 최신 틱이 맨 아래에 위치. CNN이 수직 커널로 "물량이 갑자기 사라지는" 스푸핑 패턴을 학습 가능.

#### 보상 함수 (Reward Shaping)

```
R = spread_pnl + rebate − inventory_penalty − adverse_selection_penalty

spread_pnl           = (ask − bid) × qty          [양방향 체결 시]
rebate               = REBATE_RATE × price × qty   [Maker 체결 건당]
inventory_penalty    = λ₁ × inventory² × mid       [볼록 함수 — 편향 억제]
adverse_selection    = λ₂ × |Δmid| × |inventory|  [미드 이동 × 포지션 노출]
```

재고 페널티를 **포지션의 제곱(convex)**으로 설계한 것이 핵심입니다. 포지션이 2배 늘면 페널티는 4배가 되어, 에이전트가 자연스럽게 **델타 중립(Delta-Neutral)** 전략을 향해 수렴합니다.

---

### 3-2. `feature_extractor.py`

**Dual-Stream Extractor — SB3 `BaseFeaturesExtractor` 상속**

#### Stream A: `OrderBookCNN`

```
입력 (B, 1, T=100, L=20)
   │
   ├─ Temporal Stream   Conv(5×1) × 2 → GlobalAvgPool → (B, 32)
   │    물량 소멸 속도 감지: "스푸핑은 주문을 냈다 즉시 취소"
   │
   ├─ Level Stream      Conv(1×3) × 2 → GlobalAvgPool → (B, 32)
   │    레벨 간 불균형: "특정 가격대 매물 집중 = 저항선"
   │
   └─ Mixed Stream      Conv(3×3) × 2 → GlobalAvgPool → (B, 64)
        시공간 복합: 어느 시점, 어느 레벨에서 변화가 생겼는가

   → Concat(128) → Linear → LayerNorm → GELU → Dropout(0.1)
```

#### Stream B: `IndicatorMLP` with Gated Linear Unit

```python
# GLU: 어떤 지표가 "지금 시장 국면"에서 유의미한지 에이전트가 스스로 학습
output = Linear_val(x) ⊙ σ(Linear_gate(x))

# 예시 학습 패턴:
# bb_lower_break=1 이고 RSI<30 일 때만 OBI 신호를 강하게 반응
# → gate = σ(W·[bb_break, rsi, ...]) ≈ 1.0 (해당 조건 성립 시)
# → val  = W·obi → 크게 활성화
```

#### 초기화 전략: Orthogonal Initialization

PPO 논문(Schulman et al., 2017) 권장사항에 따라 모든 Linear·Conv 레이어에 **Orthogonal 초기화** 적용. 수렴 초기 그래디언트 소실/폭발을 방지하고 Value Function 학습 안정성을 높입니다.

---

### 3-3. `train.py`

**PPO 학습 파이프라인 — 엔지니어링 상세**

#### 학습률 스케줄: Cosine Warmup

```
learning_rate
    ▲
    │   /‾‾‾‾\
    │  /      \
    │ /        \
    │/   warm   \___________cosine decay___________
    └─────────────────────────────────────────── progress
    0     5%                                    100%
```

초반 5% 구간(warmup)에서 학습률을 선형 증가시켜 PPO 초기의 불안정한 정책 탐색 구간을 완충합니다. 이후 코사인 감쇠로 수렴 속도와 최종 성능을 함께 확보합니다.

#### 병렬화 전략

```
SubprocVecEnv (n_envs=8)
   ├─ Worker 0  (seed=42)   GBM drift=0.0
   ├─ Worker 1  (seed=43)   GBM drift=0.0
   ├─ ...
   └─ Worker 7  (seed=49)   GBM drift=0.0

n_steps=512 × n_envs=8 = 4,096 샘플/업데이트
batch_size=256 → 16 미니배치/업데이트
```

각 Worker가 독립된 seed로 서로 다른 가격 경로를 생성 → **다양한 시장 국면을 동시에 경험**하여 과적합 방지.

#### 정규화 전략 종합

| 기법 | 위치 | 목적 |
|------|------|------|
| `LayerNorm` | MLP·Fusion 레이어 | 배치 크기 독립적 안정성 |
| `InstanceNorm2d` | CNN 레이어 | 가변 공간 해상도 대응 |
| `Dropout(0.1)` | 전 레이어 | 노이즈 틱 데이터 과적합 방지 |
| `VecNormalize` | 관측·보상 | 지표 간 스케일 이질성 해소 |
| `weight_decay=1e-5` | Adam 옵티마이저 | L2 파라미터 정규화 |
| `target_kl=0.02` | PPO 업데이트 | 정책 급변 시 자동 조기 종료 |

---

### 3-4. `dashboard_server.py`

**FastAPI + WebSocket 비동기 스트리밍 백엔드**

#### 비동기 설계 원칙

```python
# 시뮬레이션 루프: lifespan 훅에서 백그라운드 태스크로 실행
_sim_task = asyncio.create_task(simulation_loop(agent))

# CPU-bound 연산은 스레드풀로 위임 → 이벤트 루프 비블로킹
result = await loop.run_in_executor(None, lambda: env.step(action))

# 복수 클라이언트에 동시 브로드캐스트
await manager.broadcast(payload)   # 죽은 소켓 자동 제거
```

#### WebSocket 메시지 프로토콜

| 타입 | 트리거 | 페이로드 |
|------|--------|---------|
| `orderbook` | 매 틱 | 호가창 10레벨 + 에이전트 Bid/Ask 위치 |
| `metrics` | 매 틱 | PnL·재고·보상 분해 4요소 |
| `candle` | 60틱마다 (1분봉 완성) | OHLCV + 5가지 지표값 |
| `alert` | RSI≤25 + BB하단이탈 | 신호 강도 + 에이전트 반응 오프셋 |

#### 알람 쿨다운 시스템

```python
# ALERT_COOLDOWN=30초 — 동일 신호 반복 발화 방지
if rsi <= 25 and bb_lower_break and now - self._last_alert_time > 30:
    self._last_alert_time = now
    return alert_payload
```

---

### 3-5. `dashboard_ui.html`

**React 18 + LightweightCharts 실시간 대시보드**

#### 컴포넌트 트리

```
App
├── Header        (Mid Price · 연결 상태 · Tick Counter)
├── AlertBanner   (RSI+BB 알람 오버레이 — CSS keyframe 애니메이션)
├── CandleChart   (LightweightCharts: 캔들 + VWAP + 일목 + 피벗)
├── OrderBook     (SVG 기반: 잔량 배경바 + AGENT 태그 마킹)
└── BottomPanel
    ├── PnL Gauge       (SVG Arc Gauge)
    ├── Inventory Gauge (위험도 색상 자동 전환)
    ├── RSI + BB 상태
    └── RewardBreakdown (스프레드/리베이트/재고/역선택 미니바)
```

#### `useWebSocket` 커스텀 훅

```javascript
// 재연결 로직 내장 — 네트워크 단절 시 3초 후 자동 복구
ws.onclose = () => {
    setIsConnected(false);
    reconnectTimer.current = setTimeout(connect, 3000);
};

// 메시지 라우팅: type 필드로 컴포넌트별 상태 분기
if (type === 'candle')     setLatestCandle(data);
if (type === 'orderbook')  setOrderbook(data);
if (type === 'metrics')    setMetrics(data);
if (type === 'alert')      enqueueAlert(data);
```

#### 에이전트 주문 마킹 로직

호가창에서 에이전트의 지정가 주문이 위치한 레벨을 식별하여 `outline + AGENT 뱃지`로 시각화합니다.

```jsx
const isAgent = Math.abs(row.price - agent_bid) < 0.005;
<div className={`ob-row${isAgent ? ' agent-bid-row' : ''}`}>
  {isAgent && <span className="agent-tag bid">AGENT</span>}
</div>
```

---

## 4. Business Impact

### 4-1. 유동성 공급 전략의 수익 구조

마켓 메이킹은 **스프레드 수익 + 거래소 리베이트** 라는 두 수익원을 가집니다.

```
[수익 시뮬레이션 (가정)]
─────────────────────────────────────────────────────
일일 체결 건수          : 10,000 건 (양방향 5,000쌍)
평균 체결량             : 0.01 BTC / 건
평균 스프레드 수취      : 0.5 tick = 0.5원
Maker 리베이트 (0.02%)  : 체결가 × 0.0002

[일별 수익 추정]
스프레드 수익: 5,000 × 0.01 × 0.5  = 25,000원
리베이트:      10,000 × 0.01 × 50,000 × 0.0002 = 100,000원
─────────────────────────────────────────────────────
일일 기대 수익: ~125,000원 (리스크 비용 전)
```

핵심은 **리베이트가 스프레드 수익의 4배**라는 점입니다. RL 에이전트는 단순히 스프레드를 넓히는 것이 아니라, 리베이트 극대화를 위해 좁은 스프레드로 체결 빈도를 높이는 전략을 학습합니다.

### 4-2. 재고 위험 관리의 자동화

전통 마켓 메이커의 가장 큰 위험은 **인벤토리 리스크** — 한 방향으로 포지션이 쌓이는 동안 시장이 역방향으로 움직이는 상황입니다.

```
전통적 해결책:
  일정 포지션 초과 시 시장가로 강제 청산 → 손실 확정

RL 에이전트의 접근:
  inventory_penalty = λ × inventory² × price
  → 포지션이 쌓일수록 반대 방향 호가를 공격적으로 좁힘
  → 강제 청산 없이 자연스러운 재고 분산
```

### 4-3. 인프라 비용 관점

| 항목 | 전통 Quant 팀 | 본 시스템 |
|------|--------------|-----------|
| 전략 개발 인력 | 퀀트 애널리스트 3~5명 | 학습 파이프라인 1회 설계 |
| 파라미터 재조정 | 시장 국면 바뀔 때마다 수동 | 재학습으로 자동 적응 |
| 스푸핑 대응 | 별도 필터 로직 개발 | CNN이 패턴 직접 학습 |
| 실시간 모니터링 | 전용 리스크 담당자 | 대시보드 자동 알람 |

---

## 5. Technical Deep Dive

### Avellaneda-Stoikov 모델과 RL의 수학적 접점

Avellaneda-Stoikov(2008)는 마켓 메이커의 최적 스프레드를 다음과 같이 유도합니다.

```
최적 매도 호가 = mid + γσ²(T-t) + (1/2) * (2/γ) * ln(1 + γ/k)
최적 매수 호가 = mid - γσ²(T-t) - (1/2) * (2/γ) * ln(1 + γ/k)

여기서:
  γ   : 위험 회피 계수 (Risk Aversion)
  σ²  : 가격 분산
  T-t : 잔여 시간
  k   : 주문 체결 강도 파라미터
```

이 수식은 두 가지 가정에 의존합니다: **(1) 가격이 GBM을 따른다, (2) 주문 체결이 Poisson 과정이다.**

실제 시장에서 이 가정들은 위반됩니다. RL 에이전트는 이 수식을 알 필요 없이, **보상 함수** 라는 동일한 목적(스프레드 수익 - 재고 위험)을 최적화하면서 암묵적으로 유사한 전략을 학습합니다.

```
AS 모델 파라미터    →    RL 대응 요소
─────────────────────────────────────────────
γσ²(T-t)          →    inventory_penalty = λ × q² × S
체결 강도 k        →    체결 시뮬레이션 확률 + 체결 강도 피처 (OBI)
잔여 시간 T-t      →    에피소드 내 tick_count / episode_ticks
```

### GLU(Gated Linear Unit)가 지표 상호작용에 효과적인 이유

기술적 지표는 **조건부로** 유의미합니다. RSI 과매도 신호는 볼린저 밴드 하단 이탈이 동시에 발생할 때 훨씬 강한 매수 신호가 됩니다. 반대로 RSI 단독으로는 횡보장에서 빈번하게 오신호를 발생시킵니다.

기존 ReLU/GELU MLP는 이 **조건부 중요도**를 직접 모델링하지 않습니다. GLU는 게이트 벡터 `σ(Wx)`가 현재 시장 상태를 기반으로 어떤 특징을 활성화할지를 0~1 사이의 연속 가중치로 학습합니다.

```python
# 학습이 완료된 경우 gate 가중치 해석 예시:
gate_weights = torch.sigmoid(linear_gate(indicators))
# gate[bb_lower_break 차원] → RSI 오버매도 시 ≈ 1.0
# gate[일반 OBI 차원]       → 정상 시장 시 ≈ 0.3~0.7
```

### 역선택(Adverse Selection) 페널티 설계의 의미

역선택은 마켓 메이킹의 핵심 리스크입니다. 정보 우위를 가진 투자자(Informed Trader)가 마켓 메이커의 호가에 거래하면, 그 직후 시장 가격이 거래 방향으로 이동합니다.

```
[역선택 시나리오]
  시각 t: 에이전트가 bid=49,990에 매수 주문 제출
  시각 t+1: Informed Trader가 49,990에 매도 체결
  시각 t+5: 가격이 49,950으로 하락 → 에이전트 40원 손실

[페널티 수식의 효과]
  adverse_selection = ADV_SEL_LAMBDA × |Δmid| × |inventory|
  → 가격이 크게 움직일수록, 포지션이 클수록 페널티 증가
  → 에이전트는 "가격 변동성이 높은 구간에서 재고를 줄이는" 전략을 학습
```

---

## 6. Getting Started

### Prerequisites

```bash
Python >= 3.10
CUDA 11.8+ (선택, CPU 학습도 가능)
```

### Installation

```bash
git clone https://github.com/your-username/hft-rl-market-making.git
cd hft-rl-market-making

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**`requirements.txt`**

```
gymnasium>=0.29.0
stable-baselines3>=2.3.0
torch>=2.2.0
torchvision>=0.17.0
pandas>=2.1.0
numpy>=1.26.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
websockets>=12.0
```

### 환경 동작 확인

```bash
python hft_env.py
# === HFTMarketMakerEnv 동작 테스트 ===
# Observation Spaces:
#   image  shape : (1, 100, 20)
#   vector shape : (22,)
```

### Feature Extractor 파라미터 확인

```bash
python feature_extractor.py
# DualStreamExtractor  총 파라미터 수: 412,034
#   CNN  파라미터: 187,200
#   MLP  파라미터: 99,714
#   Fuse 파라미터: 125,120
```

### 학습

```bash
# 디버그 (30초 내 파이프라인 검증)
python train.py --fast

# 본 학습 + TensorBoard
python train.py --tb --n-envs 8 --steps 5000000

# 학습 모니터링
tensorboard --logdir logs
```

### 대시보드 실행

```bash
uvicorn dashboard_server:app --host 0.0.0.0 --port 8000
# 브라우저: http://localhost:8000
```

**학습된 모델이 없어도 Heuristic 에이전트로 즉시 대시보드 확인 가능합니다.**

---

## 7. Project Structure

```
hft-rl-market-making/
│
├── hft_env.py              # Gymnasium 환경 + TickSimulator + Feature Engineering
│   ├── HFTMarketMakerEnv   # 메인 환경 클래스
│   ├── TickSimulator       # GBM 기반 가격 시뮬레이터
│   └── compute_*()         # 5가지 기술적 지표 계산 함수
│
├── feature_extractor.py    # Dual-Stream Neural Network
│   ├── OrderBookCNN        # 3-Stream CNN (Temporal / Level / Mixed)
│   ├── IndicatorMLP        # GLU 기반 MLP (잔차 연결)
│   └── DualStreamExtractor # SB3 BaseFeaturesExtractor 상속
│
├── train.py                # PPO 학습 파이프라인
│   ├── HFTConfig           # 하이퍼파라미터 dataclass
│   ├── cosine_warmup_schedule  # LR 스케줄러
│   ├── HFTMetricsCallback  # TensorBoard HFT 지표 로깅
│   ├── EarlyStopOnDivergenceCallback
│   └── train() / evaluate()
│
├── dashboard_server.py     # FastAPI 비동기 백엔드
│   ├── ConnectionManager   # WebSocket 브로드캐스트 매니저
│   ├── SimState            # 시뮬레이션 공유 상태 + 봉 집계
│   ├── AgentWrapper        # PPO / Heuristic 폴백
│   └── simulation_loop()   # asyncio 백그라운드 태스크
│
├── dashboard_ui.html       # React 18 SPA (빌드 불필요)
│   ├── useWebSocket        # 자동 재연결 커스텀 훅
│   ├── CandleChart         # LightweightCharts (VWAP+일목+피벗)
│   ├── OrderBook           # SVG 호가창 + 에이전트 마킹
│   ├── BottomPanel         # SVG 게이지 + 보상 분해
│   └── AlertBanner         # BB+RSI 오버레이 알람
│
├── checkpoints/            # 학습 체크포인트 저장 위치
├── logs/                   # TensorBoard 로그
└── requirements.txt
```

---

## References

- **Avellaneda & Stoikov** (2008). *High-frequency trading in a limit order book.* Quantitative Finance
- **PPO**: Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- **GLU**: Dauphin et al. (2017). *Language Modeling with Gated Convolutional Networks.* ICML
- **Silent-Face Anti-Spoofing**: Minivision AI, ECCV 2022
- **LightweightCharts**: TradingView Open Source

---

<p align="center">
  Built for <strong>Toss Bank DS/MLE</strong> Portfolio | 2026
</p>
