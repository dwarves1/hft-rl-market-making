"""
hft_env.py
──────────
HFT Market Making Gymnasium 환경

관측 공간 (Dict Space):
  • image  : (1, 100, 20) 2D 호가창 히트맵 텐서  [Vision 모델용]
  • vector : (N,) 기술적 지표 + 잔고/포지션 상태  [MLP 모델용]

행동 공간 (Discrete):
  • bid_offset : 미드프라이스 기준 매수 호가를 깔 틱 수 (1 ~ MAX_OFFSET)
  • ask_offset : 미드프라이스 기준 매도 호가를 깔 틱 수 (1 ~ MAX_OFFSET)

보상 함수:
  R = spread_pnl + rebate − inventory_penalty − adverse_selection_penalty

의존성:
  pip install gymnasium pandas numpy ta
"""

from __future__ import annotations

import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
TICK_SIZE        = 0.01          # 호가 1틱 크기 (원화 환경이면 1원으로 변경)
MAX_OFFSET       = 10            # 에이전트가 선택 가능한 최대 오프셋 틱 수
N_LEVELS         = 10            # 매수/매도 호가창 깊이 (각 10단계)
HEATMAP_WINDOW   = 100           # 히트맵에 사용할 과거 틱 수
MAX_INVENTORY    = 50            # 최대 허용 순포지션 (양/음)
REBATE_RATE      = 0.0002        # 거래소 유동성 공급 리베이트 (0.02%)
TAKER_FEE        = 0.0004        # 시장가 주문 수수료 (0.04%)
INVENTORY_LAMBDA = 0.01          # 재고 페널티 가중치
ADV_SEL_LAMBDA   = 0.005         # 역선택 페널티 가중치

# 벡터 관측 차원 = 지표(18) + 잔고/포지션(4)
INDICATOR_DIM    = 18
ACCOUNT_DIM      = 4
VECTOR_DIM       = INDICATOR_DIM + ACCOUNT_DIM


# ══════════════════════════════════════════════
# 1. Feature Engineering (Pandas 기반)
# ══════════════════════════════════════════════

def compute_order_book_features(
    bid_volumes: np.ndarray,   # (N_LEVELS,)  매수 잔량
    ask_volumes: np.ndarray,   # (N_LEVELS,)  매도 잔량
    last_trade_side: int,      # 1=매수체결, -1=매도체결, 0=없음
    trade_qty: float,
) -> dict[str, float]:
    """
    (1) 매수/매도 잔량 비율 및 체결 강도

    Returns
    -------
    obi       : Order Book Imbalance  [-1, 1]  (양수=매수 우세)
    trade_int : 체결 강도  [-1, 1]  (양수=매수 체결 우세)
    """
    total_bid = bid_volumes.sum() + 1e-9
    total_ask = ask_volumes.sum() + 1e-9
    obi = (total_bid - total_ask) / (total_bid + total_ask)
    trade_int = last_trade_side * (trade_qty / (total_bid + total_ask))
    return {"obi": float(obi), "trade_intensity": float(np.clip(trade_int, -1, 1))}


def compute_vwap(df_ticks: pd.DataFrame) -> float:
    """
    (2) VWAP (Volume-Weighted Average Price)

    Parameters
    ----------
    df_ticks : columns=['price', 'volume']
    """
    if df_ticks.empty:
        return float("nan")
    pv = (df_ticks["price"] * df_ticks["volume"]).sum()
    v  = df_ticks["volume"].sum()
    return float(pv / v) if v > 0 else float("nan")


def compute_ichimoku(df_1m: pd.DataFrame) -> dict[str, float]:
    """
    (3) 일목균형표 전환선(Tenkan) / 기준선(Kijun)

    Parameters
    ----------
    df_1m : columns=['high', 'low', 'close']  (최소 26봉 필요)
    """
    if len(df_1m) < 26:
        return {"tenkan": float("nan"), "kijun": float("nan")}

    high = df_1m["high"]
    low  = df_1m["low"]

    tenkan = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2

    return {
        "tenkan": float(tenkan.iloc[-1]),
        "kijun":  float(kijun.iloc[-1]),
    }


def compute_pivot_points(
    prev_high: float, prev_low: float, prev_close: float
) -> dict[str, float]:
    """
    (4) 일간 피벗 포인트 (Classic Pivot)

    P  = (H + L + C) / 3
    R1 = 2P - L,  R2 = P + (H - L)
    S1 = 2P - H,  S2 = P - (H - L)
    """
    p  = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * p - prev_low
    r2 = p + (prev_high - prev_low)
    s1 = 2 * p - prev_high
    s2 = p - (prev_high - prev_low)
    return {"P": p, "R1": r1, "R2": r2, "S1": s1, "S2": s2}


def compute_bollinger_rsi(df_1m: pd.DataFrame) -> dict[str, float]:
    """
    (5) 볼린저 밴드 하단 이탈 여부 + RSI(14)

    Parameters
    ----------
    df_1m : columns=['close']  (최소 20봉 필요)

    Returns
    -------
    bb_lower_break : 1.0 = 하단 이탈, 0.0 = 정상
    rsi            : [0, 100]
    bb_pct         : %B = (close - lower) / (upper - lower)  [-inf, inf]
    """
    close = df_1m["close"]

    # Bollinger Band (20, 2)
    sma    = close.rolling(20).mean()
    std    = close.rolling(20).std(ddof=0)
    upper  = sma + 2 * std
    lower  = sma - 2 * std
    bb_pct = (close - lower) / (upper - lower + 1e-9)
    bb_lower_break = float(close.iloc[-1] < lower.iloc[-1])

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    rsi   = 100 - 100 / (1 + rs)

    return {
        "bb_lower_break": bb_lower_break,
        "bb_pct":         float(bb_pct.iloc[-1]),
        "rsi":            float(rsi.iloc[-1]),
    }


def build_indicator_vector(
    bid_volumes:      np.ndarray,
    ask_volumes:      np.ndarray,
    last_trade_side:  int,
    trade_qty:        float,
    df_ticks:         pd.DataFrame,
    df_1m:            pd.DataFrame,
    prev_high:        float,
    prev_low:         float,
    prev_close:       float,
    mid_price:        float,
) -> np.ndarray:
    """
    5개 지표를 하나의 float32 벡터로 패킹.

    Returns
    -------
    np.ndarray, shape (INDICATOR_DIM,) = (18,)
    """
    ob   = compute_order_book_features(bid_volumes, ask_volumes,
                                       last_trade_side, trade_qty)
    vwap = compute_vwap(df_ticks)
    ichi = compute_ichimoku(df_1m)
    pvt  = compute_pivot_points(prev_high, prev_low, prev_close)
    boll = compute_bollinger_rsi(df_1m)

    # NaN → 0 처리 후 mid_price 로 정규화 (스케일 통일)
    mp = mid_price if mid_price > 0 else 1.0
    raw = [
        ob["obi"],                                   # 1
        ob["trade_intensity"],                       # 2
        (vwap  - mp) / mp if not np.isnan(vwap) else 0.0,   # 3
        (ichi["tenkan"] - mp) / mp if not np.isnan(ichi["tenkan"]) else 0.0,  # 4
        (ichi["kijun"]  - mp) / mp if not np.isnan(ichi["kijun"])  else 0.0,  # 5
        (pvt["P"]  - mp) / mp,                       # 6
        (pvt["R1"] - mp) / mp,                       # 7
        (pvt["R2"] - mp) / mp,                       # 8
        (pvt["S1"] - mp) / mp,                       # 9
        (pvt["S2"] - mp) / mp,                       # 10
        boll["bb_lower_break"],                      # 11
        np.clip(boll["bb_pct"], -3, 3),              # 12  %B 클리핑
        np.clip(boll["rsi"] / 100.0, 0, 1),          # 13  RSI 정규화
        # 여분 5개: ask/bid 최근 3단계 불균형
        (ask_volumes[0] - bid_volumes[0]) / (ask_volumes[0] + bid_volumes[0] + 1e-9),  # 14
        (ask_volumes[1] - bid_volumes[1]) / (ask_volumes[1] + bid_volumes[1] + 1e-9),  # 15
        (ask_volumes[2] - bid_volumes[2]) / (ask_volumes[2] + bid_volumes[2] + 1e-9),  # 16
        float(np.log1p(bid_volumes.sum())),           # 17
        float(np.log1p(ask_volumes.sum())),           # 18
    ]
    return np.array(raw, dtype=np.float32)


# ══════════════════════════════════════════════
# 2. 틱 데이터 시뮬레이터
# ══════════════════════════════════════════════

class TickSimulator:
    """
    실제 틱 데이터 없이도 환경 테스트가 가능한 간단한 GBM 기반 시뮬레이터.
    실서비스에서는 실제 거래소 WebSocket 스트림으로 교체.

    Parameters
    ----------
    init_price : 초기 미드프라이스
    mu         : 일간 드리프트 (연율)
    sigma      : 일간 변동성 (연율)
    n_ticks_per_day : 하루 틱 수 (시뮬레이션 스케일)
    """

    def __init__(
        self,
        init_price: float = 50_000.0,
        mu: float = 0.0,
        sigma: float = 0.3,
        n_ticks_per_day: int = 23_400,
        seed: int | None = None,
    ) -> None:
        self.price  = init_price
        self.mu     = mu
        self.sigma  = sigma
        self.dt     = 1.0 / n_ticks_per_day
        self.rng    = np.random.default_rng(seed)

        self._tick_history: list[dict] = []
        self._1m_history:   list[dict] = []
        self._1m_open:      float = init_price
        self._1m_high:      float = init_price
        self._1m_low:       float = init_price
        self._tick_in_bar:  int   = 0
        self._ticks_per_1m: int   = 60   # 초봉 → 1분봉 집계 기준

    # ── 호가창 생성 ────────────────────────────
    def _generate_order_book(self) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
        """N_LEVELS 단계 매수/매도 호가·잔량 생성"""
        spread_ticks = self.rng.integers(1, 4)
        best_bid = self.price - spread_ticks * TICK_SIZE / 2
        best_ask = self.price + spread_ticks * TICK_SIZE / 2

        bid_prices = np.array([best_bid - i * TICK_SIZE for i in range(N_LEVELS)])
        ask_prices = np.array([best_ask + i * TICK_SIZE for i in range(N_LEVELS)])

        # 깊이 감소 지수 분포
        decay = np.exp(-0.3 * np.arange(N_LEVELS))
        bid_vols = (self.rng.exponential(100, N_LEVELS) * decay).astype(np.float32)
        ask_vols = (self.rng.exponential(100, N_LEVELS) * decay).astype(np.float32)

        return bid_prices, ask_prices, bid_vols, ask_vols

    # ── 틱 1개 진행 ────────────────────────────
    def step(self) -> dict:
        # GBM 가격 업데이트
        dW = self.rng.standard_normal()
        self.price *= np.exp(
            (self.mu - 0.5 * self.sigma ** 2) * self.dt
            + self.sigma * np.sqrt(self.dt) * dW
        )
        self.price = max(self.price, TICK_SIZE)

        bid_prices, ask_prices, bid_vols, ask_vols = self._generate_order_book()
        trade_side = int(self.rng.choice([-1, 0, 0, 1]))
        trade_qty  = float(self.rng.exponential(10)) if trade_side != 0 else 0.0

        tick = {
            "price":      self.price,
            "volume":     trade_qty,
            "side":       trade_side,
            "bid_prices": bid_prices,
            "ask_prices": ask_prices,
            "bid_vols":   bid_vols,
            "ask_vols":   ask_vols,
        }
        self._tick_history.append(tick)

        # 1분봉 집계
        self._1m_high = max(self._1m_high, self.price)
        self._1m_low  = min(self._1m_low,  self.price)
        self._tick_in_bar += 1
        if self._tick_in_bar >= self._ticks_per_1m:
            self._1m_history.append({
                "open":  self._1m_open,
                "high":  self._1m_high,
                "low":   self._1m_low,
                "close": self.price,
            })
            self._1m_open      = self.price
            self._1m_high      = self.price
            self._1m_low       = self.price
            self._tick_in_bar  = 0

        return tick

    # ── 히스토리 접근 ──────────────────────────
    def get_tick_df(self, n: int = HEATMAP_WINDOW) -> pd.DataFrame:
        recent = self._tick_history[-n:] if len(self._tick_history) >= n else self._tick_history
        return pd.DataFrame(recent)

    def get_1m_df(self, n: int = 100) -> pd.DataFrame:
        recent = self._1m_history[-n:] if len(self._1m_history) >= n else self._1m_history
        return pd.DataFrame(recent, columns=["open", "high", "low", "close"]) \
               if recent else pd.DataFrame(columns=["open", "high", "low", "close"])

    def get_prev_daily(self) -> tuple[float, float, float]:
        """전일 고가/저가/종가 (피벗 포인트용)"""
        if len(self._1m_history) < 390:
            return self.price * 1.01, self.price * 0.99, self.price
        day = self._1m_history[-390:]
        return (
            max(b["high"] for b in day),
            min(b["low"]  for b in day),
            day[-1]["close"],
        )


# ══════════════════════════════════════════════
# 3. HFTMarketMakerEnv
# ══════════════════════════════════════════════

class HFTMarketMakerEnv(gym.Env):
    """
    HFT 마켓 메이킹 Gymnasium 환경.

    Observation Space (Dict):
    ┌─────────────────────────────────────────────────────────┐
    │ image  : Box(0, 1, (1, HEATMAP_WINDOW, N_LEVELS*2))    │
    │          최근 100틱 × 20레벨 호가잔량 2D 히트맵 텐서    │
    │          채널 1: 정규화된 매수잔량 히트맵                │
    │          (실제 구현에서 채널 2 추가 가능: 매도잔량)      │
    │ vector : Box(-inf, inf, (VECTOR_DIM,))                  │
    │          5가지 기술적 지표 + 잔고/포지션 상태 (22차원)  │
    └─────────────────────────────────────────────────────────┘

    Action Space (MultiDiscrete):
      [bid_offset, ask_offset]  ∈  {1, 2, …, MAX_OFFSET}  각각

    Reward:
      R = spread_pnl + rebate − inventory_penalty − adverse_selection_penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        episode_ticks: int = 10_000,
        init_cash: float = 10_000_000.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.episode_ticks = episode_ticks
        self.init_cash     = init_cash
        self._seed         = seed

        # ── 관측 공간 ──────────────────────────
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0,
                shape=(1, HEATMAP_WINDOW, N_LEVELS * 2),
                dtype=np.float32,
            ),
            "vector": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(VECTOR_DIM,),
                dtype=np.float32,
            ),
        })

        # ── 행동 공간 ──────────────────────────
        # [bid_offset, ask_offset]  각각 1 ~ MAX_OFFSET
        self.action_space = spaces.MultiDiscrete([MAX_OFFSET, MAX_OFFSET])

        # ── 내부 상태 ──────────────────────────
        self._sim: TickSimulator | None = None
        self._reset_account()
        self._heatmap_buffer = np.zeros(
            (HEATMAP_WINDOW, N_LEVELS * 2), dtype=np.float32
        )

    # ──────────────────────────────────────────
    # 계좌 상태 초기화
    # ──────────────────────────────────────────
    def _reset_account(self) -> None:
        self.cash       = self.init_cash
        self.inventory  = 0              # 순포지션 (양=매수, 음=매도)
        self.filled_bid = 0              # 누적 매수 체결 수
        self.filled_ask = 0              # 누적 매도 체결 수
        self.tick_count = 0
        self.total_pnl  = 0.0
        self._prev_mid  = 0.0

    # ──────────────────────────────────────────
    # 관측 생성
    # ──────────────────────────────────────────
    def _build_observation(self, tick: dict) -> dict[str, np.ndarray]:
        # ── 히트맵 버퍼 업데이트 ───────────────
        # 행 = 시간축(오래된 → 최신), 열 = 호가 레벨
        # 앞쪽(매수 0~9), 뒤쪽(매도 10~19)
        new_row = np.concatenate([
            self._normalize_volumes(tick["bid_vols"]),
            self._normalize_volumes(tick["ask_vols"]),
        ])
        self._heatmap_buffer = np.roll(self._heatmap_buffer, -1, axis=0)
        self._heatmap_buffer[-1] = new_row
        image = self._heatmap_buffer[np.newaxis, :, :]  # (1, 100, 20)

        # ── 지표 벡터 구성 ─────────────────────
        sim      = self._sim
        mid      = tick["price"]
        df_ticks = sim.get_tick_df()
        df_1m    = sim.get_1m_df()
        ph, pl, pc = sim.get_prev_daily()

        indicator_vec = build_indicator_vector(
            bid_volumes     = tick["bid_vols"],
            ask_volumes     = tick["ask_vols"],
            last_trade_side = tick["side"],
            trade_qty       = tick["volume"],
            df_ticks        = df_ticks,
            df_1m           = df_1m,
            prev_high       = ph,
            prev_low        = pl,
            prev_close      = pc,
            mid_price       = mid,
        )

        # 계좌 상태 4개: 정규화된 현금비율, 순포지션, 체결 수 비율
        account_vec = np.array([
            np.clip(self.cash / self.init_cash - 1.0, -1, 1),     # 수익률
            np.clip(self.inventory / MAX_INVENTORY, -1, 1),        # 포지션 비율
            np.clip(self.filled_bid / (self.tick_count + 1), 0, 1),# 매수 체결률
            np.clip(self.filled_ask / (self.tick_count + 1), 0, 1),# 매도 체결률
        ], dtype=np.float32)

        vector = np.concatenate([indicator_vec, account_vec])

        return {"image": image, "vector": vector}

    @staticmethod
    def _normalize_volumes(vols: np.ndarray) -> np.ndarray:
        max_v = vols.max()
        return (vols / max_v).astype(np.float32) if max_v > 0 else vols.astype(np.float32)

    # ──────────────────────────────────────────
    # 주문 체결 시뮬레이션
    # ──────────────────────────────────────────
    def _simulate_fill(
        self,
        tick: dict,
        bid_price: float,
        ask_price: float,
    ) -> tuple[bool, bool]:
        """
        마켓 메이커 주문이 체결될지 여부를 판단.

        - 매수 주문: 실제 체결이 ask 방향(매도 체결)이면서 bid_price 이상이면 체결
        - 매도 주문: 실제 체결이 bid 방향(매수 체결)이면서 ask_price 이하이면 체결
        단순화된 규칙 (실제 거래소는 price-time priority 큐 구조)
        """
        market_price = tick["price"]
        trade_side   = tick["side"]

        bid_filled = (trade_side == -1) and (bid_price >= market_price - TICK_SIZE)
        ask_filled = (trade_side == 1)  and (ask_price <= market_price + TICK_SIZE)

        return bid_filled, ask_filled

    # ──────────────────────────────────────────
    # 보상 함수
    # ──────────────────────────────────────────
    def _compute_reward(
        self,
        bid_filled:  bool,
        ask_filled:  bool,
        bid_price:   float,
        ask_price:   float,
        mid_price:   float,
        trade_qty:   float,
    ) -> tuple[float, dict]:
        """
        R = spread_pnl + rebate − inventory_penalty − adverse_selection_penalty

        spread_pnl
          : 매수·매도 양쪽 모두 체결된 경우 스프레드 수익
            = (ask_price − bid_price) × qty

        rebate
          : 지정가(Maker) 주문 체결 시 거래소 유동성 공급 리베이트
            = REBATE_RATE × mid_price × qty

        inventory_penalty
          : 과도한 순포지션 보유에 대한 리스크 페널티
            = INVENTORY_LAMBDA × inventory² × mid_price

        adverse_selection_penalty
          : 역선택 – 불리한 방향으로 포지션이 누적될 때 추가 페널티
            시장 방향과 반대로 체결될수록 패널티 증가
            = ADV_SEL_LAMBDA × |Δmid| × |inventory|
        """
        qty = max(trade_qty, 1.0)  # 최소 1단위 가정

        # 스프레드 수익
        spread_pnl = 0.0
        if bid_filled and ask_filled:
            spread_pnl = (ask_price - bid_price) * qty

        # 리베이트
        rebate = 0.0
        if bid_filled:
            rebate += REBATE_RATE * bid_price * qty
        if ask_filled:
            rebate += REBATE_RATE * ask_price * qty

        # 재고 페널티 (포지션 제곱에 비례 → 볼록 함수 → 과도한 편향 억제)
        inventory_penalty = INVENTORY_LAMBDA * (self.inventory ** 2) * mid_price

        # 역선택 페널티 (미드프라이스 변동 × 순포지션)
        delta_mid = abs(mid_price - self._prev_mid)
        adverse_selection_penalty = ADV_SEL_LAMBDA * delta_mid * abs(self.inventory)

        reward = spread_pnl + rebate - inventory_penalty - adverse_selection_penalty

        info = {
            "spread_pnl":              round(spread_pnl, 6),
            "rebate":                  round(rebate, 6),
            "inventory_penalty":       round(inventory_penalty, 6),
            "adverse_selection_penalty": round(adverse_selection_penalty, 6),
        }
        return float(reward), info

    # ──────────────────────────────────────────
    # Gymnasium 인터페이스
    # ──────────────────────────────────────────
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        _seed = seed if seed is not None else self._seed

        self._sim = TickSimulator(seed=_seed)
        self._reset_account()
        self._heatmap_buffer = np.zeros(
            (HEATMAP_WINDOW, N_LEVELS * 2), dtype=np.float32
        )

        # 히트맵 버퍼를 워밍업 (HEATMAP_WINDOW 만큼 미리 진행)
        warmup_tick = None
        for _ in range(HEATMAP_WINDOW):
            warmup_tick = self._sim.step()
        self._prev_mid = self._sim.price

        obs = self._build_observation(warmup_tick)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """
        Parameters
        ----------
        action : [bid_offset, ask_offset]  각각 0-indexed (0 → 1틱)

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        bid_offset = int(action[0]) + 1   # 0-indexed → 1-indexed 틱 수
        ask_offset = int(action[1]) + 1

        tick     = self._sim.step()
        mid      = tick["price"]
        bid_price = mid - bid_offset * TICK_SIZE
        ask_price = mid + ask_offset * TICK_SIZE

        bid_filled, ask_filled = self._simulate_fill(tick, bid_price, ask_price)

        # 포지션·현금 업데이트
        trade_qty = max(tick["volume"], 1.0)
        if bid_filled:
            self.inventory += trade_qty
            self.cash      -= bid_price * trade_qty
            self.filled_bid += 1
        if ask_filled:
            self.inventory -= trade_qty
            self.cash      += ask_price * trade_qty
            self.filled_ask += 1

        reward, reward_info = self._compute_reward(
            bid_filled, ask_filled, bid_price, ask_price, mid, trade_qty
        )
        self.total_pnl += reward
        self._prev_mid  = mid
        self.tick_count += 1

        obs = self._build_observation(tick)

        # 종료 조건
        terminated = abs(self.inventory) > MAX_INVENTORY  # 포지션 한도 초과
        truncated  = self.tick_count >= self.episode_ticks

        info = {
            "tick":        self.tick_count,
            "mid_price":   round(mid, 4),
            "inventory":   self.inventory,
            "cash":        round(self.cash, 2),
            "total_pnl":   round(self.total_pnl, 4),
            "bid_filled":  bid_filled,
            "ask_filled":  ask_filled,
            "bid_price":   round(bid_price, 4),
            "ask_price":   round(ask_price, 4),
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._sim is None:
            return
        print(
            f"Tick={self.tick_count:>6d} | "
            f"Mid={self._sim.price:>10.4f} | "
            f"Inv={self.inventory:>+7.1f} | "
            f"Cash={self.cash:>15,.0f} | "
            f"PnL={self.total_pnl:>+10.4f}"
        )


# ══════════════════════════════════════════════
# 4. 빠른 동작 확인
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=== HFTMarketMakerEnv 동작 테스트 ===\n")

    env = HFTMarketMakerEnv(episode_ticks=500, seed=42)
    obs, info = env.reset()

    print(f"Observation Spaces:")
    print(f"  image  shape : {obs['image'].shape}")
    print(f"  vector shape : {obs['vector'].shape}\n")

    total_reward = 0.0
    for step_i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step_i % 50 == 0:
            env.render()

        if terminated or truncated:
            print(f"\n에피소드 종료 (terminated={terminated}, truncated={truncated})")
            break

    print(f"\n200 스텝 누적 보상: {total_reward:.4f}")
    print(f"최종 포지션      : {info['inventory']:.1f}")
    print(f"최종 현금        : {info['cash']:,.0f}")
