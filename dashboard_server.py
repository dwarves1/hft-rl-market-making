"""
dashboard_server.py
───────────────────
HFT 대시보드 FastAPI 백엔드

엔드포인트:
  GET  /              → React SPA (dashboard_ui.html)
  GET  /api/history   → 최근 캔들·지표 REST 스냅샷
  WS   /ws            → 실시간 시뮬레이션 스트림 (broadcast)

WebSocket 메시지 타입:
  candle      → 1분봉 OHLCV + VWAP/일목/피벗/볼린저/RSI
  orderbook   → 10단계 호가창 + 에이전트 Bid/Ask 마킹
  metrics     → 누적 PnL·재고·체결률
  alert       → BB 하단 이탈 + RSI≤25 신호

의존성:
  pip install fastapi uvicorn websockets pandas numpy torch
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from hft_env import (
    N_LEVELS,
    TICK_SIZE,
    HFTMarketMakerEnv,
    compute_bollinger_rsi,
    compute_ichimoku,
    compute_pivot_points,
    compute_vwap,
)

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
SIM_TICK_HZ      = 20       # 시뮬레이션 속도 (틱/초, 실시간보다 빠른 데모)
CANDLE_TICKS     = 60       # 1분봉 = 60 틱
MAX_HISTORY      = 200      # REST /api/history 가 반환할 최대 봉 수
ALERT_COOLDOWN   = 30       # 동일 알람 재발 방지 쿨다운 (초)


# ══════════════════════════════════════════════
# 1. WebSocket 연결 매니저
# ══════════════════════════════════════════════

class ConnectionManager:
    """
    복수 클라이언트 브로드캐스트 매니저.
    연결이 끊긴 소켓은 자동으로 제거.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        text = json.dumps(payload, ensure_ascii=False)
        dead: set[WebSocket] = set()
        for ws in self._clients:
            try:
                await ws.send_text(text)
            except Exception:
                dead.add(ws)
        self._clients -= dead


manager = ConnectionManager()


# ══════════════════════════════════════════════
# 2. 시뮬레이션 상태
# ══════════════════════════════════════════════

class SimState:
    """
    시뮬레이션의 공유 상태.
    lifespan 태스크와 WebSocket 핸들러가 이 객체를 통해 데이터를 공유.
    """

    def __init__(self) -> None:
        # 캔들 이력 (REST /api/history 용)
        self.candles:  deque[dict] = deque(maxlen=MAX_HISTORY)
        # 현재 틱 집계용 임시 버퍼
        self._bar_open:  float = 0.0
        self._bar_high:  float = 0.0
        self._bar_low:   float = 0.0
        self._bar_ticks: int   = 0
        self._bar_vols:  list[float] = []
        self._bar_prices: list[float] = []

        # 지표 이력 (1분봉 DataFrame용)
        self._1m_history: deque[dict] = deque(maxlen=MAX_HISTORY)

        # 마지막 브로드캐스트 데이터
        self.last_orderbook: dict = {}
        self.last_metrics:   dict = {}

        # 알람 쿨다운
        self._last_alert_time: float = 0.0

        # 전일 고/저/종 (피벗용)
        self._prev_high:  float = 0.0
        self._prev_low:   float = 0.0
        self._prev_close: float = 0.0
        self._day_tick:   int   = 0

    def update_bar(self, price: float, volume: float) -> dict | None:
        """
        틱을 1분봉으로 집계.
        봉이 완성되면 완성된 봉 dict 반환, 아니면 None.
        """
        if self._bar_ticks == 0:
            self._bar_open  = price
            self._bar_high  = price
            self._bar_low   = price
        else:
            self._bar_high = max(self._bar_high, price)
            self._bar_low  = min(self._bar_low,  price)

        self._bar_prices.append(price)
        self._bar_vols.append(volume)
        self._bar_ticks += 1
        self._day_tick  += 1

        # 전일 고/저/종 갱신 (390봉 = 하루 기준)
        if self._day_tick % 390 == 0 and len(self._1m_history) >= 390:
            day = list(self._1m_history)[-390:]
            self._prev_high  = max(b["high"]  for b in day)
            self._prev_low   = min(b["low"]   for b in day)
            self._prev_close = day[-1]["close"]

        if self._bar_ticks >= CANDLE_TICKS:
            bar = {
                "open":   self._bar_open,
                "high":   self._bar_high,
                "low":    self._bar_low,
                "close":  price,
                "volume": sum(self._bar_vols),
                "time":   int(time.time()),
            }
            # VWAP (봉 내)
            if self._bar_vols:
                bar["vwap"] = float(
                    sum(p * v for p, v in zip(self._bar_prices, self._bar_vols))
                    / (sum(self._bar_vols) + 1e-9)
                )
            self._1m_history.append(bar)
            self.candles.append(bar)
            # 버퍼 초기화
            self._bar_ticks  = 0
            self._bar_vols   = []
            self._bar_prices = []
            return bar
        return None

    def build_candle_payload(self, bar: dict, mid: float) -> dict:
        """완성된 봉에 기술적 지표를 덧붙여 브로드캐스트 페이로드 생성."""
        df = pd.DataFrame(list(self._1m_history))
        if df.empty:
            df = pd.DataFrame([bar])

        # 일목균형표
        ichi = compute_ichimoku(df) if len(df) >= 26 else {"tenkan": mid, "kijun": mid}

        # 피벗 포인트
        ph = self._prev_high  or mid * 1.01
        pl = self._prev_low   or mid * 0.99
        pc = self._prev_close or mid
        pvt = compute_pivot_points(ph, pl, pc)

        # 볼린저 밴드 & RSI
        boll = compute_bollinger_rsi(df) if len(df) >= 20 else {
            "bb_lower_break": 0, "bb_pct": 0.5, "rsi": 50.0
        }

        # 볼린저 밴드 상/하단 (시각화용)
        bb_upper = bb_lower = bb_mid = mid
        if len(df) >= 20:
            sma    = df["close"].rolling(20).mean().iloc[-1]
            std    = df["close"].rolling(20).std(ddof=0).iloc[-1]
            bb_mid   = float(sma)
            bb_upper = float(sma + 2 * std)
            bb_lower = float(sma - 2 * std)

        return {
            "type": "candle",
            "data": {
                **bar,
                "vwap":             bar.get("vwap", mid),
                "tenkan":           ichi["tenkan"],
                "kijun":            ichi["kijun"],
                "pivot_P":          pvt["P"],
                "pivot_R1":         pvt["R1"],
                "pivot_R2":         pvt["R2"],
                "pivot_S1":         pvt["S1"],
                "pivot_S2":         pvt["S2"],
                "bb_upper":         bb_upper,
                "bb_mid":           bb_mid,
                "bb_lower":         bb_lower,
                "bb_lower_break":   boll["bb_lower_break"],
                "rsi":              boll["rsi"],
            },
        }

    def check_alert(self, rsi: float, bb_lower_break: bool) -> dict | None:
        """RSI ≤ 25 + 볼린저 하단 이탈 시 알람 페이로드 반환 (쿨다운 적용)."""
        now = time.time()
        if (
            rsi <= 25
            and bb_lower_break
            and now - self._last_alert_time > ALERT_COOLDOWN
        ):
            self._last_alert_time = now
            return {
                "type": "alert",
                "data": {
                    "signal":          "bb_rsi_oversold",
                    "rsi":             round(rsi, 2),
                    "bb_lower_break":  True,
                    "timestamp":       int(now),
                },
            }
        return None


sim_state = SimState()


# ══════════════════════════════════════════════
# 3. 에이전트 래퍼 (trained or random)
# ══════════════════════════════════════════════

class AgentWrapper:
    """
    학습된 PPO 모델 또는 Heuristic 에이전트를 공통 인터페이스로 감쌈.
    모델 파일이 없으면 Heuristic으로 폴백하여 대시보드 즉시 실행 가능.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model = None
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(model_path, device="cpu")
                print(f"[Agent] PPO 모델 로드: {model_path}")
            except Exception as e:
                print(f"[Agent] 모델 로드 실패 ({e}), Heuristic으로 폴백")
        else:
            print("[Agent] Heuristic 에이전트 사용 (모델 없음)")

    def act(self, obs: dict, info: dict) -> tuple[int, int]:
        """
        Returns
        -------
        (bid_offset, ask_offset)  0-indexed (0 = 1틱)
        """
        if self._model is not None:
            action, _ = self._model.predict(obs, deterministic=True)
            return int(action[0]), int(action[1])

        # ── Heuristic 폴백 ──────────────────────
        # RSI·OBI 기반 단순 규칙
        vec = obs["vector"]
        obi = float(vec[0])   # [-1,1] 매수우세=양수
        rsi_norm = float(vec[12])  # [0,1]
        rsi = rsi_norm * 100

        # 매수 우세 시장 → ask를 좁히고 bid를 넓힘 (passive ask)
        if obi > 0.3 and rsi < 40:
            return 1, 3     # 공격적 bid, 보수적 ask
        elif obi < -0.3 or rsi > 70:
            return 3, 1     # 보수적 bid, 공격적 ask
        else:
            return 2, 2     # 대칭 스프레드


# ══════════════════════════════════════════════
# 4. 시뮬레이션 루프 (비동기 백그라운드 태스크)
# ══════════════════════════════════════════════

async def simulation_loop(agent: AgentWrapper) -> None:
    """
    HFTMarketMakerEnv를 구동하며 틱마다 WebSocket으로 브로드캐스트.

    비동기 설계:
      - run_in_executor: CPU-bound 환경 step → 이벤트 루프 비블로킹
      - await asyncio.sleep(0): 브로드캐스트 후 즉시 제어권 반환
      - 발산·종료 시 자동 reset
    """
    loop   = asyncio.get_event_loop()
    env    = HFTMarketMakerEnv(episode_ticks=50_000, seed=int(time.time()))
    obs, _ = env.reset()
    tick_interval = 1.0 / SIM_TICK_HZ

    print(f"[Sim] 시뮬레이션 시작 (speed={SIM_TICK_HZ} tick/s)")

    while True:
        t0 = loop.time()

        # ── 에이전트 액션 ──────────────────────
        bid_offset, ask_offset = agent.act(obs, {})
        action = np.array([bid_offset, ask_offset])

        # ── 환경 스텝 (executor로 비동기 실행) ─
        result = await loop.run_in_executor(
            None, lambda: env.step(action)
        )
        obs, reward, terminated, truncated, info = result

        mid       = info["mid_price"]
        price     = mid
        volume    = info.get("bid_filled", 0) + info.get("ask_filled", 0)

        # ── 1. OrderBook 페이로드 ──────────────
        tick_data = env._sim._tick_history[-1] if env._sim._tick_history else {}
        bid_vols  = tick_data.get("bid_vols",  np.zeros(N_LEVELS))
        ask_vols  = tick_data.get("ask_vols",  np.zeros(N_LEVELS))
        bid_prices = tick_data.get("bid_prices", np.array([mid - (i+1)*TICK_SIZE for i in range(N_LEVELS)]))
        ask_prices = tick_data.get("ask_prices", np.array([mid + (i+1)*TICK_SIZE for i in range(N_LEVELS)]))

        agent_bid = mid - (bid_offset + 1) * TICK_SIZE
        agent_ask = mid + (ask_offset + 1) * TICK_SIZE

        orderbook_payload = {
            "type": "orderbook",
            "data": {
                "mid_price":  round(mid, 4),
                "agent_bid":  round(agent_bid, 4),
                "agent_ask":  round(agent_ask, 4),
                "bid_offset": bid_offset + 1,
                "ask_offset": ask_offset + 1,
                "bids": [
                    {"price": round(float(p), 4), "volume": round(float(v), 2)}
                    for p, v in zip(bid_prices, bid_vols)
                ],
                "asks": [
                    {"price": round(float(p), 4), "volume": round(float(v), 2)}
                    for p, v in zip(ask_prices, ask_vols)
                ],
            },
        }
        sim_state.last_orderbook = orderbook_payload

        # ── 2. 메트릭 페이로드 ─────────────────
        metrics_payload = {
            "type": "metrics",
            "data": {
                "total_pnl":       round(info["total_pnl"], 4),
                "inventory":       round(info["inventory"], 2),
                "cash":            round(info["cash"], 0),
                "tick":            info["tick"],
                "spread_pnl":      round(info.get("spread_pnl", 0), 6),
                "rebate":          round(info.get("rebate", 0), 6),
                "inv_penalty":     round(info.get("inventory_penalty", 0), 6),
                "adv_sel_penalty": round(info.get("adverse_selection_penalty", 0), 6),
                "bid_filled":      info.get("bid_filled", False),
                "ask_filled":      info.get("ask_filled", False),
            },
        }
        sim_state.last_metrics = metrics_payload

        # ── 3. 1분봉 완성 시 캔들 페이로드 ──────
        bar = sim_state.update_bar(price, float(volume))
        if bar:
            candle_payload = sim_state.build_candle_payload(bar, mid)
            # 알람 체크
            alert = sim_state.check_alert(
                rsi            = candle_payload["data"]["rsi"],
                bb_lower_break = bool(candle_payload["data"]["bb_lower_break"]),
            )
            # 알람 발생 시 에이전트 액션 정보 첨부
            if alert:
                alert["data"]["agent_bid_offset"] = bid_offset + 1
                alert["data"]["agent_ask_offset"] = ask_offset + 1
                alert["data"]["agent_bid"]         = round(agent_bid, 4)
                alert["data"]["agent_ask"]         = round(agent_ask, 4)
                await manager.broadcast(alert)

            await manager.broadcast(candle_payload)

        # ── 4. 매 틱마다 호가창·메트릭 브로드캐스트
        await manager.broadcast(orderbook_payload)
        await manager.broadcast(metrics_payload)

        # ── 에피소드 종료 처리 ─────────────────
        if terminated or truncated:
            print(f"[Sim] 에피소드 종료 (tick={info['tick']}), 재시작")
            obs, _ = env.reset()

        # ── 틱 속도 조절 ───────────────────────
        elapsed = loop.time() - t0
        sleep_t = max(0.0, tick_interval - elapsed)
        await asyncio.sleep(sleep_t)


# ══════════════════════════════════════════════
# 5. FastAPI 앱
# ══════════════════════════════════════════════

_sim_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sim_task
    agent = AgentWrapper(model_path="checkpoints/hft_ppo_final.zip")
    _sim_task = asyncio.create_task(simulation_loop(agent))
    print("[Server] 시뮬레이션 루프 시작")
    yield
    _sim_task.cancel()
    try:
        await _sim_task
    except asyncio.CancelledError:
        pass
    print("[Server] 시뮬레이션 루프 종료")


app = FastAPI(
    title="HFT Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST ──────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    """React SPA 서빙"""
    ui_path = Path(__file__).parent / "dashboard_ui.html"
    return FileResponse(str(ui_path), media_type="text/html")


@app.get("/api/history")
async def get_history():
    """최근 캔들 & 현재 상태 스냅샷 (초기 화면 렌더링용)"""
    return JSONResponse({
        "candles":   list(sim_state.candles),
        "orderbook": sim_state.last_orderbook,
        "metrics":   sim_state.last_metrics,
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "clients": len(manager._clients)}


# ── WebSocket ─────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    클라이언트 연결 시:
      1. 현재 스냅샷 즉시 전송 (초기 렌더링)
      2. 이후 시뮬레이션 루프가 broadcast 하는 메시지 수신
    """
    await manager.connect(ws)
    try:
        # 초기 스냅샷 전송
        if sim_state.last_orderbook:
            await ws.send_text(json.dumps(sim_state.last_orderbook))
        if sim_state.last_metrics:
            await ws.send_text(json.dumps(sim_state.last_metrics))
        for bar in list(sim_state.candles)[-50:]:
            candle_payload = sim_state.build_candle_payload(bar, bar["close"])
            await ws.send_text(json.dumps(candle_payload))

        # 연결 유지 (핑/퐁)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)


# ── 진입점 ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard_server:app", host="0.0.0.0", port=8000, reload=False)
