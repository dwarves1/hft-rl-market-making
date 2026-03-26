"""
feature_extractor.py
────────────────────
Dual-Stream Feature Extractor for HFTMarketMakerEnv

Stream A  : Lightweight CNN  → 호가창 히트맵 (1, 100, 20)
              매물대 집중도·스푸핑 패턴 (급격한 잔량 소멸) 탐지
Stream B  : Gated MLP        → 기술적 지표 벡터 (22,)
              비선형 지표 상호작용 학습
              (e.g. VWAP 하방 이탈 + 피벗 P 붕괴 → 위험 가중)
Fusion    : Concat → LayerNorm → Linear → GELU

정규화 전략:
  • LayerNorm    : 배치 크기에 무관하게 안정적 (RNN/Transformer 관용)
  • Dropout(0.1) : 초단타 특성상 노이즈 과적합 방지
  • Weight decay : train.py 의 optimizer 설정으로 L2 정규화 추가

의존성:
    pip install stable-baselines3 torch gymnasium
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ──────────────────────────────────────────────
# 공통 블록
# ──────────────────────────────────────────────

class _ConvBnAct(nn.Module):
    """Conv2d + LayerNorm(channel-wise) + GELU + Dropout"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv    = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False)
        self.norm    = nn.InstanceNorm2d(out_ch, affine=True)   # H,W 가변에도 안정적
        self.act     = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.norm(self.conv(x))))


class _GatedLinear(nn.Module):
    """
    Gated Linear Unit (GLU 변형):
      output = Linear_1(x) ⊙ σ(Linear_2(x))
    지표 간 비선형 상호작용 (조건부 게이팅) 학습에 효과적.
    예: RSI 과매도 신호가 활성화될 때만 VWAP 이탈 특징을 증폭
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_val  = nn.Linear(in_dim, out_dim)
        self.linear_gate = nn.Linear(in_dim, out_dim)
        self.norm        = nn.LayerNorm(out_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val  = self.linear_val(x)
        gate = torch.sigmoid(self.linear_gate(x))
        return self.dropout(self.norm(val * gate))


# ══════════════════════════════════════════════
# Stream A : Orderbook CNN
# ══════════════════════════════════════════════

class OrderBookCNN(nn.Module):
    """
    입력  : (B, 1, T=100, L=20)  T=시간축, L=호가레벨(매수10+매도10)

    설계 의도:
    ┌─────────────────────────────────────────────────────────────┐
    │ Temporal Conv (kernel T×1): 시간축 패턴 (물량 소멸 속도)    │
    │ Level   Conv (kernel 1×L): 레벨 간 불균형 (스푸핑 탐지)     │
    │ Mixed   Conv (kernel T×L): 시공간 복합 특징                  │
    └─────────────────────────────────────────────────────────────┘

    출력  : (B, cnn_out_dim)
    """

    def __init__(self, cnn_out_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()

        # ── 시간축 패턴 감지 (수직 커널) ───────
        self.temporal_stream = nn.Sequential(
            _ConvBnAct(1,  16, kernel=(5, 1), padding=(2, 0), dropout=dropout),
            _ConvBnAct(16, 32, kernel=(5, 1), stride=(2, 1), dropout=dropout),   # T: 100→50
            _ConvBnAct(32, 32, kernel=(3, 1), stride=(2, 1), dropout=dropout),   # T: 50→25
        )

        # ── 레벨 간 불균형 감지 (수평 커널) ────
        self.level_stream = nn.Sequential(
            _ConvBnAct(1,  16, kernel=(1, 3), padding=(0, 1), dropout=dropout),
            _ConvBnAct(16, 32, kernel=(1, 3), padding=(0, 1), dropout=dropout),
            _ConvBnAct(32, 32, kernel=(1, 3), stride=(1, 2), dropout=dropout),   # L: 20→10
        )

        # ── 시공간 복합 특징 ────────────────────
        self.mixed_stream = nn.Sequential(
            _ConvBnAct(1,  16, kernel=(3, 3), padding=(1, 1), dropout=dropout),
            _ConvBnAct(16, 32, kernel=(3, 3), stride=(2, 2), dropout=dropout),   # T:50, L:10
            _ConvBnAct(32, 64, kernel=(3, 3), stride=(2, 2), dropout=dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 세 스트림 concat 후 프로젝션
        # temporal: 32ch, level: 32ch, mixed: 64ch → 128ch
        self.proj = nn.Sequential(
            nn.Linear(32 + 32 + 64, cnn_out_dim),
            nn.LayerNorm(cnn_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 100, 20)"""
        t_feat = self.global_pool(self.temporal_stream(x)).flatten(1)   # (B, 32)
        l_feat = self.global_pool(self.level_stream(x)).flatten(1)      # (B, 32)
        m_feat = self.global_pool(self.mixed_stream(x)).flatten(1)      # (B, 64)
        return self.proj(torch.cat([t_feat, l_feat, m_feat], dim=1))    # (B, 128)


# ══════════════════════════════════════════════
# Stream B : Indicator MLP
# ══════════════════════════════════════════════

class IndicatorMLP(nn.Module):
    """
    입력  : (B, vector_dim=22)

    설계 의도:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Embedding: 원시 지표 → 고차원 표현 공간 투영                  │
    │ 2. GLU Block ×2: 비선형 상호작용                                 │
    │    - Gate: 어떤 지표가 현재 시장 국면에서 유의미한지 학습        │
    │    - e.g. 볼린저 하단 이탈(bb_lower_break=1) 시에만              │
    │           RSI 과매도 → 매수 신호로 해석하는 조건부 로직          │
    │ 3. Residual Connection: 그래디언트 소실 방지                     │
    └─────────────────────────────────────────────────────────────────┘

    출력  : (B, mlp_out_dim)
    """

    def __init__(
        self,
        vector_dim: int = 22,
        hidden_dim: int = 128,
        mlp_out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # 입력 임베딩
        self.embed = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # GLU 블록 × 2 (잔차 연결)
        self.glu1 = _GatedLinear(hidden_dim, hidden_dim, dropout=dropout)
        self.glu2 = _GatedLinear(hidden_dim, hidden_dim, dropout=dropout)

        # 출력 프로젝션
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, mlp_out_dim),
            nn.LayerNorm(mlp_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 22)"""
        h  = self.embed(x)          # (B, 128)
        h  = h + self.glu1(h)       # residual
        h  = h + self.glu2(h)       # residual
        return self.out_proj(h)     # (B, 128)


# ══════════════════════════════════════════════
# Dual-Stream Feature Extractor (SB3 호환)
# ══════════════════════════════════════════════

class DualStreamExtractor(BaseFeaturesExtractor):
    """
    Stable-Baselines3 호환 Dual-Stream Feature Extractor.

    Architecture:
    ┌──────────────────────────────────────────────┐
    │  image  → OrderBookCNN → feat_A (128-D)      │
    │  vector → IndicatorMLP → feat_B (128-D)      │
    │                                               │
    │  [feat_A ‖ feat_B] → FusionHead → (256-D)    │
    │       ↓                                       │
    │   PPO Actor / Critic head                     │
    └──────────────────────────────────────────────┘

    Parameters
    ----------
    observation_space : gym.spaces.Dict
    cnn_out_dim       : CNN 스트림 출력 차원 (default 128)
    mlp_out_dim       : MLP 스트림 출력 차원 (default 128)
    fusion_dim        : 최종 Fusion 출력 차원 (= features_dim)
    dropout           : 전역 드롭아웃 비율 (default 0.1)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_out_dim: int = 128,
        mlp_out_dim: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim=fusion_dim)

        image_space  = observation_space["image"]    # (1, 100, 20)
        vector_space = observation_space["vector"]   # (22,)

        self.cnn = OrderBookCNN(cnn_out_dim=cnn_out_dim, dropout=dropout)
        self.mlp = IndicatorMLP(
            vector_dim  = vector_space.shape[0],
            hidden_dim  = mlp_out_dim,
            mlp_out_dim = mlp_out_dim,
            dropout     = dropout,
        )

        # Fusion Head: concat → cross-attention-like projection
        concat_dim = cnn_out_dim + mlp_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal 초기화 – PPO 수렴 안정성 향상"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=1.0)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        observations : {"image": (B,1,100,20), "vector": (B,22)}

        Returns
        -------
        torch.Tensor : (B, fusion_dim)
        """
        img_feat = self.cnn(observations["image"])      # (B, 128)
        vec_feat = self.mlp(observations["vector"])     # (B, 128)
        fused    = torch.cat([img_feat, vec_feat], dim=1)  # (B, 256)
        return self.fusion(fused)                          # (B, 256)


# ══════════════════════════════════════════════
# 파라미터 수 확인 (직접 실행 시)
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np

    obs_space = spaces.Dict({
        "image":  spaces.Box(0.0, 1.0, (1, 100, 20), dtype=np.float32),
        "vector": spaces.Box(-np.inf, np.inf, (22,),  dtype=np.float32),
    })

    extractor = DualStreamExtractor(obs_space)
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"DualStreamExtractor  총 파라미터 수: {total_params:,}")
    print(f"  CNN  파라미터: {sum(p.numel() for p in extractor.cnn.parameters()):,}")
    print(f"  MLP  파라미터: {sum(p.numel() for p in extractor.mlp.parameters()):,}")
    print(f"  Fuse 파라미터: {sum(p.numel() for p in extractor.fusion.parameters()):,}")

    # 순전파 테스트
    batch = {
        "image":  torch.zeros(4, 1, 100, 20),
        "vector": torch.zeros(4, 22),
    }
    out = extractor(batch)
    print(f"\n순전파 출력 shape: {out.shape}")   # (4, 256)
