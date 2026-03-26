"""
train.py
────────
HFT Market Making PPO 학습 파이프라인

구성:
  1. 하이퍼파라미터 설정 (dataclass)
  2. 환경 팩토리 & 벡터화 래퍼
  3. DualStreamExtractor 기반 CnnPolicy 정의
  4. 콜백: EvalCallback, CheckpointCallback, EarlyStopCallback
  5. 학습 루프 실행
  6. 평가 & 저장

실행:
  python train.py                   # 기본 설정
  python train.py --fast            # 빠른 디버그 실행 (짧은 에피소드)
  python train.py --tb              # TensorBoard 로깅 활성화

의존성:
  pip install stable-baselines3 tensorboard gymnasium pandas numpy torch
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from feature_extractor import DualStreamExtractor
from hft_env import VECTOR_DIM, HFTMarketMakerEnv


# ══════════════════════════════════════════════
# 1. 하이퍼파라미터
# ══════════════════════════════════════════════

@dataclass
class HFTConfig:
    # ── 환경 ──────────────────────────────────
    n_envs:          int   = 8          # 병렬 환경 수 (CPU 코어 수에 맞게 조정)
    episode_ticks:   int   = 10_000     # 에피소드 길이
    init_cash:       float = 10_000_000.0

    # ── PPO 핵심 하이퍼파라미터 ───────────────
    # n_steps × n_envs = 한 업데이트당 수집 샘플 수
    # 초단타는 샘플 효율이 중요 → rollout을 짧게 가져가 빠른 피드백
    n_steps:         int   = 512        # 환경당 rollout 길이
    batch_size:      int   = 256        # 미니배치 크기
    n_epochs:        int   = 10         # 업데이트당 epoch 수
    gamma:           float = 0.99       # 할인율 – 초단타는 단기 보상 중심
    gae_lambda:      float = 0.95       # GAE lambda
    clip_range:      float = 0.2        # PPO clip ε
    clip_range_vf:   float | None = None

    # ── 엔트로피 & 학습률 ─────────────────────
    # 마켓 메이킹에서 탐색(exploration)이 중요: 스프레드 너비 다양하게 시도
    ent_coef:        float = 0.01       # 엔트로피 보너스 계수
    vf_coef:         float = 0.5        # Value Function 손실 계수
    max_grad_norm:   float = 0.5
    learning_rate:   float = 3e-4

    # ── 네트워크 구조 ─────────────────────────
    cnn_out_dim:     int   = 128
    mlp_out_dim:     int   = 128
    fusion_dim:      int   = 256        # features_dim → Actor/Critic 입력 차원
    net_arch:        list  = field(default_factory=lambda: [256, 128])  # Actor/Critic head
    dropout:         float = 0.1

    # ── 정규화 & 안정성 ───────────────────────
    normalize_obs:   bool  = True       # VecNormalize – 관측 정규화
    normalize_rew:   bool  = True       # VecNormalize – 보상 정규화
    weight_decay:    float = 1e-5       # Adam L2 정규화
    target_kl:       float = 0.02       # KL divergence 조기 종료

    # ── 학습 일정 ─────────────────────────────
    total_timesteps: int   = 5_000_000
    eval_freq:       int   = 50_000     # 평가 주기 (스텝 단위)
    n_eval_episodes: int   = 5
    save_freq:       int   = 100_000    # 체크포인트 저장 주기

    # ── 경로 ──────────────────────────────────
    log_dir:         str   = "logs"
    save_dir:        str   = "checkpoints"
    best_model_path: str   = "best_model"

    # ── 재현성 ────────────────────────────────
    seed:            int   = 42

    # ── 디버그 모드 ───────────────────────────
    fast_mode:       bool  = False      # True → 짧은 에피소드 & 적은 타임스텝


# ══════════════════════════════════════════════
# 2. 학습률 스케줄러
# ══════════════════════════════════════════════

def linear_schedule(initial_lr: float):
    """
    학습률 선형 감쇠 스케줄.
    SB3에서 학습 진행(progress)은 1.0→0.0 으로 감소하므로 역방향 스케일링.
    """
    def _schedule(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return _schedule


def cosine_warmup_schedule(initial_lr: float, warmup_frac: float = 0.05):
    """
    Cosine 감쇠 + Warmup 스케줄.
    초반 불안정한 탐색 구간을 워밍업으로 완화.
    """
    def _schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining   # 0 → 1
        if progress < warmup_frac:
            return initial_lr * (progress / warmup_frac)
        t = (progress - warmup_frac) / (1.0 - warmup_frac)
        return initial_lr * 0.5 * (1.0 + np.cos(np.pi * t))
    return _schedule


# ══════════════════════════════════════════════
# 3. 커스텀 콜백
# ══════════════════════════════════════════════

class HFTMetricsCallback(BaseCallback):
    """
    학습 중 HFT 특화 지표를 TensorBoard에 로깅.

    로깅 지표:
      - 에피소드 평균 포지션 (inventory)
      - 매수/매도 체결률 (fill rate)
      - 스프레드 수익 vs 재고 페널티 비율
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_inventories: list[float] = []
        self._episode_pnls:        list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "inventory" in info:
                self._episode_inventories.append(abs(info["inventory"]))
            if "total_pnl" in info:
                self._episode_pnls.append(info["total_pnl"])

        # 100 스텝마다 집계 로깅
        if self.n_calls % 100 == 0 and self._episode_inventories:
            self.logger.record(
                "hft/mean_abs_inventory",
                float(np.mean(self._episode_inventories[-100:])),
            )
            self.logger.record(
                "hft/mean_episode_pnl",
                float(np.mean(self._episode_pnls[-100:])) if self._episode_pnls else 0.0,
            )
        return True


class EarlyStopOnDivergenceCallback(BaseCallback):
    """
    보상이 임계값 이하로 연속 N회 측정되면 학습 중단.
    GBM 환경에서 에이전트가 재고 청산 불능 상태에 빠지는 경우를 탐지.
    """

    def __init__(
        self,
        reward_threshold: float = -1e5,
        patience: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.threshold = reward_threshold
        self.patience  = patience
        self._bad_count = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        if rewards and np.mean(rewards) < self.threshold:
            self._bad_count += 1
            if self.verbose:
                print(f"[EarlyStop] 발산 감지 {self._bad_count}/{self.patience}")
            if self._bad_count >= self.patience:
                print("[EarlyStop] 학습 중단: 보상 발산 감지")
                return False
        else:
            self._bad_count = 0
        return True


# ══════════════════════════════════════════════
# 4. 환경 팩토리
# ══════════════════════════════════════════════

def make_env(cfg: HFTConfig, rank: int = 0, seed: int = 0):
    """
    단일 환경 생성 클로저 (make_vec_env 용).
    각 worker가 다른 seed를 사용해 시뮬레이션 다양성 확보.
    """
    def _init():
        env = HFTMarketMakerEnv(
            episode_ticks=cfg.episode_ticks,
            init_cash=cfg.init_cash,
            seed=seed + rank,
        )
        env = Monitor(env)
        return env
    set_random_seed(seed + rank)
    return _init


# ══════════════════════════════════════════════
# 5. 모델 빌드
# ══════════════════════════════════════════════

def build_ppo(cfg: HFTConfig, env) -> PPO:
    """
    DualStreamExtractor + PPO 모델 조립.

    policy_kwargs:
      features_extractor_class  : DualStreamExtractor (커스텀)
      features_extractor_kwargs : CNN/MLP 아키텍처 파라미터
      net_arch                  : Actor / Critic head 레이어 구조
      optimizer_kwargs          : weight_decay (L2 정규화)
    """
    policy_kwargs = dict(
        features_extractor_class=DualStreamExtractor,
        features_extractor_kwargs=dict(
            cnn_out_dim=cfg.cnn_out_dim,
            mlp_out_dim=cfg.mlp_out_dim,
            fusion_dim=cfg.fusion_dim,
            dropout=cfg.dropout,
        ),
        net_arch=dict(pi=cfg.net_arch, vf=cfg.net_arch),
        # PPO 의 Actor / Critic head 에도 LayerNorm 없음 → SB3 기본 제공
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(
            weight_decay=cfg.weight_decay,
            eps=1e-5,             # Adam epsilon – 수치 안정성
        ),
    )

    model = PPO(
        policy           = "MultiInputPolicy",  # Dict 관측 공간 지원
        env              = env,
        learning_rate    = cosine_warmup_schedule(cfg.learning_rate),
        n_steps          = cfg.n_steps,
        batch_size       = cfg.batch_size,
        n_epochs         = cfg.n_epochs,
        gamma            = cfg.gamma,
        gae_lambda       = cfg.gae_lambda,
        clip_range       = cfg.clip_range,
        clip_range_vf    = cfg.clip_range_vf,
        ent_coef         = cfg.ent_coef,
        vf_coef          = cfg.vf_coef,
        max_grad_norm    = cfg.max_grad_norm,
        target_kl        = cfg.target_kl,
        policy_kwargs    = policy_kwargs,
        tensorboard_log  = cfg.log_dir,
        seed             = cfg.seed,
        device           = "cuda" if torch.cuda.is_available() else "cpu",
        verbose          = 1,
    )
    return model


# ══════════════════════════════════════════════
# 6. 학습 파이프라인
# ══════════════════════════════════════════════

def train(cfg: HFTConfig, use_tensorboard: bool = False) -> PPO:
    print("=" * 60)
    print(" HFT Market Making PPO 학습 시작")
    print(f" Device  : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f" n_envs  : {cfg.n_envs}")
    print(f" steps   : {cfg.total_timesteps:,}")
    print("=" * 60)

    # ── 디렉토리 준비 ──────────────────────────
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # ── 학습 환경 벡터화 ──────────────────────
    # SubprocVecEnv: 각 환경을 별도 프로세스 → GIL 우회, CPU 병렬화
    train_env = SubprocVecEnv(
        [make_env(cfg, rank=i, seed=cfg.seed) for i in range(cfg.n_envs)],
        start_method="fork",          # macOS/Linux: fork, Windows: spawn
    )

    if cfg.normalize_obs or cfg.normalize_rew:
        train_env = VecNormalize(
            train_env,
            norm_obs     = cfg.normalize_obs,
            norm_reward  = cfg.normalize_rew,
            clip_obs     = 10.0,
            clip_reward  = 10.0,
            gamma        = cfg.gamma,
        )

    # ── 평가 환경 (단일, 정규화 파라미터 공유) ─
    eval_env = Monitor(
        HFTMarketMakerEnv(
            episode_ticks=cfg.episode_ticks,
            init_cash=cfg.init_cash,
            seed=cfg.seed + 9999,
        )
    )
    if cfg.normalize_obs or cfg.normalize_rew:
        eval_env = VecNormalize(
            make_vec_env(lambda: eval_env, n_envs=1),
            norm_obs    = cfg.normalize_obs,
            norm_reward = False,     # 평가 시 보상 정규화 OFF (실제 값 확인)
            training    = False,
        )
        # 학습 env 의 통계 동기화
        if hasattr(train_env, "obs_rms"):
            eval_env.obs_rms = train_env.obs_rms

    # ── 콜백 정의 ─────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = cfg.best_model_path,
        log_path             = os.path.join(cfg.log_dir, "eval"),
        eval_freq            = max(cfg.eval_freq // cfg.n_envs, 1),
        n_eval_episodes      = cfg.n_eval_episodes,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq  = max(cfg.save_freq // cfg.n_envs, 1),
        save_path  = cfg.save_dir,
        name_prefix= "hft_ppo",
        verbose    = 1,
    )

    hft_metrics  = HFTMetricsCallback(verbose=0)
    early_stop   = EarlyStopOnDivergenceCallback(
        reward_threshold=-1e5,
        patience=20,
        verbose=1,
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        hft_metrics,
        early_stop,
    ])

    # ── 모델 생성 ─────────────────────────────
    model = build_ppo(cfg, train_env)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n Policy 총 파라미터 수: {total_params:,}\n")

    # ── 학습 실행 ─────────────────────────────
    tb_log_name = f"PPO_HFT_{int(time.time())}" if use_tensorboard else "PPO_HFT"
    start_time  = time.time()

    try:
        model.learn(
            total_timesteps = cfg.total_timesteps,
            callback        = callbacks,
            log_interval    = 10,
            tb_log_name     = tb_log_name,
            reset_num_timesteps = True,
            progress_bar    = True,
        )
    except KeyboardInterrupt:
        print("\n[학습 중단] 키보드 인터럽트")

    elapsed = time.time() - start_time
    print(f"\n학습 완료  —  소요 시간: {elapsed / 60:.1f}분")

    # ── 최종 모델 & 정규화 통계 저장 ──────────
    final_path = os.path.join(cfg.save_dir, "hft_ppo_final")
    model.save(final_path)
    print(f"최종 모델 저장: {final_path}.zip")

    if isinstance(train_env, VecNormalize):
        norm_path = os.path.join(cfg.save_dir, "vecnormalize.pkl")
        train_env.save(norm_path)
        print(f"VecNormalize 통계 저장: {norm_path}")

    train_env.close()
    return model


# ══════════════════════════════════════════════
# 7. 평가 유틸
# ══════════════════════════════════════════════

def evaluate(
    model_path: str,
    vecnorm_path: str | None = None,
    n_episodes: int = 10,
    cfg: HFTConfig | None = None,
    render: bool = True,
) -> dict[str, float]:
    """
    저장된 모델을 불러와 평가 실행.

    Returns
    -------
    dict: mean_reward, std_reward, mean_inventory, mean_fill_rate
    """
    cfg = cfg or HFTConfig()

    env = Monitor(HFTMarketMakerEnv(
        episode_ticks=cfg.episode_ticks,
        init_cash=cfg.init_cash,
        seed=0,
    ))

    model = PPO.load(model_path, env=env, device="auto")

    episode_rewards  = []
    episode_invs     = []
    episode_fills    = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if render and ep == 0 and env.unwrapped.tick_count % 500 == 0:
                env.unwrapped.render()

        episode_rewards.append(ep_reward)
        episode_invs.append(abs(info.get("inventory", 0)))
        fill_total = info.get("bid_filled", 0) + info.get("ask_filled", 0)
        episode_fills.append(fill_total / max(info.get("tick", 1), 1))

        print(f"Episode {ep+1:>2d}: reward={ep_reward:>+10.2f}  "
              f"inventory={episode_invs[-1]:>6.1f}  "
              f"fill_rate={episode_fills[-1]:.4f}")

    result = {
        "mean_reward":    float(np.mean(episode_rewards)),
        "std_reward":     float(np.std(episode_rewards)),
        "mean_inventory": float(np.mean(episode_invs)),
        "mean_fill_rate": float(np.mean(episode_fills)),
    }

    print("\n[ 평가 결과 ]")
    for k, v in result.items():
        print(f"  {k:<20s}: {v:.4f}")

    env.close()
    return result


# ══════════════════════════════════════════════
# 8. 진입점
# ══════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HFT PPO 학습")
    parser.add_argument("--fast",     action="store_true", help="디버그용 빠른 실행")
    parser.add_argument("--tb",       action="store_true", help="TensorBoard 로깅")
    parser.add_argument("--eval",     type=str, default=None, help="평가할 모델 경로")
    parser.add_argument("--n-envs",   type=int, default=8)
    parser.add_argument("--steps",    type=int, default=5_000_000)
    parser.add_argument("--seed",     type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = HFTConfig(
        n_envs          = args.n_envs,
        total_timesteps = args.steps,
        seed            = args.seed,
    )

    if args.fast:
        # 디버그: 모든 크기를 최소화
        cfg.episode_ticks   = 500
        cfg.total_timesteps = 20_000
        cfg.n_envs          = 2
        cfg.eval_freq       = 5_000
        cfg.save_freq       = 10_000
        cfg.n_steps         = 128
        cfg.batch_size      = 64
        print("[FAST MODE] 디버그 설정 적용\n")

    if args.eval:
        # 평가 모드
        evaluate(
            model_path  = args.eval,
            n_episodes  = 10,
            cfg         = cfg,
            render      = True,
        )
    else:
        # 학습 모드
        train(cfg, use_tensorboard=args.tb)
