"""Strategy engine — parameterized, signal-generating strategies."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from indicators import ema, rsi, macd, vwap, atr, bollinger_bands, stochastic


@dataclass
class StrategyParams:
    # EMA crossover
    ema_fast: int = 12
    ema_slow: int = 26
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Risk
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 4.0
    position_size: float = 0.1          # fraction of capital
    # Strategy type
    strategy_type: str = "ema_rsi"      # ema_rsi | macd_vwap | bb_stoch
    # Regime filter
    use_regime_filter: bool = True
    allowed_regimes: tuple = ("trending", "volatile")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Strategy:
    params: StrategyParams
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0
    parent_id: str | None = None

    # Fitness (set after backtest)
    fitness: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return Series of -1, 0, 1 signals."""
        p = self.params
        signals = pd.Series(0, index=df.index)

        if p.strategy_type == "ema_rsi":
            signals = self._ema_rsi_signals(df, p)
        elif p.strategy_type == "macd_vwap":
            signals = self._macd_vwap_signals(df, p)
        elif p.strategy_type == "bb_stoch":
            signals = self._bb_stoch_signals(df, p)
        else:
            signals = self._ema_rsi_signals(df, p)

        return signals

    def _ema_rsi_signals(self, df: pd.DataFrame, p: StrategyParams) -> pd.Series:
        fast = ema(df["close"], p.ema_fast)
        slow = ema(df["close"], p.ema_slow)
        r = rsi(df["close"], p.rsi_period)

        signals = pd.Series(0, index=df.index)
        # Long: fast crosses above slow AND rsi not overbought
        long_cond = (fast > slow) & (fast.shift(1) <= slow.shift(1)) & (r < p.rsi_overbought)
        # Short: fast crosses below slow AND rsi not oversold
        short_cond = (fast < slow) & (fast.shift(1) >= slow.shift(1)) & (r > p.rsi_oversold)
        signals[long_cond] = 1
        signals[short_cond] = -1
        return signals

    def _macd_vwap_signals(self, df: pd.DataFrame, p: StrategyParams) -> pd.Series:
        macd_line, signal_line, hist = macd(df["close"], p.macd_fast, p.macd_slow, p.macd_signal)
        try:
            vw = vwap(df)
        except Exception:
            vw = df["close"].rolling(20).mean()

        signals = pd.Series(0, index=df.index)
        # Long: MACD crosses above signal and price above VWAP
        long_cond = (hist > 0) & (hist.shift(1) <= 0) & (df["close"] > vw)
        short_cond = (hist < 0) & (hist.shift(1) >= 0) & (df["close"] < vw)
        signals[long_cond] = 1
        signals[short_cond] = -1
        return signals

    def _bb_stoch_signals(self, df: pd.DataFrame, p: StrategyParams) -> pd.Series:
        upper, mid, lower = bollinger_bands(df["close"])
        k, d = stochastic(df)

        signals = pd.Series(0, index=df.index)
        # Long: price touches lower BB and stoch oversold crossing up
        long_cond = (df["close"] <= lower) & (k > d) & (k.shift(1) <= d.shift(1)) & (k < 30)
        # Short: price touches upper BB and stoch overbought crossing down
        short_cond = (df["close"] >= upper) & (k < d) & (k.shift(1) >= d.shift(1)) & (k > 70)
        signals[long_cond] = 1
        signals[short_cond] = -1
        return signals

    def complexity_penalty(self) -> float:
        """Penalize overly complex parameter combinations."""
        p = self.params
        penalty = 0.0
        # Penalize if fast >= slow
        if p.ema_fast >= p.ema_slow:
            penalty += 0.5
        if p.macd_fast >= p.macd_slow:
            penalty += 0.5
        # Penalize extreme position sizing
        if p.position_size > 0.5:
            penalty += 0.3
        return penalty

    def summary(self) -> dict:
        return {
            "id": self.id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "type": self.params.strategy_type,
            "fitness": round(self.fitness, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_return": round(self.total_return, 4),
            "win_rate": round(self.win_rate, 4),
            "num_trades": self.num_trades,
            "params": self.params.to_dict(),
        }