"""Regime detector — tag market conditions as trending, ranging, or volatile."""
from __future__ import annotations
import numpy as np
import pandas as pd
from indicators import atr, ema, rsi


REGIMES = ["trending_up", "trending_down", "ranging", "volatile", "unknown"]


class RegimeDetector:
    def __init__(
        self,
        atr_period: int = 14,
        trend_period: int = 50,
        vol_threshold: float = 1.5,
        adx_period: int = 14,
    ):
        self.atr_period = atr_period
        self.trend_period = trend_period
        self.vol_threshold = vol_threshold
        self.adx_period = adx_period

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return Series of regime labels for each bar."""
        a = atr(df, self.atr_period)
        trend_ma = ema(df["close"], self.trend_period)

        # Normalized ATR (volatility ratio)
        atr_norm = a / df["close"]
        avg_atr_norm = atr_norm.rolling(self.atr_period * 3).mean()
        vol_ratio = atr_norm / avg_atr_norm.replace(0, np.nan)

        # Slope of trend MA
        slope = trend_ma.diff(5) / trend_ma.shift(5)

        regimes = pd.Series("unknown", index=df.index)

        for i in range(len(df)):
            if np.isnan(vol_ratio.iloc[i]) or np.isnan(slope.iloc[i]):
                continue

            vr = vol_ratio.iloc[i]
            sl = slope.iloc[i]
            price = df["close"].iloc[i]
            ma = trend_ma.iloc[i]

            if vr > self.vol_threshold:
                regimes.iloc[i] = "volatile"
            elif abs(sl) > 0.002 and price > ma * 1.005:
                regimes.iloc[i] = "trending_up"
            elif abs(sl) > 0.002 and price < ma * 0.995:
                regimes.iloc[i] = "trending_down"
            else:
                regimes.iloc[i] = "ranging"

        return regimes

    def regime_distribution(self, regimes: pd.Series) -> dict:
        counts = regimes.value_counts()
        total = len(regimes)
        return {r: round(counts.get(r, 0) / total * 100, 1) for r in REGIMES}

    def current_regime(self, df: pd.DataFrame) -> str:
        regimes = self.detect(df)
        return regimes.iloc[-1]