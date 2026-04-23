"""Execution simulator — realistic fill logic with slippage and commissions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from indicators import atr


@dataclass
class Trade:
    entry_time: any
    exit_time: any
    direction: int          # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    size: float             # shares/units
    pnl: float
    pnl_pct: float
    exit_reason: str        # signal | stop_loss | take_profit | end_of_data
    regime: str = "unknown"


@dataclass
class ExecutionConfig:
    slippage_bps: float = 5.0       # basis points per trade
    commission_bps: float = 3.0     # basis points per trade
    initial_capital: float = 100_000.0


class ExecutionSimulator:
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()

    def run(self, df: pd.DataFrame, signals: pd.Series, strategy) -> dict:
        """
        Simulate trades on market replay.
        Returns dict with equity curve, trades list, and metrics.
        """
        p = strategy.params
        cfg = self.config
        capital = cfg.initial_capital
        equity_curve = [capital]
        equity_times = [df.index[0]]
        trades: list[Trade] = []

        # Precompute ATR for stop/TP
        atr_series = atr(df, 14)

        position = 0        # 0 = flat, 1 = long, -1 = short
        entry_price = 0.0
        entry_time = None
        entry_atr = 0.0
        size = 0.0

        slippage_mult = cfg.slippage_bps / 10000
        commission_mult = cfg.commission_bps / 10000

        for i in range(1, len(df)):
            bar = df.iloc[i]
            sig = signals.iloc[i]
            cur_price = bar["close"]
            cur_atr = atr_series.iloc[i]

            exit_reason = None
            exit_price = None

            if position != 0:
                # Check stop loss and take profit
                if position == 1:
                    stop = entry_price - p.stop_loss_atr * entry_atr
                    tp = entry_price + p.take_profit_atr * entry_atr
                    if bar["low"] <= stop:
                        exit_price = stop
                        exit_reason = "stop_loss"
                    elif bar["high"] >= tp:
                        exit_price = tp
                        exit_reason = "take_profit"
                    elif sig == -1:
                        exit_price = cur_price
                        exit_reason = "signal"
                else:  # short
                    stop = entry_price + p.stop_loss_atr * entry_atr
                    tp = entry_price - p.take_profit_atr * entry_atr
                    if bar["high"] >= stop:
                        exit_price = stop
                        exit_reason = "stop_loss"
                    elif bar["low"] <= tp:
                        exit_price = tp
                        exit_reason = "take_profit"
                    elif sig == 1:
                        exit_price = cur_price
                        exit_reason = "signal"

                # Last bar — force exit
                if i == len(df) - 1 and exit_reason is None:
                    exit_price = cur_price
                    exit_reason = "end_of_data"

                if exit_reason is not None:
                    # Apply slippage
                    fill = exit_price * (1 + slippage_mult * position)
                    gross_pnl = (fill - entry_price) * position * size
                    commission = (abs(fill) + abs(entry_price)) * size * commission_mult
                    net_pnl = gross_pnl - commission
                    capital += net_pnl

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=df.index[i],
                        direction=position,
                        entry_price=entry_price,
                        exit_price=fill,
                        size=size,
                        pnl=net_pnl,
                        pnl_pct=net_pnl / (entry_price * size) if size > 0 else 0,
                        exit_reason=exit_reason,
                    ))
                    position = 0

            # Enter new position
            if position == 0 and sig != 0 and not np.isnan(cur_atr) and cur_atr > 0:
                position = sig
                fill = cur_price * (1 + slippage_mult * sig * -1)
                entry_price = fill
                entry_time = df.index[i]
                entry_atr = cur_atr
                size = (capital * p.position_size) / abs(entry_price) if entry_price > 0 else 0

            equity_curve.append(capital)
            equity_times.append(df.index[i])

        equity = pd.Series(equity_curve, index=equity_times)
        metrics = self._compute_metrics(equity, trades, cfg.initial_capital)
        return {"equity": equity, "trades": trades, "metrics": metrics}

    def _compute_metrics(self, equity: pd.Series, trades: list[Trade], initial: float) -> dict:
        if len(trades) == 0:
            return {
                "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "num_trades": 0, "profit_factor": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0,
            }

        returns = equity.pct_change().dropna()
        total_return = float((equity.iloc[-1] - initial) / initial)

        # Sharpe (annualized, assuming daily bars — adjust if needed)
        periods = 252
        sharpe = 0.0
        if returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(periods))

        # Max drawdown
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_drawdown = float(drawdown.min())

        # Trade stats
        pnls = [float(t.pnl) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = float(len(wins) / len(pnls)) if pnls else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        profit_factor = float(abs(sum(wins) / sum(losses))) if losses and sum(losses) != 0 else float("inf")
        expectancy = float(win_rate * avg_win + (1 - win_rate) * avg_loss)

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": int(len(trades)),
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
        }