"""Backtest engine — runs strategy over market data with train/test split."""
from __future__ import annotations
import pandas as pd
from strategy import Strategy
from execution import ExecutionSimulator, ExecutionConfig


class Backtester:
    def __init__(self, exec_config: ExecutionConfig = None):
        self.simulator = ExecutionSimulator(exec_config or ExecutionConfig())

    def run(self, strategy: Strategy, df: pd.DataFrame, split: float = 0.7) -> dict:
        """
        Run backtest. Returns in-sample and out-of-sample results.
        split = fraction of data used for training.
        """
        n = len(df)
        split_idx = int(n * split)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # In-sample
        train_signals = strategy.generate_signals(train_df)
        train_result = self.simulator.run(train_df, train_signals, strategy)

        # Out-of-sample
        test_signals = strategy.generate_signals(test_df)
        test_result = self.simulator.run(test_df, test_signals, strategy)

        # Full
        full_signals = strategy.generate_signals(df)
        full_result = self.simulator.run(df, full_signals, strategy)

        # Combined fitness score
        fitness = self._compute_fitness(
            strategy, train_result["metrics"], test_result["metrics"]
        )

        # Update strategy
        strategy.fitness = fitness
        strategy.sharpe = test_result["metrics"]["sharpe"]
        strategy.max_drawdown = test_result["metrics"]["max_drawdown"]
        strategy.total_return = test_result["metrics"]["total_return"]
        strategy.win_rate = test_result["metrics"]["win_rate"]
        strategy.num_trades = test_result["metrics"]["num_trades"]

        return {
            "train": train_result,
            "test": test_result,
            "full": full_result,
            "fitness": fitness,
            "train_metrics": train_result["metrics"],
            "test_metrics": test_result["metrics"],
        }

    def _compute_fitness(self, strategy: Strategy, train_m: dict, test_m: dict) -> float:
        """
        Fitness = penalize overfit, reward OOS performance, penalize complexity.
        """
        oos_sharpe = test_m["sharpe"]
        oos_return = test_m["total_return"]
        oos_dd = abs(test_m["max_drawdown"])
        oos_trades = test_m["num_trades"]

        # Require minimum trades
        if oos_trades < 5:
            return -1.0

        # Core score: Sharpe adjusted for drawdown
        core = oos_sharpe * (1 - oos_dd)

        # Overfit penalty: if OOS Sharpe is much worse than IS Sharpe
        is_sharpe = train_m["sharpe"]
        overfit_gap = max(0, is_sharpe - oos_sharpe)
        overfit_penalty = overfit_gap * 0.3

        # Stability bonus: prefer strategies with more trades
        trade_bonus = min(0.2, oos_trades / 200)

        # Complexity penalty from strategy
        complexity = strategy.complexity_penalty()

        fitness = core - overfit_penalty + trade_bonus - complexity
        return round(fitness, 6)

    def equity_curve_to_dict(self, equity: pd.Series) -> list[dict]:
        return [
            {"time": str(t), "value": round(v, 2)}
            for t, v in equity.items()
        ]