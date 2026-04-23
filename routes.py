"""FastAPI — REST endpoints for the trading simulator."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import numpy as np
from strategy import Strategy, StrategyParams
from backtest import Backtester
from execution import ExecutionConfig
from genetic import EvolutionEngine, random_params
from detector import RegimeDetector
from loader import load, available_tickers
from models import (
    init_db, save_strategy, save_generation,
    get_all_strategies, get_generation_history
)

app = FastAPI(title="Autonomous Trading Simulator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
backtester = Backtester()
regime_detector = RegimeDetector()

# Global evolution state
_engine: EvolutionEngine | None = None
_evolution_running = False
_evolution_status = {"running": False, "generation": 0, "message": "idle"}


class EvolutionConfig(BaseModel):
    ticker: str = "SPY"
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.3
    elite_fraction: float = 0.2
    train_split: float = 0.7


class SingleBacktestRequest(BaseModel):
    ticker: str = "SPY"
    strategy_type: str = "ema_rsi"
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 4.0
    position_size: float = 0.1


@app.get("/")
def root():
    return {"status": "ok", "message": "Trading Simulator API"}


@app.get("/tickers")
def tickers():
    return {"tickers": available_tickers()}


@app.post("/backtest/single")
def single_backtest(req: SingleBacktestRequest):
    try:
        df = load(req.ticker, start="2020-01-01", end="2025-12-31")
        params = StrategyParams(
            strategy_type=req.strategy_type,
            ema_fast=req.ema_fast,
            ema_slow=req.ema_slow,
            rsi_period=req.rsi_period,
            stop_loss_atr=req.stop_loss_atr,
            take_profit_atr=req.take_profit_atr,
            position_size=req.position_size,
        )
        strategy = Strategy(params=params)
        result = backtester.run(strategy, df, split=0.7)

        # Build equity curve
        equity_full = backtester.equity_curve_to_dict(result["full"]["equity"])
        # Subsample for response size
        step = max(1, len(equity_full) // 200)
        equity_sampled = equity_full[::step]

        # Ensure metrics are serializable
        def py_metrics(metrics):
            return {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in metrics.items()}

        return {
            "strategy_id": strategy.id,
            "fitness": float(result["fitness"]),
            "train_metrics": py_metrics(result["train_metrics"]),
            "test_metrics": py_metrics(result["test_metrics"]),
            "equity_curve": [
                {"time": str(point["time"]), "value": float(point["value"])}
                for point in equity_sampled
            ],
            "num_trades": int(len(result["full"]["trades"])),
            "trades": [
                {
                    "entry": str(t.entry_time),
                    "exit": str(t.exit_time),
                    "direction": int(t.direction),
                    "pnl": float(round(t.pnl, 2)),
                    "pnl_pct": float(round(t.pnl_pct * 100, 3)),
                    "exit_reason": t.exit_reason,
                }
                for t in result["full"]["trades"][:50]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/evolution/start")
def start_evolution(config: EvolutionConfig, background_tasks: BackgroundTasks):
    global _engine, _evolution_running
    if _evolution_running:
        return {"message": "Evolution already running"}
    _engine = EvolutionEngine(
        population_size=config.population_size,
        mutation_rate=config.mutation_rate,
        elite_fraction=config.elite_fraction,
    )
    background_tasks.add_task(_run_evolution, config)
    return {"message": "Evolution started", "config": config.dict()}


def _run_evolution(config: EvolutionConfig):
    global _evolution_running, _evolution_status
    _evolution_running = True
    _evolution_status["running"] = True

    try:
        df = load(config.ticker, start="2020-01-01", end="2025-12-31")
        population = _engine.initialize()

        for gen in range(config.generations):
            _evolution_status["generation"] = gen
            _evolution_status["message"] = f"Evaluating generation {gen}..."

            for strategy in population:
                result = backtester.run(strategy, df, split=config.train_split)
                is_elite = strategy.fitness > 0.5
                save_strategy(strategy, config.ticker, is_elite)

            if _engine.history:
                save_generation(_engine.history[-1], config.ticker)

            population = _engine.evolve(population)
            best = _engine.best()
            _evolution_status["message"] = (
                f"Gen {gen} done. Best fitness={best.fitness:.4f} sharpe={best.sharpe:.4f}"
                if best else f"Gen {gen} done"
            )

        _evolution_status["message"] = f"Evolution complete! {config.generations} generations."
    except Exception as e:
        _evolution_status["message"] = f"Error: {str(e)}"
    finally:
        _evolution_running = False
        _evolution_status["running"] = False


@app.get("/evolution/status")
def evolution_status():
    return {
        **_evolution_status,
        "generation": _engine.generation if _engine else 0,
        "leaderboard": _engine.leaderboard(5) if _engine else [],
    }


@app.get("/strategies")
def strategies(ticker: Optional[str] = None, limit: int = 50):
    return {"strategies": get_all_strategies(ticker, limit)}


@app.get("/strategies/best")
def best_strategy(ticker: Optional[str] = None):
    strats = get_all_strategies(ticker, 1)
    if not strats:
        return {"strategy": None}
    return {"strategy": strats[0]}


@app.get("/evolution/history")
def evolution_history(ticker: Optional[str] = None):
    return {"history": get_generation_history(ticker)}


@app.get("/regime/{ticker}")
def current_regime(ticker: str):
    try:
        df = load(ticker, start="2024-01-01", end="2025-12-31")
        regime = regime_detector.current_regime(df)
        regimes = regime_detector.detect(df)
        dist = regime_detector.regime_distribution(regimes)
        return {"ticker": ticker, "current_regime": regime, "distribution": dist}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))