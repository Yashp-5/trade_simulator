"""Database models — SQLite via SQLAlchemy for storing strategies and results."""
from __future__ import annotations
import json
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, Text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
engine = create_engine("sqlite:////tmp/trading_simulator.db", echo=False)
Session = sessionmaker(bind=engine)


class StrategyRecord(Base):
    __tablename__ = "strategies"
    id = Column(String, primary_key=True)
    generation = Column(Integer, default=0)
    parent_id = Column(String, nullable=True)
    strategy_type = Column(String)
    params_json = Column(Text)
    fitness = Column(Float, default=0.0)
    sharpe = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    total_return = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    num_trades = Column(Integer, default=0)
    ticker = Column(String, default="")
    is_elite = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class GenerationRecord(Base):
    __tablename__ = "generations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation = Column(Integer)
    ticker = Column(String)
    best_fitness = Column(Float)
    avg_fitness = Column(Float)
    best_sharpe = Column(Float)
    best_return = Column(Float)
    best_strategy_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


def save_strategy(strategy, ticker: str, is_elite: bool = False):
    session = Session()
    try:
        rec = session.get(StrategyRecord, strategy.id)
        if rec is None:
            rec = StrategyRecord(id=strategy.id)
            session.add(rec)
        rec.generation = strategy.generation
        rec.parent_id = strategy.parent_id
        rec.strategy_type = strategy.params.strategy_type
        rec.params_json = json.dumps(strategy.params.to_dict())
        rec.fitness = strategy.fitness
        rec.sharpe = strategy.sharpe
        rec.max_drawdown = strategy.max_drawdown
        rec.total_return = strategy.total_return
        rec.win_rate = strategy.win_rate
        rec.num_trades = strategy.num_trades
        rec.ticker = ticker
        rec.is_elite = is_elite
        session.commit()
    finally:
        session.close()


def save_generation(gen_data: dict, ticker: str):
    session = Session()
    try:
        rec = GenerationRecord(
            generation=gen_data["generation"],
            ticker=ticker,
            best_fitness=gen_data["best_fitness"],
            avg_fitness=gen_data["avg_fitness"],
            best_sharpe=gen_data["best_sharpe"],
            best_return=gen_data["best_return"],
            best_strategy_id=gen_data["best_strategy_id"],
        )
        session.add(rec)
        session.commit()
    finally:
        session.close()


def get_all_strategies(ticker: str = None, limit: int = 100) -> list[dict]:
    session = Session()
    try:
        q = session.query(StrategyRecord)
        if ticker:
            q = q.filter(StrategyRecord.ticker == ticker)
        records = q.order_by(StrategyRecord.fitness.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "generation": r.generation,
                "type": r.strategy_type,
                "fitness": r.fitness,
                "sharpe": r.sharpe,
                "max_drawdown": r.max_drawdown,
                "total_return": r.total_return,
                "win_rate": r.win_rate,
                "num_trades": r.num_trades,
                "ticker": r.ticker,
                "is_elite": r.is_elite,
                "params": json.loads(r.params_json),
            }
            for r in records
        ]
    finally:
        session.close()


def get_generation_history(ticker: str = None) -> list[dict]:
    session = Session()
    try:
        q = session.query(GenerationRecord)
        if ticker:
            q = q.filter(GenerationRecord.ticker == ticker)
        records = q.order_by(GenerationRecord.generation.asc()).all()
        return [
            {
                "generation": r.generation,
                "best_fitness": r.best_fitness,
                "avg_fitness": r.avg_fitness,
                "best_sharpe": r.best_sharpe,
                "best_return": r.best_return,
            }
            for r in records
        ]
    finally:
        session.close()