"""Evolution engine — genetic algorithm for strategy mutation and selection."""
from __future__ import annotations
import random
import copy
from typing import List
import numpy as np
from strategy import Strategy, StrategyParams


STRATEGY_TYPES = ["ema_rsi", "macd_vwap", "bb_stoch"]


def random_params() -> StrategyParams:
    """Generate a random valid strategy parameter set."""
    ema_fast = random.randint(5, 20)
    ema_slow = random.randint(ema_fast + 5, 50)
    macd_fast = random.randint(5, 15)
    macd_slow = random.randint(macd_fast + 5, 35)
    return StrategyParams(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=random.randint(7, 21),
        rsi_oversold=random.uniform(20, 40),
        rsi_overbought=random.uniform(60, 80),
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=random.randint(5, 12),
        stop_loss_atr=random.uniform(1.0, 4.0),
        take_profit_atr=random.uniform(2.0, 8.0),
        position_size=random.uniform(0.05, 0.3),
        strategy_type=random.choice(STRATEGY_TYPES),
    )


def mutate(params: StrategyParams, mutation_rate: float = 0.3) -> StrategyParams:
    """Mutate parameters with given probability per field."""
    p = copy.deepcopy(params)

    def maybe(fn):
        if random.random() < mutation_rate:
            fn()

    maybe(lambda: setattr(p, "ema_fast", max(3, p.ema_fast + random.randint(-3, 3))))
    maybe(lambda: setattr(p, "ema_slow", max(p.ema_fast + 3, p.ema_slow + random.randint(-5, 5))))
    maybe(lambda: setattr(p, "rsi_period", max(5, p.rsi_period + random.randint(-3, 3))))
    maybe(lambda: setattr(p, "rsi_oversold", np.clip(p.rsi_oversold + random.uniform(-5, 5), 15, 45)))
    maybe(lambda: setattr(p, "rsi_overbought", np.clip(p.rsi_overbought + random.uniform(-5, 5), 55, 85)))
    maybe(lambda: setattr(p, "stop_loss_atr", np.clip(p.stop_loss_atr + random.uniform(-0.5, 0.5), 0.5, 5.0)))
    maybe(lambda: setattr(p, "take_profit_atr", np.clip(p.take_profit_atr + random.uniform(-0.5, 0.5), 1.0, 10.0)))
    maybe(lambda: setattr(p, "position_size", np.clip(p.position_size + random.uniform(-0.05, 0.05), 0.02, 0.4)))
    maybe(lambda: setattr(p, "strategy_type", random.choice(STRATEGY_TYPES)))

    return p


def crossover(parent_a: StrategyParams, parent_b: StrategyParams) -> StrategyParams:
    """Single-point crossover between two parents."""
    fields = list(StrategyParams.__dataclass_fields__.keys())
    cut = random.randint(1, len(fields) - 1)
    child_dict = {}
    for i, f in enumerate(fields):
        parent = parent_a if i < cut else parent_b
        child_dict[f] = getattr(parent, f)
    return StrategyParams.from_dict(child_dict)


class EvolutionEngine:
    def __init__(
        self,
        population_size: int = 20,
        elite_fraction: float = 0.2,
        mutation_rate: float = 0.3,
        tournament_size: int = 4,
    ):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.generation = 0
        self.population: List[Strategy] = []
        self.history: List[dict] = []       # best per generation

    def initialize(self) -> List[Strategy]:
        """Create initial random population."""
        self.population = [
            Strategy(params=random_params(), generation=0)
            for _ in range(self.population_size)
        ]
        self.generation = 0
        return self.population

    def evolve(self, evaluated_pop: List[Strategy]) -> List[Strategy]:
        """Produce next generation from evaluated population."""
        self.generation += 1

        # Sort by fitness
        sorted_pop = sorted(evaluated_pop, key=lambda s: s.fitness, reverse=True)

        # Record generation stats
        self.history.append({
            "generation": self.generation - 1,
            "best_fitness": sorted_pop[0].fitness,
            "avg_fitness": np.mean([s.fitness for s in sorted_pop]),
            "best_sharpe": sorted_pop[0].sharpe,
            "best_return": sorted_pop[0].total_return,
            "best_strategy_id": sorted_pop[0].id,
            "best_type": sorted_pop[0].params.strategy_type,
        })

        # Elite survivors
        n_elite = max(1, int(self.population_size * self.elite_fraction))
        elites = sorted_pop[:n_elite]

        # Kill strategies with negative fitness
        viable = [s for s in sorted_pop if s.fitness > 0]
        if len(viable) < 2:
            viable = sorted_pop[:max(2, n_elite)]

        new_pop = [copy.deepcopy(e) for e in elites]
        new_pop[0].generation = self.generation  # keep top alive

        while len(new_pop) < self.population_size:
            roll = random.random()

            if roll < 0.5:
                # Mutation
                parent = self._tournament_select(viable)
                child_params = mutate(parent.params, self.mutation_rate)
                child = Strategy(
                    params=child_params,
                    generation=self.generation,
                    parent_id=parent.id,
                )
            elif roll < 0.8:
                # Crossover
                p1 = self._tournament_select(viable)
                p2 = self._tournament_select(viable)
                child_params = crossover(p1.params, p2.params)
                child = Strategy(
                    params=child_params,
                    generation=self.generation,
                    parent_id=f"{p1.id}+{p2.id}",
                )
            else:
                # Random newcomer (diversity injection)
                child = Strategy(params=random_params(), generation=self.generation)

            new_pop.append(child)

        self.population = new_pop
        return new_pop

    def _tournament_select(self, population: List[Strategy]) -> Strategy:
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda s: s.fitness)

    def best(self) -> Strategy | None:
        if not self.population:
            return None
        return max(self.population, key=lambda s: s.fitness)

    def leaderboard(self, n: int = 10) -> List[dict]:
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        return [s.summary() for s in sorted_pop[:n]]