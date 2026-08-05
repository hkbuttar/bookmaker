"""Evaluates a trained RL policy against Steps 4-6's hand-tuned baselines
using identical mechanics, via `RLStrategyAdapter`: a `Strategy` wrapper
around a trained SB3 model, so it can run through the exact same
`backtest.market_maker_sim.run_backtest` + `backtest.metrics.summarize`
pipeline as every other strategy in this project.

Decision cadence is the one thing deliberately *not* left at each
strategy's native default for this comparison: the RL agent only ever
learned to act once per `decision_interval_seconds` (see rl/env.py), so
evaluating it at every background event (as Steps 4-8 do for hand-tuned
strategies) would ask it to decide far more often than it was trained
for -- not a fair test of the policy, and not even well-defined (it was
never shown observations at that granularity). `run_backtest`'s
`decision_interval_seconds` parameter (added for this) throttles the
baselines to the same cadence, so every strategy in a Step 9 comparison
runs under identical decision-frequency rules. This means Step 9's
baseline numbers are not directly comparable to Steps 4-8's own reported
numbers (which use native per-event cadence) -- a disclosed, intentional
difference, not an inconsistency.
"""

from __future__ import annotations

from collections import deque

from stable_baselines3 import DQN

from backtest.market_maker_sim import BacktestResult, run_backtest
from backtest.portfolio import equity_from
from lob.engine import LatencyModel
from rl.env import ACTION_TABLE
from rl.observation import build_observation
from strategies.base import MarketState, Quote, Strategy, round_to_tick


class RLStrategyAdapter(Strategy):
    def __init__(
        self,
        model: DQN,
        session_seconds: float,
        tick_size: float = 0.01,
        realized_vol_window: int = 20,
        inventory_norm_scale: float = 100.0,
        pnl_norm_scale: float = 50.0,
        price_change_norm_scale: float = 100.0,
        vol_scale: float = 1000.0,
        obs_bound: float = 20.0,
    ) -> None:
        self.model = model
        self.session_seconds = session_seconds
        self.tick_size = tick_size
        self.inventory_norm_scale = inventory_norm_scale
        self.pnl_norm_scale = pnl_norm_scale
        self.price_change_norm_scale = price_change_norm_scale
        self.vol_scale = vol_scale
        self.obs_bound = obs_bound
        self._mid_history: deque[float] = deque(maxlen=realized_vol_window + 1)
        self._initial_mid_price: float | None = None

    def quote(self, state: MarketState) -> Quote:
        if state.mid_price is not None:
            if self._initial_mid_price is None:
                self._initial_mid_price = state.mid_price
            self._mid_history.append(state.mid_price)

        equity = equity_from(state.cash, state.inventory, state.mid_price)
        obs = build_observation(
            inventory=state.inventory,
            mid_price=state.mid_price,
            initial_mid_price=self._initial_mid_price,
            spread=state.spread,
            imbalance=state.imbalance,
            mid_history=self._mid_history,
            equity=equity,
            t=state.time,
            session_seconds=self.session_seconds,
            tick_size=self.tick_size,
            inventory_norm_scale=self.inventory_norm_scale,
            pnl_norm_scale=self.pnl_norm_scale,
            price_change_norm_scale=self.price_change_norm_scale,
            vol_scale=self.vol_scale,
            obs_bound=self.obs_bound,
        )
        action, _ = self.model.predict(obs, deterministic=True)
        entry = ACTION_TABLE[int(action)]
        if entry is None or state.mid_price is None:
            return Quote.none()

        offset_ticks, size = entry
        bid_price = round_to_tick(state.mid_price - offset_ticks * self.tick_size, self.tick_size)
        ask_price = round_to_tick(state.mid_price + offset_ticks * self.tick_size, self.tick_size)
        return Quote(bid_price=bid_price, bid_size=size, ask_price=ask_price, ask_size=size)


def evaluate_policy(
    model: DQN,
    events,
    session_seconds: float,
    decision_interval_seconds: float = 1.0,
    tick_size: float = 0.01,
    imbalance_levels: int = 5,
    record_levels: int = 10,
    strategy_latency_model: LatencyModel | None = None,
) -> BacktestResult:
    adapter = RLStrategyAdapter(model, session_seconds=session_seconds, tick_size=tick_size)
    return run_backtest(
        events,
        adapter,
        tick_size=tick_size,
        imbalance_levels=imbalance_levels,
        record_levels=record_levels,
        strategy_latency_model=strategy_latency_model,
        decision_interval_seconds=decision_interval_seconds,
    )
