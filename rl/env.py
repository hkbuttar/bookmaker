"""Gymnasium market-making environment (Step 9).

Reuses the exact same book/portfolio/latency machinery as
`backtest.market_maker_sim` (Steps 4-8) via `backtest.execution`'s shared
fill-attribution and requote-application helpers -- the RL agent's resting
orders live at the same sentinel ids, in the same kind of OrderBook, and
under the same decision/arrival-time latency model as every hand-tuned
strategy. What's different here is the *shape* of the interaction: a
Strategy gets asked for a fresh quote after every background event; an RL
agent instead acts at a fixed decision cadence (`decision_interval_seconds`,
default 1s), because per-background-event decisions would mean hundreds of
thousands of steps per training episode at this project's recalibrated
(Step 8) event density -- intractable for CPU training in "minutes to low
hours." One `step()` call processes every background event and every due
in-flight requote arrival between the previous decision boundary and the
next, in true arrival order (the same merge-loop idea as
`market_maker_sim.run_backtest`, just bounded by a time cutoff instead of
running to exhaustion).

Fixed dataset per episode, not fresh-generated: `events` is one
pre-generated (see data.synthetic_lob), reproducible synthetic session,
replayed identically from t=0 on every reset(). This matches the plan's
"reproducible from a fixed dataset like your other backtests" -- and is a
disclosed simplification, not a hidden one: training against one fixed
realization risks the agent overfitting to that specific sequence rather
than the underlying order-flow *process*, which is exactly the kind of
generalization risk this project's own Limitations section anticipates
("RL trained on limited data may not generalize to unseen regimes").
Evaluate on a held-out, differently-seeded session (see rl/evaluate.py) to
see whether that risk actually materialized here, rather than assuming
either way.

State (7 features, compact by design for CPU trainability):
    1. inventory, normalized by `inventory_norm_scale`
    2. mid-price change from the episode's opening mid, in ticks
       (normalized by `price_change_norm_scale`) -- comparable across
       episodes since every episode starts at the same reference mid
    3. spread, in ticks
    4. order book imbalance over the top `imbalance_levels` price levels
       (see lob.features.imbalance_from_row)
    5. realized volatility of mid-price log returns over the trailing
       `realized_vol_window` *decision steps* (not raw events -- a
       coarser rolling window matched to this environment's own decision
       cadence), scaled by `vol_scale` since raw values are tiny
    6. fraction of the episode remaining, in [0, 1]
    7. unrealized mark-to-mid P&L (equity), normalized by `pnl_norm_scale`

Raw best bid/ask are deliberately omitted: they're fully recoverable from
mid_price and spread already in the state (bid = mid - spread/2, ask = mid
+ spread/2), so including them as separate features would just be
redundant, collinear inputs -- a disclosed simplification relative to the
plan's literal feature list, not an oversight.

Action (Discrete(16)): a single symmetric (bid, ask) offset from mid-price
in `{1,2,3,5,10}` ticks combined with a quote size in `{5,10,20}` shares
(small/medium/large), 15 combinations, plus one explicit "quote nothing"
action -- giving the agent the option to learn defensive pulling on its
own, the way Step 6 hand-codes it. One shared offset for both sides (not
independent bid/ask offsets) keeps the action space small enough to train
in minutes on CPU; this is the discrete action space the plan calls for as
the first thing to try before reaching for continuous actions (PPO).

Reward: realized P&L over the step (mark-to-mid equity delta) minus an
inventory-risk penalty (see rl/reward.py). The plan's stated formula is a
pure quadratic, `-lambda * inventory^2`; an initial training run with
exactly that collapsed to a near-total "never quote" policy, because rare
large-inventory excursions during exploration produced catastrophic
penalties (e.g. ~125 at inventory=500, against a typical step's P&L of a
few hundredths of a dollar) that DQN's TD bootstrapping let poison the
learned value of *any* inventory-taking path, not just the extreme ones
actually visited. The reward here uses `rl.reward.huber_inventory_penalty`
instead: quadratic (same shape, same calibration intent) below
`inventory_penalty_cap`, linear beyond it, so deterrence against large
inventory still grows without bound but can no longer dominate training
off a handful of bad episodes. `inventory_penalty_lambda` (default 1e-3)
and `inventory_penalty_cap` (default 50) are disclosed hyperparameters,
not derived optima -- calibrated by measuring a fixed-offset policy's
actual step_pnl and inventory distributions over a 10-minute episode
(step_pnl std ~0.068, inventory reaching ~30 under that policy) and
picking lambda so the quadratic-region penalty at that inventory level
(lambda * 0.5 * 30^2 ~= 0.45) lands at a few multiples of a typical
step's P&L. Different (session length, quote size, tick size)
combinations would need this re-measured, not assumed to still be
well-calibrated.

A second, later addition: `fill_bonus_per_share` (default 0.01), a small
reward bonus proportional to shares filled this step, added on top of
step_pnl. This was not in the plan's original reward formula -- it was
added after diagnosing that a trained policy, evaluated deterministically,
produced literally zero fills across seven independent evaluation windows
despite the hand-tuned baselines reliably getting 9-16 in each. Root
cause: Step 8's recalibrated (higher-frequency) background order flow
means resting depth several ticks from the touch accumulates into the
thousands of shares (measured: ~6,280 shares within 2 ticks of the best
ask vs. ~35,511 within 5 ticks, against a ~100-150 share typical market
order) -- a queue essentially unreachable without an enormous, vanishingly
rare sweep. Pure P&L-minus-penalty gives no gradient toward avoiding that
region, since sitting far from the touch and never trading yields reward
~= 0, which an undertrained Q-function can't distinguish from "acceptable"
against the risk of a fill that might go badly. The fill bonus gives a
direct, dense incentive to seek competitive (fillable) quotes; it's a
disclosed reward-shaping intervention made in response to an observed
failure mode, not part of the original design, and evaluation still
reports P&L/fill-rate/etc. from the underlying economics, not the bonus
itself, so it doesn't contaminate the actual comparison against baselines.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd

from backtest.execution import apply_requote, attribute_fills
from backtest.portfolio import Portfolio
from lob.engine import LatencyModel, MatchingEngine, zero_latency
from lob.features import imbalance_from_row, mid_and_spread_from_row
from rl.observation import N_FEATURES, build_observation
from rl.reward import huber_inventory_penalty
from strategies.base import Quote, round_to_tick

OFFSET_CHOICES_TICKS: tuple[int, ...] = (1, 2, 3, 5, 10)
SIZE_CHOICES: tuple[int, ...] = (5, 10, 20)

ACTION_TABLE: list[tuple[int, int] | None] = [
    (offset, size) for offset in OFFSET_CHOICES_TICKS for size in SIZE_CHOICES
] + [None]
N_ACTIONS = len(ACTION_TABLE)
NO_QUOTE_ACTION = N_ACTIONS - 1


class MarketMakingEnv(gym.Env):
    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        events: pd.DataFrame,
        tick_size: float = 0.01,
        decision_interval_seconds: float = 1.0,
        inventory_penalty_lambda: float = 1e-3,
        inventory_penalty_cap: float = 50.0,
        fill_bonus_per_share: float = 0.01,
        strategy_latency_model: LatencyModel | None = None,
        imbalance_levels: int = 5,
        record_levels: int = 10,
        realized_vol_window: int = 20,
        inventory_norm_scale: float = 100.0,
        pnl_norm_scale: float = 50.0,
        price_change_norm_scale: float = 100.0,
        vol_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if len(events) == 0:
            raise ValueError("events must be non-empty")

        self._events = events
        self.tick_size = tick_size
        self.decision_interval_seconds = decision_interval_seconds
        self.inventory_penalty_lambda = inventory_penalty_lambda
        self.inventory_penalty_cap = inventory_penalty_cap
        self.fill_bonus_per_share = fill_bonus_per_share
        self.strategy_latency = strategy_latency_model or zero_latency
        self.imbalance_levels = imbalance_levels
        self.record_levels = record_levels
        self.realized_vol_window = realized_vol_window
        self.inventory_norm_scale = inventory_norm_scale
        self.pnl_norm_scale = pnl_norm_scale
        self.price_change_norm_scale = price_change_norm_scale
        self.vol_scale = vol_scale

        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        # All 7 features are normalized to roughly unit scale (see
        # _get_observation); a wide but finite bound plays better with
        # SB3's preprocessing than +/-inf, and _get_observation clips into
        # it so an extreme state can never produce an out-of-bounds sample.
        self._obs_bound = 20.0
        self.observation_space = gym.spaces.Box(
            low=-self._obs_bound, high=self._obs_bound, shape=(N_FEATURES,), dtype=np.float32
        )

        self._session_seconds = float(events["time"].max())
        # A pure function of `events` + always-zero background latency --
        # computed once here, reused every episode instead of re-sorting
        # on every reset().
        self._prepared_records = MatchingEngine(tick_size=tick_size).prepare_events(events)
        self._n_records = len(self._prepared_records)

        self._engine: MatchingEngine | None = None
        self._portfolio: Portfolio | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._engine = MatchingEngine(tick_size=self.tick_size)
        self._portfolio = Portfolio()
        self._bg_idx = 0
        self._pending: list[tuple[float, int, float, Quote]] = []
        self._seq_counter = itertools.count()
        self._t = 0.0
        self._mid_history: deque[float] = deque(maxlen=self.realized_vol_window + 1)
        self._last_mid_price: float | None = None
        self._initial_mid_price: float | None = None

        self._advance()
        obs = self._get_observation()
        return obs, {}

    def step(self, action: int):
        if self._engine is None:
            raise RuntimeError("call reset() before step()")

        quote = self._decode_action(action)
        decision_time = self._t
        arrival_time = self.strategy_latency(decision_time)
        heapq.heappush(self._pending, (arrival_time, next(self._seq_counter), decision_time, quote))

        equity_before = self._portfolio.equity(self._last_mid_price)
        trades_before = len(self._portfolio.trades)

        self._advance()
        obs = self._get_observation()
        equity_after = self._portfolio.equity(self._last_mid_price)

        step_pnl = (
            0.0 if (math.isnan(equity_before) or math.isnan(equity_after)) else (equity_after - equity_before)
        )
        penalty = huber_inventory_penalty(
            self._portfolio.inventory, self.inventory_penalty_lambda, self.inventory_penalty_cap
        )
        filled_shares = sum(t.size for t in self._portfolio.trades[trades_before:])
        fill_bonus = self.fill_bonus_per_share * filled_shares
        reward = step_pnl - penalty + fill_bonus

        terminated = self._bg_idx >= self._n_records and not self._pending
        info = {
            "inventory": self._portfolio.inventory,
            "equity": equity_after,
            "step_pnl": step_pnl,
            "penalty": penalty,
            "fill_bonus": fill_bonus,
            "filled_shares": filled_shares,
        }
        return obs, float(reward), terminated, False, info

    def _advance(self) -> None:
        """Process every background event and due pending requote arrival
        between the current boundary and the next one, in true arrival
        order. If background data is exhausted, keeps draining any
        remaining in-flight requotes regardless of how far past the
        nominal boundary their arrival_time is -- an episode shouldn't end
        with orders silently still in flight.
        """
        boundary = self._t + self.decision_interval_seconds
        while True:
            next_bg_time = (
                self._prepared_records[self._bg_idx]["arrival_time"]
                if self._bg_idx < self._n_records
                else float("inf")
            )
            next_pending_time = self._pending[0][0] if self._pending else float("inf")
            next_time = min(next_bg_time, next_pending_time)

            if next_time == float("inf"):
                break
            if next_time >= boundary and self._bg_idx < self._n_records:
                break

            if next_pending_time <= next_bg_time:
                arrival_time, _, decision_time, quote = heapq.heappop(self._pending)
                apply_requote(self._engine, self._portfolio, quote, arrival_time, decision_time)
            else:
                rec = self._prepared_records[self._bg_idx]
                self._bg_idx += 1
                outcome = self._engine.process_event(rec)
                attribute_fills(outcome.fills, self._portfolio)

        self._t = boundary

    def _decode_action(self, action: int) -> Quote:
        entry = ACTION_TABLE[action]
        if entry is None or self._last_mid_price is None:
            return Quote.none()
        offset_ticks, size = entry
        bid_price = round_to_tick(self._last_mid_price - offset_ticks * self.tick_size, self.tick_size)
        ask_price = round_to_tick(self._last_mid_price + offset_ticks * self.tick_size, self.tick_size)
        return Quote(bid_price=bid_price, bid_size=size, ask_price=ask_price, ask_size=size)

    def _get_observation(self) -> np.ndarray:
        row = self._engine.book.top_levels(self.record_levels)
        mid_price, spread = mid_and_spread_from_row(row)
        imbalance = imbalance_from_row(row, self.imbalance_levels)

        if mid_price is not None:
            if self._initial_mid_price is None:
                self._initial_mid_price = mid_price
            self._mid_history.append(mid_price)
            self._last_mid_price = mid_price

        equity = self._portfolio.equity(self._last_mid_price)

        return build_observation(
            inventory=self._portfolio.inventory,
            mid_price=mid_price,
            initial_mid_price=self._initial_mid_price,
            spread=spread,
            imbalance=imbalance,
            mid_history=self._mid_history,
            equity=equity,
            t=self._t,
            session_seconds=self._session_seconds,
            tick_size=self.tick_size,
            inventory_norm_scale=self.inventory_norm_scale,
            pnl_norm_scale=self.pnl_norm_scale,
            price_change_norm_scale=self.price_change_norm_scale,
            vol_scale=self.vol_scale,
            obs_bound=self._obs_bound,
        )
