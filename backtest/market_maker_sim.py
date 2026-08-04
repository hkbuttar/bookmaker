"""Runs a Strategy against a background order-flow stream, sharing one book.

The strategy's own resting orders live in exactly the same OrderBook as the
background flow (the synthetic generator's "everyone else"), fixed at two
sentinel ids (-1 for its bid, -2 for its ask, since a strategy holds at most
one resting order per side here). That means background market orders can
fill the strategy's quotes, and -- if a strategy is ever aggressive enough
to quote through the touch -- the strategy's own submission can immediately
cross resting background liquidity. Both directions are handled uniformly
by `_attribute_fills` below, which only cares whether the sentinel ids show
up as maker or taker on a given Fill.

Per background event, in order:
  1. Apply the event to the book (via `MatchingEngine.process_event`, the
     same per-event step `lob.engine.MatchingEngine.replay` uses).
  2. Take one book snapshot (`OrderBook.top_levels`), which doubles as both
     the state fed to the strategy and the row recorded for
     `book_snapshots` -- one call, not two, matching Step 3's replay cost.
  3. Ask the strategy for its desired quote given that state.
  4. Only touch the book again if the desired quote actually changed:
     cancel whatever's resting, then submit the new bid/ask. This is a
     disclosed modeling choice, not an artifact -- requoting on every
     single background event regardless of whether anything changed would
     needlessly cost the strategy its FIFO queue priority on every tick,
     which would contaminate the Step 5/6 comparisons with a requote-churn
     effect that has nothing to do with inventory or adverse-selection
     awareness.

One consequence worth flagging: because the snapshot in step 2 is taken
*before* step 4's requote, a strategy's brand-new quote only shows up in
`book_snapshots` starting from the *next* event's row, not the row for the
event that triggered it. For research-scale analysis this one-event lag is
immaterial; it's noted here so it doesn't look like a bug later.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from backtest.portfolio import Portfolio
from lob.engine import LatencyModel, MatchingEngine
from lob.features import imbalance_from_row, mid_and_spread_from_row
from lob.models import Fill, Side
from strategies.base import MarketState, Quote, Strategy

STRATEGY_BID_ID = -1
STRATEGY_ASK_ID = -2
_STRATEGY_IDS = {STRATEGY_BID_ID, STRATEGY_ASK_ID}


@dataclasses.dataclass
class BacktestResult:
    portfolio: Portfolio
    portfolio_history: pd.DataFrame  # time, inventory, cash, equity
    book_snapshots: pd.DataFrame


def _attribute_fills(fills: list[Fill], portfolio: Portfolio) -> None:
    for f in fills:
        if f.maker_order_id in _STRATEGY_IDS:
            portfolio.apply_fill(side=f.maker_side, price=f.price, size=f.size, time=f.time, is_maker=True)
        elif f.taker_order_id in _STRATEGY_IDS:
            portfolio.apply_fill(side=f.taker_side, price=f.price, size=f.size, time=f.time, is_maker=False)


def run_backtest(
    background_events: pd.DataFrame,
    strategy: Strategy,
    tick_size: float = 0.01,
    imbalance_levels: int = 1,
    record_levels: int = 10,
    latency_model: LatencyModel | None = None,
) -> BacktestResult:
    engine = MatchingEngine(tick_size=tick_size, latency_model=latency_model)
    portfolio = Portfolio()
    records = engine.prepare_events(background_events)

    last_quote: Quote | None = None
    portfolio_rows: list[dict] = []
    book_rows: list[dict] = []

    for rec in records:
        outcome = engine.process_event(rec)
        _attribute_fills(outcome.fills, portfolio)

        row = engine.book.top_levels(record_levels)
        mid_price, spread = mid_and_spread_from_row(row)
        imbalance = imbalance_from_row(row, imbalance_levels)

        state = MarketState(
            time=rec["arrival_time"],
            best_bid=None if mid_price is None else row["bid_price_1"],
            best_ask=None if mid_price is None else row["ask_price_1"],
            mid_price=mid_price,
            spread=spread,
            imbalance=imbalance,
            inventory=portfolio.inventory,
            cash=portfolio.cash,
        )
        desired = strategy.quote(state)

        if desired != last_quote:
            if engine.book.has_order(STRATEGY_BID_ID):
                engine.book.cancel_order(STRATEGY_BID_ID, rec["arrival_time"])
            if engine.book.has_order(STRATEGY_ASK_ID):
                engine.book.cancel_order(STRATEGY_ASK_ID, rec["arrival_time"])

            if desired.bid_price is not None and desired.bid_size > 0:
                fills = engine.book.submit_limit_order(
                    STRATEGY_BID_ID, Side.BUY, desired.bid_price, desired.bid_size,
                    rec["arrival_time"], rec["arrival_time"],
                )
                _attribute_fills(fills, portfolio)
            if desired.ask_price is not None and desired.ask_size > 0:
                fills = engine.book.submit_limit_order(
                    STRATEGY_ASK_ID, Side.SELL, desired.ask_price, desired.ask_size,
                    rec["arrival_time"], rec["arrival_time"],
                )
                _attribute_fills(fills, portfolio)

            last_quote = desired

        portfolio_rows.append(
            {
                "time": rec["arrival_time"],
                "inventory": portfolio.inventory,
                "cash": portfolio.cash,
                "equity": portfolio.equity(mid_price),
            }
        )
        row["time"] = rec["arrival_time"]
        book_rows.append(row)

    return BacktestResult(
        portfolio=portfolio,
        portfolio_history=pd.DataFrame(portfolio_rows),
        book_snapshots=pd.DataFrame(book_rows),
    )
