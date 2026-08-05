"""Runs a Strategy against captured real Binance data (data.binance_capture),
producing a BacktestResult in the same shape backtest.market_maker_sim
produces, so backtest.metrics.summarize works unchanged on either -- this
is what lets synthetic and real-data results be compared with the same
metric code, not a parallel reimplementation.

This is deliberately a *different, simpler* execution model than
market_maker_sim.run_backtest, not a shortcut: Binance's public feed is
price-level aggregated (data.binance_capture's book_snapshots/trades), not
per-order, so there is no real order book to submit a resting order into
and no real queue position to preserve. Concretely:

- At each captured book_snapshot (already in the same top_levels shape
  lob.order_book.OrderBook produces, by design -- see
  data/binance_capture.py's docstring), the strategy is asked for a quote,
  exactly like the synthetic backtest.
- Fills are simulated against the real trade tape: any real trade printing
  at or through our resting price fills us, up to our quote size. This is
  an optimistic assumption, disclosed as such: it assumes we're always at
  the front of the queue at our price, which a real resting order often
  would not be. It's the standard simplification for backtesting against
  L2/trade-tape data without full order-level (L3) access, not a claim
  that this is as accurate as the synthetic backtest's real matching
  engine. Results from this module should be read as directional
  validation ("does the strategy ranking hold up on real data?"), not as
  precise real-world P&L estimates.

Optional `strategy_latency_model` applies the same decision/arrival-time
separation as `market_maker_sim.run_backtest`, simplified for this
coarser, snapshot-driven harness: at most one requote is ever in flight
(a fresh decision overwrites any earlier one that hasn't arrived yet,
rather than both being tracked and possibly arriving out of order the way
the synthetic engine's heap allows) -- a disclosed simplification, since
this harness iterates captured snapshots rather than a continuous event
stream. A pending requote "graduates" into the live resting quote as soon
as a processed snapshot's time reaches its arrival_time; until then,
fills are still simulated against whatever was resting before.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from backtest.market_maker_sim import BacktestResult
from backtest.portfolio import Portfolio
from lob.engine import LatencyModel, zero_latency
from lob.features import imbalance_from_row, mid_and_spread_from_row
from lob.models import Side
from strategies.base import MarketState, Quote, Strategy


@dataclasses.dataclass
class _RestingQuote:
    bid_price: float | None
    bid_remaining: float
    ask_price: float | None
    ask_remaining: float


def _as_resting(quote: Quote) -> _RestingQuote:
    return _RestingQuote(
        bid_price=quote.bid_price,
        bid_remaining=quote.bid_size if quote.bid_price is not None else 0.0,
        ask_price=quote.ask_price,
        ask_remaining=quote.ask_size if quote.ask_price is not None else 0.0,
    )


def run_binance_backtest(
    book_snapshots: pd.DataFrame,
    trades: pd.DataFrame,
    strategy: Strategy,
    imbalance_levels: int = 1,
    strategy_latency_model: LatencyModel | None = None,
) -> BacktestResult:
    strategy_latency = strategy_latency_model or zero_latency

    portfolio = Portfolio()
    resting = _RestingQuote(None, 0.0, None, 0.0)
    last_decided_quote: Quote | None = None
    pending: tuple[float, Quote] | None = None  # (arrival_time, quote) -- at most one in flight

    book_snapshots = book_snapshots.sort_values("time").reset_index(drop=True)
    trades = trades.sort_values("time").reset_index(drop=True)
    trade_idx = 0
    n_trades = len(trades)

    portfolio_rows: list[dict] = []

    for i, row in book_snapshots.iterrows():
        window_end = book_snapshots["time"].iloc[i + 1] if i + 1 < len(book_snapshots) else float("inf")

        # A previously-decided requote graduates into the live resting
        # quote once its arrival_time has been reached.
        if pending is not None and pending[0] <= row["time"]:
            resting = _as_resting(pending[1])
            pending = None

        mid_price, spread = mid_and_spread_from_row(row)
        imbalance = imbalance_from_row(row, imbalance_levels)

        state = MarketState(
            time=row["time"],
            best_bid=None if mid_price is None else row["bid_price_1"],
            best_ask=None if mid_price is None else row["ask_price_1"],
            mid_price=mid_price,
            spread=spread,
            imbalance=imbalance,
            inventory=portfolio.inventory,
            cash=portfolio.cash,
        )
        desired = strategy.quote(state)

        # A pending quote already in flight that matches what we want now
        # counts as "covered" -- without this check, every row before it
        # graduates would look like resting is missing it and reschedule
        # a fresh copy, repeatedly pushing the effective arrival time back.
        resting_covers_desired = (
            desired.bid_price is None or (resting.bid_price == desired.bid_price and resting.bid_remaining > 0)
        ) and (
            desired.ask_price is None or (resting.ask_price == desired.ask_price and resting.ask_remaining > 0)
        )
        pending_covers_desired = pending is not None and pending[1] == desired

        if desired != last_decided_quote:
            arrival_time = strategy_latency(row["time"])
            if arrival_time <= row["time"]:
                resting = _as_resting(desired)
                pending = None
            else:
                pending = (arrival_time, desired)
            last_decided_quote = desired
        elif not resting_covers_desired and not pending_covers_desired:
            # Target unchanged, but a full fill left resting unable to
            # satisfy it and nothing is already in flight to fix that --
            # re-submit the same target.
            arrival_time = strategy_latency(row["time"])
            if arrival_time <= row["time"]:
                resting = _as_resting(desired)
                pending = None
            else:
                pending = (arrival_time, desired)

        # Simulate fills from every real trade between this snapshot and
        # the next, against whatever's actually resting right now (which
        # may still be the *previous* quote if the new one hasn't arrived).
        while trade_idx < n_trades and trades["time"].iloc[trade_idx] < window_end:
            trade = trades.iloc[trade_idx]
            trade_idx += 1
            if resting.bid_price is not None and trade["price"] <= resting.bid_price and resting.bid_remaining > 0:
                fill_size = min(resting.bid_remaining, trade["size"])
                portfolio.apply_fill(Side.BUY, resting.bid_price, fill_size, trade["time"], is_maker=True)
                resting.bid_remaining -= fill_size
            elif resting.ask_price is not None and trade["price"] >= resting.ask_price and resting.ask_remaining > 0:
                fill_size = min(resting.ask_remaining, trade["size"])
                portfolio.apply_fill(Side.SELL, resting.ask_price, fill_size, trade["time"], is_maker=True)
                resting.ask_remaining -= fill_size

        portfolio_rows.append(
            {
                "time": row["time"],
                "inventory": portfolio.inventory,
                "cash": portfolio.cash,
                "equity": portfolio.equity(mid_price),
            }
        )

    return BacktestResult(
        portfolio=portfolio,
        portfolio_history=pd.DataFrame(portfolio_rows),
        book_snapshots=book_snapshots,
    )
