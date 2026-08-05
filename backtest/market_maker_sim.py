"""Runs a Strategy against a background order-flow stream, sharing one book.

The strategy's own resting orders live in exactly the same OrderBook as the
background flow (the synthetic generator's "everyone else"), fixed at two
sentinel ids (see backtest.execution: an agent holds at most one resting
order per side here). That means background market orders can fill the
strategy's quotes, and -- if a strategy is ever aggressive enough to quote
through the touch -- the strategy's own submission can immediately cross
resting background liquidity. Both directions are handled uniformly by
`backtest.execution.attribute_fills`, which only cares whether the
sentinel ids show up as maker or taker on a given Fill. That module (fill
attribution + requote application) is shared with `rl.env`, which
needs the exact same mechanics for its own agent.

Latency applies only to the strategy's own order submissions, not
to the background flow: `strategy_latency_model` maps the moment the
strategy *decides* on a new quote to the moment that requote actually
*arrives* at the book. The background stream is the ground-truth market
data feed being modeled, not something whose own latency this project is
studying, so it always replays at its given timestamps (zero latency at
the MatchingEngine level, regardless of `strategy_latency_model`).

This is exactly what the decision/arrival-time split built into
`lob.engine.MatchingEngine` from the start was for: a strategy's requote is
scheduled into a min-heap keyed by its arrival_time, and the main loop
merges that heap with the background event stream in true arrival order.
Concretely, per iteration the loop processes whichever comes next:

  a) A background event, exactly as before (apply it, observe the
     resulting state, ask the strategy for a quote; if that quote differs
     from the last one *decided* -- not yet necessarily arrived -- schedule
     it to arrive at `strategy_latency_model(decision_time)`).
  b) A pending requote's arrival: cancel whatever's currently resting and
     submit the new bid/ask, against whatever the book *actually* looks
     like at that moment -- which may have moved due to background events
     that landed during the delay. That's "replay book state forward
     during the delay" from the plan: it falls out of the merge ordering
     for free, rather than needing to be simulated separately.

Two disclosed modeling choices worth being explicit about:

- A requote's cancel-and-resubmit is treated as one atomic message with
  one latency draw, not two independently-delayed messages. A real system
  might send a cancel and a new order separately (with a brief window
  where stale and fresh quotes coexist, or neither does); modeling that
  would add complexity without changing the P&L-vs-latency shape this
  project is after.
- Requoting is fire-and-forget: the strategy is asked for a fresh decision
  after every background event regardless of whether its previous requote
  has arrived yet, and multiple in-flight requotes are not suppressed or
  coalesced. Under a stochastic latency model this means a later-decided
  requote can legitimately arrive before an earlier one (reordering in
  flight), which is realistic, not a bug -- whichever arrives last is what
  ends up resting.

At the default `strategy_latency_model=None` (zero latency), every pending
requote's arrival_time equals the background event that triggered it, so
it's always processed before the next background event -- reproducing the
exact pre-latency-modeling behavior (requote applied inline, same
iteration) as a special case, not a separate code path.

Book snapshots and portfolio history stay tied to background events only
(never to a requote's arrival on its own), so their row count is always
exactly `len(background_events)`. This extends the
already-disclosed one-event lag: a delayed requote's effect on the book
first shows up in the *next* background event's row, whether "next" is
one event later (zero latency) or several events later (its actual
arrival, once real latency is involved).
"""

from __future__ import annotations

import dataclasses
import heapq
import itertools

import pandas as pd

from backtest.execution import AGENT_ASK_ID, AGENT_BID_ID, apply_requote, attribute_fills
from backtest.portfolio import Portfolio
from lob.engine import LatencyModel, MatchingEngine, zero_latency
from lob.features import imbalance_from_row, mid_and_spread_from_row
from strategies.base import MarketState, Quote, Strategy


@dataclasses.dataclass
class BacktestResult:
    portfolio: Portfolio
    portfolio_history: pd.DataFrame  # time, inventory, cash, equity
    book_snapshots: pd.DataFrame


def run_backtest(
    background_events: pd.DataFrame,
    strategy: Strategy,
    tick_size: float = 0.01,
    imbalance_levels: int = 1,
    record_levels: int = 10,
    strategy_latency_model: LatencyModel | None = None,
    decision_interval_seconds: float | None = None,
) -> BacktestResult:
    """`decision_interval_seconds=None` (default): unchanged, original
    behavior -- the strategy is asked to decide after every background
    event. Set to a float to throttle decisions to fixed time boundaries
    instead (at most once per interval, on the first event whose
    arrival_time reaches the next boundary) -- this is what
    `rl.evaluate.RLStrategyAdapter` needs to put a trained RL policy
    (which only ever acted once per `decision_interval_seconds` during
    training) through the exact same execution mechanics as the hand-tuned
    strategies, for a decision-cadence-matched comparison. Since none of
    the hand-tuned strategies condition on wall-clock time, throttling their
    decisions this way has no behavioral effect on them -- only the
    cadence of *opportunities* to change a quote is reduced, and their
    quote wouldn't have changed on a "quiet" tick anyway.
    """
    engine = MatchingEngine(tick_size=tick_size)  # background flow always replays at zero latency
    portfolio = Portfolio()
    records = engine.prepare_events(background_events)
    strategy_latency = strategy_latency_model or zero_latency

    last_decided_quote: Quote | None = None
    pending: list[tuple[float, int, float, Quote]] = []  # (arrival_time, seq, decision_time, quote)
    seq_counter = itertools.count()
    next_decision_boundary = decision_interval_seconds

    portfolio_rows: list[dict] = []
    book_rows: list[dict] = []

    bg_idx = 0
    n_bg = len(records)

    while bg_idx < n_bg or pending:
        next_bg_time = records[bg_idx]["arrival_time"] if bg_idx < n_bg else float("inf")
        next_pending_time = pending[0][0] if pending else float("inf")

        if next_pending_time <= next_bg_time:
            arrival_time, _, decision_time, quote = heapq.heappop(pending)
            apply_requote(engine, portfolio, quote, arrival_time, decision_time)
            continue

        rec = records[bg_idx]
        bg_idx += 1

        outcome = engine.process_event(rec)
        attribute_fills(outcome.fills, portfolio)

        row = engine.book.top_levels(record_levels)
        mid_price, spread = mid_and_spread_from_row(row)
        imbalance = imbalance_from_row(row, imbalance_levels)

        if decision_interval_seconds is None:
            should_decide = True
        else:
            should_decide = rec["arrival_time"] >= next_decision_boundary
            if should_decide:
                while next_decision_boundary <= rec["arrival_time"]:
                    next_decision_boundary += decision_interval_seconds

        if should_decide:
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

            # Resubmit not just when the *target* quote changes, but also
            # when a side we want quoted isn't actually resting anymore --
            # e.g. it was fully filled. Comparing only against
            # last_decided_quote misses this: if background conditions
            # happen to leave best_bid/best_ask looking unchanged after a
            # full fill (something else was resting at the same price),
            # the strategy's *desired* quote doesn't change either, and it
            # would otherwise stay silently out of the market on that side
            # for the rest of the session despite having zero resting
            # orders there -- a real bug this project's own testing
            # caught, not a hypothetical.
            bid_missing = desired.bid_price is not None and not engine.book.has_order(AGENT_BID_ID)
            ask_missing = desired.ask_price is not None and not engine.book.has_order(AGENT_ASK_ID)

            if desired != last_decided_quote or bid_missing or ask_missing:
                decision_time = rec["arrival_time"]
                arrival_time = strategy_latency(decision_time)
                heapq.heappush(pending, (arrival_time, next(seq_counter), decision_time, desired))
                last_decided_quote = desired

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
