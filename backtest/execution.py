"""Shared order-submission/fill-attribution mechanics for anything that puts
a single agent's resting quotes into a live OrderBook alongside background
flow: `backtest.market_maker_sim` (Steps 4-8's strategies) and `rl.env`
(Step 9's RL policy) both need exactly this, so it lives here once rather
than being reimplemented per caller.

The agent's bid/ask live at two fixed sentinel order ids, since an agent
holds at most one resting order per side. Background flow can fill them
(the agent as maker) and, if an agent is ever aggressive enough to quote
through the touch, the agent's own submission can immediately cross
resting background liquidity (the agent as taker) -- `attribute_fills`
handles both directions uniformly by checking which side of a Fill the
sentinel ids show up on.
"""

from __future__ import annotations

from backtest.portfolio import Portfolio
from lob.engine import MatchingEngine
from lob.models import Fill, Side
from strategies.base import Quote

AGENT_BID_ID = -1
AGENT_ASK_ID = -2
AGENT_IDS = {AGENT_BID_ID, AGENT_ASK_ID}


def attribute_fills(fills: list[Fill], portfolio: Portfolio) -> None:
    for f in fills:
        if f.maker_order_id in AGENT_IDS:
            portfolio.apply_fill(side=f.maker_side, price=f.price, size=f.size, time=f.time, is_maker=True)
        elif f.taker_order_id in AGENT_IDS:
            portfolio.apply_fill(side=f.taker_side, price=f.price, size=f.size, time=f.time, is_maker=False)


def apply_requote(
    engine: MatchingEngine,
    portfolio: Portfolio,
    quote: Quote,
    arrival_time: float,
    decision_time: float,
) -> None:
    """Cancel whatever's currently resting under the agent's sentinel ids
    and submit the new bid/ask, against the book as it actually is *right
    now* -- the caller is responsible for only invoking this at the
    quote's true arrival_time (see backtest.market_maker_sim and rl.env's
    merge-loop docstrings for why that matters under latency).
    """
    if engine.book.has_order(AGENT_BID_ID):
        engine.book.cancel_order(AGENT_BID_ID, arrival_time)
    if engine.book.has_order(AGENT_ASK_ID):
        engine.book.cancel_order(AGENT_ASK_ID, arrival_time)

    if quote.bid_price is not None and quote.bid_size > 0:
        fills = engine.book.submit_limit_order(
            AGENT_BID_ID, Side.BUY, quote.bid_price, quote.bid_size, decision_time, arrival_time
        )
        attribute_fills(fills, portfolio)
    if quote.ask_price is not None and quote.ask_size > 0:
        fills = engine.book.submit_limit_order(
            AGENT_ASK_ID, Side.SELL, quote.ask_price, quote.ask_size, decision_time, arrival_time
        )
        attribute_fills(fills, portfolio)
