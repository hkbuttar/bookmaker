"""Inventory/cash accounting for a single market-making strategy.

Deliberately just bookkeeping -- it has no opinion on strategy logic, no
access to the book, and doesn't decide anything. It only records what
`backtest.market_maker_sim` tells it happened.
"""

from __future__ import annotations

import dataclasses

from lob.models import Side


@dataclasses.dataclass
class Trade:
    time: float
    side: Side  # BUY/SELL from the strategy's own perspective
    price: float
    size: int
    is_maker: bool  # True if this fill was the strategy's resting order getting hit


def equity_from(cash: float, inventory: int, mid_price: float | None) -> float:
    """Mark-to-mid equity: cash plus inventory valued at the current
    mid-price. If inventory is flat, equity is just cash and a missing
    mid-price (an empty side of the book) doesn't matter. If inventory is
    non-zero and mid_price is unavailable, equity can't be marked and this
    returns NaN rather than silently using a stale price.

    A free function, not just a Portfolio method, so risk.kill_switch can
    compute the same equity a strategy would see from MarketState's raw
    cash/inventory/mid_price fields without duplicating this formula or
    needing a Portfolio instance of its own.
    """
    if inventory == 0:
        return cash
    if mid_price is None:
        return float("nan")
    return cash + inventory * mid_price


@dataclasses.dataclass
class Portfolio:
    cash: float = 0.0
    inventory: int = 0
    trades: list[Trade] = dataclasses.field(default_factory=list)

    def apply_fill(self, side: Side, price: float, size: int, time: float, is_maker: bool) -> None:
        if side == Side.BUY:
            self.cash -= price * size
            self.inventory += size
        else:
            self.cash += price * size
            self.inventory -= size
        self.trades.append(Trade(time=time, side=side, price=price, size=size, is_maker=is_maker))

    def equity(self, mid_price: float | None) -> float:
        return equity_from(self.cash, self.inventory, mid_price)
