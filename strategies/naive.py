"""Baseline strategy: fixed spread around mid-price, fixed quote size, no
inventory or adverse-selection awareness whatsoever.

This is the control for the whole strategy comparison (Step 11): whatever
P&L, fill rate, and adverse-selection cost this produces is what ignoring
risk actually costs, which is the number Steps 5 and 6's added machinery
has to beat to justify itself.
"""

from __future__ import annotations

from strategies.base import MarketState, Quote, Strategy, round_to_tick


class NaiveSymmetricStrategy(Strategy):
    def __init__(self, half_spread: float, quote_size: int, tick_size: float = 0.01) -> None:
        if half_spread <= 0:
            raise ValueError("half_spread must be positive, or bid/ask quotes could cross")
        self.half_spread = half_spread
        self.quote_size = quote_size
        self.tick_size = tick_size

    def quote(self, state: MarketState) -> Quote:
        if state.mid_price is None:
            return Quote.none()

        bid_price = round_to_tick(state.mid_price - self.half_spread, self.tick_size)
        ask_price = round_to_tick(state.mid_price + self.half_spread, self.tick_size)

        return Quote(bid_price=bid_price, bid_size=self.quote_size, ask_price=ask_price, ask_size=self.quote_size)
