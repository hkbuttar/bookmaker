"""Inventory-aware strategy: skews both quotes off a reservation price that
shifts against the strategy's own inventory, so a long position makes both
the bid and ask cheaper (more eager to sell, less eager to buy) and a short
position does the opposite.

This is an Avellaneda-Stoikov-*style* penalty, not the full closed-form
model: the original derives a reservation price
`r = mid - inventory * gamma * sigma^2 * (T - t)` and an optimal spread that
both depend on time-to-horizon and estimated volatility. Pulling in a
volatility estimate and a session countdown would couple this strategy to
machinery (a rolling vol estimator, a notion of "session end") that doesn't
exist elsewhere in the codebase yet and isn't needed to demonstrate the
mechanism the plan asks for here: skewing quotes by inventory. So this
implementation collapses `gamma * sigma^2 * (T - t)` into a single constant,
`inventory_penalty` -- dollars of reservation-price shift per unit of
inventory. That constant is a disclosed modeling choice, not a derived
optimum: a real desk would calibrate it against realized volatility, risk
tolerance, and holding costs; here it's fixed and documented so its effect
on P&L and inventory variance can be read directly out of the strategy
comparison table.

At `inventory_penalty=0` this strategy is mathematically identical to
`NaiveSymmetricStrategy` (see tests/test_inventory_aware_strategy.py) --
it strictly generalizes the naive strategy rather than replacing it with
something unrelated.
"""

from __future__ import annotations

from strategies.base import MarketState, Quote, Strategy, round_to_tick


class InventoryAwareStrategy(Strategy):
    def __init__(
        self,
        half_spread: float,
        quote_size: int,
        inventory_penalty: float,
        tick_size: float = 0.01,
    ) -> None:
        if half_spread <= 0:
            raise ValueError("half_spread must be positive, or bid/ask quotes could cross")
        if inventory_penalty < 0:
            raise ValueError(
                "inventory_penalty must be >= 0; negative would skew quotes to *amplify* "
                "inventory risk instead of countering it"
            )
        self.half_spread = half_spread
        self.quote_size = quote_size
        self.inventory_penalty = inventory_penalty
        self.tick_size = tick_size

    def quote(self, state: MarketState) -> Quote:
        if state.mid_price is None:
            return Quote.none()

        reservation_price = state.mid_price - self.inventory_penalty * state.inventory
        bid_price = round_to_tick(reservation_price - self.half_spread, self.tick_size)
        ask_price = round_to_tick(reservation_price + self.half_spread, self.tick_size)

        return Quote(bid_price=bid_price, bid_size=self.quote_size, ask_price=ask_price, ask_size=self.quote_size)
