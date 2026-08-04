"""Wraps any Strategy with the risk layer: a hard inventory limit and a
kill-switch, composed as a drop-in Strategy itself so nothing in
backtest.market_maker_sim needs to know a risk layer exists at all --
`run_backtest(events, RiskManagedStrategy(inner_strategy, ...))` is the
entire integration point.

Scope note, disclosed rather than silently assumed: this halts *quoting*
once the kill-switch trips (both sides go to Quote.none(), same as the
plan's wording), but it does not forcibly flatten whatever inventory is
already resting via an aggressive market order the way alpha-signal-lab's
backtest engine does (portfolio.flatten_orders()). Doing that here would
mean extending the Strategy/Quote interface to let a strategy emit a
taker order, not just a resting quote -- a real architectural change,
not a Step 7-sized one. So: past the kill-switch, the strategy stops
adding to its book presence and existing inventory just sits there,
marked to whatever the market does next, until a human calls reset().
"""

from __future__ import annotations

from backtest.portfolio import equity_from
from risk.inventory_limit import clip_to_inventory_limit
from risk.kill_switch import KillSwitch
from strategies.base import MarketState, Quote, Strategy


class RiskManagedStrategy(Strategy):
    def __init__(
        self,
        inner: Strategy,
        max_inventory: int | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.inner = inner
        self.max_inventory = max_inventory
        self.kill_switch = kill_switch

    def quote(self, state: MarketState) -> Quote:
        if self.kill_switch is not None:
            equity = equity_from(state.cash, state.inventory, state.mid_price)
            if self.kill_switch.check(equity):
                return Quote.none()

        quote = self.inner.quote(state)

        if self.max_inventory is not None:
            quote = clip_to_inventory_limit(quote, state.inventory, self.max_inventory)

        return quote
