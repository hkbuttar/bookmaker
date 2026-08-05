"""Strategy interface shared by naive/inventory-aware/adverse-selection-aware
and, later, the RL policy via a thin adapter.

The split is deliberate: a Strategy only ever *decides* -- given the
current market state, what should my resting bid/ask be right now? -- and
never touches the book, order ids, or fill accounting directly. That's all
owned by `backtest.market_maker_sim`, which is what makes strategies
comparable: every strategy runs through identical order-submission,
cancellation, and P&L-accounting mechanics, so differences in the
comparison table are differences in quoting *logic*, not in simulator
plumbing.
"""

from __future__ import annotations

import abc
import dataclasses


@dataclasses.dataclass(frozen=True)
class MarketState:
    """What a strategy is allowed to see when deciding a quote. Deliberately
    narrow -- no order ids, no access to the book internals -- so a
    strategy genuinely can't cheat by peeking at anything beyond what a
    real market maker would observe (current book state + its own
    inventory/cash) at that point in time.
    """

    time: float
    best_bid: float | None
    best_ask: float | None
    mid_price: float | None
    spread: float | None
    imbalance: float | None
    inventory: int
    cash: float


@dataclasses.dataclass(frozen=True)
class Quote:
    """A strategy's desired resting quote. Either side can be omitted
    (price=None) to mean "don't quote that side right now" -- e.g. an
    adverse-selection-aware strategy pulling one side under
    one-sided order flow, or an inventory-aware strategy skewing
    so far it stops quoting into its own risk.
    """

    bid_price: float | None
    bid_size: int
    ask_price: float | None
    ask_size: int

    @staticmethod
    def none() -> "Quote":
        return Quote(bid_price=None, bid_size=0, ask_price=None, ask_size=0)


class Strategy(abc.ABC):
    @abc.abstractmethod
    def quote(self, state: MarketState) -> Quote:
        """Return the desired resting quote given the current market state.
        Called by the simulator after every processed background event;
        returning the same Quote as last time is cheap (the simulator only
        cancels/resubmits on an actual change), so a strategy can be called
        this often without needing to self-throttle.
        """
        raise NotImplementedError


def round_to_tick(price: float, tick_size: float) -> float:
    return round(price / tick_size) * tick_size
