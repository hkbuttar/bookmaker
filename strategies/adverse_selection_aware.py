"""Adverse-selection-aware strategy: widens or pulls whichever side of the
quote is at risk when order flow looks informed, using an exponentially
weighted moving average (EMA) of top-of-book order imbalance as a single
coherent signal for both facets the plan asks for -- "persistent one-sided
flow" and "order book imbalance beyond a threshold" are the same underlying
signal here, just observed at two different thresholds, rather than two
disconnected mechanisms bolted together.

Why EMA and not raw instantaneous imbalance: a single noisy event with
extreme imbalance shouldn't spook the strategy -- that's not "informed
flow," that's one order arriving. An EMA only crosses a meaningful
threshold after the book stays lopsided for a while, which is what
"persistent" actually means. This is directly tested in
tests/test_adverse_selection_aware_strategy.py: one extreme-imbalance event
is not enough to trigger a reaction; many consecutive ones are.

Direction: positive order book imbalance (more resting buy volume than
sell volume near the touch) is an established predictor of upward
near-term price pressure (see e.g. Cont, Kukanov & Stoikov, "The Price
Impact of Order Book Events," 2014) -- informed buyers tend to stack the
bid before price rises. That means heavy resting buy pressure makes the
strategy's *ask* the vulnerable side (informed flow is about to lift it
right before the price moves up), and symmetrically for negative imbalance
and the bid. This directional call is a disclosed judgment call grounded
in published microstructure literature, not a derived optimum for this
specific synthetic data.

Deliberately independent of inventory: this strategy does not skew off
inventory at all, even though InventoryAwareStrategy does. Keeping
the two mechanisms in separate classes means the comparison table can
attribute P&L/fill-rate/adverse-selection-cost differences to each
mechanism individually, rather than reporting on a strategy that already
conflates the two.
"""

from __future__ import annotations

from strategies.base import MarketState, Quote, Strategy, round_to_tick


class AdverseSelectionAwareStrategy(Strategy):
    def __init__(
        self,
        half_spread: float,
        quote_size: int,
        imbalance_ema_alpha: float,
        imbalance_threshold: float,
        widen_multiplier: float,
        pull_threshold: float | None = None,
        tick_size: float = 0.01,
    ) -> None:
        if half_spread <= 0:
            raise ValueError("half_spread must be positive, or bid/ask quotes could cross")
        if not (0.0 < imbalance_ema_alpha <= 1.0):
            raise ValueError("imbalance_ema_alpha must be in (0, 1]")
        if not (0.0 <= imbalance_threshold < 1.0):
            raise ValueError("imbalance_threshold must be in [0, 1), since |imbalance| <= 1")
        if widen_multiplier <= 1.0:
            raise ValueError("widen_multiplier must be > 1, or it wouldn't widen anything")
        if pull_threshold is not None and not (imbalance_threshold < pull_threshold <= 1.0):
            raise ValueError("pull_threshold must be > imbalance_threshold and <= 1")

        self.half_spread = half_spread
        self.quote_size = quote_size
        self.imbalance_ema_alpha = imbalance_ema_alpha
        self.imbalance_threshold = imbalance_threshold
        self.widen_multiplier = widen_multiplier
        self.pull_threshold = pull_threshold
        self.tick_size = tick_size

        self._imbalance_ema: float | None = None

    def _update_ema(self, imbalance: float | None) -> None:
        if imbalance is None:
            return  # book one-sided/empty this tick -- hold the last estimate
        if self._imbalance_ema is None:
            self._imbalance_ema = 0.0  # neutral prior: persistence has to build up, not start pre-triggered
        self._imbalance_ema = (
            self.imbalance_ema_alpha * imbalance + (1 - self.imbalance_ema_alpha) * self._imbalance_ema
        )

    def quote(self, state: MarketState) -> Quote:
        if state.mid_price is None:
            return Quote.none()

        self._update_ema(state.imbalance)
        ema = self._imbalance_ema or 0.0

        bid_half_spread = self.half_spread
        ask_half_spread = self.half_spread
        pull_bid = False
        pull_ask = False

        if ema >= self.imbalance_threshold:
            # Persistent buy-side pressure -> the ask is the vulnerable side.
            if self.pull_threshold is not None and ema >= self.pull_threshold:
                pull_ask = True
            else:
                ask_half_spread *= self.widen_multiplier
        elif ema <= -self.imbalance_threshold:
            # Persistent sell-side pressure -> the bid is the vulnerable side.
            if self.pull_threshold is not None and ema <= -self.pull_threshold:
                pull_bid = True
            else:
                bid_half_spread *= self.widen_multiplier

        bid_price = None if pull_bid else round_to_tick(state.mid_price - bid_half_spread, self.tick_size)
        ask_price = None if pull_ask else round_to_tick(state.mid_price + ask_half_spread, self.tick_size)

        return Quote(
            bid_price=bid_price,
            bid_size=0 if pull_bid else self.quote_size,
            ask_price=ask_price,
            ask_size=0 if pull_ask else self.quote_size,
        )
