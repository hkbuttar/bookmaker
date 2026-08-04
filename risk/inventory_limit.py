"""Hard inventory cap: a pure function that clips whichever side of a
strategy's desired quote would push |inventory| further past the limit.

Deliberately a risk-reducing clip, not a full halt: at the long-side cap,
the bid (which would add more long exposure) gets dropped, but the ask
(which reduces it) is left untouched -- the strategy can still trade its
way back toward flat, it just can't add more risk in the direction it's
already over-extended in. A full halt of both sides belongs to the
kill-switch (risk.kill_switch), which is a deliberately blunter,
loss-triggered instrument, not a position-size one.
"""

from __future__ import annotations

from strategies.base import Quote


def clip_to_inventory_limit(quote: Quote, inventory: int, max_inventory: int) -> Quote:
    if max_inventory <= 0:
        raise ValueError("max_inventory must be positive")

    bid_price, bid_size = quote.bid_price, quote.bid_size
    ask_price, ask_size = quote.ask_price, quote.ask_size

    if inventory >= max_inventory:
        bid_price, bid_size = None, 0
    if inventory <= -max_inventory:
        ask_price, ask_size = None, 0

    return Quote(bid_price=bid_price, bid_size=bid_size, ask_price=ask_price, ask_size=ask_size)
