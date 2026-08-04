import pytest

from risk.inventory_limit import clip_to_inventory_limit
from strategies.base import Quote


def _quote(bid=99.95, ask=100.05, size=10):
    return Quote(bid_price=bid, bid_size=size, ask_price=ask, ask_size=size)


def test_rejects_nonpositive_max_inventory():
    with pytest.raises(ValueError):
        clip_to_inventory_limit(_quote(), inventory=0, max_inventory=0)


def test_unchanged_when_within_limit():
    quote = clip_to_inventory_limit(_quote(), inventory=50, max_inventory=100)
    assert quote == _quote()


def test_drops_bid_at_long_limit_keeps_ask():
    quote = clip_to_inventory_limit(_quote(), inventory=100, max_inventory=100)
    assert quote.bid_price is None
    assert quote.bid_size == 0
    assert quote.ask_price == 100.05  # can still sell to reduce the long position
    assert quote.ask_size == 10


def test_drops_ask_at_short_limit_keeps_bid():
    quote = clip_to_inventory_limit(_quote(), inventory=-100, max_inventory=100)
    assert quote.ask_price is None
    assert quote.ask_size == 0
    assert quote.bid_price == 99.95  # can still buy to reduce the short position
    assert quote.bid_size == 10


def test_beyond_limit_also_clips():
    quote = clip_to_inventory_limit(_quote(), inventory=150, max_inventory=100)
    assert quote.bid_price is None
    assert quote.ask_price == 100.05


def test_none_quote_stays_none_at_limit():
    quote = clip_to_inventory_limit(Quote.none(), inventory=100, max_inventory=100)
    assert quote == Quote.none()
