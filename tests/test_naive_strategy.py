import pytest

from strategies.base import MarketState, Quote
from strategies.naive import NaiveSymmetricStrategy


def _state(mid_price, **overrides):
    defaults = dict(
        time=0.0, best_bid=None, best_ask=None, mid_price=mid_price,
        spread=None, imbalance=None, inventory=0, cash=0.0,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def test_rejects_nonpositive_half_spread():
    with pytest.raises(ValueError):
        NaiveSymmetricStrategy(half_spread=0.0, quote_size=10)


def test_returns_no_quote_when_mid_price_unavailable():
    strategy = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    assert strategy.quote(_state(mid_price=None)) == Quote.none()


def test_quotes_symmetrically_around_mid():
    strategy = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10, tick_size=0.01)
    quote = strategy.quote(_state(mid_price=100.00))
    assert quote.bid_price == pytest.approx(99.95)
    assert quote.ask_price == pytest.approx(100.05)
    assert quote.bid_size == 10
    assert quote.ask_size == 10


def test_ignores_inventory_and_imbalance_by_design():
    # The whole point of "naive" is that these don't move the quote at all.
    strategy = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    flat = strategy.quote(_state(mid_price=100.00, inventory=0, imbalance=0.0))
    long_skewed_flow = strategy.quote(_state(mid_price=100.00, inventory=500, imbalance=0.9))
    assert flat == long_skewed_flow


def test_prices_rounded_to_tick_grid():
    strategy = NaiveSymmetricStrategy(half_spread=0.037, quote_size=10, tick_size=0.01)
    quote = strategy.quote(_state(mid_price=100.00))
    assert quote.bid_price == pytest.approx(99.96)  # 100 - 0.037 -> rounds to nearest cent
    assert quote.ask_price == pytest.approx(100.04)
