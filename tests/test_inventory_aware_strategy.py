import pytest

from strategies.base import MarketState, Quote
from strategies.inventory_aware import InventoryAwareStrategy
from strategies.naive import NaiveSymmetricStrategy


def _state(mid_price, inventory=0, **overrides):
    defaults = dict(
        time=0.0, best_bid=None, best_ask=None, mid_price=mid_price,
        spread=None, imbalance=None, inventory=inventory, cash=0.0,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def test_rejects_nonpositive_half_spread():
    with pytest.raises(ValueError):
        InventoryAwareStrategy(half_spread=0.0, quote_size=10, inventory_penalty=0.01)


def test_rejects_negative_inventory_penalty():
    with pytest.raises(ValueError):
        InventoryAwareStrategy(half_spread=0.05, quote_size=10, inventory_penalty=-0.01)


def test_returns_no_quote_when_mid_price_unavailable():
    strategy = InventoryAwareStrategy(half_spread=0.05, quote_size=10, inventory_penalty=0.01)
    assert strategy.quote(_state(mid_price=None)) == Quote.none()


def test_flat_inventory_quotes_symmetrically_around_mid():
    strategy = InventoryAwareStrategy(half_spread=0.05, quote_size=10, inventory_penalty=0.01)
    quote = strategy.quote(_state(mid_price=100.00, inventory=0))
    assert quote.bid_price == pytest.approx(99.95)
    assert quote.ask_price == pytest.approx(100.05)


def test_long_inventory_shifts_both_quotes_down():
    strategy = InventoryAwareStrategy(half_spread=0.05, quote_size=10, inventory_penalty=0.01)
    flat = strategy.quote(_state(mid_price=100.00, inventory=0))
    long = strategy.quote(_state(mid_price=100.00, inventory=200))  # long 200 shares

    # Reservation price = 100 - 0.01*200 = 98.00 -> bid=97.95, ask=98.05.
    assert long.bid_price == pytest.approx(97.95)
    assert long.ask_price == pytest.approx(98.05)
    assert long.bid_price < flat.bid_price
    assert long.ask_price < flat.ask_price
    # Ask moved below the flat mid entirely -- the strategy is now eager
    # enough to sell that it's willing to quote through the original mid.
    assert long.ask_price < 100.00


def test_short_inventory_shifts_both_quotes_up():
    strategy = InventoryAwareStrategy(half_spread=0.05, quote_size=10, inventory_penalty=0.01)
    flat = strategy.quote(_state(mid_price=100.00, inventory=0))
    short = strategy.quote(_state(mid_price=100.00, inventory=-200))  # short 200 shares

    # Reservation price = 100 - 0.01*(-200) = 102.00 -> bid=101.95, ask=102.05.
    assert short.bid_price == pytest.approx(101.95)
    assert short.ask_price == pytest.approx(102.05)
    assert short.bid_price > flat.bid_price
    assert short.ask_price > flat.ask_price


def test_skew_scales_linearly_with_inventory():
    strategy = InventoryAwareStrategy(half_spread=0.05, quote_size=10, inventory_penalty=0.02)
    q1 = strategy.quote(_state(mid_price=100.00, inventory=100))
    q2 = strategy.quote(_state(mid_price=100.00, inventory=200))
    shift1 = 100.00 - q1.bid_price  # includes the fixed half_spread offset too
    shift2 = 100.00 - q2.bid_price
    assert (shift2 - shift1) == pytest.approx(0.02 * 100, abs=1e-9)  # doubling inventory doubles the skew


def test_zero_penalty_is_identical_to_naive_strategy():
    naive = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    inventory_aware = InventoryAwareStrategy(half_spread=0.05, quote_size=10, inventory_penalty=0.0)

    for inventory in (0, 50, -50, 1000, -1000):
        state = _state(mid_price=100.00, inventory=inventory)
        assert inventory_aware.quote(state) == naive.quote(state)
