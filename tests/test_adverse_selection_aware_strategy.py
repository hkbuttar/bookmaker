import pytest

from strategies.adverse_selection_aware import AdverseSelectionAwareStrategy
from strategies.base import MarketState, Quote
from strategies.naive import NaiveSymmetricStrategy


def _state(mid_price=100.00, imbalance=0.0, **overrides):
    defaults = dict(
        time=0.0, best_bid=None, best_ask=None, mid_price=mid_price,
        spread=None, imbalance=imbalance, inventory=0, cash=0.0,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def _strategy(**overrides):
    defaults = dict(
        half_spread=0.05, quote_size=10, imbalance_ema_alpha=0.1,
        imbalance_threshold=0.5, widen_multiplier=3.0,
    )
    defaults.update(overrides)
    return AdverseSelectionAwareStrategy(**defaults)


def test_rejects_nonpositive_half_spread():
    with pytest.raises(ValueError):
        _strategy(half_spread=0.0)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1])
def test_rejects_invalid_ema_alpha(alpha):
    with pytest.raises(ValueError):
        _strategy(imbalance_ema_alpha=alpha)


@pytest.mark.parametrize("threshold", [-0.1, 1.0, 1.5])
def test_rejects_invalid_imbalance_threshold(threshold):
    with pytest.raises(ValueError):
        _strategy(imbalance_threshold=threshold)


def test_rejects_widen_multiplier_not_greater_than_one():
    with pytest.raises(ValueError):
        _strategy(widen_multiplier=1.0)


def test_rejects_pull_threshold_not_above_imbalance_threshold():
    with pytest.raises(ValueError):
        _strategy(imbalance_threshold=0.5, pull_threshold=0.5)
    with pytest.raises(ValueError):
        _strategy(imbalance_threshold=0.5, pull_threshold=1.5)


def test_returns_no_quote_when_mid_price_unavailable():
    strategy = _strategy()
    assert strategy.quote(_state(mid_price=None)) == Quote.none()


def test_zero_imbalance_matches_naive_strategy():
    naive = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    aware = _strategy()
    for _ in range(20):
        assert aware.quote(_state(imbalance=0.0)) == naive.quote(_state(imbalance=0.0))


def test_single_extreme_imbalance_event_does_not_trigger_widening():
    # alpha=0.1: one event at imbalance=1.0 moves the EMA to only 0.1,
    # nowhere near the 0.5 threshold -- proving a single blip isn't treated
    # as "persistent" informed flow.
    strategy = _strategy()
    quote = strategy.quote(_state(imbalance=1.0))
    assert quote.ask_price == pytest.approx(100.05)  # unwidened


def test_sustained_imbalance_eventually_triggers_widening():
    strategy = _strategy()
    quote = None
    for _ in range(50):  # EMA converges toward 1.0, crossing the 0.5 threshold
        quote = strategy.quote(_state(imbalance=1.0))
    assert quote.ask_price > 100.05  # widened beyond the base half_spread
    assert quote.bid_price == pytest.approx(99.95)  # bid untouched -- only the vulnerable side widens


def test_sustained_positive_imbalance_widens_ask_not_bid():
    strategy = _strategy()
    for _ in range(50):
        quote = strategy.quote(_state(imbalance=0.9))
    assert quote.ask_price == pytest.approx(100.00 + 0.05 * 3.0)
    assert quote.bid_price == pytest.approx(99.95)


def test_sustained_negative_imbalance_widens_bid_not_ask():
    strategy = _strategy()
    for _ in range(50):
        quote = strategy.quote(_state(imbalance=-0.9))
    assert quote.bid_price == pytest.approx(100.00 - 0.05 * 3.0)
    assert quote.ask_price == pytest.approx(100.05)


def test_extreme_sustained_imbalance_pulls_ask_entirely():
    strategy = _strategy(pull_threshold=0.8)
    for _ in range(50):
        quote = strategy.quote(_state(imbalance=0.95))
    assert quote.ask_price is None
    assert quote.ask_size == 0
    assert quote.bid_price == pytest.approx(99.95)  # bid still quoted normally


def test_extreme_sustained_imbalance_pulls_bid_entirely():
    strategy = _strategy(pull_threshold=0.8)
    for _ in range(50):
        quote = strategy.quote(_state(imbalance=-0.95))
    assert quote.bid_price is None
    assert quote.bid_size == 0
    assert quote.ask_price == pytest.approx(100.05)


def test_ema_holds_last_value_when_imbalance_unavailable():
    strategy = _strategy()
    for _ in range(50):
        strategy.quote(_state(imbalance=0.9))
    ema_before = strategy._imbalance_ema
    strategy.quote(_state(imbalance=None))
    assert strategy._imbalance_ema == ema_before
