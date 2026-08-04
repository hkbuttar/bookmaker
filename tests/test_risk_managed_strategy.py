import pytest

from risk.kill_switch import KillSwitch
from risk.risk_managed_strategy import RiskManagedStrategy
from strategies.base import MarketState, Quote
from strategies.naive import NaiveSymmetricStrategy


def _state(mid_price=100.00, inventory=0, cash=0.0, **overrides):
    defaults = dict(
        time=0.0, best_bid=None, best_ask=None, mid_price=mid_price,
        spread=None, imbalance=None, inventory=inventory, cash=cash,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def test_delegates_to_inner_strategy_with_no_risk_controls():
    inner = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    wrapped = RiskManagedStrategy(inner)
    state = _state()
    assert wrapped.quote(state) == inner.quote(state)


def test_applies_inventory_limit_on_top_of_inner_quote():
    inner = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    wrapped = RiskManagedStrategy(inner, max_inventory=100)
    quote = wrapped.quote(_state(inventory=100))
    assert quote.bid_price is None  # clipped: at the long limit
    assert quote.ask_price == pytest.approx(100.05)  # inner strategy's ask untouched


def test_kill_switch_halts_quoting_once_triggered():
    inner = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    kill_switch = KillSwitch(max_drawdown=10.0)
    wrapped = RiskManagedStrategy(inner, kill_switch=kill_switch)

    assert wrapped.quote(_state(cash=0.0)) != Quote.none()  # equity=0, no drawdown yet
    quote = wrapped.quote(_state(cash=-15.0))  # $15 drawdown from the $0 peak
    assert quote == Quote.none()


def test_kill_switch_stays_halted_even_if_equity_recovers():
    inner = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    kill_switch = KillSwitch(max_drawdown=10.0)
    wrapped = RiskManagedStrategy(inner, kill_switch=kill_switch)

    wrapped.quote(_state(cash=0.0))
    wrapped.quote(_state(cash=-15.0))  # triggers
    quote = wrapped.quote(_state(cash=1000.0))  # equity recovered, still halted
    assert quote == Quote.none()


def test_manual_reset_resumes_quoting():
    inner = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    kill_switch = KillSwitch(max_drawdown=10.0)
    wrapped = RiskManagedStrategy(inner, kill_switch=kill_switch)

    wrapped.quote(_state(cash=0.0))
    wrapped.quote(_state(cash=-15.0))  # triggers
    kill_switch.reset()
    quote = wrapped.quote(_state(cash=0.0))
    assert quote != Quote.none()


def test_both_controls_compose_kill_switch_overrides_inventory_clip():
    inner = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    kill_switch = KillSwitch(max_drawdown=10.0)
    wrapped = RiskManagedStrategy(inner, max_inventory=100, kill_switch=kill_switch)

    wrapped.quote(_state(cash=0.0, inventory=100))
    quote = wrapped.quote(_state(cash=-15.0, inventory=100))  # both triggers apply
    assert quote == Quote.none()  # kill-switch wins outright, not just an inventory clip
