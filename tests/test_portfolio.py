import math

from backtest.portfolio import Portfolio
from lob.models import Side


def test_buy_fill_reduces_cash_and_increases_inventory():
    p = Portfolio()
    p.apply_fill(Side.BUY, price=100.0, size=10, time=0.0, is_maker=True)
    assert p.cash == -1000.0
    assert p.inventory == 10


def test_sell_fill_increases_cash_and_decreases_inventory():
    p = Portfolio()
    p.apply_fill(Side.SELL, price=100.0, size=10, time=0.0, is_maker=True)
    assert p.cash == 1000.0
    assert p.inventory == -10


def test_round_trip_buy_then_sell_at_same_price_is_flat():
    p = Portfolio()
    p.apply_fill(Side.BUY, price=100.0, size=10, time=0.0, is_maker=True)
    p.apply_fill(Side.SELL, price=100.0, size=10, time=1.0, is_maker=False)
    assert p.inventory == 0
    assert p.cash == 0.0


def test_equity_flat_inventory_ignores_missing_mid_price():
    p = Portfolio()
    assert p.equity(mid_price=None) == 0.0


def test_equity_marks_nonzero_inventory_to_mid():
    p = Portfolio()
    p.apply_fill(Side.BUY, price=100.0, size=10, time=0.0, is_maker=True)
    # Bought 10 @ 100 (cash -1000); mid now 101 -> equity = -1000 + 10*101 = 10
    assert p.equity(mid_price=101.0) == 10.0


def test_equity_is_nan_when_inventory_nonzero_and_mid_unavailable():
    p = Portfolio()
    p.apply_fill(Side.BUY, price=100.0, size=10, time=0.0, is_maker=True)
    assert math.isnan(p.equity(mid_price=None))


def test_trades_log_records_maker_flag():
    p = Portfolio()
    p.apply_fill(Side.BUY, price=100.0, size=10, time=5.0, is_maker=True)
    assert len(p.trades) == 1
    assert p.trades[0].is_maker is True
    assert p.trades[0].time == 5.0
