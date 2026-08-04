import pandas as pd
import pytest

from backtest.market_maker_sim import BacktestResult, run_backtest
from backtest.metrics import adverse_selection_cost, summarize
from backtest.portfolio import Portfolio, Trade
from lob.models import Side
from strategies.naive import NaiveSymmetricStrategy


def _event(order_id, time, type_, side=None, price=None, size=None):
    return {"order_id": order_id, "time": time, "type": type_, "side": side, "price": price, "size": size}


def test_summarize_matches_hand_computed_values():
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),  # mid=98.00 -> strategy quotes bid=97.5/ask=98.5
            _event(3, 2.0, "MARKET", "BUY", size=5),  # fills strategy's ask as maker
        ]
    )
    strategy = NaiveSymmetricStrategy(half_spread=0.5, quote_size=20)
    result = run_backtest(events, strategy)
    stats = summarize(result)

    assert stats["n_fills"] == 1
    assert stats["maker_fills"] == 1
    assert stats["taker_fills"] == 0
    assert stats["final_inventory"] == -5
    assert stats["final_pnl"] == pytest.approx(result.portfolio_history["equity"].iloc[-1])


def _book_snapshots(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    # rows of (time, bid_price_1, ask_price_1)
    return pd.DataFrame(rows, columns=["time", "bid_price_1", "ask_price_1"])


def _result_with_trades(trades: list[Trade], book_rows: list[tuple[float, float, float]]) -> BacktestResult:
    portfolio = Portfolio(trades=trades)
    return BacktestResult(
        portfolio=portfolio,
        portfolio_history=pd.DataFrame({"time": [], "inventory": [], "cash": [], "equity": []}),
        book_snapshots=_book_snapshots(book_rows),
    )


def test_adverse_selection_cost_none_with_no_maker_trades():
    result = _result_with_trades([], _book_snapshots([(0.0, 99.9, 100.1)]))
    assert adverse_selection_cost(result) is None


def test_adverse_selection_cost_excludes_taker_trades():
    trades = [Trade(time=0.0, side=Side.BUY, price=100.0, size=10, is_maker=False)]
    result = _result_with_trades(trades, _book_snapshots([(0.0, 99.9, 100.1), (5.0, 98.9, 99.1)]))
    assert adverse_selection_cost(result) is None


def test_adverse_selection_cost_positive_when_price_drops_after_buy():
    # Bought (as maker) at 100.00; a few snapshots later mid has dropped to 99.00.
    trades = [Trade(time=0.0, side=Side.BUY, price=100.0, size=10, is_maker=True)]
    book_rows = [
        (0.0, 99.95, 100.05),  # mid ~100.00, this is the fill's own snapshot
        (1.0, 99.95, 100.05),
        (2.0, 98.95, 99.05),  # mid now ~99.00
        (3.0, 98.95, 99.05),
    ]
    result = _result_with_trades(trades, _book_snapshots(book_rows))
    cost = adverse_selection_cost(result, horizon_events=2)
    assert cost == pytest.approx(100.0 - 99.00, abs=1e-9)  # bought at 100, price fell to 99 -> cost = 1.0


def test_adverse_selection_cost_negative_when_price_rises_after_buy():
    trades = [Trade(time=0.0, side=Side.BUY, price=100.0, size=10, is_maker=True)]
    book_rows = [
        (0.0, 99.95, 100.05),
        (1.0, 99.95, 100.05),
        (2.0, 100.95, 101.05),  # mid rose to ~101.00 -- favorable for the buyer
        (3.0, 100.95, 101.05),
    ]
    result = _result_with_trades(trades, _book_snapshots(book_rows))
    cost = adverse_selection_cost(result, horizon_events=2)
    assert cost == pytest.approx(100.0 - 101.00, abs=1e-9)  # negative == favorable


def test_adverse_selection_cost_positive_when_price_rises_after_sell():
    trades = [Trade(time=0.0, side=Side.SELL, price=100.0, size=10, is_maker=True)]
    book_rows = [
        (0.0, 99.95, 100.05),
        (1.0, 99.95, 100.05),
        (2.0, 100.95, 101.05),  # mid rose to ~101.00 -- adverse for the seller
        (3.0, 100.95, 101.05),
    ]
    result = _result_with_trades(trades, _book_snapshots(book_rows))
    cost = adverse_selection_cost(result, horizon_events=2)
    assert cost == pytest.approx(101.00 - 100.0, abs=1e-9)


def test_adverse_selection_cost_averages_across_multiple_maker_fills():
    trades = [
        Trade(time=0.0, side=Side.BUY, price=100.0, size=10, is_maker=True),  # markout cost +1.0
        Trade(time=1.0, side=Side.BUY, price=100.0, size=10, is_maker=True),  # markout cost -1.0
    ]
    book_rows = [
        (0.0, 99.95, 100.05),
        (1.0, 99.95, 100.05),
        (2.0, 98.95, 99.05),  # for the t=0 trade's horizon: mid -> 99.00 (cost +1.0)
        (3.0, 100.95, 101.05),  # for the t=1 trade's horizon: mid -> 101.00 (cost -1.0)
    ]
    result = _result_with_trades(trades, _book_snapshots(book_rows))
    cost = adverse_selection_cost(result, horizon_events=2)
    assert cost == pytest.approx(0.0, abs=1e-9)


def test_adverse_selection_cost_clips_horizon_to_end_of_book():
    trades = [Trade(time=2.0, side=Side.BUY, price=100.0, size=10, is_maker=True)]
    book_rows = [
        (0.0, 99.95, 100.05),
        (1.0, 99.95, 100.05),
        (2.0, 99.95, 100.05),
        (3.0, 98.95, 99.05),  # last row -- horizon far beyond this should just clip here
    ]
    result = _result_with_trades(trades, _book_snapshots(book_rows))
    cost = adverse_selection_cost(result, horizon_events=1000)
    assert cost == pytest.approx(100.0 - 99.00, abs=1e-9)
