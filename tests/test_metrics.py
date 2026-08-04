import pandas as pd
import pytest

from backtest.market_maker_sim import run_backtest
from backtest.metrics import summarize
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
