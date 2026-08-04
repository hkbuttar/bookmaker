import pandas as pd
import pytest

from backtest.market_maker_sim import run_backtest
from data.synthetic_lob import SyntheticLOBConfig, generate_session
from strategies.naive import NaiveSymmetricStrategy


def _event(order_id, time, type_, side=None, price=None, size=None):
    return {"order_id": order_id, "time": time, "type": type_, "side": side, "price": price, "size": size}


def test_naive_strategy_gets_filled_by_marketable_background_order():
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),  # background ask
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),  # background bid -> mid=98.00
            # After event 2, strategy (half_spread=0.5) quotes bid=97.50, ask=98.50.
            _event(3, 2.0, "MARKET", "BUY", size=5),  # should hit strategy's ask at 98.50 first
        ]
    )
    strategy = NaiveSymmetricStrategy(half_spread=0.5, quote_size=20)
    result = run_backtest(events, strategy)

    assert result.portfolio.inventory == -5  # sold 5 as maker
    assert result.portfolio.cash == pytest.approx(5 * 98.50)
    assert len(result.portfolio.trades) == 1
    assert result.portfolio.trades[0].is_maker is True


def test_no_requote_when_state_unchanged_preserves_fifo_priority():
    events = pd.DataFrame(
        [
            _event(201, 0.0, "LIMIT", "SELL", 99.00, 10),  # background ask
            _event(202, 1.0, "LIMIT", "BUY", 97.00, 10),  # background bid -> mid=98.00, strategy quotes ask=98.50
            # A second background order joins at the SAME price as the strategy's
            # ask. It shouldn't move best_bid/best_ask/mid (strategy's own quote
            # is still best/tied), so the strategy must NOT cancel+resubmit here
            # -- if it did, this new order would win time priority instead.
            _event(203, 2.0, "LIMIT", "SELL", 98.50, 5),
            # Now a market buy for exactly 5 shares: if the strategy kept its
            # original resting order (submitted before event 203), it wins FIFO
            # and gets filled. If it had needlessly requoted, order 203 (which
            # arrived first among what's left) would have been filled instead,
            # and the strategy would show zero trades.
            _event(204, 3.0, "MARKET", "BUY", size=5),
        ]
    )
    strategy = NaiveSymmetricStrategy(half_spread=0.5, quote_size=20)
    result = run_backtest(events, strategy)

    assert result.portfolio.inventory == -5
    assert len(result.portfolio.trades) == 1


def test_requote_on_mid_price_change_cancels_stale_quote():
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),  # mid=98.00 -> strategy quotes bid=97.50/ask=98.50
            _event(3, 2.0, "LIMIT", "BUY", 97.90, 5),  # improves best bid -> mid moves to 98.20, triggers requote
            # book_snapshots is taken *before* the requote each event triggers
            # (see market_maker_sim's docstring on the one-event lag), so a
            # trailing event is needed to observe the requote's effect.
            _event(999, 3.0, "CANCEL"),
        ]
    )
    strategy = NaiveSymmetricStrategy(half_spread=0.5, quote_size=20)
    result = run_backtest(events, strategy, record_levels=3)

    last_row = result.book_snapshots.iloc[-1]
    # mid at the moment of requote = (97.90 + 98.50) / 2 = 98.20 (best ask hadn't
    # moved yet), so the new ask should be 98.20 + 0.5 = 98.70, not the stale 98.50.
    assert last_row["ask_price_1"] == pytest.approx(98.70)


def test_portfolio_history_and_book_snapshots_row_aligned_with_events():
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),
        ]
    )
    strategy = NaiveSymmetricStrategy(half_spread=0.5, quote_size=20)
    result = run_backtest(events, strategy)

    assert len(result.portfolio_history) == len(events)
    assert len(result.book_snapshots) == len(events)


def test_full_synthetic_session_smoke():
    events = generate_session(SyntheticLOBConfig(session_seconds=1800.0, seed=21))
    strategy = NaiveSymmetricStrategy(half_spread=0.05, quote_size=10)
    result = run_backtest(events, strategy)

    assert len(result.portfolio_history) == len(events)
    assert len(result.portfolio.trades) > 0  # some background flow should hit the strategy's quotes
    final_equity = result.portfolio_history["equity"].iloc[-1]
    assert final_equity == final_equity or result.portfolio.inventory != 0  # not NaN unless still holding inventory
