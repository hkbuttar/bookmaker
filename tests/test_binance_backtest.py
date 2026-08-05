import pandas as pd
import pytest

from backtest.binance_backtest import run_binance_backtest
from backtest.metrics import summarize
from strategies.base import MarketState, Quote, Strategy


class _FixedQuoteStrategy(Strategy):
    def __init__(self, quote: Quote) -> None:
        self._quote = quote

    def quote(self, state: MarketState) -> Quote:
        return self._quote if state.mid_price is not None else Quote.none()


def _book_row(time, bid_price, bid_size, ask_price, ask_size):
    return {
        "time": time,
        "bid_price_1": bid_price,
        "bid_size_1": bid_size,
        "ask_price_1": ask_price,
        "ask_size_1": ask_size,
    }


def _trade(time, price, size, is_buyer_maker=False):
    return {"time": time, "price": price, "size": size, "is_buyer_maker": is_buyer_maker}


def test_trade_crossing_resting_bid_fills_as_buy():
    book = pd.DataFrame([_book_row(0.0, 99.90, 1.0, 100.10, 1.0), _book_row(2.0, 99.90, 1.0, 100.10, 1.0)])
    trades = pd.DataFrame([_trade(1.0, 99.95, 0.5)])  # trades at 99.95, crosses our bid of 100.00
    strategy = _FixedQuoteStrategy(Quote(bid_price=100.00, bid_size=1.0, ask_price=100.20, ask_size=1.0))

    result = run_binance_backtest(book, trades, strategy)

    assert len(result.portfolio.trades) == 1
    assert result.portfolio.trades[0].price == 100.00
    assert result.portfolio.trades[0].size == 0.5
    assert result.portfolio.inventory == 0.5


def test_trade_crossing_resting_ask_fills_as_sell():
    book = pd.DataFrame([_book_row(0.0, 99.90, 1.0, 100.10, 1.0), _book_row(2.0, 99.90, 1.0, 100.10, 1.0)])
    trades = pd.DataFrame([_trade(1.0, 100.05, 0.3)])  # crosses our ask of 100.00
    strategy = _FixedQuoteStrategy(Quote(bid_price=99.80, bid_size=1.0, ask_price=100.00, ask_size=1.0))

    result = run_binance_backtest(book, trades, strategy)

    assert len(result.portfolio.trades) == 1
    assert result.portfolio.trades[0].price == 100.00
    assert result.portfolio.inventory == -0.3


def test_trade_not_crossing_either_side_produces_no_fill():
    book = pd.DataFrame([_book_row(0.0, 99.90, 1.0, 100.10, 1.0), _book_row(2.0, 99.90, 1.0, 100.10, 1.0)])
    trades = pd.DataFrame([_trade(1.0, 100.00, 0.5)])  # inside our spread, doesn't cross bid=99.80/ask=100.20
    strategy = _FixedQuoteStrategy(Quote(bid_price=99.80, bid_size=1.0, ask_price=100.20, ask_size=1.0))

    result = run_binance_backtest(book, trades, strategy)
    assert len(result.portfolio.trades) == 0


def test_partial_fill_leaves_remainder_available_for_later_trades():
    book = pd.DataFrame([_book_row(0.0, 99.90, 1.0, 100.10, 1.0), _book_row(3.0, 99.90, 1.0, 100.10, 1.0)])
    trades = pd.DataFrame([_trade(1.0, 99.95, 0.4), _trade(2.0, 99.95, 0.4)])
    strategy = _FixedQuoteStrategy(Quote(bid_price=100.00, bid_size=1.0, ask_price=100.20, ask_size=1.0))

    result = run_binance_backtest(book, trades, strategy)

    assert len(result.portfolio.trades) == 2
    assert result.portfolio.inventory == pytest.approx(0.8)


def test_fully_filled_quote_does_not_fill_again_until_requoted():
    book = pd.DataFrame([_book_row(0.0, 99.90, 1.0, 100.10, 1.0), _book_row(3.0, 99.90, 1.0, 100.10, 1.0)])
    trades = pd.DataFrame([_trade(1.0, 99.95, 1.0), _trade(2.0, 99.95, 1.0)])  # first fully consumes, second has nothing to hit
    strategy = _FixedQuoteStrategy(Quote(bid_price=100.00, bid_size=1.0, ask_price=100.20, ask_size=1.0))

    result = run_binance_backtest(book, trades, strategy)

    assert len(result.portfolio.trades) == 1
    assert result.portfolio.inventory == pytest.approx(1.0)


def test_summarize_works_on_binance_backtest_result():
    book = pd.DataFrame([_book_row(0.0, 99.90, 1.0, 100.10, 1.0), _book_row(2.0, 99.90, 1.0, 100.10, 1.0)])
    trades = pd.DataFrame([_trade(1.0, 99.95, 0.5)])
    strategy = _FixedQuoteStrategy(Quote(bid_price=100.00, bid_size=1.0, ask_price=100.20, ask_size=1.0))

    result = run_binance_backtest(book, trades, strategy)
    stats = summarize(result)

    assert stats["n_fills"] == 1
    assert stats["final_inventory"] == pytest.approx(0.5)


def test_no_quote_when_mid_unavailable():
    book = pd.DataFrame(
        [_book_row(0.0, float("nan"), 0.0, 100.10, 1.0), _book_row(2.0, float("nan"), 0.0, 100.10, 1.0)]
    )
    trades = pd.DataFrame([_trade(1.0, 100.05, 0.5)])
    strategy = _FixedQuoteStrategy(Quote(bid_price=100.00, bid_size=1.0, ask_price=100.20, ask_size=1.0))

    result = run_binance_backtest(book, trades, strategy)
    assert len(result.portfolio.trades) == 0


def test_delayed_quote_does_not_fill_before_arrival():
    # Decision at t=0, latency +15 -> arrives at t=15. A trade at t=5 that
    # would otherwise cross our bid must NOT fill, since nothing is
    # resting yet.
    book = pd.DataFrame(
        [_book_row(0.0, 99.90, 1.0, 100.10, 1.0), _book_row(10.0, 99.90, 1.0, 100.10, 1.0)]
    )
    trades = pd.DataFrame([_trade(5.0, 99.95, 0.5)])
    strategy = _FixedQuoteStrategy(Quote(bid_price=100.00, bid_size=1.0, ask_price=100.20, ask_size=1.0))

    result = run_binance_backtest(book, trades, strategy, strategy_latency_model=lambda t: t + 15.0)
    assert len(result.portfolio.trades) == 0


def test_delayed_quote_fills_once_arrived():
    book = pd.DataFrame(
        [
            _book_row(0.0, 99.90, 1.0, 100.10, 1.0),
            _book_row(10.0, 99.90, 1.0, 100.10, 1.0),
            _book_row(20.0, 99.90, 1.0, 100.10, 1.0),
        ]
    )
    trades = pd.DataFrame([_trade(5.0, 99.95, 0.5), _trade(22.0, 99.95, 0.5)])
    strategy = _FixedQuoteStrategy(Quote(bid_price=100.00, bid_size=1.0, ask_price=100.20, ask_size=1.0))

    result = run_binance_backtest(book, trades, strategy, strategy_latency_model=lambda t: t + 15.0)

    assert len(result.portfolio.trades) == 1
    assert result.portfolio.trades[0].time == 22.0
    assert result.portfolio.inventory == pytest.approx(0.5)


def test_zero_latency_matches_default_none():
    book = pd.DataFrame([_book_row(0.0, 99.90, 1.0, 100.10, 1.0), _book_row(2.0, 99.90, 1.0, 100.10, 1.0)])
    trades = pd.DataFrame([_trade(1.0, 99.95, 0.5)])
    strategy_a = _FixedQuoteStrategy(Quote(bid_price=100.00, bid_size=1.0, ask_price=100.20, ask_size=1.0))
    strategy_b = _FixedQuoteStrategy(Quote(bid_price=100.00, bid_size=1.0, ask_price=100.20, ask_size=1.0))

    result_default = run_binance_backtest(book, trades, strategy_a)
    result_explicit = run_binance_backtest(book, trades, strategy_b, strategy_latency_model=lambda t: t)

    assert len(result_default.portfolio.trades) == len(result_explicit.portfolio.trades) == 1
