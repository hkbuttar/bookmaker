import pytest

from lob.models import Side
from lob.order_book import OrderBook


def test_rests_when_no_crossing_liquidity():
    book = OrderBook()
    fills = book.submit_limit_order(1, Side.BUY, 100.00, 10, decision_time=0, arrival_time=0)
    assert fills == []
    assert book.best_bid() == 100.00
    assert book.best_ask() is None


def test_marketable_limit_matches_best_price_first():
    book = OrderBook()
    # Two resting asks at different prices; the worse one should be
    # untouched while the better (lower) one absorbs the incoming buy.
    book.submit_limit_order(1, Side.SELL, 100.05, 5, 0, 0)
    book.submit_limit_order(2, Side.SELL, 100.10, 5, 0, 0)

    fills = book.submit_limit_order(3, Side.BUY, 100.10, 5, 1, 1)

    assert len(fills) == 1
    assert fills[0].price == 100.05  # traded at the resting (maker) price, not the taker's limit
    assert fills[0].maker_order_id == 1
    assert book.best_ask() == 100.10  # level 2 untouched
    assert book.best_bid() is None  # fully filled, nothing rests


def test_fifo_time_priority_within_level():
    book = OrderBook()
    book.submit_limit_order(1, Side.SELL, 100.00, 5, 0, 0)  # arrives first
    book.submit_limit_order(2, Side.SELL, 100.00, 5, 1, 1)  # arrives second, same price

    fills = book.submit_limit_order(3, Side.BUY, 100.00, 5, 2, 2)

    assert len(fills) == 1
    assert fills[0].maker_order_id == 1  # oldest at the level, not order 2
    assert fills[0].size == 5


def test_partial_fill_reduces_resting_order_size_and_keeps_queue_position():
    book = OrderBook()
    book.submit_limit_order(1, Side.SELL, 100.00, 10, 0, 0)

    fills = book.submit_limit_order(2, Side.BUY, 100.00, 4, 1, 1)

    assert fills[0].size == 4
    depth = book.depth()
    assert depth["asks"] == [(100.00, 6)]  # 10 - 4 remaining, still resting


def test_walks_multiple_price_levels():
    book = OrderBook()
    book.submit_limit_order(1, Side.SELL, 100.00, 5, 0, 0)
    book.submit_limit_order(2, Side.SELL, 100.05, 5, 0, 0)

    fills = book.submit_limit_order(3, Side.BUY, 100.05, 8, 1, 1)

    assert len(fills) == 2
    assert fills[0].price == 100.00 and fills[0].size == 5
    assert fills[1].price == 100.05 and fills[1].size == 3
    assert book.depth()["asks"] == [(100.05, 2)]


def test_market_order_partial_liquidity_drops_remainder():
    book = OrderBook()
    book.submit_limit_order(1, Side.SELL, 100.00, 5, 0, 0)

    fills = book.submit_market_order(2, Side.BUY, 20, 1, 1)

    assert len(fills) == 1
    assert fills[0].size == 5  # only 5 available; the other 15 just vanish
    assert book.best_ask() is None
    assert book.best_bid() is None  # market orders never rest


def test_market_order_no_liquidity_returns_no_fills():
    book = OrderBook()
    fills = book.submit_market_order(1, Side.BUY, 10, 0, 0)
    assert fills == []


def test_cancel_removes_order_and_updates_best_price():
    book = OrderBook()
    book.submit_limit_order(1, Side.BUY, 100.00, 5, 0, 0)
    book.submit_limit_order(2, Side.BUY, 99.99, 5, 0, 0)

    assert book.cancel_order(1, arrival_time=1) is True
    assert book.best_bid() == 99.99


def test_cancel_unknown_order_is_noop():
    book = OrderBook()
    assert book.cancel_order(999, arrival_time=0) is False


def test_price_level_removed_when_last_order_cancelled():
    book = OrderBook()
    book.submit_limit_order(1, Side.BUY, 100.00, 5, 0, 0)
    book.cancel_order(1, arrival_time=1)
    assert 100.00 not in book.bids
    assert book.best_bid() is None


def test_depth_aggregates_size_per_level_best_first():
    book = OrderBook()
    book.submit_limit_order(1, Side.BUY, 99.99, 5, 0, 0)
    book.submit_limit_order(2, Side.BUY, 100.00, 3, 0, 0)
    book.submit_limit_order(3, Side.BUY, 100.00, 2, 0, 0)  # same level as order 2
    book.submit_limit_order(4, Side.SELL, 100.05, 4, 0, 0)

    depth = book.depth()
    assert depth["bids"] == [(100.00, 5), (99.99, 5)]  # best (highest) bid first, sizes summed
    assert depth["asks"] == [(100.05, 4)]
