"""Backtest sanity checks (Step 10): does a strategy's fill rate and P&L
behave the way it obviously should under known conditions? These are
deliberately different in kind from the scenario/mechanics tests in
tests/test_market_maker_sim.py -- those pin down exact fills for hand-built
event sequences; these check aggregate, statistical properties (monotonicity,
zero-in-zero-out) against the real synthetic generator, which is closer to
how a real bug (like the one this step actually found -- see
test_market_maker_sim.py's and test_rl_env.py's FIFO-priority regression
tests) would first become visible: not a wrong individual fill, but an
aggregate number that's obviously off once you know what to expect.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.market_maker_sim import run_backtest
from data.synthetic_lob import SyntheticLOBConfig, generate_session
from strategies.naive import NaiveSymmetricStrategy


def _event(order_id, time, type_, side=None, price=None, size=None):
    return {"order_id": order_id, "time": time, "type": type_, "side": side, "price": price, "size": size}


def test_no_crossing_flow_yields_zero_fills_and_zero_pnl():
    # A background book that only ever rests far outside the strategy's
    # quotes, never crosses, and never moves. No fill should ever be
    # possible, so P&L, inventory, and fill count must all be exactly zero
    # -- not "small," exactly zero.
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 110.00, 100),  # far above where naive will ever quote
            _event(2, 1.0, "LIMIT", "BUY", 90.00, 100),  # far below where naive will ever quote
            _event(3, 2.0, "LIMIT", "SELL", 110.50, 100),
            _event(4, 3.0, "LIMIT", "BUY", 89.50, 100),
        ]
    )
    strategy = NaiveSymmetricStrategy(half_spread=0.5, quote_size=10)  # quotes ~99.5/100.5, nowhere near 90/110
    result = run_backtest(events, strategy)

    assert len(result.portfolio.trades) == 0
    assert result.portfolio.inventory == 0
    assert result.portfolio.cash == 0.0


def test_tighter_spread_never_yields_strictly_fewer_fills_than_wider_spread():
    # Same real order flow, same seed, only half_spread differs. A tighter
    # quote is strictly more marketable than a wider one at the same mid,
    # so it should never get *fewer* opportunities to fill.
    events = generate_session(SyntheticLOBConfig(session_seconds=600.0, seed=11))

    tight = NaiveSymmetricStrategy(half_spread=0.01, quote_size=10)
    wide = NaiveSymmetricStrategy(half_spread=0.20, quote_size=10)

    tight_result = run_backtest(events, tight, record_levels=5)
    wide_result = run_backtest(events, wide, record_levels=5)

    assert len(tight_result.portfolio.trades) >= len(wide_result.portfolio.trades)


def test_fill_count_is_nondecreasing_in_session_length():
    # A longer session replaying the same underlying order-flow process
    # (same seed) can only accumulate at least as many fills as a shorter
    # prefix of it -- fills don't get undone by more time passing.
    cfg_short = SyntheticLOBConfig(session_seconds=300.0, seed=22)
    cfg_long = SyntheticLOBConfig(session_seconds=1200.0, seed=22)

    short_events = generate_session(cfg_short)
    long_events = generate_session(cfg_long)

    strategy_short = NaiveSymmetricStrategy(half_spread=0.02, quote_size=10)
    strategy_long = NaiveSymmetricStrategy(half_spread=0.02, quote_size=10)

    short_result = run_backtest(short_events, strategy_short, record_levels=5)
    long_result = run_backtest(long_events, strategy_long, record_levels=5)

    assert len(long_result.portfolio.trades) >= len(short_result.portfolio.trades)


def test_larger_quote_size_yields_at_least_as_much_filled_volume():
    # Same order flow, same price offsets, only size differs. A strategy
    # quoting a larger size at every fill opportunity should never end up
    # with *less* total filled volume than a smaller-size version of
    # itself -- each fill it does get can only be as large or larger.
    events = generate_session(SyntheticLOBConfig(session_seconds=600.0, seed=33))

    small = NaiveSymmetricStrategy(half_spread=0.02, quote_size=5)
    large = NaiveSymmetricStrategy(half_spread=0.02, quote_size=50)

    small_result = run_backtest(events, small, record_levels=5)
    large_result = run_backtest(events, large, record_levels=5)

    small_volume = sum(t.size for t in small_result.portfolio.trades)
    large_volume = sum(t.size for t in large_result.portfolio.trades)
    assert large_volume >= small_volume


def test_maker_only_strategy_has_no_taker_fills_on_real_data():
    # Naive never crosses its own quotes by construction (bid < mid < ask
    # always, for a positive half_spread) -- on real, dense synthetic
    # order flow, every one of its fills should be as maker, never taker.
    events = generate_session(SyntheticLOBConfig(session_seconds=600.0, seed=44))
    strategy = NaiveSymmetricStrategy(half_spread=0.02, quote_size=10)
    result = run_backtest(events, strategy, record_levels=5)

    assert len(result.portfolio.trades) > 0  # sanity: this scenario should produce real fills
    assert all(t.is_maker for t in result.portfolio.trades)
