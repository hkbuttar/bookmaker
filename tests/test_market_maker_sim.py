import pandas as pd
import pytest

from backtest.market_maker_sim import run_backtest
from data.synthetic_lob import SyntheticLOBConfig, generate_session
from lob.models import Side
from strategies.base import MarketState, Quote, Strategy
from strategies.naive import NaiveSymmetricStrategy


def _event(order_id, time, type_, side=None, price=None, size=None):
    return {"order_id": order_id, "time": time, "type": type_, "side": side, "price": price, "size": size}


class _FixedQuoteStrategy(Strategy):
    """Always returns the same Quote once a mid-price exists, and never
    changes it again -- isolates latency/merge-loop tests from any real
    strategy's own reactive behavior, so at most one requote is ever
    in flight unless the test explicitly arranges more.
    """

    def __init__(self, quote: Quote) -> None:
        self._quote = quote

    def quote(self, state: MarketState) -> Quote:
        return self._quote if state.mid_price is not None else Quote.none()


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


def test_delayed_requote_interacts_with_book_as_it_actually_was_at_arrival():
    # The strategy decides ONE quote (bid=97.50) at t=1.0, when the only
    # resting ask is 99.00 -- at zero latency this would just rest
    # passively, no fill. But its arrival is delayed to t=51.0, and at
    # t=2.0 a background order posts a *better* ask (97.20) that the
    # strategy's decision never saw. If the delayed order is matched
    # against the book as it was at *decision* time, nothing crosses. If
    # it's matched against the book as it actually is at *arrival* time
    # (the plan's "replay book state forward" claim), it should
    # immediately cross the 97.20 ask as a taker -- proving
    # latency-induced staleness has a real, observable consequence, not
    # just a delayed timestamp.
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),  # mid=98.00 -> decision at t=1.0
            _event(3, 2.0, "LIMIT", "SELL", 97.20, 5),  # better ask appears AFTER the decision
        ]
    )
    fixed_quote = Quote(bid_price=97.50, bid_size=20, ask_price=98.50, ask_size=20)
    strategy = _FixedQuoteStrategy(fixed_quote)

    result = run_backtest(events, strategy, strategy_latency_model=lambda t: t + 50.0)

    assert len(result.portfolio.trades) == 1
    trade = result.portfolio.trades[0]
    assert trade.is_maker is False  # arrived and crossed as a taker, not rested as a maker
    assert trade.side == Side.BUY
    assert trade.price == pytest.approx(97.20)
    assert trade.size == 5
    assert result.portfolio.inventory == 5


def test_out_of_order_arrivals_last_to_arrive_wins_not_last_decided():
    # Decision A (t=1.0) is delayed heavily (arrives at t=51); decision B
    # (t=2.0, decided *after* A) is delayed lightly (arrives at t=3) --
    # so B lands first, then A lands later and overwrites it. The engine
    # should end up resting at A's prices, proving arrival order controls
    # final book state, not decision order (same claim as the matching
    # engine's own test, now exercised through the full interactive sim).
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),  # mid=98.00 -> decision A: bid=97.50/ask=98.50
            _event(3, 2.0, "LIMIT", "BUY", 97.20, 5),  # improves best bid -> decision B: bid=97.60/ask=98.60
            _event(999, 60.0, "CANCEL"),  # trailing event so both arrivals (t=3, t=51) show up in a row
        ]
    )
    strategy = NaiveSymmetricStrategy(half_spread=0.5, quote_size=20)

    def latency_model(decision_time: float) -> float:
        return decision_time + 50.0 if decision_time == 1.0 else decision_time + 1.0

    result = run_backtest(events, strategy, strategy_latency_model=latency_model)

    last_row = result.book_snapshots.iloc[-1]
    assert last_row["bid_price_1"] == pytest.approx(97.50)  # A's price, not B's 97.60
    assert last_row["ask_price_1"] == pytest.approx(98.50)


def test_default_latency_matches_explicit_none():
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),
            _event(3, 2.0, "MARKET", "BUY", size=5),
        ]
    )
    strategy_a = NaiveSymmetricStrategy(half_spread=0.5, quote_size=20)
    strategy_b = NaiveSymmetricStrategy(half_spread=0.5, quote_size=20)

    result_default = run_backtest(events, strategy_a)
    result_explicit_none = run_backtest(events, strategy_b, strategy_latency_model=None)

    assert result_default.portfolio.inventory == result_explicit_none.portfolio.inventory
    assert result_default.portfolio.cash == result_explicit_none.portfolio.cash


class _CountingStrategy(Strategy):
    """Delegates to an inner strategy but records how many times quote()
    was actually called -- used to verify decision_interval_seconds
    throttles decision *opportunities*, for rl.evaluate's fixed-cadence
    comparison against RL policies.
    """

    def __init__(self, inner: Strategy) -> None:
        self.inner = inner
        self.call_times: list[float] = []

    def quote(self, state: MarketState) -> Quote:
        self.call_times.append(state.time)
        return self.inner.quote(state)


def test_decision_interval_throttles_quote_calls_to_fixed_boundaries():
    events = pd.DataFrame(
        [_event(i, float(i) * 0.2, "LIMIT", "BUY", 97.00, 1) for i in range(1, 30)]
        + [_event(100, 0.05, "LIMIT", "SELL", 99.00, 1)]
    )
    events = events.sort_values("time").reset_index(drop=True)

    unthrottled = _CountingStrategy(NaiveSymmetricStrategy(half_spread=0.5, quote_size=10))
    run_backtest(events, unthrottled)  # decision_interval_seconds=None -> every event

    throttled = _CountingStrategy(NaiveSymmetricStrategy(half_spread=0.5, quote_size=10))
    run_backtest(events, throttled, decision_interval_seconds=1.0)

    assert len(unthrottled.call_times) == len(events)
    assert len(throttled.call_times) < len(unthrottled.call_times)
    # At most one decision per 1-second boundary crossed by the data (times span ~0.05-5.8s).
    assert len(throttled.call_times) <= 6
    # Decisions happen on the first event to reach each boundary, so
    # consecutive recorded times should be spaced >= the interval apart.
    for earlier, later in zip(throttled.call_times, throttled.call_times[1:]):
        assert later - earlier >= 1.0 - 1e-9


def test_decision_interval_none_is_default_and_unthrottled():
    events = pd.DataFrame(
        [_event(1, 0.0, "LIMIT", "SELL", 99.00, 10), _event(2, 0.001, "LIMIT", "BUY", 97.00, 10)]
    )
    counting = _CountingStrategy(NaiveSymmetricStrategy(half_spread=0.5, quote_size=10))
    run_backtest(events, counting)
    assert len(counting.call_times) == 2


def test_resubmits_after_full_fill_even_when_market_state_looks_unchanged():
    # Regression test for a real bug found during this project's testing: the
    # "only requote when the desired quote changes" optimization compared
    # against the *decided* quote, not whether it was still resting. If a
    # strategy's resting order got fully filled but another order (here,
    # background's) happened to sit at the exact same price afterward,
    # best_bid/best_ask looked unchanged from the outside, so the
    # strategy's *desired* quote never changed either -- and it would
    # silently stay out of the market on that side for the rest of the
    # session despite having zero resting orders there.
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 100),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 100),  # mid=98.00
            _event(3, 2.0, "LIMIT", "SELL", 98.50, 50),  # background also rests at our ask price
            _event(4, 3.0, "MARKET", "BUY", size=10),  # fully consumes our ask (FIFO: ours was first)
            # Background's order is now ahead of our (correctly) resubmitted
            # one in the queue; a big enough sweep should reach it.
            _event(5, 4.0, "MARKET", "BUY", size=60),
        ]
    )
    strategy = NaiveSymmetricStrategy(half_spread=0.5, quote_size=10)
    result = run_backtest(events, strategy)

    assert len(result.portfolio.trades) == 2
    assert result.portfolio.trades[0].price == pytest.approx(98.50)
    assert result.portfolio.trades[1].price == pytest.approx(98.50)
    assert result.portfolio.inventory == -20
