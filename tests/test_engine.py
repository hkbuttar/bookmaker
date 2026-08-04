import pandas as pd

from data.synthetic_lob import SyntheticLOBConfig, generate_session
from lob.engine import MatchingEngine


def _event(order_id, time, type_, side=None, price=None, size=None):
    return {"order_id": order_id, "time": time, "type": type_, "side": side, "price": price, "size": size}


def test_zero_latency_processes_in_decision_order():
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "BUY", 100.00, 10),
            _event(2, 1.0, "LIMIT", "SELL", 100.00, 10),
        ]
    )
    engine = MatchingEngine()
    result = engine.replay(events)

    assert len(result.fills) == 1
    # Order 1 decided (and, at zero latency, arrived) first, so it was
    # resting when order 2 showed up: order 1 is the maker.
    assert result.fills[0].maker_order_id == 1
    assert result.fills[0].taker_order_id == 2


def test_custom_latency_can_reorder_arrival_and_flip_maker_taker():
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "BUY", 100.00, 10),  # decided first...
            _event(2, 1.0, "LIMIT", "SELL", 100.00, 10),  # ...decided second
        ]
    )

    # Order 1 (decided at t=0) is delayed by 10s; order 2 (decided at t=1)
    # has no delay. So order 2 actually *arrives* first (t=1 < t=10),
    # despite having a later decision_time. If the engine matched on
    # decision order, order 1 would always be the maker; matching on
    # arrival order instead, order 2 rests first and becomes the maker.
    def latency_model(decision_time: float) -> float:
        return decision_time + 10.0 if decision_time == 0.0 else decision_time

    engine = MatchingEngine(latency_model=latency_model)
    result = engine.replay(events)

    assert len(result.fills) == 1
    assert result.fills[0].maker_order_id == 2
    assert result.fills[0].taker_order_id == 1


def test_cancel_of_unknown_order_is_counted_not_raised():
    events = pd.DataFrame([_event(1, 0.0, "CANCEL")])
    engine = MatchingEngine()
    result = engine.replay(events)
    assert result.fills == []
    assert result.unknown_cancels == 1


def test_cancel_then_resubmit_same_price_does_not_crash_and_book_stays_consistent():
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "BUY", 100.00, 10),
            _event(1, 1.0, "CANCEL"),  # cancels order 1 -- CANCEL reuses the target order's id
            _event(2, 2.0, "LIMIT", "SELL", 99.00, 5),  # crosses nothing since order 1 is gone
        ]
    )
    engine = MatchingEngine()
    result = engine.replay(events)
    assert result.fills == []
    assert result.unknown_cancels == 0
    assert engine.book.best_bid() is None
    assert engine.book.best_ask() == 99.00


def test_replay_full_synthetic_session_smoke():
    events = generate_session(SyntheticLOBConfig(session_seconds=600.0, seed=11))
    engine = MatchingEngine()
    result = engine.replay(events)

    assert isinstance(result.fills, list)
    assert len(result.fills) > 0  # market orders in the session should cross something

    best_bid = engine.book.best_bid()
    best_ask = engine.book.best_ask()
    if best_bid is not None and best_ask is not None:
        assert best_bid < best_ask  # matching must never leave a crossed book
