import pytest

from data.binance_capture import LocalOrderBook, SequenceGapError


def make_snapshot():
    return {
        "lastUpdateId": 100,
        "bids": [["10.00", "5.0"], ["9.99", "3.0"]],
        "asks": [["10.01", "4.0"], ["10.02", "2.0"]],
    }


def test_apply_snapshot_sets_book_state():
    book = LocalOrderBook()
    book.apply_snapshot(make_snapshot())
    assert book.bids == {10.00: 5.0, 9.99: 3.0}
    assert book.asks == {10.01: 4.0, 10.02: 2.0}
    assert book.last_update_id == 100


def test_discards_diff_entirely_covered_by_snapshot():
    book = LocalOrderBook()
    book.apply_snapshot(make_snapshot())
    stale_event = {"U": 90, "u": 99, "b": [["10.00", "999"]], "a": []}
    assert book.should_discard(stale_event)
    book.apply_diff(stale_event)
    # Stale event must not have touched the book.
    assert book.bids[10.00] == 5.0


def test_first_event_must_bridge_snapshot():
    book = LocalOrderBook()
    book.apply_snapshot(make_snapshot())
    bridging_event = {"U": 98, "u": 105, "b": [], "a": []}
    non_bridging_event = {"U": 102, "u": 105, "b": [], "a": []}
    assert book.is_first_valid_event(bridging_event)
    assert not book.is_first_valid_event(non_bridging_event)


def test_apply_diff_updates_price_levels():
    book = LocalOrderBook()
    book.apply_snapshot(make_snapshot())
    event = {
        "U": 101,
        "u": 102,
        "b": [["9.99", "0.0"], ["9.98", "1.5"]],  # remove 9.99, add 9.98
        "a": [["10.01", "6.0"]],  # update size at 10.01
    }
    book.apply_diff(event)
    assert 9.99 not in book.bids
    assert book.bids[9.98] == 1.5
    assert book.asks[10.01] == 6.0
    assert book.last_update_id == 102


def test_apply_diff_raises_on_sequence_gap():
    book = LocalOrderBook()
    book.apply_snapshot(make_snapshot())
    book.apply_diff({"U": 101, "u": 102, "b": [], "a": []})
    gapped_event = {"U": 110, "u": 115, "b": [], "a": []}  # skips 103-109
    with pytest.raises(SequenceGapError):
        book.apply_diff(gapped_event)


def test_top_levels_sorted_best_first_and_pads_missing():
    book = LocalOrderBook()
    book.apply_snapshot(make_snapshot())
    row = book.top_levels(3)
    assert row["bid_price_1"] == 10.00
    assert row["bid_price_2"] == 9.99
    assert row["bid_price_3"] != row["bid_price_3"]  # NaN, only 2 bid levels
    assert row["bid_size_3"] == 0.0
    assert row["ask_price_1"] == 10.01
    assert row["ask_price_2"] == 10.02
