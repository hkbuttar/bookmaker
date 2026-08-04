import numpy as np
import pytest

from data.synthetic_lob import SyntheticLOBConfig, generate_session


def test_generate_session_reproducible_with_seed():
    cfg = SyntheticLOBConfig(session_seconds=300.0, seed=42)
    a = generate_session(cfg)
    b = generate_session(cfg)
    pd_equal = a.equals(b)
    assert pd_equal


def test_events_sorted_by_time():
    df = generate_session(SyntheticLOBConfig(session_seconds=300.0, seed=1))
    assert df["time"].is_monotonic_increasing


def test_event_types_and_prices_on_tick_grid():
    cfg = SyntheticLOBConfig(session_seconds=300.0, tick_size=0.01, seed=2)
    df = generate_session(cfg)
    assert set(df["type"].unique()) <= {"LIMIT", "MARKET", "CANCEL"}

    limit_prices = df.loc[df["type"] == "LIMIT", "price"]
    ticks = limit_prices / cfg.tick_size
    assert np.allclose(ticks, np.round(ticks), atol=1e-6)


def test_market_and_cancel_events_have_no_price():
    df = generate_session(SyntheticLOBConfig(session_seconds=300.0, seed=3))
    non_limit = df[df["type"] != "LIMIT"]
    assert non_limit["price"].isna().all()


def test_cancels_reference_a_prior_limit_order_id():
    df = generate_session(SyntheticLOBConfig(session_seconds=600.0, seed=4))
    limit_ids = set(df.loc[df["type"] == "LIMIT", "order_id"])
    cancels = df[df["type"] == "CANCEL"]
    assert set(cancels["order_id"]) <= limit_ids

    # Every cancel must occur strictly after its limit order's submission.
    limit_times = df[df["type"] == "LIMIT"].set_index("order_id")["time"]
    for _, row in cancels.iterrows():
        assert row["time"] > limit_times.loc[row["order_id"]]


def test_event_counts_scale_with_session_length():
    short = generate_session(SyntheticLOBConfig(session_seconds=300.0, seed=5))
    long = generate_session(SyntheticLOBConfig(session_seconds=3000.0, seed=5))
    assert len(long) > len(short)


def test_stressed_regime_increases_arrival_rate():
    normal = generate_session(SyntheticLOBConfig(session_seconds=1000.0, regime="normal", seed=6))
    stressed = generate_session(SyntheticLOBConfig(session_seconds=1000.0, regime="stressed", seed=6))
    assert len(stressed) > len(normal)


def test_sizes_are_positive_integers():
    df = generate_session(SyntheticLOBConfig(session_seconds=300.0, seed=8))
    sized = df[df["type"] != "CANCEL"]
    assert (sized["size"] > 0).all()
    assert (sized["size"] == sized["size"].astype(int)).all()
