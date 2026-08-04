import numpy as np
import pandas as pd
import pytest

from lob.features import compute_features


def _row(bid_price_1, bid_size_1, ask_price_1, ask_size_1, bid_price_2=np.nan, bid_size_2=0,
         ask_price_2=np.nan, ask_size_2=0):
    return {
        "bid_price_1": bid_price_1,
        "bid_size_1": bid_size_1,
        "ask_price_1": ask_price_1,
        "ask_size_1": ask_size_1,
        "bid_price_2": bid_price_2,
        "bid_size_2": bid_size_2,
        "ask_price_2": ask_price_2,
        "ask_size_2": ask_size_2,
    }


def test_mid_price_and_spread():
    df = pd.DataFrame([_row(99.90, 10, 100.10, 5)])
    out = compute_features(df)
    assert out.loc[0, "mid_price"] == pytest.approx(100.00)
    assert out.loc[0, "spread"] == pytest.approx(0.20)


def test_mid_price_and_spread_nan_when_one_side_empty():
    df = pd.DataFrame([_row(np.nan, 0, 100.10, 5)])
    out = compute_features(df)
    assert pd.isna(out.loc[0, "mid_price"])
    assert pd.isna(out.loc[0, "spread"])


def test_imbalance_top_of_book():
    # More resting buy interest than sell -> positive imbalance.
    df = pd.DataFrame([_row(99.90, 30, 100.10, 10)])
    out = compute_features(df, imbalance_levels=1)
    assert out.loc[0, "imbalance"] == pytest.approx((30 - 10) / (30 + 10))


def test_imbalance_symmetric_book_is_zero():
    df = pd.DataFrame([_row(99.90, 20, 100.10, 20)])
    out = compute_features(df, imbalance_levels=1)
    assert out.loc[0, "imbalance"] == pytest.approx(0.0)


def test_imbalance_both_sides_empty_is_nan():
    df = pd.DataFrame([_row(np.nan, 0, np.nan, 0)])
    out = compute_features(df, imbalance_levels=1)
    assert pd.isna(out.loc[0, "imbalance"])


def test_imbalance_aggregates_multiple_levels():
    df = pd.DataFrame(
        [_row(99.90, 10, 100.10, 5, bid_price_2=99.85, bid_size_2=10, ask_price_2=100.15, ask_size_2=5)]
    )
    out_l1 = compute_features(df, imbalance_levels=1)
    out_l2 = compute_features(df, imbalance_levels=2)
    assert out_l1.loc[0, "imbalance"] == pytest.approx((10 - 5) / 15)
    assert out_l2.loc[0, "imbalance"] == pytest.approx((20 - 10) / 30)


def test_imbalance_levels_beyond_available_columns_raises():
    df = pd.DataFrame([_row(99.90, 10, 100.10, 5)])
    with pytest.raises(ValueError):
        compute_features(df, imbalance_levels=5)


def test_realized_vol_matches_manual_rolling_std():
    # A small, deterministic mid-price path so the rolling std is easy to
    # hand-check against pandas' own std, independent of the feature code.
    mids = [100.00, 100.10, 99.95, 100.05, 100.00, 99.90, 100.10, 100.00]
    rows = [_row(m - 0.01, 10, m + 0.01, 10) for m in mids]
    df = pd.DataFrame(rows)

    out = compute_features(df, vol_window=4)

    expected_log_returns = np.log(pd.Series(mids)).diff()
    expected_vol = expected_log_returns.rolling(4, min_periods=2).std()

    pd.testing.assert_series_equal(
        out["realized_vol"], expected_vol.rename("realized_vol"), check_exact=False
    )


def test_realized_vol_nan_for_first_row_no_prior_return():
    df = pd.DataFrame([_row(99.99, 10, 100.01, 10), _row(100.09, 10, 100.11, 10)])
    out = compute_features(df, vol_window=4)
    assert pd.isna(out.loc[0, "realized_vol"])
