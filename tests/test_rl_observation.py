from collections import deque

import numpy as np
import pytest

from rl.observation import build_observation, realized_vol_from_history


def test_realized_vol_needs_at_least_three_points():
    assert realized_vol_from_history(deque([100.0, 100.1])) == 0.0


def test_realized_vol_matches_manual_std():
    prices = deque([100.0, 100.1, 99.9, 100.2])
    expected = np.std(np.diff(np.log(prices)))
    assert realized_vol_from_history(prices) == pytest.approx(expected)


def _kwargs(**overrides):
    defaults = dict(
        inventory=0,
        mid_price=100.00,
        initial_mid_price=100.00,
        spread=0.02,
        imbalance=0.0,
        mid_history=deque([100.00, 100.01, 99.99]),
        equity=0.0,
        t=0.0,
        session_seconds=600.0,
        tick_size=0.01,
        inventory_norm_scale=100.0,
        pnl_norm_scale=50.0,
        price_change_norm_scale=100.0,
        vol_scale=1000.0,
        obs_bound=20.0,
    )
    defaults.update(overrides)
    return defaults


def test_observation_shape_and_dtype():
    obs = build_observation(**_kwargs())
    assert obs.shape == (7,)
    assert obs.dtype == np.float32


def test_inventory_and_pnl_normalization():
    obs = build_observation(**_kwargs(inventory=50, inventory_norm_scale=100.0, equity=25.0, pnl_norm_scale=50.0))
    assert obs[0] == pytest.approx(0.5)
    assert obs[6] == pytest.approx(0.5)


def test_missing_mid_price_gives_neutral_price_change_and_spread():
    obs = build_observation(**_kwargs(mid_price=None, spread=None, imbalance=None))
    assert obs[1] == 0.0  # price_change_ticks
    assert obs[2] == 0.0  # spread_ticks
    assert obs[3] == 0.0  # imbalance


def test_time_remaining_fraction():
    obs = build_observation(**_kwargs(t=150.0, session_seconds=600.0))
    assert obs[5] == pytest.approx(0.75)


def test_time_remaining_clamped_nonnegative_past_session_end():
    obs = build_observation(**_kwargs(t=900.0, session_seconds=600.0))
    assert obs[5] == 0.0


def test_extreme_values_clipped_to_obs_bound():
    obs = build_observation(**_kwargs(inventory=100_000, inventory_norm_scale=1.0, obs_bound=20.0))
    assert obs[0] == 20.0
