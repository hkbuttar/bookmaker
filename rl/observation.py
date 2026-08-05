"""Shared observation-vector construction for the RL environment (rl/env.py)
and the trained-policy evaluation adapter (rl/evaluate.py). Both need to
build the *exact* same 7-feature vector a trained model was shown during
training -- if evaluation reconstructed it with even a slightly different
formula, any performance gap would be confounded by a train/eval mismatch
instead of reflecting the policy itself. See rl/env.py's module docstring
for the feature list and the disclosed rationale for each one.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

N_FEATURES = 7


def realized_vol_from_history(mid_history: deque) -> float:
    if len(mid_history) < 3:
        return 0.0
    prices = np.array(mid_history, dtype=np.float64)
    log_returns = np.diff(np.log(prices))
    return float(np.std(log_returns))


def build_observation(
    inventory: int,
    mid_price: float | None,
    initial_mid_price: float | None,
    spread: float | None,
    imbalance: float | None,
    mid_history: deque,
    equity: float,
    t: float,
    session_seconds: float,
    tick_size: float,
    inventory_norm_scale: float,
    pnl_norm_scale: float,
    price_change_norm_scale: float,
    vol_scale: float,
    obs_bound: float,
) -> np.ndarray:
    price_change_ticks = (
        0.0 if (initial_mid_price is None or mid_price is None) else (mid_price - initial_mid_price) / tick_size
    )
    spread_ticks = 0.0 if spread is None else spread / tick_size
    imbalance_value = 0.0 if imbalance is None else imbalance
    realized_vol = realized_vol_from_history(mid_history)
    time_remaining_frac = max(0.0, 1.0 - t / session_seconds)
    pnl_norm = 0.0 if math.isnan(equity) else equity / pnl_norm_scale

    obs = np.array(
        [
            inventory / inventory_norm_scale,
            price_change_ticks / price_change_norm_scale,
            spread_ticks,
            imbalance_value,
            realized_vol * vol_scale,
            time_remaining_frac,
            pnl_norm,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -obs_bound, obs_bound)
