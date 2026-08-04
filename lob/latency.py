"""Stochastic latency models mapping decision_time -> arrival_time.

Real network + exchange latency is right-skewed: most round trips cluster
near a floor set by physical distance and processing overhead, but a fat
tail of slow ones always exists (a queueing delay, a retransmit, a GC
pause). A lognormal distribution is the standard choice to capture that
shape -- bounded below by construction (no negative latency), with the
long right tail the plan calls for.

`LATENCY_PRESETS` gives four illustrative settings (0 / 5 / 20 / 50ms
median), loosely calibrated to the real spread between co-located and
retail connectivity, not fitted to any specific measured dataset -- a
disclosed assumption, same as every other calibration choice in this
project (see data/synthetic_lob.py's docstring for the same pattern).
Real published figures put exchange co-location round trips in the
low-single-digit milliseconds and retail/cross-region retail connections
in the tens of milliseconds; these four points are chosen to span that
range plausibly; they are not a claim about any particular venue.
"""

from __future__ import annotations

import math

import numpy as np

from lob.engine import LatencyModel, zero_latency


def lognormal_latency_model(
    median_seconds: float,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> LatencyModel:
    """A latency model whose delay ~ Lognormal(mu, sigma), with mu chosen
    so the distribution's median is exactly `median_seconds`.

    `sigma` is the lognormal shape parameter (log-space standard
    deviation): larger sigma means a heavier right tail (more occasional
    very-slow round trips) relative to the median.
    """
    if median_seconds <= 0:
        raise ValueError("median_seconds must be positive; use lob.engine.zero_latency for exact zero delay")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    rng = rng if rng is not None else np.random.default_rng()
    mu = math.log(median_seconds)

    def model(decision_time: float) -> float:
        return decision_time + rng.lognormal(mean=mu, sigma=sigma)

    return model


# (median_seconds, sigma) -- sigma grows with the median here on the
# (disclosed, illustrative) assumption that slower/longer network paths
# also see proportionally more jitter, not just a shifted floor.
LATENCY_PRESET_PARAMS: dict[str, tuple[float, float]] = {
    "0ms": (0.0, 0.0),
    "5ms": (0.005, 0.3),
    "20ms": (0.020, 0.4),
    "50ms": (0.050, 0.5),
}


def make_latency_model(preset: str, rng: np.random.Generator | None = None) -> LatencyModel:
    if preset not in LATENCY_PRESET_PARAMS:
        raise ValueError(f"Unknown latency preset {preset!r}; choose from {sorted(LATENCY_PRESET_PARAMS)}")
    median_seconds, sigma = LATENCY_PRESET_PARAMS[preset]
    if median_seconds == 0.0:
        return zero_latency
    return lognormal_latency_model(median_seconds, sigma, rng)
