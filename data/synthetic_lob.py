"""Synthetic limit order book event generator (primary data source).

Data source decision (see README "Data" section for full rationale): this
project's core value is mechanism, not proving a real trading edge exists
-- matching-engine correctness, latency effects, and RL-vs-heuristics
comparisons all depend on controllable, unlimited order flow, not on any
specific real instrument. Synthetic data is the primary source because it
gives full control over regime (calm vs. stressed order flow) for Step 8's
latency sweep and Step 11's robustness testing, with zero licensing
friction. Real LOBSTER equity data was investigated and ruled out on cost
(it now requires a paid subscription or metered "lobster-data-coin"
credits). Real order book data is still used, as a secondary validation
layer, via `binance_capture.py` -- see that module's docstring.

The calibration parameters below (arrival rates, depth/size
distributions) are starting points drawn from published market
microstructure literature (e.g. Cont, Stoikov & Talreja's queue-reactive
model; Avellaneda-Stoikov-style order placement), not fitted to any
specific stock. That is a disclosed judgment call, not an empirical fact.

Model (a simplified queue-reactive / reduced-form model):

- A reference mid-price follows a discrete-time random walk in tick units.
  It only exists to *place* incoming limit orders realistically -- once
  these events are replayed through the actual matching engine (lob/), the
  engine's own best bid/ask emerges from the events and is free to drift
  from this reference. That divergence is expected and disclosed as a
  limitation, not a bug: this generator does not attempt to be
  self-consistent with the book it produces.
- New limit orders arrive as a Poisson process. Each is placed at a depth
  (in ticks) from the reference best bid/ask drawn from a geometric
  distribution, so most volume sits near the touch and depth thins out
  further away, matching the qualitative shape of a real book.
- Each limit order is independently assigned an exponential lifetime; if
  that lifetime elapses before the session ends, a CANCEL event is emitted
  for it. The matching engine is responsible for no-op'ing cancels for
  orders that were already filled -- this generator does not know about
  fills, since it has no matching logic itself.
- Market orders arrive as an independent Poisson process with their own
  side and lognormal size distribution.
- A `regime` multiplier lets Step 8/11 stress the same model (higher
  arrival rates, higher volatility, thinner depth) without a different
  code path.

Output is a single time-sorted DataFrame of events, order_id-linked, ready
to be replayed through lob/ (Step 2/3).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["order_id", "time", "type", "side", "price", "size"]

REGIME_MULTIPLIERS = {
    # (arrival_rate_mult, mid_vol_mult, lifetime_mult)
    "calm": (1.0, 1.0, 1.0),
    "normal": (1.0, 1.0, 1.0),
    "stressed": (2.5, 3.0, 0.4),
}


@dataclasses.dataclass
class SyntheticLOBConfig:
    session_seconds: float = 23_400.0  # 6.5h regular trading session
    tick_size: float = 0.01
    initial_mid_price: float = 100.00

    # Reference mid-price random walk, in ticks, per sqrt(second).
    mid_price_vol_ticks: float = 0.5

    # Fixed half-spread (in ticks) used only to place new limit orders
    # relative to the reference mid.
    half_spread_ticks: float = 2.0

    # Poisson rates, events per second.
    limit_order_rate: float = 2.0
    market_order_rate: float = 0.3

    # Depth (in ticks) of a new limit order beyond the reference best
    # bid/ask: depth ~ Geometric(depth_p), so mean depth = (1-p)/p.
    depth_p: float = 0.35

    # Order size ~ lognormal(mu, sigma), rounded to a lot size.
    size_mu: float = 4.6  # exp(4.6) ~= 100 shares
    size_sigma: float = 0.8
    lot_size: int = 1

    # Mean resting-order lifetime in seconds, before an independent cancel.
    mean_order_lifetime: float = 30.0

    # "calm" / "normal" / "stressed" -- see REGIME_MULTIPLIERS.
    regime: str = "normal"

    seed: int | None = None

    def scaled(self) -> "SyntheticLOBConfig":
        """Return a copy with regime multipliers applied."""
        rate_mult, vol_mult, lifetime_mult = REGIME_MULTIPLIERS[self.regime]
        return dataclasses.replace(
            self,
            limit_order_rate=self.limit_order_rate * rate_mult,
            market_order_rate=self.market_order_rate * rate_mult,
            mid_price_vol_ticks=self.mid_price_vol_ticks * vol_mult,
            mean_order_lifetime=self.mean_order_lifetime * lifetime_mult,
        )


def _draw_sizes(rng: np.random.Generator, n: int, cfg: SyntheticLOBConfig) -> np.ndarray:
    sizes = rng.lognormal(mean=cfg.size_mu, sigma=cfg.size_sigma, size=n)
    sizes = np.maximum(cfg.lot_size, np.round(sizes / cfg.lot_size) * cfg.lot_size)
    return sizes.astype(np.int64)


def generate_session(cfg: SyntheticLOBConfig | None = None) -> pd.DataFrame:
    """Generate one synthetic trading session as a time-sorted event log."""
    cfg = (cfg or SyntheticLOBConfig()).scaled()
    rng = np.random.default_rng(cfg.seed)

    n_limit = rng.poisson(cfg.limit_order_rate * cfg.session_seconds)
    n_market = rng.poisson(cfg.market_order_rate * cfg.session_seconds)

    limit_times = np.sort(rng.uniform(0.0, cfg.session_seconds, size=n_limit))
    market_times = np.sort(rng.uniform(0.0, cfg.session_seconds, size=n_market))

    # Reference mid-price path, sampled at every order arrival time via a
    # running random walk (cumulative sum of independent increments scaled
    # by sqrt(dt) since the previous point in the merged, sorted timeline).
    all_times = np.sort(np.concatenate([limit_times, market_times]))
    dt = np.diff(all_times, prepend=0.0)
    increments = rng.normal(0.0, cfg.mid_price_vol_ticks, size=len(all_times)) * np.sqrt(
        np.maximum(dt, 1e-9)
    )
    mid_path_ticks = np.cumsum(increments)

    def mid_price(t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(all_times, t, side="left")
        idx = np.clip(idx, 0, len(mid_path_ticks) - 1)
        return cfg.initial_mid_price + mid_path_ticks[idx] * cfg.tick_size

    events: list[dict] = []
    order_id = 0

    # --- Limit orders + their independent cancels ---
    limit_sides = rng.choice(["BUY", "SELL"], size=n_limit)
    limit_depths = rng.geometric(cfg.depth_p, size=n_limit) - 1  # >= 0
    limit_sizes = _draw_sizes(rng, n_limit, cfg)
    limit_mids = mid_price(limit_times)
    lifetimes = rng.exponential(cfg.mean_order_lifetime, size=n_limit)

    for i in range(n_limit):
        order_id += 1
        side = limit_sides[i]
        depth_ticks = limit_depths[i]
        half_spread = cfg.half_spread_ticks * cfg.tick_size
        if side == "BUY":
            price = limit_mids[i] - half_spread - depth_ticks * cfg.tick_size
        else:
            price = limit_mids[i] + half_spread + depth_ticks * cfg.tick_size
        price = round(price / cfg.tick_size) * cfg.tick_size

        events.append(
            {
                "order_id": order_id,
                "time": limit_times[i],
                "type": "LIMIT",
                "side": side,
                "price": price,
                "size": limit_sizes[i],
            }
        )

        cancel_time = limit_times[i] + lifetimes[i]
        if cancel_time < cfg.session_seconds:
            events.append(
                {
                    "order_id": order_id,
                    "time": cancel_time,
                    "type": "CANCEL",
                    "side": side,
                    "price": np.nan,
                    "size": np.nan,
                }
            )

    # --- Market orders ---
    market_sides = rng.choice(["BUY", "SELL"], size=n_market)
    market_sizes = _draw_sizes(rng, n_market, cfg)
    for i in range(n_market):
        order_id += 1
        events.append(
            {
                "order_id": order_id,
                "time": market_times[i],
                "type": "MARKET",
                "side": market_sides[i],
                "price": np.nan,
                "size": market_sizes[i],
            }
        )

    df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    df = df.sort_values(["time", "order_id"], kind="stable").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import pathlib

    out_path = pathlib.Path(__file__).parent / "synthetic_session_sample.csv"
    frame = generate_session(SyntheticLOBConfig(session_seconds=600.0, seed=7))
    frame.to_csv(out_path, index=False)
    print(f"Wrote {len(frame)} events to {out_path}")
    print(frame.head(10))
