"""Features every strategy conditions on, computed from book-state snapshots.

Consumes the flat `top_levels`-shaped DataFrame both data sources produce
(`lob.engine.MatchingEngine.replay(...).book_snapshots` for synthetic/
matching-engine replay, `data.binance_capture.capture_session(...).book_snapshots`
for the Binance validation layer) -- same column schema, same feature code,
no per-source glue needed by the dual-source comparison.

Definitions (disclosed as choices, not the only valid ones):

- mid_price: (best_bid + best_ask) / 2. NaN whenever either side of the
  book is empty -- that's a real state (no two-sided market), not missing
  data, and callers should decide how to handle it rather than have it
  silently imputed here.
- spread: best_ask - best_bid. Same NaN behavior as mid_price.
- imbalance: (bid_volume - ask_volume) / (bid_volume + ask_volume) summed
  over the top `imbalance_levels` price levels per side (default 1, i.e.
  top-of-book only). Ranges [-1, 1]; positive means more resting buy
  interest near the touch. NaN when both sides are empty at those levels.
- realized_vol: rolling standard deviation of mid-price log returns over
  the trailing `vol_window` events (an event-count window, not a fixed
  time window -- simpler, and fine for "short-term" signal purposes here).
  Not annualized: this is a per-step microstructure feature, not a
  portfolio-level risk statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(
    book_snapshots: pd.DataFrame,
    imbalance_levels: int = 1,
    vol_window: int = 50,
) -> pd.DataFrame:
    """Return a copy of `book_snapshots` with mid_price, spread, imbalance,
    and realized_vol columns added.
    """
    for level in range(1, imbalance_levels + 1):
        for col in (f"bid_size_{level}", f"ask_size_{level}"):
            if col not in book_snapshots.columns:
                raise ValueError(
                    f"imbalance_levels={imbalance_levels} needs column {col!r}, "
                    f"but book_snapshots only has {book_snapshots.shape[1]} columns. "
                    "Record more levels (e.g. a higher record_levels/depth_levels) upstream."
                )

    df = book_snapshots.copy()

    df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]

    bid_vol = sum(df[f"bid_size_{lvl}"] for lvl in range(1, imbalance_levels + 1))
    ask_vol = sum(df[f"ask_size_{lvl}"] for lvl in range(1, imbalance_levels + 1))
    total_vol = (bid_vol + ask_vol).replace(0, np.nan)
    df["imbalance"] = (bid_vol - ask_vol) / total_vol

    log_returns = np.log(df["mid_price"]).diff()
    df["realized_vol"] = log_returns.rolling(vol_window, min_periods=max(2, vol_window // 5)).std()

    return df


def mid_and_spread_from_row(row: dict) -> tuple[float | None, float | None]:
    """Scalar counterpart to `compute_features`'s mid_price/spread columns,
    for callers building state one event at a time (backtest.market_maker_sim)
    rather than post-processing a whole DataFrame. Same definitions,
    intentionally kept in sync with the vectorized version above.
    """
    bid, ask = row["bid_price_1"], row["ask_price_1"]
    if np.isnan(bid) or np.isnan(ask):
        return None, None
    return (bid + ask) / 2.0, ask - bid


def imbalance_from_row(row: dict, levels: int) -> float | None:
    """Scalar counterpart to `compute_features`'s imbalance column -- see
    mid_and_spread_from_row's docstring for why this exists separately."""
    bid_vol = sum(row[f"bid_size_{lvl}"] for lvl in range(1, levels + 1))
    ask_vol = sum(row[f"ask_size_{lvl}"] for lvl in range(1, levels + 1))
    total_vol = bid_vol + ask_vol
    if total_vol <= 0:
        return None
    return (bid_vol - ask_vol) / total_vol
