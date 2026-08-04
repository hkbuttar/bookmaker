"""Summary statistics for comparing strategy backtest runs.

A plain dict of numbers, not a report -- Step 6 needs to compare fill rate
and P&L against Steps 4-5's baselines, and Step 11 needs the same numbers
assembled into a full strategy x latency x data-source table. Both want the
same handful of stats out of a BacktestResult, so it lives here once rather
than getting recomputed ad hoc at each comparison site.
"""

from __future__ import annotations

from backtest.market_maker_sim import BacktestResult


def summarize(result: BacktestResult) -> dict:
    trades = result.portfolio.trades
    inventory_series = result.portfolio_history["inventory"]
    equity_series = result.portfolio_history["equity"]

    maker_fills = sum(1 for t in trades if t.is_maker)
    taker_fills = len(trades) - maker_fills

    # Equity starts at 0 (flat, no cash) by construction, so the final
    # mark-to-mid equity *is* total P&L -- not annualized, not
    # risk-adjusted, just the number this run actually produced.
    final_pnl = equity_series.iloc[-1] if len(equity_series) else float("nan")

    return {
        "n_fills": len(trades),
        "maker_fills": maker_fills,
        "taker_fills": taker_fills,
        "final_inventory": result.portfolio.inventory,
        "final_pnl": final_pnl,
        "inventory_mean_abs": inventory_series.abs().mean(),
        "inventory_std": inventory_series.std(),
        "equity_std": equity_series.std(),
    }
