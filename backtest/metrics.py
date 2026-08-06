"""Summary statistics for comparing strategy backtest runs.

A plain dict of numbers, not a report -- comparing fill rate and P&L across
strategies, and assembling those numbers into a full strategy x latency x
data-source table, both want the same handful of stats out of a
BacktestResult, so it lives here once rather than getting recomputed ad
hoc at each comparison site.
"""

from __future__ import annotations

import numpy as np

from backtest.market_maker_sim import BacktestResult
from lob.models import Side


def adverse_selection_cost(result: BacktestResult, horizon_events: int = 20) -> float | None:
    """Mean markout cost, in dollars per share, over the strategy's maker
    fills: for each fill, how much the mid-price moved against the
    strategy's new position over the next `horizon_events` book snapshots.

    cost = fill_price - future_mid   for a BUY (bought too high if price fell)
    cost = future_mid - fill_price   for a SELL (sold too low if price rose)

    Positive = net adverse selection cost (informed flow picked the
    strategy off); negative = the fills were, on average, favorably timed.
    Only maker fills count -- adverse selection is specifically about
    resting orders getting hit by better-informed flow, not about a
    strategy's own (currently nonexistent, for the hand-tuned strategies)
    aggressive taker orders. Returns None if there are no maker fills, or none with
    a defined future mid-price to markout against (e.g. right at the end
    of the session).
    """
    maker_trades = [t for t in result.portfolio.trades if t.is_maker]
    if not maker_trades:
        return None

    times = result.book_snapshots["time"].to_numpy()
    mids = ((result.book_snapshots["bid_price_1"] + result.book_snapshots["ask_price_1"]) / 2.0).to_numpy()

    costs = []
    for trade in maker_trades:
        idx = np.searchsorted(times, trade.time, side="left")
        future_idx = min(idx + horizon_events, len(mids) - 1)
        future_mid = mids[future_idx]
        if np.isnan(future_mid):
            continue
        cost = (trade.price - future_mid) if trade.side == Side.BUY else (future_mid - trade.price)
        costs.append(cost)

    return float(np.mean(costs)) if costs else None


def sharpe_ratio(result: BacktestResult) -> float | None:
    """Per-step Sharpe: mean(equity changes) / std(equity changes) across
    consecutive portfolio_history rows. Deliberately NOT annualized --
    sessions here range from minutes to hours, not trading years, and
    "decision steps" aren't a standard annualization unit -- this is a
    within-session risk-adjusted-return proxy for comparing strategies
    against each other on the *same* session, not a figure comparable to
    a conventional annualized Sharpe ratio from other contexts.

    Rows where equity is NaN (inventory held but unmarkable, see
    backtest.portfolio.equity_from) are dropped before differencing.
    Returns None if fewer than 2 valid equity observations remain, or
    equity never changed at all (e.g. a strategy with zero fills, where
    "risk-adjusted return" is undefined, not zero).
    """
    equity = result.portfolio_history["equity"].dropna()
    if len(equity) < 2:
        return None
    changes = equity.diff().dropna()
    if len(changes) < 2 or changes.std() == 0:
        return None
    return float(changes.mean() / changes.std())


def summarize(result: BacktestResult, adverse_selection_horizon_events: int = 20) -> dict:
    trades = result.portfolio.trades
    inventory_series = result.portfolio_history["inventory"]
    equity_series = result.portfolio_history["equity"]

    maker_fills = sum(1 for t in trades if t.is_maker)
    taker_fills = len(trades) - maker_fills

    # Equity starts at 0 (flat, no cash) by construction, so the final
    # mark-to-mid equity *is* total P&L -- not annualized, not
    # risk-adjusted, just the number this run actually produced.
    final_pnl = equity_series.iloc[-1] if len(equity_series) else float("nan")

    # float()/int() everywhere below: pandas reductions (.mean, .std,
    # .iloc) return numpy scalar types, not plain Python ones. SQLite
    # tolerates numpy.float64 silently (it subclasses float), which let
    # this slip through every local run and test -- psycopg2/Postgres
    # does not, and fails opaquely (a stringified "np.float64(...)"
    # ends up embedded in the literal SQL under numpy>=2.0's repr,
    # raising an unrelated-looking InvalidSchemaName). DB rows should
    # hold plain Python types regardless of driver quirks.
    return {
        "n_fills": len(trades),
        "maker_fills": maker_fills,
        "taker_fills": taker_fills,
        "final_inventory": float(result.portfolio.inventory),
        "final_pnl": float(final_pnl),
        "inventory_mean_abs": float(inventory_series.abs().mean()),
        "inventory_std": float(inventory_series.std()),
        "equity_std": float(equity_series.std()),
        "adverse_selection_cost": adverse_selection_cost(result, adverse_selection_horizon_events),
        "sharpe_ratio": sharpe_ratio(result),
    }
