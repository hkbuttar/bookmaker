"""Runs a backtest from API request parameters and persists the result.

Deliberately thin: this just wires together data.synthetic_lob,
strategies.*, lob.latency, backtest.market_maker_sim, and
backtest.metrics -- the same machinery every other part of this project
uses -- rather than reimplementing anything backend-specific.
"""

from __future__ import annotations

import math

from sqlalchemy.orm import Session

from backend.db_models import BookSnapshotRow, SimulationRun
from backend.schemas import LATENCY_PRESETS, STRATEGY_NAMES, SimulateRequest
from backtest.market_maker_sim import BacktestResult, run_backtest
from backtest.metrics import summarize
from data.synthetic_lob import SyntheticLOBConfig, generate_session
from lob.latency import make_latency_model
from strategies.adverse_selection_aware import AdverseSelectionAwareStrategy
from strategies.base import Strategy
from strategies.inventory_aware import InventoryAwareStrategy
from strategies.naive import NaiveSymmetricStrategy

_STRATEGY_DEFAULTS = {
    "naive": dict(half_spread=0.02, quote_size=10),
    "inventory_aware": dict(half_spread=0.02, quote_size=10, inventory_penalty=0.0005),
    "adverse_selection_aware": dict(
        half_spread=0.02, quote_size=10, imbalance_ema_alpha=0.05,
        imbalance_threshold=0.3, widen_multiplier=2.5, pull_threshold=0.7,
    ),
}

_STRATEGY_CLASSES: dict[str, type[Strategy]] = {
    "naive": NaiveSymmetricStrategy,
    "inventory_aware": InventoryAwareStrategy,
    "adverse_selection_aware": AdverseSelectionAwareStrategy,
}


class InvalidSimulationRequest(ValueError):
    pass


def build_strategy(strategy_name: str, strategy_params: dict, tick_size: float) -> Strategy:
    if strategy_name not in _STRATEGY_CLASSES:
        raise InvalidSimulationRequest(f"Unknown strategy_name {strategy_name!r}; choose from {STRATEGY_NAMES}")
    params = {**_STRATEGY_DEFAULTS[strategy_name], **strategy_params, "tick_size": tick_size}
    try:
        return _STRATEGY_CLASSES[strategy_name](**params)
    except (TypeError, ValueError) as e:
        # TypeError: unknown/missing kwarg. ValueError: a known strategy
        # constructor rejecting an out-of-range value (e.g. half_spread <= 0).
        raise InvalidSimulationRequest(f"Invalid strategy_params for {strategy_name!r}: {e}") from e


def run_simulation(db: Session, request: SimulateRequest) -> SimulationRun:
    if request.latency_preset not in LATENCY_PRESETS:
        raise InvalidSimulationRequest(f"Unknown latency_preset {request.latency_preset!r}; choose from {LATENCY_PRESETS}")
    if request.data_source != "synthetic":
        raise InvalidSimulationRequest("Only data_source='synthetic' can be simulated on demand; Binance requires a prior capture on disk")

    strategy = build_strategy(request.strategy_name, request.strategy_params, request.tick_size)

    run = SimulationRun(
        strategy_name=request.strategy_name,
        strategy_params=request.strategy_params,
        latency_preset=request.latency_preset,
        data_source=request.data_source,
        session_seconds=request.session_seconds,
        seed=request.seed,
        status="running",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        events = generate_session(
            SyntheticLOBConfig(session_seconds=request.session_seconds, seed=request.seed, tick_size=request.tick_size)
        )
        latency_model = None if request.latency_preset == "0ms" else make_latency_model(request.latency_preset)
        result: BacktestResult = run_backtest(
            events,
            strategy,
            tick_size=request.tick_size,
            record_levels=request.record_levels,
            strategy_latency_model=latency_model,
        )
        stats = summarize(result)

        for field in (
            "n_fills", "maker_fills", "taker_fills", "final_inventory", "final_pnl",
            "inventory_mean_abs", "inventory_std", "equity_std", "adverse_selection_cost", "sharpe_ratio",
        ):
            value = stats[field]
            setattr(run, field, None if (isinstance(value, float) and math.isnan(value)) else value)
        run.status = "completed"

        _persist_book_snapshots(db, run.id, result, request.record_levels)

    except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure should mark the run failed, not 500 silently
        run.status = "failed"
        run.error_message = str(e)

    db.commit()
    db.refresh(run)
    return run


def _persist_book_snapshots(db: Session, run_id: int, result: BacktestResult, record_levels: int) -> None:
    # book_snapshots and portfolio_history are row-aligned by construction
    # in run_backtest (appended together in the same loop iteration), so
    # zip by position -- not a time-based join, which could misbehave if
    # two events ever land on the exact same float timestamp.
    book_rows = result.book_snapshots.to_dict("records")
    portfolio_rows = result.portfolio_history.to_dict("records")

    for book_row, portfolio_row in zip(book_rows, portfolio_rows):
        bids = [
            [book_row[f"bid_price_{lvl}"], book_row[f"bid_size_{lvl}"]]
            for lvl in range(1, record_levels + 1)
            if not math.isnan(book_row[f"bid_price_{lvl}"])
        ]
        asks = [
            [book_row[f"ask_price_{lvl}"], book_row[f"ask_size_{lvl}"]]
            for lvl in range(1, record_levels + 1)
            if not math.isnan(book_row[f"ask_price_{lvl}"])
        ]
        db.add(
            BookSnapshotRow(
                run_id=run_id,
                time=book_row["time"],
                bids=bids,
                asks=asks,
                inventory=portfolio_row["inventory"],
                cash=portfolio_row["cash"],
                equity=None if math.isnan(portfolio_row["equity"]) else portfolio_row["equity"],
            )
        )
    db.commit()
