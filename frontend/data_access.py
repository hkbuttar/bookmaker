"""Read-only query helpers against the backend's own database.

This dashboard talks directly to the same store `backend/` writes to (see
backend/db_models.py's docstring) -- no separate JSON API layer for the
dashboard itself, per the plan's Bokeh architecture decision. Every
function here opens its own short-lived session and returns plain
data (dicts/lists), not ORM objects or a held-open session, since Bokeh's
periodic callbacks re-query on a timer rather than holding a live
connection across the app's lifetime.
"""

from __future__ import annotations

import pandas as pd

from backend.db import SessionLocal
from backend.db_models import BookSnapshotRow, ComparisonResult, SimulationRun, TrainingRunRecord


def list_simulation_runs() -> pd.DataFrame:
    with SessionLocal() as db:
        runs = db.query(SimulationRun).filter(SimulationRun.status == "completed").order_by(SimulationRun.id.desc()).all()
        return pd.DataFrame(
            [
                {
                    "id": r.id,
                    "label": f"#{r.id} {r.strategy_name} / {r.latency_preset} (seed={r.seed})",
                    "strategy_name": r.strategy_name,
                    "latency_preset": r.latency_preset,
                    "seed": r.seed,
                }
                for r in runs
            ]
        )


def get_book_snapshots(run_id: int) -> pd.DataFrame:
    with SessionLocal() as db:
        rows = (
            db.query(BookSnapshotRow)
            .filter(BookSnapshotRow.run_id == run_id)
            .order_by(BookSnapshotRow.time)
            .all()
        )
        return pd.DataFrame(
            [{"time": r.time, "bids": r.bids, "asks": r.asks, "inventory": r.inventory, "equity": r.equity} for r in rows]
        )


def get_comparison_table(data_source: str = "synthetic") -> pd.DataFrame:
    with SessionLocal() as db:
        rows = db.query(ComparisonResult).filter(ComparisonResult.data_source == data_source).all()
        return pd.DataFrame(
            [
                {
                    "strategy_name": r.strategy_name,
                    "latency_preset": r.latency_preset,
                    "n_fills": r.n_fills,
                    "final_pnl": r.final_pnl,
                    "sharpe_ratio": r.sharpe_ratio,
                    "inventory_std": r.inventory_std,
                    "adverse_selection_cost": r.adverse_selection_cost,
                }
                for r in rows
            ]
        )


def get_training_runs() -> dict[str, pd.DataFrame]:
    """One DataFrame of (episode_index, reward, inventory_std) per label."""
    with SessionLocal() as db:
        records = db.query(TrainingRunRecord).all()
        out: dict[str, pd.DataFrame] = {}
        for record in records:
            out[record.label] = pd.DataFrame(
                [
                    {"episode_index": e.episode_index, "reward": e.reward, "inventory_std": e.inventory_std}
                    for e in sorted(record.episodes, key=lambda e: e.episode_index)
                ]
            )
        return out
