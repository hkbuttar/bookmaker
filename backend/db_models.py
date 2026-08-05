"""SQLAlchemy ORM models. This is the schema the (future) dashboard reads
directly against the same database this backend writes to -- no separate
JSON API layer needed for the dashboard itself, per the frontend plan --
so table/column shapes here are a real integration contract, not private
implementation detail to be refactored freely later.
"""

from __future__ import annotations

import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String, nullable=False)
    strategy_params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    latency_preset: Mapped[str] = mapped_column(String, nullable=False, default="0ms")
    data_source: Mapped[str] = mapped_column(String, nullable=False, default="synthetic")
    session_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    # Flattened backtest.metrics.summarize() output -- nullable until the
    # run completes (status starts "running").
    n_fills: Mapped[int | None] = mapped_column(Integer, nullable=True)
    maker_fills: Mapped[int | None] = mapped_column(Integer, nullable=True)
    taker_fills: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_inventory: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    inventory_mean_abs: Mapped[float | None] = mapped_column(Float, nullable=True)
    inventory_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    equity_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    adverse_selection_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    book_snapshots: Mapped[list["BookSnapshotRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class BookSnapshotRow(Base):
    __tablename__ = "book_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("simulation_runs.id"), nullable=False, index=True)
    time: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    # [[price, size], ...] best-first, per side -- matches
    # lob.order_book.OrderBook.depth()'s shape directly, and supports
    # however many levels a run recorded without a fixed-column schema.
    bids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    asks: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    inventory: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash: Mapped[float | None] = mapped_column(Float, nullable=True)
    equity: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped["SimulationRun"] = relationship(back_populates="book_snapshots")


class ComparisonResult(Base):
    """One row per (strategy, latency, data_source) cell of the strategy
    comparison table -- populated by backend/populate.py, not computed
    live on request (a full sweep takes minutes, not response-time).
    """

    __tablename__ = "comparison_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String, nullable=False)
    latency_preset: Mapped[str] = mapped_column(String, nullable=False)
    data_source: Mapped[str] = mapped_column(String, nullable=False, default="synthetic")
    n_fills: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    inventory_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    adverse_selection_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


class TrainingRunRecord(Base):
    """One row per trained RL policy -- populated by backend/populate.py
    from rl.train.train_dqn's output.
    """

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String, nullable=False)
    total_timesteps: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    episodes: Mapped[list["TrainingEpisode"]] = relationship(
        back_populates="training_run", cascade="all, delete-orphan"
    )


class TrainingEpisode(Base):
    __tablename__ = "training_episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(
        ForeignKey("training_runs.id"), nullable=False, index=True
    )
    episode_index: Mapped[int] = mapped_column(Integer, nullable=False)
    reward: Mapped[float] = mapped_column(Float, nullable=False)
    inventory_std: Mapped[float] = mapped_column(Float, nullable=False)

    training_run: Mapped["TrainingRunRecord"] = relationship(back_populates="episodes")
