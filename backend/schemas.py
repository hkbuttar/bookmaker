"""Pydantic request/response schemas for the API -- kept separate from
backend/db_models.py's SQLAlchemy ORM models on purpose: the API's shape
(what a client sends/receives) and the storage schema (what the dashboard
queries directly) are allowed to diverge, even though today they're
close, so a future change to one doesn't force a breaking change to the
other.
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, Field

STRATEGY_NAMES = ("naive", "inventory_aware", "adverse_selection_aware")
LATENCY_PRESETS = ("0ms", "5ms", "20ms", "50ms")
DATA_SOURCES = ("synthetic",)  # Binance simulate-on-demand isn't supported (needs a prior capture on disk)


class SimulateRequest(BaseModel):
    strategy_name: str = Field(..., description=f"One of {STRATEGY_NAMES}")
    strategy_params: dict = Field(
        default_factory=dict,
        description="Strategy constructor kwargs beyond tick_size, e.g. half_spread, quote_size, "
        "inventory_penalty, imbalance_ema_alpha, widen_multiplier, pull_threshold.",
    )
    latency_preset: str = Field("0ms", description=f"One of {LATENCY_PRESETS}")
    data_source: str = Field("synthetic", description=f"One of {DATA_SOURCES}")
    session_seconds: float = Field(600.0, gt=0, le=3600.0, description="Capped to keep API calls responsive")
    seed: int | None = Field(None, description="Synthetic data generator seed; None picks a random one")
    record_levels: int = Field(5, ge=1, le=20)
    tick_size: float = Field(0.01, gt=0)


class SummaryStats(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    n_fills: int | None
    maker_fills: int | None
    taker_fills: int | None
    final_inventory: float | None
    final_pnl: float | None
    inventory_mean_abs: float | None
    inventory_std: float | None
    equity_std: float | None
    adverse_selection_cost: float | None
    sharpe_ratio: float | None


class SimulateResponse(BaseModel):
    run_id: int
    status: str
    strategy_name: str
    latency_preset: str
    data_source: str
    summary: SummaryStats | None
    error_message: str | None = None


class BookSnapshotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    time: float
    bids: list[list[float]]
    asks: list[list[float]]
    inventory: float | None
    cash: float | None
    equity: float | None


class BookStateResponse(BaseModel):
    run_id: int
    n_snapshots: int
    snapshots: list[BookSnapshotOut]


class ComparisonRow(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    strategy_name: str
    latency_preset: str
    data_source: str
    n_fills: int | None
    final_pnl: float | None
    sharpe_ratio: float | None
    inventory_std: float | None
    adverse_selection_cost: float | None
    computed_at: datetime.datetime


class ComparisonResponse(BaseModel):
    rows: list[ComparisonRow]


class TrainingEpisodeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    episode_index: int
    reward: float
    inventory_std: float


class TrainingRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    label: str
    total_timesteps: int
    created_at: datetime.datetime
    episodes: list[TrainingEpisodeOut]


class TrainingResponse(BaseModel):
    runs: list[TrainingRunOut]
