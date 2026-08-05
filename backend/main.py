"""FastAPI backend: four endpoints over the same database the (future)
Bokeh dashboard reads directly.

- POST /simulate          trigger a new backtest run with chosen parameters
- GET  /book/state/{id}   book snapshots for a completed run
- GET  /strategies/compare  the precomputed strategy comparison table
- GET  /rl/training       RL training curve data

No streaming infrastructure: this is a backtest/simulation system, not a
live feed, so a request/response API is all it needs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db import get_db, init_db
from backend.db_models import BookSnapshotRow, ComparisonResult, SimulationRun, TrainingRunRecord
from backend.schemas import (
    BookSnapshotOut,
    BookStateResponse,
    ComparisonResponse,
    ComparisonRow,
    SimulateRequest,
    SimulateResponse,
    SummaryStats,
    TrainingResponse,
    TrainingRunOut,
)
from backend.simulate import InvalidSimulationRequest, run_simulation


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="BookMaker API", description="Limit order book market-making simulator API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness check for the deployment platform -- deliberately doesn't
    touch the database, so a slow/cold DB connection doesn't get read as
    the process being down.
    """
    return {"status": "ok"}


@app.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest, db: Session = Depends(get_db)) -> SimulateResponse:
    try:
        run = run_simulation(db, request)
    except InvalidSimulationRequest as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    summary = SummaryStats.model_validate(run) if run.status == "completed" else None
    return SimulateResponse(
        run_id=run.id,
        status=run.status,
        strategy_name=run.strategy_name,
        latency_preset=run.latency_preset,
        data_source=run.data_source,
        summary=summary,
        error_message=run.error_message,
    )


@app.get("/book/state/{run_id}", response_model=BookStateResponse)
def book_state(
    run_id: int,
    at_time: float | None = Query(None, description="If given, return only the snapshot at/just before this time"),
    db: Session = Depends(get_db),
) -> BookStateResponse:
    run = db.get(SimulationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No simulation run with id {run_id}")

    query = db.query(BookSnapshotRow).filter(BookSnapshotRow.run_id == run_id).order_by(BookSnapshotRow.time)
    if at_time is not None:
        snapshot = query.filter(BookSnapshotRow.time <= at_time).order_by(BookSnapshotRow.time.desc()).first()
        snapshots = [snapshot] if snapshot is not None else []
    else:
        snapshots = query.all()

    return BookStateResponse(
        run_id=run_id,
        n_snapshots=len(snapshots),
        snapshots=[BookSnapshotOut.model_validate(s) for s in snapshots],
    )


@app.get("/strategies/compare", response_model=ComparisonResponse)
def strategies_compare(
    data_source: str | None = Query(None, description="Filter to one data source, e.g. 'synthetic' or 'binance'"),
    db: Session = Depends(get_db),
) -> ComparisonResponse:
    query = db.query(ComparisonResult)
    if data_source is not None:
        query = query.filter(ComparisonResult.data_source == data_source)
    rows = query.order_by(ComparisonResult.strategy_name, ComparisonResult.latency_preset).all()
    return ComparisonResponse(rows=[ComparisonRow.model_validate(r) for r in rows])


@app.get("/rl/training", response_model=TrainingResponse)
def rl_training(
    label: str | None = Query(None, description="Filter to one policy label, e.g. 'latency_naive'"),
    db: Session = Depends(get_db),
) -> TrainingResponse:
    query = db.query(TrainingRunRecord)
    if label is not None:
        query = query.filter(TrainingRunRecord.label == label)
    runs = query.order_by(TrainingRunRecord.created_at.desc()).all()
    return TrainingResponse(runs=[TrainingRunOut.model_validate(r) for r in runs])
