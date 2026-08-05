import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import Base, get_db
from backend.db_models import ComparisonResult, TrainingEpisode, TrainingRunRecord
from backend.main import app


@pytest.fixture()
def client(tmp_path):
    # A fresh file-based SQLite DB per test, wired in via FastAPI's
    # dependency-override mechanism -- the app's own lifespan still calls
    # init_db() against the default DATABASE_URL on startup (a harmless
    # side effect: it may create an empty ./bookmaker.db in the working
    # directory), but every request in this test goes through the
    # override below, not that default engine.
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        test_client.session_factory = TestingSessionLocal
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


def _simulate_payload(**overrides):
    payload = dict(
        strategy_name="naive",
        strategy_params={"half_spread": 0.05, "quote_size": 10},
        latency_preset="0ms",
        data_source="synthetic",
        session_seconds=30.0,
        seed=1,
        record_levels=3,
        tick_size=0.01,
    )
    payload.update(overrides)
    return payload


def test_simulate_creates_completed_run(client):
    response = client.post("/simulate", json=_simulate_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["strategy_name"] == "naive"
    assert body["summary"] is not None
    assert body["summary"]["n_fills"] is not None
    assert isinstance(body["run_id"], int)


def test_simulate_unknown_strategy_name_rejected(client):
    response = client.post("/simulate", json=_simulate_payload(strategy_name="not_a_strategy"))
    assert response.status_code == 422


def test_simulate_unknown_latency_preset_rejected(client):
    response = client.post("/simulate", json=_simulate_payload(latency_preset="100ms"))
    assert response.status_code == 422


def test_simulate_binance_data_source_rejected(client):
    response = client.post("/simulate", json=_simulate_payload(data_source="binance"))
    assert response.status_code == 422


def test_simulate_invalid_strategy_params_rejected(client):
    response = client.post(
        "/simulate", json=_simulate_payload(strategy_params={"half_spread": -1.0, "quote_size": 10})
    )
    assert response.status_code == 422


def test_simulate_session_seconds_over_cap_rejected(client):
    response = client.post("/simulate", json=_simulate_payload(session_seconds=10_000.0))
    assert response.status_code == 422


def test_book_state_after_simulate_has_snapshots(client):
    run_id = client.post("/simulate", json=_simulate_payload()).json()["run_id"]

    response = client.get(f"/book/state/{run_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run_id
    assert body["n_snapshots"] > 0
    first = body["snapshots"][0]
    assert "time" in first and "bids" in first and "asks" in first


def test_book_state_unknown_run_id_404s(client):
    response = client.get("/book/state/999999")
    assert response.status_code == 404


def test_book_state_at_time_returns_single_snapshot(client):
    run_id = client.post("/simulate", json=_simulate_payload()).json()["run_id"]
    full = client.get(f"/book/state/{run_id}").json()
    mid_time = full["snapshots"][len(full["snapshots"]) // 2]["time"]

    response = client.get(f"/book/state/{run_id}", params={"at_time": mid_time})
    body = response.json()
    assert body["n_snapshots"] == 1
    assert body["snapshots"][0]["time"] <= mid_time


def test_strategies_compare_empty_when_unpopulated(client):
    response = client.get("/strategies/compare")
    assert response.status_code == 200
    assert response.json()["rows"] == []


def test_strategies_compare_returns_populated_rows(client):
    db = client.session_factory()
    db.add(
        ComparisonResult(
            strategy_name="naive", latency_preset="0ms", data_source="synthetic",
            n_fills=10, final_pnl=1.5, sharpe_ratio=0.01, inventory_std=5.0, adverse_selection_cost=-0.01,
        )
    )
    db.commit()
    db.close()

    response = client.get("/strategies/compare")
    rows = response.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["strategy_name"] == "naive"
    assert rows[0]["final_pnl"] == 1.5


def test_strategies_compare_filters_by_data_source(client):
    db = client.session_factory()
    db.add(ComparisonResult(strategy_name="naive", latency_preset="0ms", data_source="synthetic"))
    db.add(ComparisonResult(strategy_name="naive", latency_preset="0ms", data_source="binance"))
    db.commit()
    db.close()

    response = client.get("/strategies/compare", params={"data_source": "binance"})
    rows = response.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["data_source"] == "binance"


def test_rl_training_empty_when_unpopulated(client):
    response = client.get("/rl/training")
    assert response.status_code == 200
    assert response.json()["runs"] == []


def test_rl_training_returns_populated_episodes(client):
    db = client.session_factory()
    record = TrainingRunRecord(label="latency_naive", total_timesteps=300_000)
    db.add(record)
    db.flush()
    db.add(TrainingEpisode(training_run_id=record.id, episode_index=0, reward=-10.0, inventory_std=5.0))
    db.add(TrainingEpisode(training_run_id=record.id, episode_index=1, reward=-5.0, inventory_std=3.0))
    db.commit()
    db.close()

    response = client.get("/rl/training")
    runs = response.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["label"] == "latency_naive"
    assert len(runs[0]["episodes"]) == 2


def test_rl_training_filters_by_label(client):
    db = client.session_factory()
    db.add(TrainingRunRecord(label="latency_naive", total_timesteps=1))
    db.add(TrainingRunRecord(label="latency_aware", total_timesteps=1))
    db.commit()
    db.close()

    response = client.get("/rl/training", params={"label": "latency_aware"})
    runs = response.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["label"] == "latency_aware"
