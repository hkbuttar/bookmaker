"""Database engine/session setup.

SQLite by default (zero external service needed for local development or
tests), configurable via the DATABASE_URL env var so the exact same models
point at Postgres in deployment. The schema here (backend/db_models.py) is
a public integration point, not private backend implementation: the
dashboard queries it directly rather than round-tripping through this
API's own endpoints, so it stays queryable on its own.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./bookmaker.db")

# Render (like Heroku before it) hands out Postgres URLs with the
# `postgres://` scheme, which SQLAlchemy's psycopg2 dialect no longer
# accepts -- normalize to `postgresql://` so the same env var works
# whether it was typed by hand or copied from the dashboard.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_is_sqlite = DATABASE_URL.startswith("sqlite")
# check_same_thread=False: FastAPI can hand a request to a different
# thread than the one that opened the connection; only relevant for
# SQLite, which is otherwise single-connection-thread-affine by default.
# pool_pre_ping: Postgres deployments (Render's included) recycle idle
# connections server-side; without this, the first query after a lull
# gets a stale connection and fails instead of transparently reconnecting.
_connect_args = {"check_same_thread": False} if _is_sqlite else {}
_engine_kwargs = {} if _is_sqlite else {"pool_pre_ping": True}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't already exist. Idempotent."""
    from backend import db_models  # noqa: F401  (registers models on Base.metadata)

    Base.metadata.create_all(bind=engine)
