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
import time

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
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
# connect_timeout: without it, a connection attempt to an unreachable
# host hangs on the OS-level TCP retransmission timeout (a minute or
# more), which defeats init_db()'s retry loop below -- each attempt
# needs to fail fast so the *loop's* backoff schedule is what controls
# total wait time, not the kernel's.
# pool_pre_ping: Postgres deployments (Render's included) recycle idle
# connections server-side; without this, the first query after a lull
# gets a stale connection and fails instead of transparently reconnecting.
_connect_args = {"check_same_thread": False} if _is_sqlite else {"connect_timeout": 5}
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


def init_db(max_attempts: int = 6, initial_delay: float = 1.0) -> None:
    """Create all tables if they don't already exist. Idempotent.

    Retries with exponential backoff on OperationalError: on a fresh
    Render Blueprint deploy, the web service and its Postgres database
    are provisioned together, and the database isn't always done coming
    up (or its DNS/networking hasn't settled) by the time this runs at
    app startup -- a bare first attempt can lose that race and take the
    whole deploy down over what's really just a few seconds of "not
    ready yet." ~1+2+4+8+16 = 31s of total backoff comfortably covers
    that without masking a genuinely unreachable database forever.
    """
    from backend import db_models  # noqa: F401  (registers models on Base.metadata)

    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError:
            if attempt == max_attempts:
                raise
            time.sleep(delay)
            delay *= 2
