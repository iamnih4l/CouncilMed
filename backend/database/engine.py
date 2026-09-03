"""
CouncilMed — Database Engine
================================
Async SQLite database with session management and initialization.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import Base

logger = logging.getLogger(__name__)

from ..paths import get_base_dir

# Default database path
DEFAULT_DB_PATH = get_base_dir() / "data" / "councilmed.db"


def get_database_url(db_path: str | Path | None = None) -> str:
    """Get the async SQLite database URL."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{path}"


# Create engine and session factory
_engine = None
_session_factory = None


async def init_database(db_path: str | Path | None = None) -> None:
    """Initialize the database engine and create tables."""
    global _engine, _session_factory
    
    url = get_database_url(db_path)
    logger.info("Initializing database: %s", url)
    
    _engine = create_async_engine(
        url,
        echo=False,
        pool_pre_ping=True,
    )
    
    # Enable WAL mode for better concurrent read performance
    @event.listens_for(_engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()
    
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Create all tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database initialized successfully")


async def get_session() -> AsyncSession:
    """Get a database session."""
    if _session_factory is None:
        await init_database()
    return _session_factory()


async def close_database() -> None:
    """Close the database engine."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Database closed")
