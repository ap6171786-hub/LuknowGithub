"""SQLite engine, SQLAlchemy session management, and initialization."""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import datetime
import json
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""


def _database_url() -> str:
    """Read the configured database URL and prepare a local SQLite directory."""
    url = os.getenv("DATABASE_URL", "sqlite:///./data/insight_engine.db")
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        database_path = Path(url.removeprefix("sqlite:///"))
        if database_path.parent != Path("."):
            database_path.parent.mkdir(parents=True, exist_ok=True)
    return url


DATABASE_URL = _database_url()
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped database session and always close it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create database tables that do not yet exist."""
    from app.db import models

    _ = models
    legacy_rows: list[dict[str, object]] = []
    if DATABASE_URL.startswith("sqlite"):
        legacy_rows = _migrate_legacy_dataset_table()
    Base.metadata.create_all(bind=engine)
    if legacy_rows:
        _restore_legacy_datasets(legacy_rows)


def _migrate_legacy_dataset_table() -> list[dict[str, object]]:
    """Preserve starter-API dataset rows while replacing its incompatible table."""
    inspector = inspect(engine)
    if "datasets" not in inspector.get_table_names():
        return []
    columns = {column["name"] for column in inspector.get_columns("datasets")}
    if not {"filename", "profile_json"} <= columns:
        return []
    with engine.connect() as connection:
        result = connection.execute(
            text(
                "SELECT id, filename, uploaded_at, row_count, column_count, profile_json "
                "FROM datasets"
            )
        )
        rows = [dict(row._mapping) for row in result]
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE datasets RENAME TO datasets_legacy"))
    return rows


def _restore_legacy_datasets(rows: list[dict[str, object]]) -> None:
    """Copy legacy metadata and profiles to the normalized dataset tables."""
    from app.db.models import Dataset, Profile

    with SessionLocal() as session:
        for row in rows:
            raw_date = row.get("uploaded_at")
            uploaded_at = (
                datetime.fromisoformat(str(raw_date))
                if raw_date is not None
                else datetime.now()
            )
            try:
                old_profile = json.loads(str(row.get("profile_json") or "{}"))
            except json.JSONDecodeError:
                old_profile = {"legacy_profile": str(row.get("profile_json") or "")}
            dataset = Dataset(
                id=int(row["id"]),
                name=str(row.get("filename") or "legacy-dataset"),
                uploaded_at=uploaded_at,
                row_count=int(row.get("row_count") or 0),
                column_count=int(row.get("column_count") or 0),
                file_path="",
                owner="local",
            )
            dataset.profiles.append(Profile(profile_json=old_profile))
            session.add(dataset)
        session.commit()
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE datasets_legacy"))
