"""Tests for database initialization and legacy schema migration."""

from __future__ import annotations

import json

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database
from app.db.models import Dataset, Profile


def test_init_db_migrates_legacy_dataset_records(monkeypatch) -> None:
    """Legacy dataset metadata and profile JSON survive the schema upgrade."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_sessions = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    with test_engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE datasets (
                    id INTEGER PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    uploaded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    profile_json TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO datasets "
                "(id, filename, row_count, column_count, profile_json) "
                "VALUES (1, 'legacy.csv', 8, 2, :profile)"
            ),
            {"profile": json.dumps({"row_count": 8, "columns": []})},
        )
    monkeypatch.setattr(database, "DATABASE_URL", "sqlite://")
    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(database, "SessionLocal", test_sessions)

    database.init_db()

    with test_sessions() as session:
        dataset = session.get(Dataset, 1)
        assert dataset is not None
        assert dataset.name == "legacy.csv"
        assert dataset.row_count == 8
        profile = session.query(Profile).filter_by(dataset_id=1).one()
        assert profile.profile_json["row_count"] == 8
    with test_engine.connect() as connection:
        assert connection.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'datasets_legacy'"
            )
        ).first() is None
    test_engine.dispose()
