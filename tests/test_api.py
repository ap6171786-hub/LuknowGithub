"""API integration tests for the persisted dataset and model workflow."""

from __future__ import annotations

import asyncio

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes
from app.api.server import app
from app.db.database import Base, get_db


def test_dataset_upload_train_predict_and_delete(monkeypatch) -> None:
    """Exercise dataset persistence and the model lifecycle over the HTTP API."""
    monkeypatch.setenv("INSIGHT_API_KEY", "test-api-key-that-is-at-least-32-chars")
    openapi = app.openapi()
    assert "ApiKeyAuth" in openapi["components"]["securitySchemes"]
    assert openapi["paths"]["/datasets"]["get"]["security"] == [{"ApiKeyAuth": []}]
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    test_sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        """Yield an isolated in-memory database session."""
        session = test_sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    csv_rows = ["size,group,target"]
    csv_rows.extend(
        f"{index},{'a' if index % 2 else 'b'},{index * 2 + 1}"
        for index in range(30)
    )
    content = ("\n".join(csv_rows) + "\n").encode()

    async def exercise_api() -> None:
        """Run requests using HTTPX's in-process ASGI transport."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"X-API-Key": "test-api-key-that-is-at-least-32-chars"},
        ) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            preflight = await client.options(
                "/health",
                headers={
                    "Origin": "http://localhost:8501",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert preflight.headers["access-control-allow-origin"] == (
                "http://localhost:8501"
            )

            monkeypatch.delenv("INSIGHT_API_KEY")
            unconfigured = await client.get("/datasets")
            assert unconfigured.status_code == 503
            monkeypatch.setenv(
                "INSIGHT_API_KEY", "test-api-key-that-is-at-least-32-chars"
            )
            unauthenticated = await client.get(
                "/datasets", headers={"X-API-Key": "wrong-key"}
            )
            assert unauthenticated.status_code == 401
            unauthenticated_upload = await client.post(
                "/datasets/upload",
                files={"file": ("sample.csv", content, "text/csv")},
                headers={"X-API-Key": "wrong-key"},
            )
            assert unauthenticated_upload.status_code == 401

            monkeypatch.setattr(routes, "MAX_UPLOAD_SIZE_BYTES", 8)
            oversized = await client.post(
                "/datasets/upload",
                files={"file": ("too-large.csv", b"x" * 9, "text/csv")},
            )
            assert oversized.status_code == 413
            monkeypatch.setattr(routes, "MAX_UPLOAD_SIZE_BYTES", 25 * 1024 * 1024)

            uploaded = await client.post(
                "/datasets/upload",
                files={"file": ("sample.csv", content, "text/csv")},
            )
            assert uploaded.status_code == 201, uploaded.text
            dataset_id = uploaded.json()["id"]
            assert uploaded.json()["profile"]["row_count"] == 30

            listed = await client.get("/datasets?limit=10&offset=0")
            assert listed.status_code == 200
            assert listed.json()[0]["id"] == dataset_id

            detail = await client.get(f"/datasets/{dataset_id}")
            assert detail.status_code == 200
            assert detail.json()["profile"]["column_count"] == 3

            trained = await client.post(
                f"/datasets/{dataset_id}/train",
                json={
                    "target": "target",
                    "task_type": "regression",
                    "algorithm": "ridge",
                },
            )
            assert trained.status_code == 201, trained.text
            model_id = trained.json()["model_run_id"]

            model = await client.get(f"/models/{model_id}")
            assert model.status_code == 200
            assert "mae" in model.json()["metrics"]

            prediction = await client.post(
                f"/models/{model_id}/predict",
                json={"features": {"size": 5, "group": "a"}},
            )
            assert prediction.status_code == 200, prediction.text
            assert prediction.json()["prediction"] is not None

            deleted = await client.delete(f"/datasets/{dataset_id}")
            assert deleted.status_code == 204
            missing = await client.get(f"/datasets/{dataset_id}")
            assert missing.status_code == 404

    try:
        asyncio.run(exercise_api())
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
