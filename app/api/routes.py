"""Dataset, model training, prediction, and health endpoints."""

from __future__ import annotations

import io
import logging
import os
import uuid
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.schemas import (
    DatasetDetail,
    DatasetSummary,
    ModelDetail,
    PredictRequest,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)
from app.api.security import require_api_key
from app.core.ingest import DataIngestor, IngestionError
from app.core.model import AutoMLPipeline
from app.core.profile import DataProfiler
from app.db.database import get_db
from app.db.models import Dataset, ModelRun, Prediction, Profile

logger = logging.getLogger(__name__)
router = APIRouter()
protected_router = APIRouter(dependencies=[Depends(require_api_key)])
DatabaseSession = Annotated[Session, Depends(get_db)]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "uploads"
MODEL_DIRECTORY = PROJECT_ROOT / "data" / "models"
ALLOWED_SUFFIXES = {".csv", ".xls", ".xlsx", ".xlsm", ".xlsb", ".json", ".jsonl", ".ndjson", ".parquet", ".pq"}
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25")) * 1024 * 1024


@router.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Return a basic liveness response."""
    return {"status": "ok"}


@protected_router.post(
    "/datasets/upload",
    response_model=DatasetDetail,
    status_code=status.HTTP_201_CREATED,
    tags=["datasets"],
)
async def upload_dataset(
    db: DatabaseSession,
    file: Annotated[UploadFile, File(description="CSV, Excel, JSON, or Parquet dataset")],
) -> DatasetDetail:
    """Persist an uploaded dataset and its generated profile."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload CSV, Excel, JSON, or Parquet.",
        )

    content_buffer = bytearray()
    while chunk := await file.read(UPLOAD_CHUNK_SIZE):
        content_buffer.extend(chunk)
        if len(content_buffer) > MAX_UPLOAD_SIZE_BYTES:
            await file.close()
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    "Upload exceeds the configured limit of "
                    f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
                ),
            )
    content = bytes(content_buffer)
    safe_name = Path(file.filename).name
    destination = UPLOAD_DIRECTORY / f"{uuid.uuid4().hex}_{safe_name}"
    try:
        dataframe, profile_data = await run_in_threadpool(
            _load_and_profile, content, suffix
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    try:
        dataset = Dataset(
            name=safe_name,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            file_path=str(destination),
            owner="local",
        )
        dataset.profiles.append(Profile(profile_json=profile_data))
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
    except SQLAlchemyError as exc:
        db.rollback()
        destination.unlink(missing_ok=True)
        logger.exception("Failed to persist uploaded dataset")
        raise HTTPException(status_code=500, detail="Could not save dataset metadata.") from exc

    logger.info("Uploaded dataset id=%s name=%s", dataset.id, dataset.name)
    return DatasetDetail.model_validate(
        {
            **DatasetSummary.model_validate(dataset).model_dump(),
            "profile": profile_data,
        }
    )


def _load_and_profile(content: bytes, file_type: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse an upload and compute its profile off the async event loop."""
    dataframe = DataIngestor().load_file(io.BytesIO(content), file_type)
    return dataframe, DataProfiler(dataframe).profile().to_dict()


@protected_router.get("/datasets", response_model=list[DatasetSummary], tags=["datasets"])
def list_datasets(
    db: DatabaseSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DatasetSummary]:
    """Return datasets ordered newest first with limit/offset pagination."""
    datasets = db.scalars(
        select(Dataset).order_by(Dataset.uploaded_at.desc(), Dataset.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [DatasetSummary.model_validate(dataset) for dataset in datasets]


@protected_router.get("/datasets/{dataset_id}", response_model=DatasetDetail, tags=["datasets"])
def get_dataset(dataset_id: int, db: DatabaseSession) -> DatasetDetail:
    """Return one dataset and its latest profile."""
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    latest_profile = db.scalar(
        select(Profile)
        .where(Profile.dataset_id == dataset_id)
        .order_by(Profile.created_at.desc(), Profile.id.desc())
        .limit(1)
    )
    return DatasetDetail(
        **DatasetSummary.model_validate(dataset).model_dump(),
        profile=latest_profile.profile_json if latest_profile else None,
    )


@protected_router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["datasets"])
def delete_dataset(dataset_id: int, db: DatabaseSession) -> None:
    """Delete a dataset and associated database records and stored artifacts."""
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    paths = [Path(dataset.file_path)]
    paths.extend(Path(run.model_path) for run in dataset.model_runs)
    try:
        db.delete(dataset)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to delete dataset id=%s", dataset_id)
        raise HTTPException(status_code=500, detail="Could not delete dataset.") from exc
    for path in paths:
        if path.is_file() and path.resolve().is_relative_to(PROJECT_ROOT):
            path.unlink(missing_ok=True)


@protected_router.post(
    "/datasets/{dataset_id}/train",
    response_model=TrainResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["models"],
)
def train_dataset(
    dataset_id: int, request: TrainRequest, db: DatabaseSession
) -> TrainResponse:
    """Train and persist a model run for a saved dataset."""
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        dataframe = DataIngestor().load_file(
            dataset.file_path, Path(dataset.file_path).suffix
        )
        result = AutoMLPipeline().train(
            dataframe,
            request.target,
            task_type=request.task_type,
            algorithm=request.algorithm,
        )
    except (IngestionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIRECTORY / f"model_{uuid.uuid4().hex}.joblib"
    AutoMLPipeline.save(result.pipeline, model_path)
    metrics_json: dict[str, Any] = {
        "metrics": result.metrics,
        "feature_importances": result.feature_importances,
        "confusion_matrix": result.confusion_matrix,
        "class_labels": result.class_labels,
        "residuals": result.residuals,
        "actual_values": result.actual_values,
        "predicted_values": result.predicted_values,
        "roc_curve": result.roc_curve,
        "cv_scores": result.cv_scores,
        "feature_columns": result.feature_columns,
    }
    run = ModelRun(
        dataset_id=dataset_id,
        target_column=request.target,
        task_type=result.task,
        algorithm=result.algorithm,
        metrics_json=metrics_json,
        model_path=str(model_path),
        status="completed",
    )
    try:
        db.add(run)
        db.commit()
        db.refresh(run)
    except SQLAlchemyError as exc:
        db.rollback()
        model_path.unlink(missing_ok=True)
        logger.exception("Failed to persist model run for dataset id=%s", dataset_id)
        raise HTTPException(status_code=500, detail="Could not save model run.") from exc
    return TrainResponse(
        model_run_id=run.id,
        algorithm=run.algorithm,
        task_type=run.task_type,
        metrics=result.metrics,
    )


@protected_router.get("/models/{model_id}", response_model=ModelDetail, tags=["models"])
def get_model(model_id: int, db: DatabaseSession) -> ModelDetail:
    """Return model metrics and transformed feature importance values."""
    run = db.get(ModelRun, model_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Model run not found.")
    stored = run.metrics_json
    return ModelDetail(
        id=run.id,
        dataset_id=run.dataset_id,
        target_column=run.target_column,
        task_type=run.task_type,
        algorithm=run.algorithm,
        status=run.status,
        metrics=stored.get("metrics", {}),
        feature_importances=stored.get("feature_importances", {}),
    )


@protected_router.post(
    "/models/{model_id}/predict",
    response_model=PredictionResponse,
    tags=["models"],
)
def predict(
    model_id: int, request: PredictRequest, db: DatabaseSession
) -> PredictionResponse:
    """Generate and persist a single-record prediction."""
    run = db.get(ModelRun, model_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Model run not found.")
    if run.status != "completed" or not Path(run.model_path).is_file():
        raise HTTPException(status_code=409, detail="The model artifact is unavailable.")
    try:
        model = AutoMLPipeline.load(run.model_path)
        result = AutoMLPipeline.predict(model, request.features)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.exception("Prediction failed for model id=%s", model_id)
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc
    prediction = Prediction(
        model_run_id=run.id,
        input_json=request.features,
        output_json=result,
    )
    try:
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to persist prediction for model id=%s", model_id)
        raise HTTPException(status_code=500, detail="Could not save prediction.") from exc
    return PredictionResponse(
        prediction_id=prediction.id,
        prediction=result["prediction"],
        confidence=result.get("confidence"),
        probabilities=result.get("probabilities"),
        confidence_interval=result.get("confidence_interval"),
    )


router.include_router(protected_router)
