"""Pydantic v2 request and response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetSummary(BaseModel):
    """Paginated dataset list entry."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    uploaded_at: datetime
    row_count: int
    column_count: int
    owner: str


class DatasetDetail(DatasetSummary):
    """Dataset information with its most recent profile."""

    profile: dict[str, Any] | None = None


class TrainRequest(BaseModel):
    """Model training options for a dataset."""

    target: str = Field(min_length=1)
    task_type: Literal["auto", "classification", "regression"] = "auto"
    algorithm: str = "auto"


class TrainResponse(BaseModel):
    """Identifier and selected model details after training."""

    model_run_id: int
    algorithm: str
    task_type: str
    metrics: dict[str, float]


class ModelDetail(BaseModel):
    """Training metrics and feature importance for a saved model."""

    id: int
    dataset_id: int
    target_column: str
    task_type: str
    algorithm: str
    status: str
    metrics: dict[str, Any]
    feature_importances: dict[str, float] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    """Features for a single-record prediction."""

    features: dict[str, Any]


class PredictionResponse(BaseModel):
    """Prediction result and optional confidence details."""

    prediction_id: int
    prediction: Any
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    confidence_interval: list[float | None] | None = None

