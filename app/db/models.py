"""SQLAlchemy 2.0 models for datasets, profiles, model runs, and predictions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Dataset(Base):
    """Uploaded dataset and its persisted file location."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    owner: Mapped[str] = mapped_column(String(255), nullable=False, default="local")

    profiles: Mapped[list[Profile]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    model_runs: Mapped[list[ModelRun]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class Profile(Base):
    """Generated JSON profile associated with a dataset."""

    __tablename__ = "profiles"
    __table_args__ = (Index("ix_profiles_dataset_id", "dataset_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="profiles")


class ModelRun(Base):
    """A trained model artifact and its evaluation metrics."""

    __tablename__ = "model_runs"
    __table_args__ = (Index("ix_model_runs_dataset_id", "dataset_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    model_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")

    dataset: Mapped[Dataset] = relationship(back_populates="model_runs")
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="model_run", cascade="all, delete-orphan"
    )


class Prediction(Base):
    """A prediction request and response associated with a model run."""

    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_model_run_id", "model_run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_run_id: Mapped[int] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False
    )
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    model_run: Mapped[ModelRun] = relationship(back_populates="predictions")

