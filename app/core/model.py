"""Automated preprocessing, model selection, training, and inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)
TaskType = Literal["classification", "regression"]


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Extract calendar components from dataframe datetime feature columns."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> DateTimeFeatures:
        """Record names of the datetime columns."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return year, month, day, and weekday features per datetime column."""
        transformed: list[np.ndarray] = []
        for column in X.columns:
            values = pd.to_datetime(X[column], errors="coerce", format="mixed")
            transformed.extend(
                [
                    values.dt.year.to_numpy(dtype=float, na_value=np.nan),
                    values.dt.month.to_numpy(dtype=float, na_value=np.nan),
                    values.dt.day.to_numpy(dtype=float, na_value=np.nan),
                    values.dt.dayofweek.to_numpy(dtype=float, na_value=np.nan),
                ]
            )
        if not transformed:
            return np.empty((len(X), 0), dtype=float)
        return np.column_stack(transformed)

    def get_feature_names_out(
        self, input_features: np.ndarray | None = None
    ) -> np.ndarray:
        """Name extracted calendar component columns."""
        features = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(
            [
                f"{name}_{part}"
                for name in features
                for part in ("year", "month", "day", "weekday")
            ],
            dtype=object,
        )


@dataclass
class TrainingResult:
    """Fitted estimator and cross-validated training evaluation."""

    pipeline: Pipeline
    task: TaskType
    metrics: dict[str, float]
    feature_columns: list[str]
    algorithm: str = "logistic_regression"
    feature_importances: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    residuals: list[float] | None = None
    actual_values: list[float] | None = None
    predicted_values: list[float] | None = None
    class_labels: list[str] | None = None
    roc_curve: dict[str, list[float]] | None = None
    cv_scores: dict[str, float] = field(default_factory=dict)
    target_column: str = ""
    task_type: str = ""


class AutoMLPipeline:
    """Prepare tabular features and select supervised estimators by CV score."""

    def __init__(self, *, random_state: int = 42, cv_folds: int = 5) -> None:
        """Configure deterministic split/model selection defaults."""
        self.random_state = random_state
        self.cv_folds = cv_folds

    @staticmethod
    def detect_task_type(y: pd.Series) -> TaskType:
        """Infer task type from target dtype and number of distinct values."""
        if pd.api.types.is_bool_dtype(y.dtype) or not pd.api.types.is_numeric_dtype(
            y.dtype
        ):
            return "classification"
        non_null = y.dropna()
        if non_null.empty:
            raise ValueError("Target must contain at least one non-missing value.")
        unique_ratio = non_null.nunique() / len(non_null)
        if non_null.nunique() <= 2 or (
            non_null.nunique() <= 20 and unique_ratio <= 0.1
        ):
            return "classification"
        return "regression"

    @staticmethod
    def build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
        """Build imputation, scaling, categorical encoding, and datetime steps."""
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} was not found.")
        features = df.drop(columns=[target])
        numeric = features.select_dtypes(include=np.number).columns.tolist()
        boolean = features.select_dtypes(include="bool").columns.tolist()
        numeric = [column for column in numeric if column not in boolean]
        datetime = features.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        remaining = [column for column in features.columns if column not in numeric + datetime]
        datetime_text = [
            column
            for column in remaining
            if AutoMLPipeline._is_datetime_column(features[column])
        ]
        datetime.extend(datetime_text)
        categorical = [column for column in remaining if column not in datetime_text]

        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical,
                )
            )
        if datetime:
            transformers.append(
                (
                    "datetime",
                    Pipeline(
                        [
                            ("extract", DateTimeFeatures()),
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    datetime,
                )
            )
        if not transformers:
            raise ValueError("At least one usable feature column is required.")
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def train(
        self,
        df: pd.DataFrame,
        target: str,
        task_type: str = "auto",
        algorithm: str = "auto",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> TrainingResult:
        """Select and fit a model, returning CV metrics and diagnostics."""
        logger.info("Starting model training for target=%s algorithm=%s", target, algorithm)
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} was not found.")
        usable = df.dropna(subset=[target]).copy()
        if len(usable) < 4:
            raise ValueError("At least four rows with a non-missing target are required.")
        y = usable[target]
        inferred_task = self.detect_task_type(y)
        normalized_task = inferred_task if task_type == "auto" else task_type.lower()
        if normalized_task not in {"classification", "regression"}:
            raise ValueError("task_type must be 'auto', 'classification', or 'regression'.")
        task: TaskType = normalized_task  # validated literal
        if task == "classification" and y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        if task == "regression" and y.nunique() < 2:
            raise ValueError("Regression requires at least two distinct target values.")

        features = usable.drop(columns=[target])
        features = features.loc[:, ~features.isna().all(axis=0)]
        if features.empty:
            raise ValueError("At least one feature column with non-missing data is required.")
        preprocessor = self.build_preprocessor(usable.loc[:, [*features.columns, target]], target)
        cv, scoring, n_splits = self._cross_validator(y, task)
        candidates = self._candidates(task, algorithm)
        cv_scores: dict[str, float] = {}
        best_name = ""
        best_score = -np.inf
        best_pipeline: Pipeline | None = None

        for candidate_index, (name, estimator) in enumerate(candidates, start=1):
            candidate = Pipeline(
                [("preprocessor", preprocessor), ("estimator", estimator)]
            )
            try:
                scores = cross_val_score(
                    candidate, features, y, cv=cv, scoring=scoring, error_score="raise"
                )
                score = float(np.mean(scores))
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping %s after CV failure: %s", name, exc)
                continue
            cv_scores[name] = score
            logger.info("CV score for %s: %.4f", name, score)
            if progress_callback is not None:
                progress_callback(candidate_index, len(candidates), name)
            if score > best_score:
                best_name, best_score, best_pipeline = name, score, candidate

        if best_pipeline is None:
            raise ValueError("No candidate model could be trained on this dataset.")

        try:
            validation_predictions = cross_val_predict(
                best_pipeline, features, y, cv=cv, method="predict"
            )
            validation_probabilities: np.ndarray | None = None
            if task == "classification":
                validation_probabilities = cross_val_predict(
                    best_pipeline, features, y, cv=cv, method="predict_proba"
                )
            best_pipeline.fit(features, y)
        except (ValueError, TypeError) as exc:
            logger.exception("Selected model fitting or validation prediction failed")
            raise ValueError(f"Could not complete model training: {exc}") from exc

        metrics, confusion, residuals, actual, predicted, labels, roc = self._evaluate(
            y, validation_predictions, task, validation_probabilities
        )
        if residuals is not None:
            best_pipeline.residual_std_ = float(np.std(residuals, ddof=1)) if len(
                residuals
            ) > 1 else 0.0
        importances = self._feature_importances(best_pipeline)
        logger.info("Completed training with %s; metrics=%s", best_name, metrics)
        return TrainingResult(
            pipeline=best_pipeline,
            task=task,
            metrics=metrics,
            feature_columns=[str(column) for column in features.columns],
            algorithm=best_name,
            feature_importances=importances,
            confusion_matrix=confusion,
            residuals=residuals,
            actual_values=actual,
            predicted_values=predicted,
            class_labels=labels,
            roc_curve=roc,
            cv_scores=cv_scores,
            target_column=target,
            task_type=task,
        )

    @staticmethod
    def save(model: Any, path: str | Path) -> None:
        """Persist a fitted model with joblib, creating parent directories."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination)
        logger.info("Saved model to %s", destination)

    @staticmethod
    def load(path: str | Path) -> Any:
        """Load a model previously persisted with :meth:`save`."""
        logger.info("Loading model from %s", path)
        return joblib.load(path)

    @staticmethod
    def predict(model: Any, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict one record and include class probability or interval estimate."""
        logger.info("Generating prediction for %d input feature(s)", len(input_dict))
        row = pd.DataFrame([input_dict])
        prediction = model.predict(row)[0]
        result: dict[str, Any] = {"prediction": _scalar(prediction)}
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            classes = getattr(model, "classes_", range(len(probabilities)))
            result["probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(classes, probabilities, strict=False)
            }
            result["confidence"] = float(np.max(probabilities))
        elif hasattr(model, "residual_std_"):
            residual_std = float(model.residual_std_)
            result["confidence_interval"] = [
                float(prediction - 1.96 * residual_std),
                float(prediction + 1.96 * residual_std),
            ]
        elif hasattr(model, "estimators_"):
            estimator_predictions = np.asarray(
                [estimator.predict(row)[0] for estimator in model.estimators_],
                dtype=float,
            )
            standard_error = (
                float(estimator_predictions.std(ddof=1))
                if len(estimator_predictions) > 1
                else 0.0
            )
            result["confidence_interval"] = [
                float(prediction - 1.96 * standard_error),
                float(prediction + 1.96 * standard_error),
            ]
        else:
            result["confidence_interval"] = [None, None]
        logger.info("Prediction complete")
        return result

    @staticmethod
    def _is_datetime_column(series: pd.Series) -> bool:
        """Test whether at least 80% of non-null values parse as timestamps."""
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            return False
        values = series.dropna()
        if values.empty:
            return False
        return bool(
            pd.to_datetime(values, errors="coerce", format="mixed").notna().mean()
            >= 0.8
        )

    def _cross_validator(
        self, y: pd.Series, task: TaskType
    ) -> tuple[StratifiedKFold | KFold, str, int]:
        """Build a deterministic CV splitter sized for the available samples."""
        if task == "classification":
            minimum_class = int(y.value_counts().min())
            n_splits = min(self.cv_folds, minimum_class)
            if n_splits < 2:
                raise ValueError("Classification requires at least two samples per class.")
            return (
                StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=self.random_state
                ),
                "accuracy",
                n_splits,
            )
        n_splits = min(self.cv_folds, len(y))
        if n_splits < 2:
            raise ValueError("Regression requires at least two samples.")
        return (
            KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state),
            "r2",
            n_splits,
        )

    @staticmethod
    def _candidates(task: TaskType, algorithm: str) -> list[tuple[str, Any]]:
        """Return candidate estimators appropriate to the requested task."""
        options: dict[str, Any]
        if task == "classification":
            options = {
                "logistic_regression": LogisticRegression(max_iter=2000),
                "random_forest": RandomForestClassifier(
                    n_estimators=150, random_state=42, class_weight="balanced"
                ),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
            }
        else:
            options = {
                "ridge": Ridge(),
                "random_forest": RandomForestRegressor(
                    n_estimators=150, random_state=42
                ),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
            }
        normalized = algorithm.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "logistic": "logistic_regression",
            "linear": "ridge",
            "randomforest": "random_forest",
            "gradientboosting": "gradient_boosting",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized == "auto":
            return list(options.items())
        if normalized not in options:
            raise ValueError(
                f"Algorithm {algorithm!r} is not supported for {task}."
            )
        return [(normalized, options[normalized])]

    @staticmethod
    def _evaluate(
        y: pd.Series,
        predictions: np.ndarray,
        task: TaskType,
        probabilities: np.ndarray | None,
    ) -> tuple[
        dict[str, float],
        list[list[int]] | None,
        list[float] | None,
        list[float] | None,
        list[float] | None,
        list[str] | None,
        dict[str, list[float]] | None,
    ]:
        """Calculate task metrics and serializable diagnostic series."""
        if task == "classification":
            metrics: dict[str, float] = {
                "accuracy": float(accuracy_score(y, predictions)),
                "f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
            }
            labels = [str(value) for value in pd.unique(y)]
            matrix = confusion_matrix(y, predictions, labels=pd.unique(y)).tolist()
            roc: dict[str, list[float]] | None = None
            if probabilities is not None:
                try:
                    if y.nunique() == 2:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y, probabilities[:, 1])
                        )
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(
                                y,
                                probabilities,
                                multi_class="ovr",
                                average="weighted",
                            )
                        )
                except ValueError as exc:
                    logger.warning("ROC AUC could not be calculated: %s", exc)
                if y.nunique() == 2:
                    from sklearn.metrics import roc_curve

                    classes = np.unique(y)
                    false_positive, true_positive, _ = roc_curve(
                        y, probabilities[:, 1], pos_label=classes[1]
                    )
                    roc = {
                        "false_positive_rate": false_positive.tolist(),
                        "true_positive_rate": true_positive.tolist(),
                    }
            return metrics, matrix, None, None, None, labels, roc

        actual = y.to_numpy(dtype=float)
        predicted = np.asarray(predictions, dtype=float)
        residuals = actual - predicted
        return (
            {
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "r2": float(r2_score(actual, predicted)),
            },
            None,
            residuals.tolist(),
            actual.tolist(),
            predicted.tolist(),
            None,
            None,
        )

    @staticmethod
    def _feature_importances(pipeline: Pipeline) -> dict[str, float]:
        """Map transformed feature names to estimator importance magnitudes."""
        estimator = pipeline.named_steps["estimator"]
        importance = getattr(estimator, "feature_importances_", None)
        if importance is None and hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_)
            importance = (
                np.abs(coefficients)
                if coefficients.ndim == 1
                else np.mean(np.abs(coefficients), axis=0)
            )
        if importance is None:
            return {}
        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        try:
            names = preprocessor.get_feature_names_out()
        except (AttributeError, ValueError):
            names = [f"feature_{index}" for index in range(len(importance))]
        return {
            str(name): float(value)
            for name, value in zip(names, importance, strict=False)
        }


def _scalar(value: Any) -> Any:
    """Convert NumPy scalar predictions into JSON-friendly values."""
    return value.item() if isinstance(value, np.generic) else value


def train_model(
    dataframe: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Backward-compatible model training helper using AutoML cross-validation."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")
    result = AutoMLPipeline(random_state=random_state).train(dataframe, target)
    if result.task == "regression":
        result.metrics["mean_absolute_error"] = result.metrics["mae"]
    return result
