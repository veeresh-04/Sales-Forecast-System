"""
Model registry and automatic model selection.

`ModelSelector` trains all registered models on a state's training data,
evaluates them on the validation set, selects the best by RMSE, and
caches the winner for fast inference.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import MODEL_DIR, REPORT_DIR, VALIDATION_WEEKS
from app.data_pipeline import train_val_split
from app.models.base import BaseForecaster, ForecastResult
from app.models.lstm_model import LSTMForecaster
from app.models.prophet_model import ProphetForecaster
from app.models.sarima import SARIMAForecaster
from app.models.xgboost_model import XGBoostForecaster

logger = logging.getLogger(__name__)

# All model classes — add new models here and they participate automatically
_MODEL_CLASSES: list[type[BaseForecaster]] = [
    SARIMAForecaster,
    ProphetForecaster,
    XGBoostForecaster,
    LSTMForecaster,
]

_SELECTION_METRIC = "RMSE"   # lower is better


class ModelSelector:
    """Train, evaluate, and persist the best model per state."""

    def __init__(self, model_dir: Path = MODEL_DIR, report_dir: Path = REPORT_DIR) -> None:
        self._model_dir = model_dir
        self._report_dir = report_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, BaseForecaster] = {}

    # ── Public interface ───────────────────────────────────────────────────────

    def fit_state(
        self, df: pd.DataFrame, state: str, val_weeks: int = VALIDATION_WEEKS
    ) -> dict:
        """Train all models, pick the best one, persist it, return a report."""
        train, val = train_val_split(df, state, val_weeks)
        report: dict[str, dict] = {}

        best_model: Optional[BaseForecaster] = None
        best_model_class: Optional[type[BaseForecaster]] = None
        best_score = float("inf")

        for ModelClass in _MODEL_CLASSES:
            model = ModelClass()
            try:
                model.fit(train)
                metrics = model.evaluate(val)
                report[model.name] = metrics
                logger.info("  %s → %s RMSE=%.2f", state, model.name, metrics["RMSE"])

                if metrics[_SELECTION_METRIC] < best_score:
                    best_score = metrics[_SELECTION_METRIC]
                    best_model = model
                    best_model_class = ModelClass

            except Exception as exc:  # noqa: BLE001
                logger.warning("  %s | %s failed: %s", state, model.name, exc)
                report[model.name] = {"error": str(exc)}

        if best_model is None:
            raise RuntimeError(f"All models failed for state '{state}'")
        if best_model_class is None:
            raise RuntimeError(f"No model class selected for state '{state}'")

        # Selection uses the holdout split; serving uses a fresh winner refit on
        # the full state history so forecasts start after the latest known week.
        full_state = df[df["state"] == state].sort_values("date").reset_index(drop=True)
        serving_model = best_model_class().fit(full_state)

        result = {
            "state": state,
            "best_model": best_model.name,
            "best_rmse": best_score,
            "all_models": report,
        }

        self._persist(state, serving_model)
        self._persist_report(state, result)
        self._cache[state] = serving_model
        logger.info("  ✓ Best for %s: %s (RMSE=%.2f)", state, best_model.name, best_score)

        return result

    def predict_state(self, state: str, horizon: int) -> pd.DataFrame:
        model = self._cache.get(state) or self._load(state)
        return model.predict(horizon)

    def get_best_model_name(self, state: str) -> str:
        model = self._cache.get(state) or self._load(state)
        return model.name

    def get_model_report(self, state: str) -> dict:
        path = self._report_path(state)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        model = self._cache.get(state) or self._load(state)
        return {
            "state": state,
            "best_model": model.name,
            "best_rmse": None,
            "all_models": None,
        }

    def trained_states(self) -> list[str]:
        return [p.stem for p in self._model_dir.glob("*.pkl")]

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist(self, state: str, model: BaseForecaster) -> None:
        path = self._model_dir / f"{state}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def _persist_report(self, state: str, report: dict) -> None:
        with open(self._report_path(state), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _report_path(self, state: str) -> Path:
        return self._report_dir / f"{state}.json"

    def _load(self, state: str) -> BaseForecaster:
        path = self._model_dir / f"{state}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model found for '{state}'. Run /train first."
            )
        with open(path, "rb") as f:
            model = pickle.load(f)
        self._cache[state] = model
        return model
