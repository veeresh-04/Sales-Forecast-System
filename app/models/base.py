"""
Abstract base class for all forecasting models.

Every model adapter must implement `fit`, `predict`, and `name`.
The common `evaluate` method computes MAE, RMSE, and MAPE on a held-out set.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import DATE_COL, FORECAST_HORIZON_WEEKS, TARGET_COL


@dataclass
class ForecastResult:
    model_name: str
    state: str
    predictions: pd.DataFrame          # columns: date, forecast
    metrics: dict[str, float] = field(default_factory=dict)
    is_best: bool = False


class BaseForecaster(abc.ABC):
    """Defines the interface every model adapter must honour."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def fit(self, train: pd.DataFrame) -> "BaseForecaster": ...

    @abc.abstractmethod
    def predict(self, horizon: int = FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
        """Return a DataFrame with columns [date, forecast]."""
        ...

    # ── Shared evaluation ──────────────────────────────────────────────────────

    def evaluate(self, val: pd.DataFrame) -> dict[str, float]:
        """Compute MAE, RMSE, MAPE against validation actuals."""
        pred_df = self.predict(horizon=len(val))
        actuals = val[TARGET_COL].values

        # Align by position (both are sorted chronologically)
        forecasts = pred_df["forecast"].values[: len(actuals)]

        mae = float(np.mean(np.abs(actuals - forecasts)))
        rmse = float(np.sqrt(np.mean((actuals - forecasts) ** 2)))
        nonzero = actuals != 0
        mape = (
            float(np.mean(np.abs((actuals[nonzero] - forecasts[nonzero]) / actuals[nonzero])) * 100)
            if nonzero.any()
            else float("inf")
        )
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
