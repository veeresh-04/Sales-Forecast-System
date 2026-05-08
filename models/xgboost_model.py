"""
XGBoost forecaster with engineered lag and rolling features.

Prediction is recursive: each forecast is appended to the in-memory history so
future lag and rolling features are built from the latest known/predicted sales.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.config import (
    DATE_COL,
    FORECAST_HORIZON_WEEKS,
    LAG_DAYS,
    LAG_PERIODS,
    ROLLING_WINDOWS,
    TARGET_COL,
    XGB_PARAMS,
)
from app.models.base import BaseForecaster

logger = logging.getLogger(__name__)

_FEATURE_COLS = (
    [f"lag_t_{lag}" for lag in LAG_PERIODS]
    + [f"lag_{days}d" for days in LAG_DAYS]
    + [f"roll_mean_{window}w" for window in ROLLING_WINDOWS]
    + [f"roll_std_{window}w" for window in ROLLING_WINDOWS]
    + ["day_of_week", "month", "week_of_year", "quarter", "is_holiday", "is_month_end"]
)


class XGBoostForecaster(BaseForecaster):
    name = "XGBoost"

    def __init__(self) -> None:
        self._model: XGBRegressor | None = None
        self._last_obs: pd.Series | None = None
        self._history: list[float] = []

    def fit(self, train: pd.DataFrame) -> "XGBoostForecaster":
        clean = train.dropna(subset=_FEATURE_COLS + [TARGET_COL])
        X = clean[_FEATURE_COLS]
        y = np.log1p(clean[TARGET_COL])

        self._model = XGBRegressor(**XGB_PARAMS)
        self._model.fit(X, y, verbose=False)
        self._last_obs = train.iloc[-1].copy()
        self._history = train[TARGET_COL].astype(float).tolist()
        return self

    def predict(self, horizon: int = FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
        if self._model is None or self._last_obs is None:
            raise RuntimeError("XGBoost model has not been fitted.")

        results = []
        current_date = self._last_obs[DATE_COL]
        history = list(self._history)

        for _ in range(horizon):
            next_date = current_date + pd.DateOffset(weeks=1)
            row = _build_next_row(history, next_date)
            X_pred = pd.DataFrame([row])[_FEATURE_COLS]
            log_pred = self._model.predict(X_pred)[0]
            forecast = max(float(np.expm1(log_pred)), 0.0)

            results.append({"date": next_date, "forecast": forecast})
            history.append(forecast)
            current_date = next_date

        return pd.DataFrame(results)


def _build_next_row(history: list[float], next_date: pd.Timestamp) -> dict:
    import holidays as _hol

    us_holidays = _hol.US()
    row: dict = {
        "day_of_week": next_date.dayofweek,
        "month": next_date.month,
        "week_of_year": next_date.isocalendar().week,
        "quarter": next_date.quarter,
        "is_holiday": int(next_date in us_holidays),
        "is_month_end": int(next_date.is_month_end),
    }

    for lag in LAG_PERIODS:
        row[f"lag_t_{lag}"] = _lag_value(history, lag)

    for days in LAG_DAYS:
        lag_weeks = max(1, round(days / 7))
        row[f"lag_{days}d"] = _lag_value(history, lag_weeks)

    for window_size in ROLLING_WINDOWS:
        window = history[-window_size:]
        row[f"roll_mean_{window_size}w"] = float(np.mean(window)) if window else 0.0
        row[f"roll_std_{window_size}w"] = (
            float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
        )

    return row


def _lag_value(history: list[float], periods: int) -> float:
    if len(history) >= periods:
        return float(history[-periods])
    return float(history[-1]) if history else 0.0
