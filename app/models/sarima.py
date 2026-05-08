"""SARIMA model adapter built on statsmodels SARIMAX."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.config import (
    ARIMA_SEASONAL_PERIOD,
    DATE_COL,
    FORECAST_HORIZON_WEEKS,
    TARGET_COL,
)
from app.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class SARIMAForecaster(BaseForecaster):
    """SARIMA with yearly seasonality and plain-ARIMA fallback."""

    name = "SARIMA"

    def __init__(self) -> None:
        self._model = None
        self._last_date: pd.Timestamp | None = None
        self._freq = "W-SAT"

    def fit(self, train: pd.DataFrame) -> "SARIMAForecaster":
        series = train.set_index(DATE_COL)[TARGET_COL].asfreq(self._freq)

        order = (1, 1, 1)
        seasonal_candidates = [
            (1, 1, 0, ARIMA_SEASONAL_PERIOD),
            (0, 0, 0, 0),
        ]

        for seasonal_order in seasonal_candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._model = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=200)
                self._last_date = series.index[-1]
                return self
            except Exception as exc:  # noqa: BLE001
                logger.warning("SARIMA seasonal_order=%s failed: %s", seasonal_order, exc)

        raise RuntimeError("SARIMA fitting failed for all configured orders.")

    def predict(self, horizon: int = FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
        forecast = self._model.forecast(steps=horizon)
        future_dates = pd.date_range(
            start=self._last_date + pd.DateOffset(weeks=1),
            periods=horizon,
            freq=self._freq,
        )
        return pd.DataFrame(
            {"date": future_dates, "forecast": np.maximum(forecast.values, 0)}
        )
