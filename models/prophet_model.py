"""
Facebook Prophet model adapter.

Prophet handles missing dates, trend changes, and multi-level seasonality
natively. We pass US holidays to improve accuracy on holiday weeks.
"""

from __future__ import annotations

import logging

import holidays
import numpy as np
import pandas as pd
from prophet import Prophet

from app.config import DATE_COL, FORECAST_HORIZON_WEEKS, PROPHET_PARAMS, TARGET_COL
from app.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):

    name = "Prophet"

    def __init__(self) -> None:
        self._model: Prophet | None = None
        self._last_date: pd.Timestamp | None = None
        self._freq = "W-SAT"

    def fit(self, train: pd.DataFrame) -> "ProphetForecaster":
        df_prophet = train[[DATE_COL, TARGET_COL]].rename(
            columns={DATE_COL: "ds", TARGET_COL: "y"}
        )

        m = Prophet(**PROPHET_PARAMS)
        m.add_country_holidays(country_name="US")

        import logging as _log
        _log.getLogger("prophet").setLevel(_log.WARNING)
        _log.getLogger("cmdstanpy").setLevel(_log.WARNING)

        m.fit(df_prophet)
        self._model = m
        self._last_date = train[DATE_COL].max()
        return self

    def predict(self, horizon: int = FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
        future = self._model.make_future_dataframe(
            periods=horizon, freq=self._freq, include_history=False
        )
        forecast = self._model.predict(future)
        return pd.DataFrame(
            {
                "date": forecast["ds"].values,
                "forecast": np.maximum(forecast["yhat"].values, 0),
            }
        )