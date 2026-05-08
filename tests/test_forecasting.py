"""
Integration and unit tests for the forecasting system.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import FORECAST_HORIZON_WEEKS
from app.data_pipeline import get_states, load_and_prepare, train_val_split
from app.models.sarima import SARIMAForecaster
from app.models.prophet_model import ProphetForecaster
from app.models.xgboost_model import XGBoostForecaster
from app.models.lstm_model import LSTMForecaster


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def full_df():
    return load_and_prepare()


@pytest.fixture(scope="session")
def california_train(full_df):
    train, _ = train_val_split(full_df, "California")
    return train


@pytest.fixture(scope="session")
def california_val(full_df):
    _, val = train_val_split(full_df, "California")
    return val


# ── Data pipeline tests ────────────────────────────────────────────────────────


class TestDataPipeline:
    def test_load_returns_dataframe(self, full_df):
        assert isinstance(full_df, pd.DataFrame)
        assert len(full_df) > 0

    def test_required_columns_present(self, full_df):
        for col in ("state", "date", "sales"):
            assert col in full_df.columns

    def test_no_negative_sales(self, full_df):
        assert (full_df["sales"] >= 0).all()

    def test_feature_columns_present(self, full_df):
        expected = ["lag_7d", "lag_14d", "lag_30d", "roll_mean_4w", "is_holiday"]
        for col in expected:
            assert col in full_df.columns, f"Missing feature: {col}"

    def test_states_count(self, full_df):
        assert full_df["state"].nunique() == 43

    def test_train_val_no_overlap(self, full_df):
        train, val = train_val_split(full_df, "Texas")
        assert train["date"].max() < val["date"].min()

    def test_val_size(self, full_df):
        _, val = train_val_split(full_df, "Texas", val_weeks=12)
        assert len(val) == 12


# ── Model tests ────────────────────────────────────────────────────────────────


class TestSARIMA:
    def test_fit_predict(self, california_train):
        m = SARIMAForecaster()
        m.fit(california_train)
        pred = m.predict(horizon=8)
        assert len(pred) == 8
        assert (pred["forecast"] >= 0).all()

    def test_metrics_keys(self, california_train, california_val):
        m = SARIMAForecaster().fit(california_train)
        metrics = m.evaluate(california_val)
        assert {"MAE", "RMSE", "MAPE"} == set(metrics.keys())


class TestProphet:
    def test_fit_predict(self, california_train):
        m = ProphetForecaster()
        m.fit(california_train)
        pred = m.predict(horizon=8)
        assert len(pred) == 8
        assert (pred["forecast"] >= 0).all()


class TestXGBoost:
    def test_fit_predict(self, california_train):
        m = XGBoostForecaster()
        m.fit(california_train)
        pred = m.predict(horizon=8)
        assert len(pred) == 8

    def test_no_negative_forecasts(self, california_train):
        m = XGBoostForecaster().fit(california_train)
        pred = m.predict(horizon=8)
        assert (pred["forecast"] >= 0).all()


class TestLSTM:
    def test_fit_predict(self, california_train):
        m = LSTMForecaster()
        m.fit(california_train)
        pred = m.predict(horizon=8)
        assert len(pred) == 8
        assert (pred["forecast"] >= 0).all()


# ── API integration tests ──────────────────────────────────────────────────────


class TestAPI:
    """Smoke tests against the live FastAPI app via TestClient."""

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as c:
            yield c

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_states_list(self, client):
        r = client.get("/states")
        assert r.status_code == 200
        assert "states" in r.json()
        assert len(r.json()["states"]) == 43

    def test_train_single_state(self, client):
        r = client.post("/train", json={"states": ["California"]})
        assert r.status_code == 200
        data = r.json()
        assert data["results"][0]["state"] == "California"
        assert "best_model" in data["results"][0]

    def test_predict_after_train(self, client):
        client.post("/train", json={"states": ["California"]})
        r = client.get("/predict/California?horizon=8")
        assert r.status_code == 200
        body = r.json()
        assert body["horizon_weeks"] == 8
        assert len(body["forecast"]) == 8

    def test_predict_unknown_state_returns_404(self, client):
        r = client.get("/predict/Atlantis")
        assert r.status_code == 404

    def test_model_info_endpoint(self, client):
        client.post("/train", json={"states": ["California"]})
        r = client.get("/models/California")
        assert r.status_code == 200
        assert "model_name" in r.json()
