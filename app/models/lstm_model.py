"""
LSTM forecaster using PyTorch.

Architecture: stacked LSTM with dropout → linear output head.
Training uses early stopping on a held-out split to avoid over-fitting
small state-level time series.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from app.config import DATE_COL, FORECAST_HORIZON_WEEKS, LSTM_PARAMS, TARGET_COL
from app.models.base import BaseForecaster

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Neural network definition ──────────────────────────────────────────────────


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Forecaster adapter ─────────────────────────────────────────────────────────


class LSTMForecaster(BaseForecaster):

    name = "LSTM"

    def __init__(self) -> None:
        self._net: Optional[_LSTMNet] = None
        self._scaler = MinMaxScaler()
        self._lookback: int = LSTM_PARAMS["lookback"]
        self._last_window: Optional[np.ndarray] = None
        self._last_date: Optional[pd.Timestamp] = None
        self._freq = "W-SAT"

    def fit(self, train: pd.DataFrame) -> "LSTMForecaster":
        series = train[TARGET_COL].values.astype(float)
        scaled = self._scaler.fit_transform(series.reshape(-1, 1)).flatten()

        X, y = _make_sequences(scaled, self._lookback)
        if len(X) < 4:
            logger.warning("Insufficient data for LSTM; skipping.")
            return self

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)

        # Internal 80/20 split for early stopping
        n_train = max(1, int(len(X_t) * 0.8))
        loader = DataLoader(
            TensorDataset(X_t[:n_train], y_t[:n_train]),
            batch_size=LSTM_PARAMS["batch_size"],
            shuffle=False,
        )
        val_X, val_y = X_t[n_train:], y_t[n_train:]

        self._net = _LSTMNet(
            input_size=1,
            hidden=LSTM_PARAMS["hidden_size"],
            layers=LSTM_PARAMS["num_layers"],
            dropout=LSTM_PARAMS["dropout"],
        ).to(DEVICE)

        opt = torch.optim.Adam(self._net.parameters(), lr=LSTM_PARAMS["lr"])
        loss_fn = nn.MSELoss()
        best_val, patience_counter = float("inf"), 0

        self._net.train()
        for epoch in range(LSTM_PARAMS["epochs"]):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                opt.step()

            if len(val_X) > 0:
                self._net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self._net(val_X), val_y).item()
                self._net.train()

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= LSTM_PARAMS["patience"]:
                        logger.debug("Early stopping at epoch %d", epoch)
                        break

        self._last_window = scaled[-self._lookback:]
        self._last_date = train[DATE_COL].max()
        return self

    def predict(self, horizon: int = FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
        if self._net is None or self._last_window is None:
            return pd.DataFrame(columns=["date", "forecast"])

        self._net.eval()
        window = self._last_window.copy()
        results = []

        for step in range(horizon):
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            with torch.no_grad():
                pred_scaled = self._net(x).item()

            pred = float(self._scaler.inverse_transform([[pred_scaled]])[0][0])
            next_date = self._last_date + pd.DateOffset(weeks=step + 1)
            results.append({"date": next_date, "forecast": max(pred, 0)})

            window = np.append(window[1:], pred_scaled)

        return pd.DataFrame(results)


# ── Utilities ──────────────────────────────────────────────────────────────────


def _make_sequences(data: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback: i])
        y.append(data[i])
    return np.array(X), np.array(y)
