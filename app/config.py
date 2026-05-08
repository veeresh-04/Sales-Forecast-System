"""Application configuration - single source of truth for tunable parameters."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DATA_FILE = DATA_DIR / "Forecasting_Case_Study.xlsx"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Forecast
FORECAST_HORIZON_WEEKS = 8
VALIDATION_WEEKS = 12          # last N weeks held out for model selection
RANDOM_SEED = 42

# Feature engineering
LAG_PERIODS = [1, 7, 30]       # required t-1, t-7, t-30 row lags
LAG_DAYS = [7, 14, 30]         # legacy weekly lag windows for old model artifacts
ROLLING_WINDOWS = [4, 8]       # rolling windows in weeks
TARGET_COL = "sales"
DATE_COL = "date"
STATE_COL = "state"

# Model hyper-parameters
ARIMA_MAX_P = 3
ARIMA_MAX_Q = 3
ARIMA_SEASONAL_PERIOD = 52     # weekly data -> yearly seasonality

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LSTM_PARAMS = {
    "lookback": 12,            # weeks
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 16,
    "lr": 1e-3,
    "patience": 15,
}

PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "seasonality_mode": "multiplicative",
    "changepoint_prior_scale": 0.05,
}
