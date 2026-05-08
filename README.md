# Sales Forecasting System
A multi-model time-series forecasting service for weekly
state-level beverage sales. Forecasts the next **8 weeks** of sales for
each of 43 US states using an ensemble of SARIMA, Prophet, XGBoost, and LSTM.

---

## Architecture

```
forecasting_system/
├── app/
│   ├── config.py          # All tuneable parameters — single source of truth
│   ├── data_pipeline.py   # Ingestion, cleaning, feature engineering
│   ├── model_selector.py  # Training orchestration + best-model selection
│   ├── schemas.py         # Pydantic request / response models
│   ├── main.py            # FastAPI application
│   └── models/
│       ├── base.py        # Abstract base class + shared evaluate()
│       ├── sarima.py      # SARIMA (statsmodels SARIMAX)
│       ├── prophet_model.py  # Facebook Prophet
│       ├── xgboost_model.py  # XGBoost with recursive multi-step
│       └── lstm_model.py     # PyTorch LSTM with early stopping
├── scripts/
│   └── train.py           # Offline CLI training script
├── tests/
│   └── test_forecasting.py  # Pytest unit + integration tests
├── data/
│   └── Forecasting_Case_Study.xlsx
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train models (CLI)

```bash
# Train all 43 states
python scripts/train.py

# Train specific states only
python scripts/train.py --states California Texas Florida

# Save a JSON training report
python scripts/train.py --states California --output report.json
```

### 3. Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs at: http://localhost:8000/docs

---

## API Reference

| Method | Endpoint             | Description                                   |
|--------|----------------------|-----------------------------------------------|
| GET    | `/health`            | Liveness + readiness check                    |
| GET    | `/states`            | List all 43 available states                  |
| POST   | `/train`             | Train & persist best model per state          |
| GET    | `/predict/{state}`   | Get 8-week forecast for a state               |
| GET    | `/models/{state}`    | Show which model was selected for a state     |

### POST /train

```json
{
  "states": ["California", "Texas"]   // omit to train all states
}
```

Response:
```json
{
  "message": "Training complete for 2 state(s).",
  "results": [
    {
      "state": "California",
      "best_model": "Prophet",
      "best_rmse": 12345678.9,
      "all_models": {
        "SARIMA":   {"MAE": ..., "RMSE": ..., "MAPE": ...},
        "Prophet":  {"MAE": ..., "RMSE": ..., "MAPE": ...},
        "XGBoost":  {"MAE": ..., "RMSE": ..., "MAPE": ...},
        "LSTM":     {"MAE": ..., "RMSE": ..., "MAPE": ...}
      }
    }
  ]
}
```

### GET /predict/{state}?horizon=8

```json
{
  "state": "California",
  "model_used": "Prophet",
  "horizon_weeks": 8,
  "forecast": [
    {"date": "2024-01-06", "forecast": 456123456.78},
    {"date": "2024-01-13", "forecast": 461234567.89}
  ]
}
```

---

## Data Pipeline

### Input
- **File**: `Forecasting_Case_Study.xlsx`
- **Columns**: State, Date, Total (sales), Category
- **Coverage**: Jan 2019 – Dec 2023, 43 US states, weekly (irregular)

### Preprocessing
1. Rename columns to snake_case
2. Aggregate duplicate state+date rows (sum)
3. Drop zero/negative sales rows
4. Reindex each state to a continuous `W-SAT` weekly grid
5. Forward-fill + backward-fill missing weeks

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `lag_7d` | Sales 1 week ago |
| `lag_14d` | Sales 2 weeks ago |
| `lag_30d` | Sales ~4 weeks ago |
| `roll_mean_4w` | 4-week rolling mean (lagged) |
| `roll_mean_8w` | 8-week rolling mean (lagged) |
| `roll_std_4w` | 4-week rolling std (lagged) |
| `roll_std_8w` | 8-week rolling std (lagged) |
| `day_of_week` | 0–6 |
| `month` | 1–12 |
| `week_of_year` | 1–52 |
| `quarter` | 1–4 |
| `is_holiday` | US federal holiday flag |
| `is_month_end` | Month-end flag |
| `log_sales` | log1p transform of target |

---

## Model Details

### SARIMA
- Order: (1,1,1)(1,1,0)[52] — seasonal period = 52 weeks
- Fitted via statsmodels SARIMAX
- Forecast: direct multi-step

### Prophet
- Multiplicative seasonality (yearly + weekly)
- US holidays injected
- Changepoint prior scale: 0.05

### XGBoost
- 300 estimators, lr=0.05, max_depth=4
- Trained on all engineered features in log-space
- Forecast: recursive (each step's prediction becomes input lag)

### LSTM
- 2-layer stacked LSTM, hidden_size=64, dropout=0.2
- Lookback window: 12 weeks
- Early stopping (patience=15) on internal 80/20 split
- Target scaled with MinMaxScaler; predictions inverse-transformed

---

## Model Selection

All models are trained on the training split. The last `val_weeks=12`
observations form the validation set (no shuffling, no leakage).

Selection metric: **RMSE** (lower is better). The winning model is
persisted to `models/<state>.pkl` and served by the API.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All parameters live in `app/config.py`. Notable settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FORECAST_HORIZON_WEEKS` | 8 | Default prediction horizon |
| `VALIDATION_WEEKS` | 12 | Hold-out window for model selection |
| `LAG_DAYS` | [7, 14, 30] | Lag feature windows |
| `ROLLING_WINDOWS` | [4, 8] | Rolling stat windows (weeks) |
| `XGB_PARAMS` | — | XGBoost hyperparameters |
| `LSTM_PARAMS` | — | LSTM architecture + training settings |
| `PROPHET_PARAMS` | — | Prophet seasonality settings |
