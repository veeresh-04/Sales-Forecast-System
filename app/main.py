"""
Sales Forecasting REST API.

Endpoints
---------
GET  /                    - Service entry point
GET  /health              - Service health check
GET  /states              - List all available states
GET  /models              - List states with persisted models
GET  /models/{state}      - Show selected model and model-comparison metrics
POST /train               - Train models for all or selected states
GET  /predict/{state}     - Forecast next N weeks for a state

Design notes
------------
* The ModelSelector singleton is stored in app.state so all requests share the
  same in-memory model cache.
* Training is synchronous for this case study. In a larger production service,
  dispatch /train work to a background queue such as Celery, RQ, or ARQ.
* Data is loaded once at startup through FastAPI lifespan.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from app.config import FORECAST_HORIZON_WEEKS
from app.data_pipeline import get_states, load_and_prepare
from app.model_selector import ModelSelector
from app.schemas import (
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
    ServiceInfoResponse,
    TrainRequest,
    TrainResponse,
    TrainStateResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading dataset ...")
    app.state.df = load_and_prepare()
    app.state.selector = ModelSelector()
    logger.info("Startup complete - %d states available.", app.state.df["state"].nunique())
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "Multi-model time-series forecasting service for weekly state-level "
        "sales. Supports SARIMA, Prophet, XGBoost, and LSTM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=ServiceInfoResponse, tags=["Meta"])
def root():
    """Return entry-point links for browsers, checks, and demos."""
    return ServiceInfoResponse(
        service="Sales Forecasting API",
        status="ok",
        docs_url="/docs",
        health_url="/health",
        states_url="/states",
    )


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Service liveness and readiness check."""
    selector: ModelSelector = app.state.selector
    return HealthResponse(
        status="ok",
        trained_states=len(selector.trained_states()),
        available_models=["SARIMA", "Prophet", "XGBoost", "LSTM"],
    )


@app.get("/states", tags=["Meta"])
def list_states():
    """Return all states present in the dataset."""
    return {"states": get_states(app.state.df)}


@app.get("/models", tags=["Meta"])
def list_models():
    """Return all states with persisted trained models."""
    selector: ModelSelector = app.state.selector
    return {"trained_states": selector.trained_states()}


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train(body: TrainRequest = TrainRequest()):
    """
    Train and evaluate all models for the requested states.

    The best model per state is selected by RMSE on a held-out validation
    window, persisted to disk, and cached in memory for fast inference.
    """
    df = app.state.df
    selector: ModelSelector = app.state.selector

    available_states = get_states(df)
    target_states = body.states or available_states
    unknown = [state for state in target_states if state not in available_states]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown states: {unknown}")

    results: list[TrainStateResult] = []
    for state in target_states:
        logger.info("Training models for: %s", state)
        try:
            report = selector.fit_state(df, state)
            results.append(TrainStateResult(**report))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training failed for %s", state)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TrainResponse(
        message=f"Training complete for {len(results)} state(s).",
        results=results,
    )


@app.get("/predict/{state}", response_model=ForecastResponse, tags=["Inference"])
def predict(
    state: str,
    horizon: int = Query(
        default=FORECAST_HORIZON_WEEKS,
        ge=1,
        le=52,
        description="Number of weeks to forecast (1-52)",
    ),
):
    """
    Return weekly sales forecasts for the requested state.

    The endpoint uses whichever model was selected as best during training.
    Run /train first if a persisted model does not exist yet.
    """
    selector: ModelSelector = app.state.selector

    try:
        pred_df = selector.predict_state(state, horizon)
        model_name = selector.get_best_model_name(state)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed for %s", state)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    forecast_points = [
        ForecastPoint(date=row["date"].date(), forecast=round(row["forecast"], 2))
        for _, row in pred_df.iterrows()
    ]
    return ForecastResponse(
        state=state,
        model_used=model_name,
        horizon_weeks=horizon,
        forecast=forecast_points,
    )


@app.get("/models/{state}", response_model=ModelInfoResponse, tags=["Meta"])
def model_info(state: str):
    """Return the selected model and saved comparison metrics for one state."""
    selector: ModelSelector = app.state.selector
    try:
        report = selector.get_model_report(state)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ModelInfoResponse(
        state=state,
        model_name=report["best_model"],
        metrics=report.get("all_models"),
    )
