"""Pydantic v2 request / response schemas for the forecasting API."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    date: date
    forecast: float = Field(..., description="Predicted weekly sales")


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    horizon_weeks: int
    forecast: list[ForecastPoint]


class TrainRequest(BaseModel):
    states: Optional[list[str]] = Field(
        default=None,
        description="Subset of states to train. Trains all states if omitted.",
    )


class TrainStateResult(BaseModel):
    state: str
    best_model: str
    best_rmse: float
    all_models: dict[str, dict]


class TrainResponse(BaseModel):
    message: str
    results: list[TrainStateResult]


class ServiceInfoResponse(BaseModel):
    service: str
    status: str
    docs_url: str
    health_url: str
    states_url: str


class ModelInfoResponse(BaseModel):
    state: str
    model_name: str
    status: str = "ready"
    metrics: dict[str, dict] | None = None


class HealthResponse(BaseModel):
    status: str
    trained_states: int
    available_models: list[str]
