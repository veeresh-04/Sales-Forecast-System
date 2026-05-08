"""
Data ingestion, cleaning, and feature engineering pipeline.

Design principles
-----------------
* All transformations are stateless functions — easy to test and compose.
* A single `load_and_prepare` entry-point returns a clean, feature-rich DataFrame
  ready for model training.
* Missing dates are imputed via forward-fill after reindexing to a continuous
  weekly frequency so every model receives a regular time series.
"""

from __future__ import annotations

import logging
from typing import Optional

import holidays
import numpy as np
import pandas as pd

from app.config import (
    DATA_FILE,
    DATE_COL,
    LAG_DAYS,
    LAG_PERIODS,
    RANDOM_SEED,
    ROLLING_WINDOWS,
    STATE_COL,
    TARGET_COL,
    VALIDATION_WEEKS,
)

logger = logging.getLogger(__name__)

# ── Public API ─────────────────────────────────────────────────────────────────


def load_and_prepare(path=DATA_FILE) -> pd.DataFrame:
    """Return a cleaned, feature-engineered DataFrame indexed by (state, date)."""
    raw = _read_raw(path)
    df = _clean(raw)
    df = _reindex_weekly(df)
    df = _add_features(df)
    logger.info("Dataset ready: %d rows, %d states", len(df), df[STATE_COL].nunique())
    return df


def train_val_split(
    df: pd.DataFrame, state: str, val_weeks: int = VALIDATION_WEEKS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-series aware split — validation is always the *last* val_weeks rows.
    No shuffling, no leakage.
    """
    sdf = df[df[STATE_COL] == state].sort_values(DATE_COL).reset_index(drop=True)
    train = sdf.iloc[:-val_weeks]
    val = sdf.iloc[-val_weeks:]
    return train, val


def get_states(df: pd.DataFrame) -> list[str]:
    return sorted(df[STATE_COL].unique().tolist())


# ── Private helpers ────────────────────────────────────────────────────────────


def _read_raw(path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={"State": STATE_COL, "Date": DATE_COL, "Total": TARGET_COL}
    )
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df[[STATE_COL, DATE_COL, TARGET_COL]].copy()

    # Aggregate duplicates (same state+date → sum)
    df = (
        df.groupby([STATE_COL, DATE_COL], as_index=False)[TARGET_COL]
        .sum()
    )

    # Drop rows with non-positive sales (data quality guard)
    df = df[df[TARGET_COL] > 0].copy()
    return df


def _reindex_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each state, reindex to a continuous weekly (W-SAT) frequency.
    Missing weeks are forward-filled (carry-forward last known sales).
    """
    frames = []
    for state, grp in df.groupby(STATE_COL):
        grp = grp.set_index(DATE_COL).sort_index()

        # Build weekly grid from first to last observed date
        weekly_idx = pd.date_range(
            start=grp.index.min(), end=grp.index.max(), freq="W-SAT"
        )

        # Reindex + forward-fill + backward-fill for any leading NaN
        grp = grp.reindex(grp.index.union(weekly_idx)).sort_index()
        grp[TARGET_COL] = grp[TARGET_COL].ffill().bfill()
        grp = grp.reindex(weekly_idx)
        grp[STATE_COL] = state
        grp.index.name = DATE_COL
        frames.append(grp.reset_index())

    return pd.concat(frames, ignore_index=True)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features inside each state group."""
    frames = []
    us_holidays = holidays.US()

    for _, grp in df.groupby(STATE_COL):
        grp = grp.sort_values(DATE_COL).reset_index(drop=True)

        # Calendar
        grp["day_of_week"] = grp[DATE_COL].dt.dayofweek
        grp["month"] = grp[DATE_COL].dt.month
        grp["week_of_year"] = grp[DATE_COL].dt.isocalendar().week.astype(int)
        grp["quarter"] = grp[DATE_COL].dt.quarter
        grp["is_holiday"] = grp[DATE_COL].apply(
            lambda d: int(d in us_holidays)
        )
        grp["is_month_end"] = grp[DATE_COL].dt.is_month_end.astype(int)

        # Lag features (expressed in weeks; 1 week ≈ 7 days)
        for lag in LAG_PERIODS:
            grp[f"lag_t_{lag}"] = grp[TARGET_COL].shift(lag)

        for lag_days in LAG_DAYS:
            lag_weeks = max(1, round(lag_days / 7))
            grp[f"lag_{lag_days}d"] = grp[TARGET_COL].shift(lag_weeks)

        # Rolling statistics
        for window in ROLLING_WINDOWS:
            grp[f"roll_mean_{window}w"] = (
                grp[TARGET_COL].shift(1).rolling(window).mean()
            )
            grp[f"roll_std_{window}w"] = (
                grp[TARGET_COL].shift(1).rolling(window).std()
            )

        # Log-transform (stabilises variance; models use log-space internally)
        grp["log_sales"] = np.log1p(grp[TARGET_COL])

        frames.append(grp)

    result = pd.concat(frames, ignore_index=True)
    return result
