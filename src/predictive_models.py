# src/predictive_models.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# feature engineering, helpers

def _prepare_features(laps: pd.DataFrame,
                      target_col: str = "lap_time_s") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select numeric features and target for lap-time prediction.
    Assumes a processed lap_features CSV (like barber_r2_GR86-002-000_lap_features.csv).
    """

    if target_col not in laps.columns:
        raise KeyError(f"Target column '{target_col}' not found in laps columns: {list(laps.columns)}")

    # Target
    y = laps[target_col].astype(float)

    # Start from numeric columns only
    X = laps.select_dtypes(include=[np.number]).copy()

    drop_cols = [
        target_col,
        "is_pit_lap",       
        "lap_end_time_s",    
    ]
    for col in drop_cols:
        if col in X.columns:
            X = X.drop(columns=[col])

    # If stint_lap is missing but lap_no exists, we can fall back to it
    if "stint_lap" not in X.columns and "lap_no" in X.columns:
        X["stint_lap"] = laps["lap_no"].astype(float)

    if X.empty:
        raise ValueError("No numeric features left after filtering – check your lap_features CSV.")

    return X, y


#  training 

def train_lap_time_model(
    laps: pd.DataFrame,
    *,
    track_id: str = "barber",
    car_id: str = "GR86-002-000",
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train a RandomForestRegressor to predict lap_time_s from lap features.

    Returns:
        model_bundle: dict with estimator + feature names + metadata
        metrics: dict with basic validation metrics
    """

    # Only use non-pit laps for modelling
    if "is_pit_lap" in laps.columns:
        laps = laps[~laps["is_pit_lap"]].copy()

    X, y = _prepare_features(laps, target_col="lap_time_s")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    rmse = float(mean_squared_error(y_val, y_pred, squared=False))
    r2 = float(r2_score(y_val, y_pred))

    metrics = {
        "track_id": track_id,
        "car_id": car_id,
        "n_laps": int(len(laps)),
        "n_features": int(X.shape[1]),
        "rmse_val_s": rmse,
        "r2_val": r2,
    }

    model_bundle: Dict[str, Any] = {
        "estimator": rf,
        "feature_names": list(X.columns),
        "track_id": track_id,
        "car_id": car_id,
    }

    print(f"[lap-time model] {track_id} / {car_id}")
    print(f"  laps: {metrics['n_laps']}, features: {metrics['n_features']}")
    print(f"  val RMSE: {rmse:.3f} s, R²: {r2:.3f}")

    return model_bundle, metrics


def _default_model_path(track_id: str, car_id: str) -> Path:
    fname = f"lap_time_model_{track_id}_{car_id}.joblib"
    return MODEL_DIR / fname


def save_lap_time_model(
    model_bundle: Dict[str, Any],
    *,
    track_id: str = "barber",
    car_id: str = "GR86-002-000",
    path: Path | None = None,
) -> Path:
    """
    Save the trained model bundle to data/models/.
    """
    if path is None:
        path = _default_model_path(track_id, car_id)

    joblib.dump(model_bundle, path)
    print(f"[lap-time model] Saved to: {path}")
    return path


def load_lap_time_model(
    *,
    track_id: str = "barber",
    car_id: str = "GR86-002-000",
    path: Path | None = None,
) -> Dict[str, Any]:
    """
    Load a previously saved lap-time model bundle.
    """
    if path is None:
        path = _default_model_path(track_id, car_id)

    bundle = joblib.load(path)
    return bundle