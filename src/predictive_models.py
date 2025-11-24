"""
Predictive models for lap-time estimation.

This module trains and serves simple ML models (currently RandomForestRegressor)
that map per-lap summary features (throttle, brake, sector times, etc.)
to lap-time predictions. Models are saved under models/ so they can be
reused from notebooks, CLI tools, or the Streamlit app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Choose reasonable feature columns from a lap-features DataFrame.

    Prefers throttle + brake + sector times if available, otherwise falls back
    to a smaller subset, and finally to "anything except obvious non-feature columns".
    """
    ranked_sets = [
        ["aps_mean", "pbrake_f_mean", "pbrake_r_mean",
         "sector1_time_s", "sector2_time_s", "sector3_time_s"],
        ["aps_mean", "pbrake_f_mean",
         "sector1_time_s", "sector2_time_s", "sector3_time_s"],
        ["aps_mean", "pbrake_f_mean"],
    ]

    for cand in ranked_sets:
        cols = [c for c in cand if c in df.columns]
        if len(cols) >= 2:
            return cols

    # Fallback: anything that is not clearly an ID / target column
    drop = {"lap", "lap_no", "lap_time_s", "race_id", "is_race_pace_lap"}
    return [c for c in df.columns if c not in drop]


def _model_filename(track_id: str, car_id: str) -> str:
    safe_track = track_id.replace("/", "_")
    safe_car = car_id.replace("/", "_")
    return f"lap_time_{safe_track}_{safe_car}.joblib"


# --------------------------------------------------------------------------------------
# Core training API
# --------------------------------------------------------------------------------------

def train_lap_time_model(
    laps: pd.DataFrame,
    track_id: str = "unknown",
    car_id: str = "unknown",
    random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Train a simple RandomForest lap-time model.

    Parameters
    ----------
    laps : pd.DataFrame
        Must contain at least:
          - 'lap_time_s' (float, target)
          - some of: 'aps_mean', 'pbrake_f_mean', 'pbrake_r_mean',
                     'sector1_time_s', 'sector2_time_s', 'sector3_time_s'
    track_id : str
        e.g. 'barber'
    car_id : str
        e.g. 'GR86-002-000'
    random_state : int
        For reproducibility.

    Returns
    -------
    model_bundle : dict
        {
          "model": fitted RandomForestRegressor,
          "feature_names": [...],
          "track_id": ...,
          "car_id": ...,
          "metrics": {...}
        }
    metrics : dict
        Basic validation metrics.
    """
    df = laps.copy().reset_index(drop=True)

    if "lap_time_s" not in df.columns:
        raise ValueError("laps DataFrame must contain 'lap_time_s' column.")

    # Drop rows with NaN target
    df = df[np.isfinite(df["lap_time_s"])]

    if len(df) < 6:
        raise ValueError(f"Need at least 6 laps to train a model, got {len(df)}")

    feature_cols = _select_feature_cols(df)
    if len(feature_cols) == 0:
        raise ValueError("Could not find any usable feature columns.")

    X = df[feature_cols].astype(float).to_numpy()
    y = df["lap_time_s"].astype(float).to_numpy()

    # Simple train/val split
    test_size = max(2, len(df) // 4)  # at least 2 laps in val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = float(np.sqrt(mse))  # compatible with older sklearn
    r2 = float(r2_score(y_val, y_pred))

    metrics: Dict = {
        "track_id": track_id,
        "car_id": car_id,
        "rmse_val": rmse,
        "r2_val": r2,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "features": feature_cols,
    }

    model_bundle: Dict = {
        "model": rf,
        "feature_names": feature_cols,
        "track_id": track_id,
        "car_id": car_id,
        "metrics": metrics,
    }

    return model_bundle, metrics


# --------------------------------------------------------------------------------------
# Prediction + IO helpers
# --------------------------------------------------------------------------------------

def predict_lap_times(laps: pd.DataFrame, model_bundle: Dict) -> pd.DataFrame:
    """
    Add a 'lap_time_pred_s' column to a laps DataFrame using a trained model_bundle.

    Parameters
    ----------
    laps : pd.DataFrame
        Must contain the feature columns used at training time.
    model_bundle : dict
        As returned by train_lap_time_model().

    Returns
    -------
    pd.DataFrame
        Copy of laps with an extra 'lap_time_pred_s' column.
    """
    if model_bundle is None:
        raise ValueError("model_bundle is None – train or load a model first.")

    feature_cols: List[str] = model_bundle.get("feature_names", [])
    if not feature_cols:
        raise ValueError("model_bundle is missing 'feature_names'.")

    missing = [c for c in feature_cols if c not in laps.columns]
    if missing:
        raise ValueError(
            f"Input laps DataFrame is missing feature columns: {missing}"
        )

    X = laps[feature_cols].astype(float).to_numpy()
    preds = model_bundle["model"].predict(X)

    out = laps.copy()
    out["lap_time_pred_s"] = preds
    return out


def save_lap_time_model(
    model_bundle: Dict,
    model_dir: Path | None = None,
) -> Path:
    """
    Persist a model_bundle to disk as a .joblib file and a small JSON sidecar.

    Returns
    -------
    Path
        Path to the saved .joblib file.
    """
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir.mkdir(exist_ok=True)

    track_id = model_bundle.get("track_id", "unknown")
    car_id = model_bundle.get("car_id", "unknown")

    model_path = model_dir / _model_filename(track_id, car_id)
    joblib.dump(model_bundle, model_path)

    # Optional: write metrics as JSON next to the model
    metrics = model_bundle.get("metrics", {})
    meta_path = model_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return model_path


def load_lap_time_model(
    track_id: str,
    car_id: str,
    model_dir: Path | None = None,
) -> Dict:
    """
    Load a previously saved model_bundle from disk.
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    model_path = model_dir / _model_filename(track_id, car_id)
    if not model_path.exists():
        raise FileNotFoundError(f"No model file found at {model_path}")

    model_bundle: Dict = joblib.load(model_path)
    return model_bundle


# --------------------------------------------------------------------------------------
# Convenience helper
# --------------------------------------------------------------------------------------

def predict_lap_times_for(
    track_id: str,
    car_id: str,
    laps: pd.DataFrame,
    model_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Convenience helper: load the lap-time model for (track_id, car_id)
    and apply it to a laps DataFrame.

    This is mainly for use in Streamlit or small scripts where you just
    want `laps` back with a ``lap_time_pred_s`` column.

    Parameters
    ----------
    track_id : str
        Track identifier, e.g. "barber".
    car_id : str
        Car identifier, e.g. "GR86-002-000".
    laps : pd.DataFrame
        Lap-features DataFrame containing the feature columns used at training time.
    model_dir : Path or None
        Optional override for the model directory.

    Returns
    -------
    pd.DataFrame
        A copy of `laps` with an added ``lap_time_pred_s`` column.
    """
    model_bundle = load_lap_time_model(
        track_id=track_id,
        car_id=car_id,
        model_dir=model_dir,
    )
    return predict_lap_times(laps, model_bundle)


__all__ = [
    "train_lap_time_model",
    "predict_lap_times",
    "save_lap_time_model",
    "load_lap_time_model",
    "predict_lap_times_for",
]