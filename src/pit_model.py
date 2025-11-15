from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DegradationModel:
    """
    Simple linear degradation model:
        lap_time_s ≈ base + slope * stint_lap
    """
    base: float
    slope: float

    def predict(self, stint_lap: np.ndarray | float) -> np.ndarray:
        return self.base + self.slope * np.asarray(stint_lap)


def detect_pit_laps(
    lap_df: pd.DataFrame,
    lap_time_col: str = "lap_time_s",
    threshold_factor: float = 1.6,
) -> pd.DataFrame:
    """
    Detect pit laps based on unusually long lap times.

    Parameters
    ----------
    lap_df : DataFrame
        One row per (vehicle_id, lap_index), must include lap_time_s.
    lap_time_col : str
        Column name for lap time in seconds.
    threshold_factor : float
        Laps with lap_time > threshold_factor * median for that car
        are flagged as pit laps.

    Returns
    -------
    DataFrame
        Copy of lap_df with new boolean column 'is_pit_lap'.
    """
    df = lap_df.copy()
    lt = lap_time_col

    if lt not in df.columns:
        raise ValueError(f"Expected lap time column {lt!r} in lap_df")

    # We assume there is some vehicle identifier column
    veh_col_candidates = [c for c in df.columns if c.lower() in ("vehicle", "car", "car_id", "carnumber", "car_number")]
    if not veh_col_candidates:
        raise ValueError("Could not infer vehicle id column in lap_df.")
    veh_col = veh_col_candidates[0]

    df["is_pit_lap"] = False

    for veh, g in df.groupby(veh_col):
        median_lt = g[lt].median()
        if not np.isfinite(median_lt) or median_lt <= 0:
            continue

        threshold = threshold_factor * median_lt
        mask = g[lt] > threshold
        df.loc[g.index[mask], "is_pit_lap"] = True

    return df


def add_stint_index(
    lap_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    From lap_df with 'is_pit_lap', derive:
      - 'stint_id' (increments after each pit lap)
      - 'stint_lap' (0,1,2,... within each stint)
    """
    if "is_pit_lap" not in lap_df.columns:
        raise ValueError("lap_df must contain 'is_pit_lap' column.")

    df = lap_df.copy()

    # infer vehicle column
    veh_col_candidates = [c for c in df.columns if c.lower() in ("vehicle", "car", "car_id", "carnumber", "car_number")]
    if not veh_col_candidates:
        raise ValueError("Could not infer vehicle id column in lap_df.")
    veh_col = veh_col_candidates[0]

    stint_ids = np.zeros(len(df), dtype="int32")
    stint_laps = np.zeros(len(df), dtype="int32")

    for veh, g in df.groupby(veh_col, sort=True):
        # We assume laps are already sorted by lap_index
        g_sorted = g.sort_values("lap_index")
        idx = g_sorted.index.to_numpy()
        is_pit = g_sorted["is_pit_lap"].to_numpy()

        stint = 0
        lap_in_stint = 0

        for i, row_idx in enumerate(idx):
            stint_ids[row_idx] = stint
            stint_laps[row_idx] = lap_in_stint

            if is_pit[i]:
                stint += 1
                lap_in_stint = 0
            else:
                lap_in_stint += 1

    df["stint_id"] = stint_ids
    df["stint_lap"] = stint_laps
    return df


def fit_degradation_model(
    lap_df: pd.DataFrame,
    lap_time_col: str = "lap_time_s",
    min_stint_laps: int = 4,
) -> Tuple[DegradationModel, pd.DataFrame]:
    """
    Fit a simple linear degradation model:
        lap_time_s ≈ base + slope * stint_lap

    Uses only non-pit laps, and only stints with at least min_stint_laps.

    Returns
    -------
    model : DegradationModel
        Fitted base and slope across all stints.
    df_used : DataFrame
        The subset of lap_df used for fitting.
    """
    required_cols = {"stint_id", "stint_lap", lap_time_col}
    missing = required_cols - set(lap_df.columns)
    if missing:
        raise ValueError(f"lap_df is missing required columns: {missing}")

    df = lap_df.copy()

    # drop pit laps if is_pit_lap is present
    if "is_pit_lap" in df.columns:
        df = df[~df["is_pit_lap"].astype(bool)].copy()

    # filter out very short stints
    stint_counts = df.groupby("stint_id")["stint_lap"].max() + 1
    valid_stints = stint_counts[stint_counts >= min_stint_laps].index
    df_used = df[df["stint_id"].isin(valid_stints)].copy()

    if df_used.empty:
        raise ValueError("No valid stints found to fit degradation model.")

    x = df_used["stint_lap"].to_numpy(dtype=float)
    y = df_used[lap_time_col].to_numpy(dtype=float)

    # Simple linear regression (least squares)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, base = np.linalg.lstsq(A, y, rcond=None)[0]

    model = DegradationModel(base=base, slope=slope)
    return model, df_used