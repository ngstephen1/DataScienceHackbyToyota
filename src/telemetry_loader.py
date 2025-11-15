from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from .track_utils import sector_from_distance
from .track_meta import TRACK_METAS


# ---------- helpers ----------

def _find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first existing column name from a list of candidates (case-insensitive),
    or None if none are found.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        col = lower_map.get(cand.lower())
        if col is not None:
            return col
    return None


@dataclass
class TelemetryColumns:
    vehicle_id: str
    meta_time: str
    distance: str                     # Laptrigger-like distance from SF (meters)
    lap_raw: Optional[str] = None     # raw lap column if present


# ---------- load telemetry ----------

def load_raw_telemetry(csv_path: str | Path) -> tuple[pd.DataFrame, TelemetryColumns]:
    """
    Generic telemetry loader.

    Assumptions (based on TRD docs / screenshots):
      - There is some column for car/vehicle id (Vehicle, CarNumber, etc.).
      - There is a 'meta_time' style column (message receive time).
      - There is a Laptrigger-like distance column in meters.
      - There may be a 'lap' column, but it can be noisy (e.g., 32768).

    This function:
      - Reads the CSV.
      - Infers key column names.
      - Parses meta_time as datetime.
      - Sorts by (vehicle_id, meta_time).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    vehicle_col = _find_first_column(
        df,
        ["Vehicle", "VEHICLE", "Car", "CAR", "car_id", "CarNumber", "CAR_NUMBER"],
    )
    if vehicle_col is None:
        raise ValueError(
            "Could not find a vehicle id column. "
            "Please update the candidate list in telemetry_loader.py "
            "once you know the exact column name."
        )

    meta_time_col = _find_first_column(df, ["meta_time", "Meta_Time", "META_TIME"])
    if meta_time_col is None:
        raise ValueError(
            "Could not find meta_time column. "
            "Add the correct name to the candidate list in telemetry_loader.py."
        )

    distance_col = _find_first_column(
        df,
        ["Laptrigger_lapdist_dls", "laptrigger_lapdist_dls", "LAPTRIGGER_LAPDIST_DLS"],
    )
    if distance_col is None:
        raise ValueError(
            "Could not find Laptrigger-style distance column. "
            "Check the CSV and update the candidate list."
        )

    lap_raw_col = _find_first_column(df, ["lap", "Lap", "LAP", "lapctr", "Lapctr"])

    # parse time + sort
    df[meta_time_col] = pd.to_datetime(df[meta_time_col], errors="coerce")
    df = df.sort_values([vehicle_col, meta_time_col]).reset_index(drop=True)

    cols = TelemetryColumns(
        vehicle_id=vehicle_col,
        meta_time=meta_time_col,
        distance=distance_col,
        lap_raw=lap_raw_col,
    )
    return df, cols


# ---------- lap reconstruction from distance ----------

def assign_laps_from_distance(
    df: pd.DataFrame,
    cols: TelemetryColumns,
    min_lap_fraction: float = 0.2,
) -> pd.DataFrame:
    """
    Use Laptrigger distance to reconstruct lap_index per vehicle.

    For each vehicle:
      - Approximate lap length from the 95th percentile of distance.
      - Start at lap_index = 0.
      - Each time distance drops from "reasonably large" to "near zero",
        increment lap_index.

    Returns a copy of df with a new integer column 'lap_index'.
    """
    df = df.copy()

    v_col = cols.vehicle_id
    t_col = cols.meta_time
    d_col = cols.distance

    df[d_col] = pd.to_numeric(df[d_col], errors="coerce")

    lap_indices = np.zeros(len(df), dtype="int32")

    for veh, g in df.groupby(v_col, sort=False):
        idx = g.index.to_numpy()
        dist = g[d_col].to_numpy()

        if np.all(np.isnan(dist)):
            lap_indices[idx] = 0
            continue

        lap_length = np.nanpercentile(dist, 95)
        if not np.isfinite(lap_length) or lap_length <= 0:
            lap_indices[idx] = 0
            continue

        lap = 0
        lap_indices[idx[0]] = lap

        for i in range(1, len(idx)):
            prev_d = dist[i - 1]
            cur_d = dist[i]

            if np.isnan(prev_d) or np.isnan(cur_d):
                lap_indices[idx[i]] = lap
                continue

            # simple reset rule: fell from reasonably large to near zero
            if prev_d > lap_length * min_lap_fraction and cur_d < lap_length * 0.1:
                lap += 1

            lap_indices[idx[i]] = lap

    df["lap_index"] = lap_indices
    return df


# ---------- sector labeling (optional, requires track_id) ----------

def add_sector_labels(
    df: pd.DataFrame,
    cols: TelemetryColumns,
    track_id: str,
) -> pd.DataFrame:
    """
    Attach a 'sector_index' column (0=S1,1=S2,2=S3) to each row
    using Laptrigger distance and TrackMeta.

    NOTE: this assumes the distance column is in meters and comparable
    to TrackMeta.circuit_length_m.
    """
    if track_id not in TRACK_METAS:
        raise ValueError(f"Unknown track_id {track_id!r}. Available: {list(TRACK_METAS.keys())}")

    df = df.copy()
    d_col = cols.distance

    def _map_row(x: float) -> int:
        try:
            return sector_from_distance(track_id, float(x))
        except Exception:
            return -1  # unknown / invalid

    df["sector_index"] = df[d_col].apply(_map_row)
    return df


# ---------- per-lap summary ----------

def build_lap_summary(
    df: pd.DataFrame,
    cols: TelemetryColumns,
) -> pd.DataFrame:
    """
    Compress high-frequency telemetry into one row per (vehicle_id, lap_index).

    Outputs at least:
      - vehicle_id
      - lap_index
      - lap_start_time, lap_end_time, lap_time_s
      - optional averages like speed, throttle, brake pressures, etc.
    """
    v_col = cols.vehicle_id
    t_col = cols.meta_time

    # try to find some common metrics
    speed_col = _find_first_column(df, ["Speed", "SPEED"])
    throttle_col = _find_first_column(df, ["ath", "ATH"])
    accel_pedal_col = _find_first_column(df, ["aps", "APS"])
    brake_f_col = _find_first_column(df, ["pbrake_f", "PBrake_F", "PBRK_F"])
    brake_r_col = _find_first_column(df, ["pbrake_r", "PBrake_R", "PBRK_R"])

    agg_dict: dict[str, List[str]] = {
        t_col: ["min", "max"],
    }
    if speed_col:
        agg_dict[speed_col] = ["mean", "max"]
    if throttle_col:
        agg_dict[throttle_col] = ["mean"]
    if accel_pedal_col:
        agg_dict[accel_pedal_col] = ["mean"]
    if brake_f_col:
        agg_dict[brake_f_col] = ["mean", "max"]
    if brake_r_col:
        agg_dict[brake_r_col] = ["mean", "max"]

    grouped = df.groupby([v_col, "lap_index"], sort=True).agg(agg_dict)

    # flatten MultiIndex columns
    grouped.columns = ["__".join([c for c in col if c]) for col in grouped.columns.values]
    grouped = grouped.reset_index()

    grouped.rename(
        columns={
            f"{t_col}__min": "lap_start_time",
            f"{t_col}__max": "lap_end_time",
        },
        inplace=True,
    )
    grouped["lap_time_s"] = (grouped["lap_end_time"] - grouped["lap_start_time"]).dt.total_seconds()

    return grouped


# ---------- convenience pipeline ----------

def build_lap_table_from_csv(
    csv_path: str | Path,
    track_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    One-shot helper for later:

      1. Load telemetry CSV.
      2. Assign lap_index per vehicle.
      3. (Optionally) attach sector_index if track_id is given.
      4. Build per-lap summary table.

    NOTE: You can call this once you have access to telemetry CSVs.
    """
    df_raw, cols = load_raw_telemetry(csv_path)
    df_laps = assign_laps_from_distance(df_raw, cols)

    if track_id is not None:
        df_laps = add_sector_labels(df_laps, cols, track_id=track_id)

    lap_summary = build_lap_summary(df_laps, cols)
    return lap_summary


if __name__ == "__main__":
    # This is just a placeholder demo call.
    # Once you have a real CSV, run e.g.:
    #   python -m src.telemetry_loader data/raw/vir/telemetry.csv virgina-international-raceway
    import sys

    if len(sys.argv) >= 2:
        path = sys.argv[1]
        track = sys.argv[2] if len(sys.argv) >= 3 else None
        print(f"Building lap table from {path!r} (track_id={track!r})...")
        laps = build_lap_table_from_csv(path, track)
        print(laps.head())
    else:
        print("Usage: python -m src.telemetry_loader <csv_path> [track_id]")