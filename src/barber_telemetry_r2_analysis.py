#!/usr/bin/env python3
"""
Barber R2 sector analysis script.

This script:
- Loads raw telemetry CSV
- Converts sector time columns to seconds (float)
- Computes mean, min, std for each car
- Highlights a target car (CAR_NUM)
- Generates boxplots comparing sector times across field
- Saves results to processed/ directory
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- CONFIG --------------------------------- #

# Column names from original CSV
VEH_COL = "NUMBER"       # or "DRIVER_NUMBER" depending on dataset
LAP_COL = "LAP_NUMBER"

S1_COL = "S1_SECONDS"
S2_COL = "S2_SECONDS"
S3_COL = "S3_SECONDS"

# Default paths
RAW_DATA_PATH = Path(
    "../notebooks/data/raw/barber/23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV")

PROCESSED_DIR = Path("../data/processed/barber")


# --------------------------- DATA LOADING ------------------------------ #

def load_sector_data(csv_path: Path) -> pd.DataFrame:
    """Load sector data and convert columns to numeric format."""
    print(f"[INFO] Loading: {csv_path} (exists: {csv_path.exists()})")

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, sep=";")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV: {e}") from e

    # Normalize column names
    df.columns = df.columns.str.strip()

    print(f"[INFO] Data loaded: shape={df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Convert sector time columns
    for col, new_col in [
        (S1_COL, "S1_s"),
        (S2_COL, "S2_s"),
        (S3_COL, "S3_s"),
    ]:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in CSV")

        try:
            df[new_col] = df[col].astype(float)
        except ValueError as e:
            raise ValueError(f"Cannot convert {col} to float: {e}") from e

    # Remove rows with missing sector times
    before = len(df)
    df = df.dropna(subset=["S1_s", "S2_s", "S3_s"])
    after = len(df)
    if after < before:
        print(
            f"[INFO] Removed {before - after} rows with missing sector times")

    return df


# -------------------------- ANALYSIS LOGIC ----------------------------- #

def calculate_sector_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sector statistics (mean, min, std) per car."""
    print("[INFO] Calculating sector statistics...")

    grouped = (
        df.groupby(VEH_COL)[["S1_s", "S2_s", "S3_s"]]
        .agg(["mean", "min", "std"])
    )

    grouped.columns = [
        f"{base}_{stat}" for (base, stat) in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()

    print(f"[INFO] Stats computed: shape={grouped.shape}")
    return grouped


def add_is_our_car_flag(sector_stats: pd.DataFrame, car_num: int) -> pd.DataFrame:
    """Add boolean flag marking our car for visualization."""
    sector_stats = sector_stats.copy()
    sector_stats["is_our_car"] = sector_stats[VEH_COL] == car_num

    if not sector_stats["is_our_car"].any():
        print(f"[WARN] Car {car_num} not found.")
    else:
        print(f"[INFO] Car {car_num} found.")
    return sector_stats


def rank_column(sector_stats: pd.DataFrame, car_num: int, colname: str) -> float:
    """Return ranking for a specific sector column (1 = best)."""
    ranks = sector_stats[colname].rank(method="min")  # lower is better
    mask = sector_stats[VEH_COL] == car_num

    if not mask.any():
        raise ValueError(f"Car {car_num} not found when ranking.")

    return ranks.loc[mask].iloc[0]


def build_summary_for_car(sector_stats: pd.DataFrame, car_num: int) -> pd.DataFrame:
    """Create summary table for selected car."""
    mask = sector_stats[VEH_COL] == car_num
    if not mask.any():
        print(f"[WARN] No data for car {car_num}.")
        return pd.DataFrame()

    row = sector_stats.loc[mask].iloc[0]

    summary = pd.DataFrame([
        {
            "car_number": car_num,
            "S1_mean_s": row["S1_s_mean"],
            "S1_mean_rank": rank_column(sector_stats, car_num, "S1_s_mean"),
            "S2_mean_s": row["S2_s_mean"],
            "S2_mean_rank": rank_column(sector_stats, car_num, "S2_s_mean"),
            "S3_mean_s": row["S3_s_mean"],
            "S3_mean_rank": rank_column(sector_stats, car_num, "S3_s_mean"),
        }
    ])

    return summary


# ------------------------------ PLOTS ---------------------------------- #

def plot_sector_boxplots(sector_stats: pd.DataFrame, car_num: int, out_dir: Path) -> None:
    """Plot sector mean times and highlight target car."""
    if sector_stats.empty:
        print("[WARN] No data to plot.")
        return

    print("[INFO] Creating sector boxplots...")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    sectors = [
        ("S1", "S1_s_mean"),
        ("S2", "S2_s_mean"),
        ("S3", "S3_s_mean"),
    ]

    for i, (label, col) in enumerate(sectors):
        ax = axes[i]
        vals = sector_stats[col].values

        ax.boxplot(vals, vert=True, labels=[label])

        our_car = sector_stats[sector_stats["is_our_car"]]
        if not our_car.empty:
            our_val = our_car[col].iloc[0]
            ax.scatter(1, our_val, color="red", zorder=3)
            ax.set_title(f"{label} mean time (car {car_num} marked)")
        else:
            ax.set_title(f"{label} mean time (car {car_num} not found)")

        ax.set_ylabel("Seconds")

    plt.suptitle("Barber R2 – Sector Mean Times Across Field")
    plt.tight_layout()

    out_path = out_dir / f"barber_r2_sector_means_car{car_num}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Plot saved → {out_path}")


# ---------------------------- OUTPUT SAVE ------------------------------ #

def save_outputs(sector_stats: pd.DataFrame, summary_df: pd.DataFrame, car_num: int, out_dir: Path):
    """Write results to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cars_path = out_dir / "barber_r2_sector_stats_all_cars.csv"
    sector_stats.to_csv(all_cars_path, index=False)
    print(f"[INFO] Saved sector stats → {all_cars_path}")

    if not summary_df.empty:
        summary_path = out_dir / f"barber_r2_car{car_num}_sector_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Saved summary → {summary_path}")
    else:
        print(f"[WARN] Summary empty → not saved.")


# ------------------------------ MAIN ----------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sector analysis for Barber R2 telemetry.")
    parser.add_argument("--csv", type=str,
                        default=str(RAW_DATA_PATH), help="Path to CSV file.")
    parser.add_argument("--car", type=int, default=2,
                        help="Target car number.")
    parser.add_argument("--outdir", type=str,
                        default=str(PROCESSED_DIR), help="Output directory.")
    return parser.parse_args()


def main():
    print("=== PROGRAM STARTED ===")

    args = parse_args()
    csv_path = Path(args.csv)
    car_num = args.car
    out_dir = Path(args.outdir)

    df_sec = load_sector_data(csv_path)
    sector_stats = calculate_sector_stats(df_sec)
    sector_stats = add_is_our_car_flag(sector_stats, car_num)

    summary_df = build_summary_for_car(sector_stats, car_num)

    if not summary_df.empty:
        print("[INFO] Summary for selected car:")
        try:
            print(summary_df.to_markdown(index=False))
        except Exception:
            print(summary_df.to_string(index=False))

    save_outputs(sector_stats, summary_df, car_num, out_dir)
    plot_sector_boxplots(sector_stats, car_num, out_dir)

    print("=== PROGRAM FINISHED ===")


if __name__ == "__main__":
    main()
