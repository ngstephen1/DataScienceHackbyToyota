from __future__ import annotations

"""
hello wo
"""

from dataclasses import dataclass
from typing import Iterable, Optional

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from .predictive_models import predict_lap_times_for


@dataclass
class PlotRequirements:
    cols: Iterable[str]
    message: str


def _check_columns(df: pd.DataFrame, req: PlotRequirements) -> bool:
    """
    Return True if df contains all required columns; otherwise show a Streamlit
    info message and return False.
    """
    missing = [c for c in req.cols if c not in df.columns]
    if missing:
        st.info(f"{req.message} (missing columns: {', '.join(missing)})")
        return False
    return True


# Lap time profile + best laps


def render_lap_time_profile(
    lap_df: pd.DataFrame,
    title: str | None = None,
    show_stats: bool = True,
    height: int = 260,
) -> None:
    """
    Standard lap-time vs lap profile.

    Expects at minimum:
      - lap
      - lap_time_s
    Optionally uses:
      - is_pit_lap
    """
    req = PlotRequirements(
        cols=["lap", "lap_time_s"],
        message="Lap-time profile unavailable.",
    )
    if not _check_columns(lap_df, req):
        return

    race_laps = lap_df.copy().sort_values("lap")

    if "is_pit_lap" in race_laps.columns:
        clean = race_laps[~race_laps["is_pit_lap"]]
    else:
        clean = race_laps

    if clean.empty:
        st.info("No non-pit laps available to plot.")
        return

    if show_stats:
        med_lt = float(clean["lap_time_s"].median())
        best_lt = float(clean["lap_time_s"].min())
        std_lt = float(clean["lap_time_s"].std())

        st.markdown(
            f"""
- Laps considered: **{len(clean)}**  
- Best clean lap: **{best_lt:.3f} s**  
- Median clean lap: **{med_lt:.3f} s**  
- Lap-time spread (std dev): **{std_lt:.3f} s**  
"""
        )

    if title:
        st.markdown(f"#### {title}")

    # Streamlit-native chart; fast & interactive
    st.line_chart(
        race_laps[["lap", "lap_time_s"]].set_index("lap"),
        height=height,
    )
    st.caption(
        "Lap time vs lap number. Dips = strong laps, spikes often = traffic, "
        "caution, or driver mistakes."
    )


def render_best_laps_table(
    lap_df: pd.DataFrame,
    top_n: int = 5,
    include_caption: bool = True,
) -> None:
    """
    Simple best-laps table.

    Expects:
      - lap_time_s
      - lap
    Optionally:
      - is_pit_lap
    """
    req = PlotRequirements(
        cols=["lap", "lap_time_s"],
        message="Cannot compute best laps without lap_time_s.",
    )
    if not _check_columns(lap_df, req):
        return

    race_laps = lap_df.copy()
    if "is_pit_lap" in race_laps.columns:
        race_laps = race_laps[~race_laps["is_pit_lap"]]

    if race_laps.empty:
        st.info("No non-pit laps available to rank.")
        return

    best_laps = race_laps.nsmallest(top_n, "lap_time_s")[["lap", "lap_time_s"]]
    st.dataframe(best_laps, use_container_width=True)

    if include_caption:
        st.caption(
            f"Top {top_n} clean laps ranked by lap time. "
            "Use this to spot peak performance and benchmark stints."
        )


# Predictive model overlays

def render_actual_vs_predicted_overlay(
    lap_df: pd.DataFrame,
    track_id_short: str,
    race: str,
    car_id: str,
    height: int = 260,
) -> None:
    """
    Overlay actual lap times vs predicted lap times from the trained model.

    - track_id_short: short ID used in predictive_models (e.g. 'barber').
    - race: 'R1' or 'R2' (if your model is race-specific).
    - car_id: telemetry car ID.

    This is meant for the Driver Insights tab (Streamlit line_chart).
    """
    req = PlotRequirements(
        cols=["lap", "lap_time_s"],
        message="Cannot overlay predictions – lap_time_s or lap missing.",
    )
    if not _check_columns(lap_df, req):
        return

    base = lap_df[["lap", "lap_time_s"]].dropna()

    try:
        pred_df = predict_lap_times_for(
            track_id=track_id_short,
            race=race,
            car_id=car_id,
        )
    except FileNotFoundError:
        st.caption(
            "No trained lap-time model found yet for this car/track. "
            "Train and save a model in `11_barber_predictive_model.ipynb` first."
        )
        return
    except Exception as e:
        st.caption(f"Predictive model unavailable in this session: {e}")
        return

    if "lap" not in pred_df.columns or "lap_time_pred_s" not in pred_df.columns:
        st.caption(
            "Predictive model output does not contain expected columns "
            "(`lap`, `lap_time_pred_s`). Check `predictive_models.py`."
        )
        return

    merged = (
        base.merge(pred_df[["lap", "lap_time_pred_s"]], on="lap", how="left")
        .set_index("lap")
    )

    st.markdown("### Predictive lap-time model overlay")
    st.line_chart(
        merged[["lap_time_s", "lap_time_pred_s"]],
        height=height,
    )
    st.caption(
        "Overlay: actual lap times vs Random Forest model prediction using throttle + "
        "brake features. Separation = modelling error or unmodelled effects (traffic, "
        "caution, driver mistakes)."
    )


def render_predicted_vs_actual_matplotlib(
    laps_df: pd.DataFrame,
    track_id_short: str,
    car_id: str,
    race: Optional[str] = None,
    height_px: int = 320,
) -> None:
    """
    Matplotlib version of actual vs predicted lap times (used in Predictive Models tab).

    This uses the predictive_models.predict_lap_times_for() that can accept an
    explicit laps_df if you’ve already loaded and filtered it.

    - laps_df: dataframe with at least lap and lap_time_s.
    - track_id_short: e.g. 'barber'.
    - car_id: car identifier string.
    - race: optional; depends on how your predictive_models is wired.
    """
    req = PlotRequirements(
        cols=["lap", "lap_time_s"],
        message="Cannot plot predicted vs actual – lap_time_s or lap missing.",
    )
    if not _check_columns(laps_df, req):
        return

    try:
        laps_pred = predict_lap_times_for(
            track_id=track_id_short,
            car_id=car_id,
            laps=laps_df,
        )
    except FileNotFoundError:
        st.warning(
            "No trained model file found yet. Train a model in the notebook and "
            "save it under `models/`."
        )
        return
    except Exception as e:
        st.error(f"Problem running predictions: {e}")
        return

    req2 = PlotRequirements(
        cols=["lap", "lap_time_s", "lap_time_pred_s"],
        message="Unexpected columns from predictive model; cannot plot.",
    )
    if not _check_columns(laps_pred, req2):
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(laps_pred["lap"], laps_pred["lap_time_s"], "o-", label="Actual")
    ax.plot(
        laps_pred["lap"],
        laps_pred["lap_time_pred_s"],
        "s--",
        label="Predicted",
    )
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    title = f"Actual vs predicted lap times – {track_id_short}, {car_id}"
    if race is not None:
        title += f" ({race})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7)

    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Scatter/line plot of actual vs predicted lap times. Large deviations or "
        "systematic trends may indicate missing features, traffic, or unmodelled "
        "race dynamics."
    )

# VIR sector evolution and sector comparison helpers


def render_vir_sector_pace(
    driver_sector_df: pd.DataFrame,
    car_label: str = "Car #2",
) -> None:
    """
    Plot VIR sector mean times by race for a single car.

    Expected columns (example):
      - race (e.g. 'R1', 'R2')
      - S1_mean_s
      - S2_mean_s
      - S3_mean_s
    """
    req = PlotRequirements(
        cols=["race", "S1_mean_s", "S2_mean_s", "S3_mean_s"],
        message="VIR sector summary missing expected columns.",
    )
    if not _check_columns(driver_sector_df, req):
        return

    st.subheader(f"VIR – {car_label} sector pace (R1 vs R2)")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=False)
    for ax, sec in zip(axes, ["S1", "S2", "S3"]):
        ax.bar(driver_sector_df["race"], driver_sector_df[f"{sec}_mean_s"])
        ax.set_title(f"{sec} mean time")
        ax.set_ylabel("time (s)")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        """
- Differences between **R1** and **R2** by sector highlight where pace was found or lost.  
- Use this view alongside lap-time profiles to decide whether setup or driving changes
  helped specific parts of the lap (launch, mid-lap, final sector).
"""
    )


def render_sector_comparison_for_car(
    sector_df: pd.DataFrame,
    car_id: str,
    lap_col_prefix: str = "S",
    time_suffix: str = "_mean_s",
) -> None:
    """
    Render a simple textual summary of sector stats for a given car from a
    precomputed all-cars sector summary file.

    The intent is to be reused in Driver Insights OR triggered by the chatbot.

    Expected columns:
      - vehicle_id
      - <lap_col_prefix><i><time_suffix>, e.g. S1_mean_s, S2_mean_s, S3_mean_s
    """
    if "vehicle_id" not in sector_df.columns:
        st.info(
            "Sector stats file doesn't contain `vehicle_id` column. "
            "Cannot filter by car."
        )
        return

    car_rows = sector_df[sector_df["vehicle_id"] == car_id]
    if car_rows.empty:
        st.info(
            f"Sector stats file loaded, but no row for vehicle_id `{car_id}`. "
            "Check that IDs match between telemetry and sector export."
        )
        return

    car_row = car_rows.iloc[0]
    st.markdown("**Sector stats for this car (from precomputed file):**")
    st.write(car_row)

    # Optionally: bar chart of sector means if columns exist
    sec_cols = [
        c for c in sector_df.columns if c.startswith(lap_col_prefix) and c.endswith(time_suffix)
    ]
    if sec_cols:
        vals = [car_row[c] for c in sec_cols]
        labels = [c.replace(lap_col_prefix, "S").replace(time_suffix, "") for c in sec_cols]

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(labels, vals)
        ax.set_ylabel("mean time (s)")
        ax.set_title("Sector mean times for this car")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Sector-level mean lap times for this car. Combine this with VIR or Barber "
            "sector analysis to understand where the car is strong or weak."
        )


def render_plot_by_intent(
    intent: str,
    *,
    lap_df: Optional[pd.DataFrame] = None,
    track_id_short: Optional[str] = None,
    race: Optional[str] = None,
    car_id: Optional[str] = None,
    sector_df: Optional[pd.DataFrame] = None,
    vir_sector_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Convenience router so the chat assistant (or other tools) can request a
    high-level plot by name and we dispatch to the correct helper.

    Example intents (you can extend in src/chat_assistant.py):
      - "lap_time_profile"
      - "actual_vs_predicted"
      - "predicted_vs_actual_detailed"
      - "vir_sector_pace"
      - "sector_comparison"
    """
    intent = (intent or "").strip().lower()

    if intent == "lap_time_profile":
        if lap_df is None:
            st.info("Lap data not provided to plot lap-time profile.")
            return
        render_lap_time_profile(lap_df)

    elif intent == "actual_vs_predicted":
        if lap_df is None or not track_id_short or not race or not car_id:
            st.info("Need lap_df, track_id_short, race, and car_id for overlay plot.")
            return
        render_actual_vs_predicted_overlay(
            lap_df=lap_df,
            track_id_short=track_id_short,
            race=race,
            car_id=car_id,
        )

    elif intent == "predicted_vs_actual_detailed":
        if lap_df is None or not track_id_short or not car_id:
            st.info("Need lap_df, track_id_short, and car_id for detailed plot.")
            return
        render_predicted_vs_actual_matplotlib(
            laps_df=lap_df,
            track_id_short=track_id_short,
            car_id=car_id,
            race=race,
        )

    elif intent == "vir_sector_pace":
        if vir_sector_df is None:
            st.info("VIR sector dataframe not provided.")
            return
        render_vir_sector_pace(vir_sector_df)

    elif intent == "sector_comparison":
        if sector_df is None or not car_id:
            st.info("Need sector_df and car_id to show sector comparison.")
            return
        render_sector_comparison_for_car(sector_df, car_id=car_id)

    else:
        st.info(
            f"Plot intent `{intent}` not recognised yet. "
            "You can extend `render_plot_by_intent` and the chat intent classifier "
            "to support more plot types."
        )