from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Import local modules
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from track_meta import TRACK_METAS  # type: ignore
from strategy_engine import (      # type: ignore
    make_config_from_meta,
    simulate_strategy_with_caution,
    run_multiverse,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_lap_features(track_id: str, race: str, car_id: str) -> pd.DataFrame:
    """
    Load per-lap features for a given track / race / car.

    Convention (per track):

        data/processed/<short_name>/<short_name>_<race_lower>_<CAR_ID>_lap_features.csv

    Example for Barber R2, GR86-002-000:
        data/processed/barber/barber_r2_GR86-002-000_lap_features.csv

    You need to generate the corresponding files for VIR, COTA, etc.
    """

    track_dirs = {
        "barber-motorsports-park": "barber",
        "virginia-international-raceway": "virginia",
        "circuit-of-the-americas": "cota",
    }

    if track_id not in track_dirs:
        raise ValueError(f"Track {track_id} not wired yet in load_lap_features().")

    short = track_dirs[track_id]

    # race is like "R1" or "R2" from the UI
    race_lower = race.lower()  # "r1" / "r2"

    proc_dir = ROOT / "data" / "processed" / short
    fname = f"{short}_{race_lower}_{car_id}_lap_features.csv"
    path = proc_dir / fname

    if not path.exists():
        raise FileNotFoundError(
            f"Lap feature file not found for {track_id}, {race}, {car_id}: {path}"
        )

    return pd.read_csv(path)


def build_strategies(total_laps: int) -> dict[str, list[int]]:
    """
    Simple 3-strategy set for now.
      - 1-stop_early : earlier than mid-race
      - 1-stop_mid   : around mid-race
      - 2-stop       : two shorter stints
    """
    one_stop_mid = [total_laps // 2]
    one_stop_early = [max(3, total_laps // 2 - 4)]
    two_stop = [total_laps // 3, 2 * total_laps // 3]

    return {
        "1-stop_early": one_stop_early,
        "1-stop_mid": one_stop_mid,
        "2-stop": two_stop,
    }


# ---------------------------------------------------------------------
# Streamlit UI setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Racing Hokies – GR RaceCraft Copilot",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏁 Racing Hokies – GR RaceCraft Copilot (Barber MVP)")

st.markdown(
    """
**Team:** Tue Tran Minh & Nguyen Phan Nguyen – Virginia Tech 🦃  

Real-time strategy sandbox for the **GR Cup Series** using TRD telemetry.

For this MVP we focus on **Barber Motorsports Park, Race 2, Car #2** and:
- Compare **pit strategies** (1-stop early, 1-stop mid, 2-stop)  
- Let you drop a **caution window** and see which strategy wins  
- Run a small **“mini multiverse”** (Monte Carlo) of random cautions  
- Provide a basic **Driver Insights** view for lap-time and consistency analysis  
"""
)

# ---------------------------------------------------------------------
# Sidebar – configuration
# ---------------------------------------------------------------------
st.sidebar.header("Configuration")

track_options = {
    "Barber Motorsports Park": "barber-motorsports-park",
    "Virginia International Raceway": "virginia-international-raceway",
    "Circuit of the Americas": "circuit-of-the-americas",  # future
}
track_label = st.sidebar.selectbox("Track", list(track_options.keys()), index=0)
track_id = track_options[track_label]

race = st.sidebar.selectbox("Race", ["R2", "R1"], index=0)
car_id = st.sidebar.text_input("Car ID", value="GR86-002-000")

n_sims = st.sidebar.slider(
    "Mini multiverse size (simulations)",
    min_value=100,
    max_value=1000,
    value=500,
    step=50,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Caution scenario")

use_caution = st.sidebar.checkbox("Include specific caution window?", value=True)

# ---------------------------------------------------------------------
# Load data & basic config
# ---------------------------------------------------------------------
try:
    lap_df = load_lap_features(track_id, race, car_id)
    meta = TRACK_METAS[track_id]
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

total_laps = int(lap_df["lap"].max())
cfg = make_config_from_meta(lap_df, pit_lane_time_s=meta.pit_lane_time_s)
strategies = build_strategies(total_laps)

# Now that we know lap count, define caution sliders (if enabled)
if use_caution:
    caution_lap = st.sidebar.slider(
        "Caution start lap",
        min_value=2,
        max_value=max(3, total_laps - 3),
        value=10,
        step=1,
    )
    caution_len = st.sidebar.slider(
        "Caution length (laps)",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
    )
else:
    caution_lap = None
    caution_len = 0

# ---------------------------------------------------------------------
# Tabs: Strategy Brain + Driver Insights
# ---------------------------------------------------------------------
tab_strategy, tab_driver = st.tabs(["🧠 Strategy Brain", "📊 Driver Insights"])

# ========================== STRATEGY BRAIN TAB =========================
with tab_strategy:
    col_info, col_deterministic, col_multiverse = st.columns([1.1, 1.4, 1.5])

    # --- track & car context ------------------------------------------------
    with col_info:
        st.subheader("Track & car context")

        st.markdown(
            f"""
**Track:** {meta.name}  
**Race:** {race}  
**Car:** `{car_id}`  

- Laps: **{total_laps}**  
- Base race pace (trimmed median): **{cfg.base_lap_s:.2f} s**  
- Pit lane loss (green): **{cfg.pit_loss_s:.2f} s**  
"""
        )

        st.markdown("**Strategies defined**")
        for name, pits in strategies.items():
            pit_str = ", ".join(str(p) for p in pits)
            st.markdown(f"- `{name}` → pit on lap(s): {pit_str}")

    # --- deterministic scenario (specific caution) --------------------------
    with col_deterministic:
        st.subheader("Scenario: specific caution window")

        if use_caution and caution_lap is not None:
            st.markdown(
                f"Caution from **lap {caution_lap}** "
                f"for **{caution_len} laps** (reduced pit loss during yellow)."
            )

            rows: list[dict[str, float | str]] = []
            for name, pits in strategies.items():
                total_time = simulate_strategy_with_caution(
                    n_laps=total_laps,
                    pit_laps=pits,
                    cfg=cfg,
                    caution_start=caution_lap,
                    caution_len=caution_len,
                )
                rows.append(
                    {
                        "strategy": name,
                        "total_time_s": total_time,
                        "total_time_min": total_time / 60.0,
                    }
                )

            det_df = pd.DataFrame(rows).sort_values("total_time_s")
            det_df["delta_vs_best_s"] = det_df["total_time_s"] - det_df["total_time_s"].min()

            st.dataframe(det_df, use_container_width=True)

            best_row = det_df.iloc[0]
            st.success(
                f"**Best in this scenario:** `{best_row['strategy']}`  "
                f"(≈ {best_row['total_time_min']:.2f} min, "
                f"{best_row['delta_vs_best_s']:.2f} s faster than alternatives)"
            )
        else:
            st.info(
                "Turn on **'Include specific caution window?'** in the sidebar "
                "to compare strategies for a particular yellow."
            )

    # --- multiverse (random cautions) ---------------------------------------
    with col_multiverse:
        st.subheader("Mini multiverse – random cautions")

        summary = run_multiverse(
            strategies=strategies,
            n_laps=total_laps,
            cfg=cfg,
            n_sims=n_sims,
        )

        st.dataframe(summary, use_container_width=True)

        st.bar_chart(summary["win_prob"])

        best_name = summary["win_prob"].idxmax()
        best_row = summary.loc[best_name]

        st.success(
            f"**Baseline recommendation:** `{best_name}`  \n"
            f"- Win probability across universes: **{best_row['win_prob']:.1%}**  \n"
            f"- Mean race time: **{best_row['mean_time_s']/60:.2f} min**"
        )

# ========================== DRIVER INSIGHTS TAB =========================
with tab_driver:
    st.subheader("Lap-time and consistency insights")

    # --- Lap-time profile ---------------------------------------------------
    st.markdown("### Lap-time profile")

    if "lap_time_s" in lap_df.columns:
        display_df = lap_df[["lap", "lap_time_s"]].copy()
        display_df = display_df.sort_values("lap")

        # Basic stats (ignoring pit laps if we have that flag)
        race_laps = lap_df.copy()
        if "is_pit_lap" in race_laps.columns:
            race_laps_nopit = race_laps[~race_laps["is_pit_lap"]]
        else:
            race_laps_nopit = race_laps

        if not race_laps_nopit.empty:
            med_lt = race_laps_nopit["lap_time_s"].median()
            best_lt = race_laps_nopit["lap_time_s"].min()
            std_lt = race_laps_nopit["lap_time_s"].std()

            st.markdown(
                f"""
- Laps considered (no pit laps if flag available): **{len(race_laps_nopit)}**  
- Best clean lap: **{best_lt:.3f} s**  
- Median clean lap: **{med_lt:.3f} s**  
- Lap-time spread (std dev): **{std_lt:.3f} s**  
"""
            )

        st.line_chart(
            display_df.set_index("lap"),
            height=260,
        )

        st.caption(
            "Line chart: lap time vs lap number. Dips = good laps, spikes often = traffic, caution, or mistakes."
        )
    else:
        st.info("`lap_time_s` column not found in lap_features – driver insights are limited.")

    # --- Best laps table ----------------------------------------------------
    st.markdown("### Best laps")

    if "lap_time_s" in lap_df.columns:
        race_laps = lap_df.copy()
        if "is_pit_lap" in race_laps.columns:
            race_laps = race_laps[~race_laps["is_pit_lap"]]

        if not race_laps.empty:
            best_laps = race_laps.nsmallest(5, "lap_time_s")[["lap", "lap_time_s"]]
            st.dataframe(best_laps, use_container_width=True)
        else:
            st.info("No non-pit laps available to rank.")
    else:
        st.info("Cannot compute best laps without `lap_time_s`.")

    # --- Optional sector comparison hook -----------------------------------
    st.markdown("### (Optional) Sector comparison vs field")

    # We only show something if a precomputed sector file exists
    sector_path = ROOT / "data" / "processed" / "barber" / f"barber_{race.lower()}_sector_stats_all_cars.csv"
    if sector_path.exists():
        try:
            sec_df = pd.read_csv(sector_path)
            if "vehicle_id" in sec_df.columns:
                car_sec = sec_df[sec_df["vehicle_id"] == car_id]
                if not car_sec.empty:
                    car_row = car_sec.iloc[0]
                    st.markdown("Sector stats loaded from precomputed file.")
                    st.write(car_row)
                else:
                    st.info(
                        f"Sector stats file found, but no row for vehicle_id `{car_id}`. "
                        "Check that IDs match."
                    )
            else:
                st.info(
                    "Sector stats file found, but it doesn't contain a `vehicle_id` column. "
                    "Adjust column names in the driver insights tab if needed."
                )
        except Exception as e:
            st.info(f"Could not parse sector stats file: {e}")
    else:
        st.caption(
            "No sector comparison file found yet. You can generate one from the Barber R1/R2 "
            "section notebooks and save as "
            "`data/processed/barber/barber_<race>_sector_stats_all_cars.csv`."
        )

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Racing Hokies – Barber-only MVP. Strategy engine is track-agnostic; "
    "support for VIR, COTA, Sebring, Sonoma, and Virginia International Raceway "
    "can be added by wiring additional lap_features tables and TRACK_METAS entries."
)