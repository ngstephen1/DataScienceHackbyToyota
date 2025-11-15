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
    For now we only have Barber wired with:
        data/processed/barber/barber_r{1,2}_{CAR_ID}_lap_features.csv
    """
    if track_id != "barber-motorsports-park":
        raise ValueError("Currently only Barber is wired – more tracks coming soon.")

    proc_dir = ROOT / "data" / "processed" / "barber"
    fname = f"barber_{race.lower()}_{car_id}_lap_features.csv"
    path = proc_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Lap feature file not found: {path}")
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
# Streamlit UI
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
"""
)

# ---------------------------------------------------------------------
# Sidebar – configuration
# ---------------------------------------------------------------------
st.sidebar.header("Configuration")

track_options = {
    "Barber Motorsports Park": "barber-motorsports-park",
    # Later: VIR, COTA, Sebring, etc.
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

# ---------------------------------------------------------------------
# Sidebar – caution controls (need lap count)
# ---------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Caution scenario")

use_caution = st.sidebar.checkbox("Include specific caution window?", value=True)

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
# Main layout
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Racing Hokies – Barber-only MVP. Strategy engine is track-agnostic; "
    "support for VIR, COTA, Sebring, Sonoma, and Virginia International Raceway "
    "can be added by wiring additional lap_features tables and TRACK_METAS entries."
)