from __future__ import annotations
from src.predictive_models import predict_lap_times_for
from pathlib import Path
import sys
import time
import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

DATA_ROOT = ROOT / "data"
DATA_PROCESSED = DATA_ROOT / "processed"
TRACK_GEOM_DIR = DATA_ROOT / "track_geom"

# ---------- Local live-state writer (shared with Tk car-map viewer) ----------
LIVE_DIR = DATA_ROOT / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)


def get_live_state_path(session_id: str = "barber") -> Path:
    safe_id = session_id.replace("/", "_")
    return LIVE_DIR / f"live_state_{safe_id}.json"


def save_live_state(session_id: str, payload: dict) -> None:
    stamped = dict(payload)
    stamped.setdefault("session_id", session_id)
    stamped["saved_at"] = time.time()

    out_path = get_live_state_path(session_id)
    tmp_path = out_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(stamped, f, ensure_ascii=False, indent=2)
    tmp_path.replace(out_path)


from track_meta import TRACK_METAS  # type: ignore
from strategy_engine import (      # type: ignore
    make_config_from_meta,
    simulate_strategy_with_caution,
    run_multiverse,
)


# ---------- Optional Barber polyline for car position ----------
BARBER_GEOM: np.ndarray | None = None
barber_geom_path = TRACK_GEOM_DIR / "barber_track_xy.csv"
if barber_geom_path.exists():
    try:
        df_geom = pd.read_csv(barber_geom_path)
        if {"x_norm", "y_norm"}.issubset(df_geom.columns):
            BARBER_GEOM = df_geom[["x_norm", "y_norm"]].to_numpy()
    except Exception:
        BARBER_GEOM = None


def barber_lap_to_xy(lap: int, max_lap: int) -> tuple[float | None, float | None]:
    """
    Map race progress (lap / max_lap) to a point on the Barber polyline (0–1 normalised).
    This is race-progress based, not “per lap around the track”.
    """
    global BARBER_GEOM
    if BARBER_GEOM is None or len(BARBER_GEOM) == 0 or max_lap <= 1:
        return None, None

    s = (lap - 1) / float(max_lap - 1)  # 0 → 1 over race
    s = min(max(s, 0.0), 1.0)
    idx = int(round(s * (len(BARBER_GEOM) - 1)))
    idx = min(max(idx, 0), len(BARBER_GEOM) - 1)
    x, y = BARBER_GEOM[idx]
    return float(x), float(y)


# ---------- Gemini setup and helpers ----------
def init_gemini_model() -> genai.GenerativeModel | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        return None


def make_live_feature_vector(
    lap_df: pd.DataFrame,
    current_lap: int,
    cfg,
    strategy_name: str,
    pit_laps: list[int],
    caution_lap: int | None,
    caution_len: int,
    push_factor: float,
    risk_pref: float,
) -> dict:
    clean = lap_df.copy()
    if "is_pit_lap" in clean.columns:
        clean = clean[~clean["is_pit_lap"]]
    clean = clean.sort_values("lap")

    best_lap = float(clean["lap_time_s"].min()) if "lap_time_s" in clean.columns else None
    median_lap = float(clean["lap_time_s"].median()) if "lap_time_s" in clean.columns else None

    window = clean[(clean["lap"] <= current_lap) & (clean["lap"] >= max(1, current_lap - 4))]
    last5_avg = float(window["lap_time_s"].mean()) if not window.empty else None
    last5_delta_vs_best = (
        last5_avg - best_lap if (last5_avg is not None and best_lap is not None) else None
    )

    stint_laps = None
    if "stint_lap" in lap_df.columns:
        row = lap_df[lap_df["lap"] == current_lap]
        if not row.empty:
            stint_laps = int(row["stint_lap"].iloc[0])

    return {
        "current_lap": int(current_lap),
        "total_laps": int(clean["lap"].max()),
        "strategy_name": strategy_name,
        "pit_laps": list(pit_laps),
        "caution_lap": int(caution_lap) if caution_lap is not None else None,
        "caution_len": int(caution_len) if caution_lap is not None else 0,
        "best_lap_s": best_lap,
        "median_lap_s": median_lap,
        "last5_avg_s": last5_avg,
        "last5_delta_vs_best_s": last5_delta_vs_best,
        "stint_lap": stint_laps,
        "push_factor": float(push_factor),
        "risk_pref": float(risk_pref),
        "base_lap_s": float(getattr(cfg, "base_lap_s", 0.0)) if cfg is not None else None,
        "pit_loss_s": float(getattr(cfg, "pit_loss_s", 0.0)) if cfg is not None else None,
    }


def gemini_live_insight(model: genai.GenerativeModel | None, features: dict) -> str:
    if model is None:
        # Fallback deterministic insight if Gemini is not configured
        lap = features.get("current_lap")
        total = features.get("total_laps")
        last5 = features.get("last5_delta_vs_best_s")
        strategy = features.get("strategy_name")
        base = features.get("base_lap_s")
        pit_loss = features.get("pit_loss_s")

        parts: list[str] = []
        parts.append(f"Lap {lap}/{total}: running {strategy} blueprint.")
        if last5 is not None:
            if last5 < 0.2:
                parts.append("Tyre performance stable – last 5 laps are near best pace.")
            elif last5 < 1.0:
                parts.append("We are slowly drifting away from peak pace – mild degradation.")
            else:
                parts.append("Degradation is significant – bring the pit window forward.")
        if base is not None and pit_loss is not None:
            parts.append(
                f"Base lap ~{base:.2f}s, pit loss ~{pit_loss:.1f}s – any stop must earn that back in the next stint."
            )
        return " ".join(parts)

    prompt = f"""
You are an elite GT race engineer watching a live telemetry stream.

Here is the current live snapshot (Python dict):

{json.dumps(features, indent=2)}

In **2–4 short bullet points**, answer for this instant:
- What matters most right now about tyre & pace,
- Whether we should move the pit window (earlier / later / hold),
- How we should think about a possible caution in the next 3 laps.

Use concise race-radio language (no fluff). If the situation is normal, say so briefly.
"""
    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text
    except Exception:
        return "(Gemini insight unavailable – using fallback heuristics.)"


# ---------- Data loading ----------
def load_lap_features(track_id: str, race: str, car_id: str) -> pd.DataFrame:
    track_dirs = {
        "barber-motorsports-park": "barber",
        "virginia-international-raceway": "virginia",
        "circuit-of-the-americas": "cota",
    }

    if track_id not in track_dirs:
        raise ValueError(f"Track {track_id} not wired yet in load_lap_features().")

    short = track_dirs[track_id]
    race_lower = race.lower()
    proc_dir = DATA_PROCESSED / short
    fname = f"{short}_{race_lower}_{car_id}_lap_features.csv"
    path = proc_dir / fname

    if not path.exists():
        raise FileNotFoundError(
            f"Lap feature file not found for {track_id}, {race}, {car_id}: {path}"
        )

    return pd.read_csv(path)


def build_strategies(total_laps: int) -> dict[str, list[int]]:
    one_stop_mid = [total_laps // 2]
    one_stop_early = [max(3, total_laps // 2 - 4)]
    two_stop = [total_laps // 3, 2 * total_laps // 3]

    return {
        "1-stop_early": one_stop_early,
        "1-stop_mid": one_stop_mid,
        "2-stop": two_stop,
    }


# ---------- Streamlit layout ----------
st.set_page_config(
    page_title="Racing Hokies – GR RaceCraft Copilot",
    page_icon="🏎️",
    layout="wide",
)

GEMINI_MODEL = init_gemini_model()

if GEMINI_MODEL is None:
    st.sidebar.warning(
        "Gemini model not initialised – set GEMINI_API_KEY and install google-generativeai. "
        "Using fallback heuristic insights."
    )
else:
    st.sidebar.success("Gemini live insights enabled (gemini-2.5-flash).")

if "live_insights" not in st.session_state:
    st.session_state["live_insights"] = []

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
- Stream a **live engineering feed** powered by heuristics + Gemini  
"""
)

# ----- Sidebar config -----
st.sidebar.header("Configuration")

track_options = {
    "Barber Motorsports Park": "barber-motorsports-park",
    "Virginia International Raceway": "virginia-international-raceway",
    "Circuit of the Americas": "circuit-of-the-americas",
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

lap_df: pd.DataFrame | None = None
strategy_available = True

try:
    lap_df = load_lap_features(track_id, race, car_id)
except FileNotFoundError as e:
    if track_id == "virginia-international-raceway":
        st.warning(
            "Lap feature file for VIR not found. Strategy Brain and lap-time plots "
            "are disabled for this track, but sector-based driver insights are available below."
        )
        strategy_available = False
    else:
        st.error(f"Error loading data: {e}")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

meta = TRACK_METAS[track_id]

if strategy_available and lap_df is not None:
    total_laps = int(lap_df["lap"].max())
    cfg = make_config_from_meta(lap_df, pit_lane_time_s=meta.pit_lane_time_s)
    strategies = build_strategies(total_laps)

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
else:
    total_laps = 0
    cfg = None
    strategies: dict[str, list[int]] = {}
    caution_lap = None
    caution_len = 0

tab_strategy, tab_driver, tab_live = st.tabs(
    [
        "🧠 Strategy Brain",
        "📊 Driver Insights",
        "📡 Live Race Copilot",
    ]
)

# ---------- Strategy Brain ----------
with tab_strategy:
    if not strategy_available or lap_df is None:
        st.info(
            "Strategy Brain is currently wired only for tracks with per-lap telemetry "
            "exports (Barber). VIR/COTA views will be enabled once lap_features are generated."
        )
    else:
        col_info, col_deterministic, col_multiverse = st.columns([1.1, 1.4, 1.5])

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

# ---------- Driver Insights ----------
with tab_driver:
    st.subheader("Lap-time and consistency insights")
    st.markdown("### Lap-time profile")

    if lap_df is not None and "lap_time_s" in lap_df.columns:
        display_df = lap_df[["lap", "lap_time_s"]].copy()
        display_df = display_df.sort_values("lap")

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
        st.info("Lap-time profile unavailable (no per-lap telemetry for this track).")

    st.markdown("### Best laps")

    if lap_df is not None and "lap_time_s" in lap_df.columns:
        race_laps = lap_df.copy()
        if "is_pit_lap" in race_laps.columns:
            race_laps = race_laps[~race_laps["is_pit_lap"]]

        if not race_laps.empty:
            best_laps = race_laps.nsmallest(5, "lap_time_s")[["lap", "lap_time_s"]]
            st.dataframe(best_laps, use_container_width=True)
        else:
            st.info("No non-pit laps available to rank.")
    else:
        st.info("Cannot compute best laps without per-lap telemetry.")

    st.markdown("### Track map animation (Barber only)")

    if lap_df is not None and "lap" in lap_df.columns:
        n_laps_anim = int(lap_df["lap"].max())

        if st.button("Run map animation"):
            placeholder = st.empty()
            text_placeholder = st.empty()

            # Try to load the Barber background map
            bg_img = None
            if track_id == "barber-motorsports-park":
                map_path = DATA_ROOT / "track_maps" / "barber_map.png"
                if map_path.exists():
                    bg_img = plt.imread(map_path)

            for lap in range(1, n_laps_anim + 1):
                fig, ax = plt.subplots(figsize=(6, 6))

                # Draw background map
                if bg_img is not None:
                    ax.imshow(bg_img, extent=[0, 1, 0, 1])
                else:
                    # fallback: just draw the polyline
                    if BARBER_GEOM is not None:
                        ax.plot(
                            BARBER_GEOM[:, 0],
                            BARBER_GEOM[:, 1],
                            linewidth=3,
                            color="lime",
                        )

                # Draw track polyline and car position if we have geometry
                if BARBER_GEOM is not None:
                    ax.plot(
                        BARBER_GEOM[:, 0],
                        BARBER_GEOM[:, 1],
                        linewidth=3,
                        color="lime",
                    )
                    x, y = barber_lap_to_xy(lap, n_laps_anim)
                    if x is not None:
                        ax.scatter([x], [y], s=300, color="orange")

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                ax.set_title(f"Lap {lap}/{n_laps_anim}")

                placeholder.pyplot(fig)
                text_placeholder.markdown(
                    f"**Car position on map:** lap {lap} / {n_laps_anim}"
                )
                plt.close(fig)

                time.sleep(0.15)  # slow it down so you can see it

    else:
        st.caption("Track map animation unavailable (no lap index).")

    # VIR sector comparison remains unchanged
    if track_id == "virginia-international-raceway":
        vir_dir = DATA_PROCESSED / "vir"
        try:
            driver_vir = pd.read_csv(vir_dir / "vir_car2_sector_summary.csv")

            st.subheader("VIR – Car #2 sector pace (R1 vs R2)")

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
            for ax, sec in zip(axes, ["S1", "S2", "S3"]):
                ax.bar(driver_vir["race"], driver_vir[f"{sec}_mean_s"])
                ax.set_title(f"{sec} mean time")
                ax.set_ylabel("time (s)")
                ax.grid(True, axis="y")

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(
                """
- **Race 1 → Race 2 evolution (VIR):**  
  Car #2 improves its overall sector balance in Race 2, especially in **S1** and **S2**,  
  while keeping **S3** competitive. This suggests better launch + mid-lap execution at VIR.
"""
            )
        except FileNotFoundError:
            st.info(
                "VIR sector summary file `vir_car2_sector_summary.csv` not found in "
                "`data/processed/vir/`. Run the VIR notebook export first."
            )
        except Exception as e:
            st.info(f"Could not load VIR sector summary: {e}")

    st.markdown("### (Optional) Sector comparison vs field")

    sector_path = (
        DATA_PROCESSED
        / "barber"
        / f"barber_{race.lower()}_sector_stats_all_cars.csv"
    )
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

# ---------- Live Race Copilot ----------
with tab_live:
    st.subheader("📡 Live Race Copilot – real-time what-if sandbox")

    if lap_df is None or not strategy_available:
        st.info("Live copilot currently available only for tracks with per-lap telemetry (e.g. Barber).")
    else:
        col_cfg, col_live = st.columns([0.9, 1.6])

        with col_cfg:
            st.markdown("#### Live control panel")

            live_strategy_name = st.selectbox(
                "Active strategy",
                options=list(strategies.keys()),
                index=0,
                help="Which pre-defined pit blueprint we are currently following.",
            )
            live_pits = strategies[live_strategy_name]

            push_factor = st.slider(
                "Driver push level (0 = fuel save, 1 = normal, 2 = quali mode)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
            )
            risk_pref = st.slider(
                "Risk appetite (0 = very conservative, 1 = all-in)",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
            )

            live_caution_prob = st.slider(
                "Subjective chance of caution in next 3 laps (%)",
                min_value=0,
                max_value=100,
                value=20,
                step=5,
            )

            tick_delay = st.slider(
                "Update interval (seconds, local sim)",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.1,
            )

            st.markdown("---")
            start_live = st.button("🚦 Start live sim from lap 1")

        with col_live:
            st.markdown("#### Live car state + engineering feed")

            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            feed_placeholder = st.empty()

            if start_live:
                st.session_state["live_insights"] = []

                clean = lap_df.copy()
                if "is_pit_lap" in clean.columns:
                    clean = clean[~clean["is_pit_lap"]]
                clean = clean.sort_values("lap")
                max_lap = int(clean["lap"].max())

                for lap in range(1, max_lap + 1):
                    feats = make_live_feature_vector(
                        lap_df=lap_df,
                        current_lap=lap,
                        cfg=cfg,
                        strategy_name=live_strategy_name,
                        pit_laps=live_pits,
                        caution_lap=caution_lap,
                        caution_len=caution_len,
                        push_factor=push_factor,
                        risk_pref=risk_pref,
                    )

                    last5_delta = feats.get("last5_delta_vs_best_s")
                    tyre_flag = "🟢 Stable"
                    if last5_delta is not None:
                        if last5_delta > 1.0:
                            tyre_flag = "🔴 Heavy deg"
                        elif last5_delta > 0.4:
                            tyre_flag = "🟡 Mild deg"

                    est_remaining = None
                    if feats.get("base_lap_s") is not None:
                        est_remaining = (max_lap - lap + 1) * feats["base_lap_s"] / 60.0

                    with metrics_placeholder.container():
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Lap", f"{lap}/{max_lap}")
                        m2.metric("Tyre state", tyre_flag)
                        if est_remaining is not None:
                            m3.metric("Est. time to flag", f"{est_remaining:.1f} min")
                        else:
                            m3.metric("Est. time to flag", "–")

                    hist = clean[clean["lap"] <= lap][["lap", "lap_time_s"]].set_index("lap")
                    chart_placeholder.line_chart(hist, height=180)

                    insight_text = gemini_live_insight(GEMINI_MODEL, feats)
                    st.session_state["live_insights"].append(
                        {
                            "lap": lap,
                            "text": insight_text,
                        }
                    )

                    # ---- write shared live state for the Barber map animation ----
                    x_norm, y_norm = None, None
                    if track_id == "barber-motorsports-park":
                        x_norm, y_norm = barber_lap_to_xy(lap, max_lap)

                    live_state_payload = {
                        "track_id": track_id,
                        "race": race,
                        "car_id": car_id,
                        "lap": lap,
                        "max_lap": max_lap,
                        "x_norm": x_norm,
                        "y_norm": y_norm,
                        "tyre_flag": tyre_flag,
                        "last5_delta_vs_best_s": last5_delta,
                        "push_factor": push_factor,
                        "risk_pref": risk_pref,
                        "subjective_caution_prob_pct": live_caution_prob,
                        "pit_laps": live_pits,
                        "caution_lap": caution_lap,
                        "caution_len": caution_len,
                        "gemini_insight": insight_text,
                        "timestamp": time.time(),
                    }
                    # single shared “barber” session (you can make this per-combo if you want)
                    save_live_state("barber", live_state_payload)

                    # ---- engineering radio feed ----
                    with feed_placeholder.container():
                        st.markdown("**Engineering radio feed (latest first):**")
                        for msg in reversed(st.session_state["live_insights"][-12:]):
                            st.markdown(f"- **Lap {msg['lap']}:** {msg['text']}")

                    time.sleep(tick_delay)

st.markdown("---")
st.caption(
    "Racing Hokies – Barber-focused MVP. Strategy engine is track-agnostic; "
    "support for VIR, COTA, Sebring, Sonoma, and other circuits can be added "
    "by wiring additional lap_features tables and TRACK_METAS entries."
)