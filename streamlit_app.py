from __future__ import annotations
from src import race_plots
from src.chat_assistant import build_chat_context, answer_engineer
from src.live_state import load_live_state
from src.decision_reviewer import review_decision
from src.predictive_models import predict_lap_times_for
from src.vision_gemini import analyze_race_image
from src.live_state import get_live_state_path, save_live_state  # <-- shared live-state helpers
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
VISION_DIR = DATA_ROOT / "vision"
VISION_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VISION_IMG = VISION_DIR / "sample_gr86_barber.png"

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
def load_lap_features(track_id: str, race: str, car: str) -> pd.DataFrame:
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
    fname = f"{short}_{race_lower}_{car}_lap_features.csv"
    path = proc_dir / fname

    if not path.exists():
        raise FileNotFoundError(
            f"Lap feature file not found for {track_id}, {race}, {car}: {path}"
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

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

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
- Offer a **Strategy Chat** where you can ask the copilot questions in natural language  
- Add a **Computer Vision** view, where Gemini 2.5 reads race frames and estimates line / risk  
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

# Shared live session id for this (track, race, car) combo
SESSION_ID = f"{track_id}_{race}_{car_id}"

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

tab_strategy, tab_driver, tab_predict, tab_vision, tab_chat, tab_live = st.tabs(
    [
        "🧠 Strategy Brain",
        "📊 Driver Insights",
        "📈 Predictive Models",
        "👁️ Vision",
        "💬 Strategy Chat",
        "🎙 Live Race Copilot",
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

        # ---------- Predictive lap-time overlay (Random Forest model) ----------
        # Only wired for Barber right now, where the model was trained.
        if track_id == "barber-motorsports-park":
            try:
                # In predictive_models we trained with short track_id "barber"
                pred_df = predict_lap_times_for(track_id="barber", race=race, car_id=car_id)
                if "lap" in pred_df.columns and "lap_time_pred_s" in pred_df.columns:
                    merged = (
                        display_df.merge(
                            pred_df[["lap", "lap_time_pred_s"]],
                            on="lap",
                            how="left",
                        )
                        .set_index("lap")
                    )

                    st.markdown("### Predictive lap-time model overlay")
                    st.line_chart(
                        merged[["lap_time_s", "lap_time_pred_s"]],
                        height=260,
                    )
                    st.caption(
                        "Overlay: actual lap times vs Random Forest model prediction using throttle + brake features."
                    )
                else:
                    st.caption(
                        "Predictive model output does not contain expected columns; "
                        "check `predictive_models.py` implementation."
                    )
            except FileNotFoundError:
                st.caption(
                    "No trained lap-time model found yet for this car/track. "
                    "Train it in the notebook and save under `models/`."
                )
            except Exception as e:
                st.caption(f"Predictive model unavailable in this session: {e}")
        else:
            st.caption(
                "Predictive lap-time overlay currently enabled only for Barber where a model has been trained."
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

# ---------- Predictive Models ----------
with tab_predict:
    st.subheader("Predictive Models – Lap Time Forecast")
    st.markdown(
        "We use a Random Forest trained on per-lap features "
        "(`aps_mean`, `pbrake_f_mean`, …) to predict lap times."
    )

    laps_path = DATA_PROCESSED / "barber" / "barber_r2_GR86-002-000_lap_features.csv"
    if not laps_path.exists():
        st.warning(f"Lap-features file not found at `{laps_path}`.")
    else:
        laps_df = pd.read_csv(laps_path)
        st.caption("Sample of lap features used as model inputs:")
        st.dataframe(laps_df.head())

        try:
            laps_pred = predict_lap_times_for(
                track_id="barber",
                car_id="GR86-002-000",
                laps=laps_df,
            )
            st.caption("Predicted vs actual lap times:")
            st.dataframe(
                laps_pred[["lap", "lap_time_s", "lap_time_pred_s"]].head()
            )

            # quick plot
            fig, ax = plt.subplots()
            ax.plot(laps_pred["lap"], laps_pred["lap_time_s"], "o-", label="Actual")
            ax.plot(
                laps_pred["lap"],
                laps_pred["lap_time_pred_s"],
                "s--",
                label="Predicted",
            )
            ax.set_xlabel("Lap")
            ax.set_ylabel("Lap time (s)")
            ax.set_title("Actual vs predicted lap times")
            ax.legend()
            st.pyplot(fig)

        except FileNotFoundError:
            st.warning(
                "No trained model found yet. "
                "Train and save a model in the `11_barber_predictive_model.ipynb` notebook first."
            )
        except Exception as e:
            st.error(f"Problem running predictions: {e}")

# ---------- Vision (Gemini 2.5 CV) ----------
with tab_vision:
    st.subheader("👁️ Computer Vision – race frame analysis")

    if GEMINI_MODEL is None:
        st.info(
            "Gemini is not configured. Set `GEMINI_API_KEY` in your environment "
            "to enable the vision assistant."
        )
    else:
        uploaded = st.file_uploader(
            "Upload a race frame (PNG/JPG). If you skip this, we'll use the sample GR86 image.",
            type=["png", "jpg", "jpeg"],
        )

        img_path: Path | None = None

        if uploaded is not None:
            # Save uploaded file into data/vision
            dest = VISION_DIR / uploaded.name
            with dest.open("wb") as f:
                f.write(uploaded.getbuffer())
            img_path = dest
        elif SAMPLE_VISION_IMG.exists():
            st.caption("No image uploaded – using sample frame `sample_gr86_barber.png`.")
            img_path = SAMPLE_VISION_IMG

        if img_path is None:
            st.info("Upload a race frame or add `sample_gr86_barber.png` to `data/vision/`.")
        else:
            st.image(str(img_path), caption=f"Frame: {img_path.name}", use_column_width=True)

            with st.spinner("Letting Gemini 2.5 read the frame…"):
                try:
                    stats = analyze_race_image(GEMINI_MODEL, img_path)
                except Exception as e:
                    st.error(f"Vision analysis failed: {e}")
                    stats = None

            if stats is not None:
                st.markdown("#### Structured view (for engineers / ML):")
                st.json(
                    {
                        "car_detected": stats.car_detected,
                        "num_cars": stats.num_cars,
                        "car_color": stats.car_color,
                        "track_curvature_deg": stats.track_curvature_deg,
                        "lane_position_norm": stats.lane_position_norm,
                        "distance_to_inside_m": stats.distance_to_inside_m,
                        "distance_to_outside_m": stats.distance_to_outside_m,
                        "est_speed_kmh": stats.est_speed_kmh,
                        "visibility_score": stats.visibility_score,
                        "risk_score": stats.risk_score,
                    }
                )

                st.markdown("#### Quick engineering summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Cars in frame", stats.num_cars)
                if stats.est_speed_kmh is not None:
                    col2.metric("Estimated speed", f"{stats.est_speed_kmh:.0f} km/h")
                else:
                    col2.metric("Estimated speed", "–")

                if stats.risk_score is not None:
                    col3.metric("Off-track risk", f"{stats.risk_score:.2f} (0–1)")
                else:
                    col3.metric("Off-track risk", "–")

                # Simple plot turning Gemini numbers into visuals
                st.markdown("#### Line & risk visualisation from Gemini outputs")
                fig, axes = plt.subplots(1, 2, figsize=(8, 3))

                # Lane position
                ax1 = axes[0]
                ax1.barh(["lane"], [stats.lane_position_norm])
                ax1.set_xlim(-1, 1)
                ax1.axvline(0, linestyle="--", linewidth=1)
                ax1.set_xlabel("Inside  ←  lane position  →  Outside")
                ax1.set_title("Lane position (−1 inside, +1 outside)")

                # Risk score
                ax2 = axes[1]
                ax2.bar(["risk"], [stats.risk_score])
                ax2.set_ylim(0, 1)
                ax2.set_ylabel("Risk (0–1)")
                ax2.set_title("Gemini off-track risk estimate")

                fig.tight_layout()
                st.pyplot(fig)

# ---------- Strategy Chat ----------
with tab_chat:
    st.subheader("💬 Strategy Chat – talk to the RaceCraft Copilot")

    if GEMINI_MODEL is None:
        st.info(
            "Gemini is not configured. Set `GEMINI_API_KEY` in your environment "
            "to enable the chat assistant."
        )
    elif lap_df is None or not strategy_available:
        st.info(
            "Strategy Chat is currently wired only for tracks with per-lap telemetry "
            "(Barber). Enable once lap_features are available."
        )
    else:
        # Build a static context once per session (optional optimisation)
        context = build_chat_context(
            lap_df=lap_df,
            track_meta=meta,
            cfg=cfg,
            strategies=strategies,
            race=race,
            car_id=car_id,
        )

        st.markdown(
            "Ask anything like:  \n"
            "- *“If we box this lap, do we undercut the car ahead?”*  \n"
            "- *“How risky is it to stay out if a caution comes in the next 3 laps?”*  \n"
            "- *“Which stint looks weakest so far and why?”*  \n"
            "- Or even just say **hi** or ask a general question – the copilot will tell you "
            "when something is off-topic and what it *can* help with."
        )

        user_q = st.text_area(
            "Your question to the engineer copilot:",
            height=100,
            placeholder="Example: If we stay out 5 more laps on this stint, how much time do we likely lose?",
        )
        ask_button = st.button("Ask the copilot")

        if ask_button and user_q.strip():
            with st.spinner("Thinking like a race engineer..."):
                # Optional: bring in latest live state if running the Live tab in parallel
                live_state = None
                live_state_path = get_live_state_path(SESSION_ID)
                if live_state_path.exists():
                    try:
                        live_state = json.loads(live_state_path.read_text(encoding="utf-8"))
                    except Exception:
                        live_state = None

                answer = answer_engineer(
                    model=GEMINI_MODEL,
                    question=user_q,
                    context=context,
                    live_state=live_state,
                    chat_history=st.session_state["chat_history"],
                )

                st.session_state["chat_history"].append(
                    {"role": "user", "content": user_q}
                )
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": answer}
                )

        # Render chat history (latest at top)
        if st.session_state["chat_history"]:
            st.markdown("#### Conversation")

            for msg in reversed(st.session_state["chat_history"]):
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Engineer Copilot:** {msg['content']}")

            # ---- Refinement tools for the latest answer ----
            # Find latest user + assistant messages
            last_user = None
            last_assistant = None
            for msg in reversed(st.session_state["chat_history"]):
                if msg["role"] == "assistant" and last_assistant is None:
                    last_assistant = msg
                elif msg["role"] == "user" and last_user is None:
                    last_user = msg
                if last_user is not None and last_assistant is not None:
                    break

            if last_user is not None and last_assistant is not None:
                st.markdown("#### Refine latest answer")
                c1, c2, c3, c4 = st.columns(4)
                shorten_click = c1.button("Shorten", key="refine_shorten")
                viz_click = c2.button("Visualize in plots", key="refine_viz")
                more_click = c3.button("More detail", key="refine_more")
                explain_click = c4.button("Explain like I'm new", key="refine_explain")

                action_instruction = None
                want_local_plots = False

                if shorten_click:
                    action_instruction = (
                        "Please shorten and tighten your previous answer, keeping only the key "
                        "recommendation and rationale for the race engineer."
                    )
                elif viz_click:
                    action_instruction = (
                        "Please propose a few simple plots or tables (described in text) that "
                        "would help visualize your previous answer. Do NOT write code; just "
                        "describe the visuals and what they would show."
                    )
                    # also try to build local figures from current Barber data
                    want_local_plots = True
                elif more_click:
                    action_instruction = (
                        "Please expand your previous answer with more technical detail, edge "
                        "cases, and concrete examples, while keeping it grounded in the data "
                        "and strategy context."
                    )
                elif explain_click:
                    action_instruction = (
                        "Please explain your previous answer in simpler, beginner-friendly "
                        "language, avoiding jargon. Assume the person is new to racing and "
                        "telemetry."
                    )

                if action_instruction is not None:
                    followup_prompt = (
                        f"{action_instruction}\n\n"
                        f"Original question:\n{last_user['content']}\n\n"
                        f"Your previous answer:\n{last_assistant['content']}"
                    )

                    with st.spinner("Refining the last answer..."):
                        live_state = None
                        live_state_path = get_live_state_path(SESSION_ID)
                        if live_state_path.exists():
                            try:
                                live_state = json.loads(
                                    live_state_path.read_text(encoding="utf-8")
                                )
                            except Exception:
                                live_state = None

                        # Use the same engine, but with a refinement meta-question
                        followup_answer = answer_engineer(
                            model=GEMINI_MODEL,
                            question=followup_prompt,
                            context=context,
                            live_state=live_state,
                            chat_history=st.session_state["chat_history"],
                        )

                    # Log refinement as a new turn
                    st.session_state["chat_history"].append(
                        {"role": "user", "content": f"[Refinement] {action_instruction}"}
                    )
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": followup_answer}
                    )

                    st.info("Refinement added to the conversation. Latest response is shown at the top.")

                    # If the user requested plots, optionally generate local visuals too
                    if want_local_plots and lap_df is not None:
                        with st.expander("📊 Quick plots from current Barber data", expanded=True):
                            # If you implemented helpers in race_plots, use them here.
                            # This block is guarded with hasattr so it won't break if not present.
                            if hasattr(race_plots, "render_quick_insights"):
                                try:
                                    race_plots.render_quick_insights(
                                        st=st,
                                        lap_df=lap_df,
                                        question=last_user["content"],
                                        answer=last_assistant["content"],
                                    )
                                except Exception as e:
                                    st.warning(
                                        f"Could not render custom race plots for this answer: {e}"
                                    )
                            else:
                                # Fallback: simple generic lap-time visual
                                try:
                                    fig, ax = plt.subplots()
                                    df_plot = lap_df.sort_values("lap")
                                    ax.plot(
                                        df_plot["lap"],
                                        df_plot["lap_time_s"],
                                        "o-",
                                        label="Lap time (s)",
                                    )
                                    ax.set_xlabel("Lap")
                                    ax.set_ylabel("Lap time (s)")
                                    ax.set_title("Generic lap-time evolution (Barber)")
                                    ax.legend()
                                    st.pyplot(fig)
                                    st.caption(
                                        "Local plotting helper in `race_plots` is not wired yet; "
                                        "showing a basic lap-time evolution plot as a fallback."
                                    )
                                except Exception as e:
                                    st.warning(
                                        f"Could not draw fallback lap-time plot: {e}"
                                    )

# ---------- Live Race Copilot + Decision Reviewer ----------
with tab_live:
    st.subheader("📡 Live Race Copilot – real-time what-if sandbox")

    if lap_df is None or not strategy_available:
        st.info("Live copilot currently available only for tracks with per-lap telemetry (e.g. Barber).")
    else:
        # Precompute clean laps + max_lap for both live sim + reviewer
        clean_live = lap_df.copy()
        if "is_pit_lap" in clean_live.columns:
            clean_live = clean_live[~clean_live["is_pit_lap"]]
        clean_live = clean_live.sort_values("lap")
        max_lap = int(clean_live["lap"].max())

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

                    hist = clean_live[clean_live["lap"] <= lap][["lap", "lap_time_s"]].set_index("lap")
                    chart_placeholder.line_chart(hist, height=180)

                    insight_text = gemini_live_insight(GEMINI_MODEL, feats)
                    st.session_state["live_insights"].append(
                        {
                            "lap": lap,
                            "text": insight_text,
                        }
                    )

                    # ---- write shared live state for the Barber map animation and chat ----
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
                    # shared session id for this combo
                    save_live_state(SESSION_ID, live_state_payload)

                    # ---- engineering radio feed ----
                    with feed_placeholder.container():
                        st.markdown("**Engineering radio feed (latest first):**")
                        for msg in reversed(st.session_state["live_insights"][-12:]):
                            st.markdown(f"- **Lap {msg['lap']}:** {msg['text']}")

                    time.sleep(tick_delay)

            # ---------- AI Decision Reviewer (static what-if) ----------
            st.markdown("---")
            st.markdown("#### 🧾 Decision Reviewer – sanity-check a strategy call")

            if GEMINI_MODEL is None:
                st.info(
                    "Set `GEMINI_API_KEY` to enable AI decision review. "
                    "Right now only heuristic live insights are available."
                )
            else:
                if "lap" in lap_df.columns:
                    review_lap = st.slider(
                        "Lap to review",
                        min_value=1,
                        max_value=max_lap,
                        value=max_lap,
                        step=1,
                        key="review_lap_slider",
                    )
                    default_decision = "Box now for 4 tyres and fuel to the end."
                    decision_text = st.text_input(
                        "Describe your intended call (engineer radio-style):",
                        value=default_decision,
                        key="decision_text_input",
                    )
                    review_button = st.button(
                        "Review this call",
                        key="review_button",
                        help="Ask the AI engineer to agree/disagree and highlight risks.",
                    )

                    if review_button and decision_text.strip():
                        with st.spinner("Reviewing the call like a race engineer..."):
                            feats_review = make_live_feature_vector(
                                lap_df=lap_df,
                                current_lap=review_lap,
                                cfg=cfg,
                                strategy_name=live_strategy_name,
                                pit_laps=live_pits,
                                caution_lap=caution_lap,
                                caution_len=caution_len,
                                push_factor=push_factor,
                                risk_pref=risk_pref,
                            )

                            review_text = review_decision(
                                GEMINI_MODEL,
                                decision_text,
                                feats_review,
                            )

                        st.markdown("**AI Decision Review:**")
                        st.markdown(review_text)
                else:
                    st.info("Lap index not available – cannot run decision review on this dataset.")


st.markdown("---")
st.caption(
    "Racing Hokies – Barber-focused MVP. Strategy engine is track-agnostic; "
    "support for VIR, COTA, Sebring, Sonoma, and other circuits can be added "
    "by wiring additional lap_features tables and TRACK_METAS entries."
)