from __future__ import annotations

from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from matplotlib.widgets import Slider

from live_state import save_live_state  # NEW: for live JSON export

# ---------- Gemini ----------

try:
    import google.generativeai as genai

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    HAVE_GEMINI = bool(GEMINI_API_KEY)
    if HAVE_GEMINI:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL_NAME = "gemini-2.5-flash"
    else:
        GEMINI_MODEL_NAME = ""
except Exception:
    HAVE_GEMINI = False
    GEMINI_API_KEY = ""
    GEMINI_MODEL_NAME = ""

# ---------- paths ----------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

TRACK_MAP_PATH = REPO_ROOT / "data" / "track_maps" / "barber_map.png"
TRACK_GEOM_CSV_S = REPO_ROOT / "data" / "track_geom" / "barber_track_xy_s.csv"
TRACK_GEOM_CSV = REPO_ROOT / "data" / "track_geom" / "barber_track_xy.csv"

CAR_ICON_PATH = REPO_ROOT / "data" / "track_maps" / "car_icon.png"

BARBER_PROC = REPO_ROOT / "data" / "processed" / "barber"
LAP_FEATURES_PATH = BARBER_PROC / "barber_r2_GR86-002-000_lap_features.csv"
STRAT_SUMMARY_PATH = BARBER_PROC / "barber_r2_strategy_multiverse_summary.csv"

BARBER_INSIGHTS_PATH = REPO_ROOT / "notes" / "barber_notebook_insights.txt"


# ---------- track geometry ----------

def load_track_points() -> tuple[np.ndarray, np.ndarray]:
    # Use smoothed points if available
    if TRACK_GEOM_CSV_S.exists():
        df = pd.read_csv(TRACK_GEOM_CSV_S)
    else:
        df = pd.read_csv(TRACK_GEOM_CSV)
    x = df["x_px"].to_numpy()
    y = df["y_px"].to_numpy()

    # Interpolate to a smooth, dense polyline for animation
    t = np.linspace(0.0, 1.0, len(x))
    t_new = np.linspace(0.0, 1.0, 800)
    x_i = np.interp(t_new, t, x)
    y_i = np.interp(t_new, t, y)
    return x_i, y_i


def load_car_icon(path: Path) -> np.ndarray:
    return mpimg.imread(path)


# ---------- lap data & strategy ----------

def _find_col(df: pd.DataFrame, keywords: list[str], default: str | None = None) -> str:
    for c in df.columns:
        lc = c.lower()
        if all(k in lc for k in keywords):
            return c
    if default is not None and default in df.columns:
        return default
    raise KeyError(
        f"Could not find column with keywords {keywords} in {list(df.columns)}"
    )


def load_lap_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Lap-features CSV not found at {path}\n"
            "Run your Barber processing notebook to regenerate it."
        )
    df = pd.read_csv(path)

    # Lap number
    try:
        lap_col = _find_col(df, ["lap"], default="lap")
    except KeyError:
        df["lap"] = np.arange(1, len(df) + 1)
        lap_col = "lap"
    df["lap_no"] = df[lap_col].astype(int)

    # Lap time
    try:
        time_col = _find_col(df, ["lap", "time"])
    except KeyError:
        time_col = "lap_time_s" if "lap_time_s" in df.columns else df.columns[0]
    df["lap_time_s"] = df[time_col].astype(float)

    # Stint lap (fallback)
    if "stint_lap" not in df.columns:
        df["stint_lap"] = df["lap_no"]

    # Pit flag
    pit_col = None
    for cand in ["is_pit_lap", "pit", "in_pit"]:
        for c in df.columns:
            if cand in c.lower():
                pit_col = c
                break
        if pit_col:
            break
    if pit_col is None:
        df["is_pit_lap"] = False
    else:
        df["is_pit_lap"] = df[pit_col].astype(bool)

    # Track status
    status_col = None
    for c in df.columns:
        if "status" in c.lower() or "flag" in c.lower():
            status_col = c
            break
    if status_col is not None:
        df["track_status"] = df[status_col].astype(str)
    else:
        df["track_status"] = "GREEN"

    # Gaps / position if available
    gap_ahead_col = None
    gap_behind_col = None
    pos_col = None
    for c in df.columns:
        lc = c.lower()
        if "gap" in lc and "ahead" in lc:
            gap_ahead_col = c
        elif "gap" in lc and "behind" in lc:
            gap_behind_col = c
        elif lc in {"pos", "position"}:
            pos_col = c

    df["gap_ahead_s"] = (
        pd.to_numeric(df[gap_ahead_col], errors="coerce")
        if gap_ahead_col
        else np.nan
    )
    df["gap_behind_s"] = (
        pd.to_numeric(df[gap_behind_col], errors="coerce")
        if gap_behind_col
        else np.nan
    )
    df["position"] = (
        pd.to_numeric(df[pos_col], errors="coerce") if pos_col else np.nan
    )

    return df


def load_strategy_summary(path: Path, total_laps: int) -> int:
    # Ideal pit lap from multiverse; fallback to mid-race
    if not path.exists():
        return max(6, total_laps // 2)

    df = pd.read_csv(path)
    time_col = None
    for c in df.columns:
        lc = c.lower()
        if "time" in lc and ("total" in lc or "race" in lc):
            time_col = c
            break
    if time_col is None:
        return max(6, total_laps // 2)

    best = df.sort_values(time_col).iloc[0]
    for c in df.columns:
        lc = c.lower()
        if "stop" in lc and "lap" in lc:
            try:
                val = int(best[c])
            except Exception:
                continue
            if 1 <= val <= total_laps:
                return val

    return max(6, total_laps // 2)


def build_metrics_table(laps: pd.DataFrame, ideal_pit_lap: int) -> list[dict]:
    laps = laps.copy()
    total_laps = int(laps["lap_no"].max())
    race_laps = laps[~laps["is_pit_lap"]].copy()
    best_lap = race_laps["lap_time_s"].min()

    # Rolling references
    tmp = laps["lap_time_s"].where(~laps["is_pit_lap"])
    laps["rolling5_mean"] = tmp.rolling(window=5, min_periods=1).mean()
    laps["delta_vs_rolling5"] = laps["lap_time_s"] - laps["rolling5_mean"]
    laps["delta_vs_best"] = laps["lap_time_s"] - best_lap

    # Simple linear degradation model
    try:
        slope, intercept = np.polyfit(
            race_laps["stint_lap"].astype(float), race_laps["lap_time_s"], 1
        )
    except Exception:
        slope, intercept = 0.0, best_lap

    # Caution laps
    caution_laps: set[int] = set()
    if "track_status" in laps.columns:
        for row in laps.itertuples(index=False):
            if str(row.track_status).upper() in {"SC", "SAFETY CAR", "FCY", "YELLOW"}:
                caution_laps.add(int(row.lap_no))
    if not caution_laps:
        if total_laps >= 10:
            caution_laps = {10, 11}
        else:
            caution_laps = {max(2, total_laps // 3)}

    pit_lane_loss = 23.0
    free_stop_gain = pit_lane_loss * 0.7

    race_lap_times = race_laps.set_index("lap_no")["lap_time_s"]
    lap_var_3 = race_lap_times.rolling(window=3, min_periods=2).std()

    metrics_per_lap: list[dict] = []

    for row in laps.itertuples(index=False):
        lap_no = int(row.lap_no)
        stint_lap = int(row.stint_lap)
        lap_time = float(row.lap_time_s)
        is_pit = bool(row.is_pit_lap)

        # Tyre model
        tyre_life = max(
            0.0, 100.0 - (stint_lap - 1) * (100.0 / (total_laps // 2 + 4))
        )
        delta_best = float(
            laps.loc[laps["lap_no"] == lap_no, "delta_vs_best"].iloc[0]
        )
        delta_roll5 = float(
            laps.loc[laps["lap_no"] == lap_no, "delta_vs_rolling5"].iloc[0]
        )

        deg_loss_now = max(0.0, slope * (stint_lap - 1))
        laps_remaining = max(0, total_laps - lap_no)
        projected_deg_loss_future = max(0.0, slope * laps_remaining)

        projected_lap_in_5_stay = intercept + slope * (stint_lap + 5)
        projected_lap_in_5_pit = best_lap + max(0.0, slope * 3)

        if stint_lap <= 2:
            tyre_phase = "warm-up"
        elif stint_lap >= ideal_pit_lap - 2 or deg_loss_now > 1.5:
            tyre_phase = "degradation"
        else:
            tyre_phase = "stable"

        if tyre_life > 60:
            tyre_light = "GREEN"
        elif tyre_life > 30:
            tyre_light = "YELLOW"
        else:
            tyre_light = "RED"

        # Simple net-gain model vs pit lane loss
        def net_gain_if_pit_at(candidate_lap: int) -> float:
            candidate_lap = int(np.clip(candidate_lap, 1, total_laps))
            laps_rem_if_wait = max(0, total_laps - candidate_lap)
            deg_if_wait = max(0.0, slope * laps_rem_if_wait)
            return deg_if_wait - pit_lane_loss

        net_gain_now = net_gain_if_pit_at(lap_no)
        net_gain_minus2 = net_gain_if_pit_at(lap_no - 2)
        net_gain_plus2 = net_gain_if_pit_at(lap_no + 2)

        in_caution = lap_no in caution_laps
        caution_in_next_3 = any(
            (lap_no + k) in caution_laps
            for k in range(1, 4)
            if (lap_no + k) <= total_laps
        )

        time_gain_if_pitted_before_caution = max(
            0.0, free_stop_gain - projected_deg_loss_future
        )

        gap_ahead = float(getattr(row, "gap_ahead_s", np.nan))
        gap_behind = float(getattr(row, "gap_behind_s", np.nan))

        clean_air = True
        if not np.isnan(gap_ahead) and gap_ahead < 2.5:
            clean_air = False
        if not np.isnan(gap_behind) and gap_behind < 0.7:
            clean_air = False

        fuel_laps_remaining_if_pit_now = max(
            0, total_laps - max(lap_no + 1, ideal_pit_lap)
        )
        can_one_stop_from_here = lap_no >= ideal_pit_lap - 2

        var3 = float(lap_var_3.get(lap_no, np.nan))
        if np.isnan(var3):
            consistency_flag = "UNKNOWN"
        elif var3 < 0.3:
            consistency_flag = "VERY STABLE"
        elif var3 < 0.8:
            consistency_flag = "STABLE"
        else:
            consistency_flag = "ERRATIC"

        reliability_flag = "NORMAL"
        if stint_lap > (ideal_pit_lap + 3):
            reliability_flag = "TEMPS RISK / LONG STINT"

        strategy_drift_laps = lap_no - ideal_pit_lap
        in_window = (ideal_pit_lap - 2) <= lap_no <= (ideal_pit_lap + 2)

        # Recommendation text
        if is_pit:
            recommendation = "Already boxing – focus on marks and a clean release."
        elif in_caution and in_window:
            recommendation = (
                "BOX NOW: free stop in ideal window – turn tyre life into track position."
            )
        elif in_caution and lap_no < ideal_pit_lap - 2:
            recommendation = (
                "Early caution – only box if stuck in traffic or we can jump a pack."
            )
        elif in_caution and lap_no > ideal_pit_lap + 2:
            recommendation = (
                "Late caution – short-fill or splash option; check fuel-to-end and tyre delta."
            )
        elif not in_caution and lap_no < ideal_pit_lap - 3:
            recommendation = (
                "Build gap on this stint; keep clean air and protect the tyres for the window."
            )
        elif not in_caution and (ideal_pit_lap - 3) <= lap_no < ideal_pit_lap:
            recommendation = (
                "Approaching pit window – prep stop settings and be ready to react to traffic."
            )
        elif not in_caution and in_window and net_gain_now > 0.5:
            recommendation = (
                f"In window – pitting now saves ~{net_gain_now:.1f}s if we rejoin in clear air."
            )
        elif not in_caution and in_window:
            recommendation = (
                "In window – delta is neutral; decide based on traffic and driver balance."
            )
        else:
            recommendation = (
                "Window passed – commit to tyre management and watch for a late Safety Car."
            )

        metrics_per_lap.append(
            {
                "lap_no": lap_no,
                "total_laps": total_laps,
                "stint_lap": stint_lap,
                "lap_time_s": lap_time,
                "delta_best": delta_best,
                "delta_vs_rolling5": delta_roll5,
                "tyre_life": tyre_life,
                "tyre_light": tyre_light,
                "tyre_phase": tyre_phase,
                "deg_loss_now": deg_loss_now,
                "projected_deg_loss_future": projected_deg_loss_future,
                "projected_lap_in_5_stay": projected_lap_in_5_stay,
                "projected_lap_in_5_pit": projected_lap_in_5_pit,
                "net_gain_if_pit_now": net_gain_now,
                "net_gain_if_pit_minus2": net_gain_minus2,
                "net_gain_if_pit_plus2": net_gain_plus2,
                "is_pit_lap": is_pit,
                "in_caution": in_caution,
                "caution_in_next_3": caution_in_next_3,
                "time_gain_if_pitted_before_caution": time_gain_if_pitted_before_caution,
                "gap_ahead_s": gap_ahead,
                "gap_behind_s": gap_behind,
                "clean_air": clean_air,
                "fuel_laps_remaining_if_pit_now": fuel_laps_remaining_if_pit_now,
                "can_one_stop_from_here": can_one_stop_from_here,
                "consistency_flag": consistency_flag,
                "reliability_flag": reliability_flag,
                "strategy_drift_laps": strategy_drift_laps,
                "ideal_pit_lap": ideal_pit_lap,
                "pit_lane_loss": pit_lane_loss,
                "recommendation": recommendation,
            }
        )

    return metrics_per_lap


# ---------- notebooks + gemini ----------

def load_notebook_insights() -> str:
    notes = []

    # Optional text file you created
    if BARBER_INSIGHTS_PATH.exists():
        try:
            notes.append(BARBER_INSIGHTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            notes.append(BARBER_INSIGHTS_PATH.read_text(errors="ignore"))

    # Pull markdown from any *barber*.ipynb
    notebooks_dir = REPO_ROOT / "notebooks"
    if notebooks_dir.exists():
        from glob import glob

        for nb_path_str in glob(str(notebooks_dir / "*barber*.ipynb")):
            nb_path = Path(nb_path_str)
            try:
                with nb_path.open("r", encoding="utf-8") as f:
                    nb_json = json.load(f)
                for cell in nb_json.get("cells", []):
                    if cell.get("cell_type") == "markdown":
                        cell_text = "".join(cell.get("source", []))
                        if cell_text.strip():
                            notes.append(f"# From {nb_path.name}\n{cell_text}")
            except Exception:
                continue

    return "\n\n".join(notes) if notes else ""


def gemini_insight(metrics: dict, base_notes: str) -> str:
    if not HAVE_GEMINI or not GEMINI_API_KEY:
        return ""

    prompt = f"""
You are a race engineer assistant for the GR Cup at Barber Motorsports Park.

CURRENT LAP METRICS (JSON):
{json.dumps(metrics, indent=2)}

Study notes from previous Barber notebooks:
\"\"\"{base_notes[:7000]}\"\"\"

Give at most 3 short bullet points with:
- Tyre & pace phase and which sectors to focus on.
- Pit window thinking (undercut/overcut, caution in next 3 laps).
- Concrete driver instruction (push/save, where, and why).

Return ONLY the bullet points.
"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception:
        return ""


# ---------- main animation ----------

def main() -> None:
    if not TRACK_MAP_PATH.exists():
        raise FileNotFoundError(f"Track map not found at {TRACK_MAP_PATH}")
    if not CAR_ICON_PATH.exists():
        raise FileNotFoundError(f"Car icon not found at {CAR_ICON_PATH}")

    laps = load_lap_data(LAP_FEATURES_PATH)
    total_laps = int(laps["lap_no"].max())
    ideal_pit_lap = load_strategy_summary(STRAT_SUMMARY_PATH, total_laps)
    metrics_table = build_metrics_table(laps, ideal_pit_lap)

    base_notes = load_notebook_insights()
    gem_cache: dict[int, str] = {}

    bg_img = plt.imread(TRACK_MAP_PATH)
    xs, ys = load_track_points()
    car_img = load_car_icon(CAR_ICON_PATH)

    fig, (ax_map, ax_info, ax_chart) = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(bottom=0.16)

    h, w = bg_img.shape[0], bg_img.shape[1]

    # Map + car icon
    ax_map.set_title("Barber – live lap with car icon", fontsize=18, pad=14)
    ax_map.imshow(bg_img, extent=[0, w, h, 0])
    xs_plot, ys_plot = xs, ys
    (path_line,) = ax_map.plot([], [], lw=2.0, alpha=0.8)
    ax_map.set_xlim(0, w)
    ax_map.set_ylim(h, 0)
    ax_map.axis("off")

    icon_h = 45
    icon_w = 90
    initial_x = xs_plot[0]
    initial_y = ys_plot[0]
    car_im = ax_map.imshow(
        car_img,
        extent=[
            initial_x - icon_w / 2,
            initial_x + icon_w / 2,
            initial_y + icon_h / 2,
            initial_y - icon_h / 2,
        ],
        zorder=10,
    )

    # Info panel
    ax_info.set_title("Race Engineer – Real-Time Strategy Console", fontsize=16, pad=14)
    ax_info.axis("off")

    # Lap-time chart
    lap_nums = np.array([m["lap_no"] for m in metrics_table])
    lap_times = np.array([m["lap_time_s"] for m in metrics_table])
    ax_chart.set_title("Lap times (live marker)", fontsize=14)
    ax_chart.set_xlabel("Lap")
    ax_chart.set_ylabel("Lap time (s)")
    ax_chart.plot(lap_nums, lap_times, lw=1.5)
    (lap_marker,) = ax_chart.plot([lap_nums[0]], [lap_times[0]], "ro")
    ax_chart.grid(True, alpha=0.3)

    # Sliders (added Speed x slider)
    slider_ax_deg = plt.axes([0.10, 0.05, 0.18, 0.03])
    slider_ax_pit = plt.axes([0.32, 0.05, 0.18, 0.03])
    slider_ax_risk = plt.axes([0.54, 0.05, 0.18, 0.03])
    slider_ax_speed = plt.axes([0.76, 0.05, 0.18, 0.03])

    s_deg = Slider(slider_ax_deg, "Tyre deg x", 0.5, 1.5, valinit=1.0)
    s_pit = Slider(slider_ax_pit, "Pit loss x", 0.5, 1.5, valinit=1.0)
    s_risk = Slider(slider_ax_risk, "Risk mode", 0.0, 2.0, valinit=1.0)
    s_speed = Slider(slider_ax_speed, "Speed x", 0.25, 3.0, valinit=1.0)

    slider_vals = {
        "deg_scale": 1.0,
        "pit_scale": 1.0,
        "risk": 1.0,
        "speed": 1.0,
    }

    def _on_slider_change(_):
        slider_vals["deg_scale"] = float(s_deg.val)
        slider_vals["pit_scale"] = float(s_pit.val)
        slider_vals["risk"] = float(s_risk.val)
        slider_vals["speed"] = float(s_speed.val)

    s_deg.on_changed(_on_slider_change)
    s_pit.on_changed(_on_slider_change)
    s_risk.on_changed(_on_slider_change)
    s_speed.on_changed(_on_slider_change)

    FPS = 25
    INTERVAL_MS = int(1000 / FPS)

    state = {
        "lap_idx": 0,
        "frame_in_lap": 0,
        "frames_this_lap": max(5, int(FPS * metrics_table[0]["lap_time_s"])),
    }

    def frame_gen():
        f = 0
        while True:
            yield f
            f += 1

    def update(frame: int):
        lap_idx = state["lap_idx"]
        frames_this_lap = state["frames_this_lap"]
        fi = state["frame_in_lap"]

        metrics = metrics_table[lap_idx]
        lap_no = metrics["lap_no"]
        lap_time = metrics["lap_time_s"]

        frames_this_lap = max(5, int(FPS * lap_time))
        state["frames_this_lap"] = frames_this_lap

        speed = slider_vals["speed"]

        # progress scaled by speed slider
        progress = (fi * speed) / frames_this_lap

        if progress >= 1.0:
            # Complete one circuit -> advance to next lap
            lap_idx = (lap_idx + 1) % len(metrics_table)
            state["lap_idx"] = lap_idx
            state["frame_in_lap"] = 0
            metrics = metrics_table[lap_idx]
            lap_no = metrics["lap_no"]
            lap_time = metrics["lap_time_s"]
            frames_this_lap = max(5, int(FPS * lap_time))
            state["frames_this_lap"] = frames_this_lap
            progress = 0.0
            fi = 0
        else:
            state["frame_in_lap"] = fi + 1

        # 0–1 along path -> map to track polyline
        i_path = int(progress * (len(xs_plot) - 1))
        i_path = max(0, min(len(xs_plot) - 1, i_path))
        x = xs_plot[i_path]
        y = ys_plot[i_path]

        car_im.set_extent(
            [
                x - icon_w / 2,
                x + icon_w / 2,
                y + icon_h / 2,
                y - icon_h / 2,
            ]
        )
        path_line.set_data(xs_plot[: i_path + 1], ys_plot[: i_path + 1])

        # Live marker on lap-time chart
        lap_marker.set_data([lap_no], [metrics["lap_time_s"]])

        # Apply slider scales
        deg_scale = slider_vals["deg_scale"]
        pit_scale = slider_vals["pit_scale"]
        risk = slider_vals["risk"]

        tyre_life_eff = metrics["tyre_life"]
        deg_now_eff = metrics["deg_loss_now"] * deg_scale
        deg_future_eff = metrics["projected_deg_loss_future"] * deg_scale
        pit_loss_eff = metrics["pit_lane_loss"] * pit_scale
        net_gain_now_eff = deg_future_eff - pit_loss_eff

        if risk < 0.8:
            risk_mode = "SAFE"
        elif risk > 1.2:
            risk_mode = "ATTACK"
        else:
            risk_mode = "NEUTRAL"

        time_into_lap = progress * lap_time

        # Info panel rebuild each frame
        ax_info.clear()
        ax_info.set_title(
            "Race Engineer – Real-Time Strategy Console", fontsize=16, pad=14
        )
        ax_info.axis("off")

        y0 = 0.96
        dy = 0.055

        # Tyre & pace / basic context
        lines_top = [
            f"Lap: {metrics['lap_no']} / {metrics['total_laps']}   "
            f"(lap progress: {progress*100:4.0f}% | t ≈ {time_into_lap:4.1f}s of {lap_time:4.1f}s)",
            f"Stint lap: {metrics['stint_lap']}   (ideal pit lap ≈ {metrics['ideal_pit_lap']})",
            f"Last lap: {metrics['lap_time_s']:.3f} s (Δbest {metrics['delta_best']:+.3f} s, "
            f"Δrolling5 {metrics['delta_vs_rolling5']:+.3f} s)",
            f"Tyre life: {tyre_life_eff:.0f}% [{metrics['tyre_light']}] – phase: {metrics['tyre_phase']}",
            f"Deg now (scaled): {deg_now_eff:.2f} s   "
            f"Future deg if stay out (scaled): {deg_future_eff:.2f} s",
            f"Projected lap in 5 laps – stay: {metrics['projected_lap_in_5_stay']:.2f} s, "
            f"if box now: {metrics['projected_lap_in_5_pit']:.2f} s",
        ]
        for i, text in enumerate(lines_top):
            ax_info.text(
                0.03,
                y0 - i * dy,
                text,
                transform=ax_info.transAxes,
                fontsize=10.5,
                va="top",
            )

        # Pit window & fuel
        y_pit = y0 - len(lines_top) * dy - 0.01
        pit_lines = [
            f"Pit window & undercut/overcut (risk mode: {risk_mode}):",
            f"• Net gain if BOX NOW (scaled): {net_gain_now_eff:+.2f} s  "
            f"(−2 laps: {metrics['net_gain_if_pit_minus2']:+.2f} s, "
            f"+2 laps: {metrics['net_gain_if_pit_plus2']:+.2f} s)",
            f"• Fuel proxy – laps remaining if final stop now: "
            f"{metrics['fuel_laps_remaining_if_pit_now']}  "
            f"| One-stop from here? {'YES' if metrics['can_one_stop_from_here'] else 'NO'}",
            f"• Strategy drift vs offline plan: {metrics['strategy_drift_laps']:+d} laps "
            f"around ideal pit lap {metrics['ideal_pit_lap']}.",
        ]
        for i, text in enumerate(pit_lines):
            ax_info.text(
                0.03,
                y_pit - i * dy,
                text,
                transform=ax_info.transAxes,
                fontsize=10,
                va="top",
            )

        # Caution / Safety Car
        y_caution = y_pit - len(pit_lines) * dy - 0.01
        status_line = (
            "Track status: CAUTION – Safety Car / yellow flag"
            if metrics["in_caution"]
            else "Track status: GREEN FLAG"
        )
        caution_lines = [
            status_line,
            f"If a caution in next 3 laps: "
            f"{'HIGH RISK' if metrics['caution_in_next_3'] else 'low model probability'}   "
            f"| Time gain if we had already pitted: ~{metrics['time_gain_if_pitted_before_caution']:.1f} s",
        ]
        for i, text in enumerate(caution_lines):
            ax_info.text(
                0.03,
                y_caution - i * dy,
                text,
                transform=ax_info.transAxes,
                fontsize=10,
                va="top",
            )

        # Traffic & driver state
        y_traffic = y_caution - len(caution_lines) * dy - 0.01
        clean_air_txt = (
            "Clean air" if metrics["clean_air"] else "In traffic / dirty air risk"
        )
        ga = (
            "n/a"
            if np.isnan(metrics["gap_ahead_s"])
            else f"{metrics['gap_ahead_s']:.1f}s"
        )
        gb = (
            "n/a"
            if np.isnan(metrics["gap_behind_s"])
            else f"{metrics['gap_behind_s']:.1f}s"
        )
        traffic_lines = [
            "Traffic & driver:",
            f"• Air: {clean_air_txt} | gap ahead: {ga} | gap behind: {gb}",
            f"• Driver consistency: {metrics['consistency_flag']}  "
            f"| Reliability: {metrics['reliability_flag']}",
        ]
        for i, text in enumerate(traffic_lines):
            ax_info.text(
                0.03,
                y_traffic - i * dy,
                text,
                transform=ax_info.transAxes,
                fontsize=10,
                va="top",
            )

        # Engineer call & mental checklist
        ax_info.text(
            0.03,
            0.20,
            "Engineer call this lap:\n"
            + metrics["recommendation"],
            transform=ax_info.transAxes,
            fontsize=10,
            va="top",
            wrap=True,
        )
        ax_info.text(
            0.03,
            0.06,
            "Mental checklist:\n"
            "• Are we in clean air or stuck behind traffic?\n"
            "• Push vs save? Do we need to move the pit window based on risk slider?\n"
            "• What if a Safety Car appears in the next 2–3 laps?\n"
            "• If we pit now, do we rejoin into a gap or into a pack?",
            transform=ax_info.transAxes,
            fontsize=9,
            va="top",
            wrap=True,
        )

        # Gemini – once per lap, cached
        if fi == 0 and lap_no not in gem_cache:
            extra = gemini_insight(metrics, base_notes)
            if extra:
                gem_cache[lap_no] = extra

        gem_text = gem_cache.get(lap_no, "")
        if gem_text:
            ax_info.text(
                0.52,
                0.20,
                "Gemini insight:\n" + gem_text,
                transform=ax_info.transAxes,
                fontsize=9,
                va="top",
                wrap=True,
            )

        # ---------- NEW: write live state JSON for Streamlit / tools ----------

        live_state = {
            "track_id": "barber",
            "lap_no": int(metrics["lap_no"]),
            "total_laps": int(metrics["total_laps"]),
            "stint_lap": int(metrics["stint_lap"]),
            "lap_progress": float(progress),          # 0–1 within the current lap
            "time_into_lap_s": float(time_into_lap),  # seconds into this lap
            "lap_time_s": float(lap_time),
            "car_x_px": float(x),
            "car_y_px": float(y),
            "slider_deg_scale": float(deg_scale),
            "slider_pit_scale": float(pit_scale),
            "slider_risk_mode": float(risk),
            "slider_speed": float(speed),
            "metrics": metrics,
        }

        try:
            save_live_state("barber", live_state)
        except Exception:
            # Don't kill the animation if JSON write fails
            pass

        return car_im, path_line, lap_marker

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_gen(),
        interval=INTERVAL_MS,
        blit=False,
        repeat=True,
    )

    print("Showing Barber race-engineer simulation… Close the window to exit.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()