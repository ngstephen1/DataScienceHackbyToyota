from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import google.generativeai as genai
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------------
# Model + data structures
# ---------------------------------------------------------------------

GEMINI_MODEL_NAME = "gemini-2.5-flash"


@dataclass
class FrameStats:
    """Structured summary of one race image returned by Gemini."""
    image_path: Path
    car_detected: bool
    num_cars: int
    car_color: str | None
    track_curvature_deg: float | None
    lane_position_norm: float | None  # -1 = very inside, +1 = very outside
    distance_to_inside_m: float | None
    distance_to_outside_m: float | None
    est_speed_kmh: float | None
    visibility_score: float | None    # 0–1 (0 = poor, 1 = clear)
    risk_score: float | None          # 0–1 (0 = safe, 1 = very risky)


def init_gemini_from_env() -> genai.GenerativeModel:
    """
    Configure Gemini using GEMINI_API_KEY and return a multimodal model.
    Re-uses the same pattern you already have in streamlit_app.py.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


# ---------------------------------------------------------------------
# Core CV → numbers call
# ---------------------------------------------------------------------

_ANALYSIS_PROMPT = """
You are a computer-vision assistant for a race engineer.
You will be given a single image from a race track.

Return ONLY a single JSON object with the following keys:

- car_detected: true/false
- num_cars: integer (0,1,2,…)
- car_color: short string (e.g. "red", "white/blue")
- track_curvature_deg: approximate curvature at the car's position
  in degrees (0 = straight, >0 = turning; e.g. 30 for medium corner).
- lane_position_norm: number between -1 and +1:
  -1 = very close to inside kerb,
   0 = roughly center of lane,
  +1 = close to outside edge of track.
- distance_to_inside_m: approximate meters from the car to inside edge/kerb.
- distance_to_outside_m: approximate meters to outside edge/grass/wall.
- est_speed_kmh: rough estimated speed (km/h).
- visibility_score: 0–1 (0 = very poor, 1 = very clear conditions).
- risk_score: 0–1 (0 = very safe, 1 = very risky for going off).

If you are unsure about a numeric value,
still output a best guess instead of null.

Example JSON:
{
  "car_detected": true,
  "num_cars": 1,
  "car_color": "red",
  "track_curvature_deg": 32.0,
  "lane_position_norm": -0.3,
  "distance_to_inside_m": 1.8,
  "distance_to_outside_m": 6.2,
  "est_speed_kmh": 115,
  "visibility_score": 0.92,
  "risk_score": 0.22
}
"""


def _call_gemini_on_image(
    model: genai.GenerativeModel,
    image_path: Path,
) -> dict[str, Any]:
    """Low-level Gemini call that returns the raw JSON dict."""
    img = Image.open(image_path)

    # For the legacy google.generativeai SDK:
    response = model.generate_content([_ANALYSIS_PROMPT, img])

    text = (response.text or "").strip()
    # Gemini sometimes wraps JSON in markdown; strip fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        # Drop possible leading 'json\n'
        if "\n" in text:
            text = text.split("\n", 1)[1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini did not return valid JSON. Raw text:\n{text}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got: {data!r}")

    return data


def analyze_race_image(
    model: genai.GenerativeModel,
    image_path: str | Path,
) -> FrameStats:
    """
    Run Gemini on a single image and return structured FrameStats.

    Parameters
    ----------
    model : genai.GenerativeModel
        Output of init_gemini_from_env().
    image_path : str or Path
        Path to .png/.jpg frame.

    """
    p = Path(image_path)
    data = _call_gemini_on_image(model, p)

    def _get(name: str, default=None):
        return data.get(name, default)

    return FrameStats(
        image_path=p,
        car_detected=bool(_get("car_detected", False)),
        num_cars=int(_get("num_cars", 0) or 0),
        car_color=_get("car_color"),
        track_curvature_deg=float(_get("track_curvature_deg", 0.0) or 0.0),
        lane_position_norm=float(_get("lane_position_norm", 0.0) or 0.0),
        distance_to_inside_m=float(_get("distance_to_inside_m", 0.0) or 0.0),
        distance_to_outside_m=float(_get("distance_to_outside_m", 0.0) or 0.0),
        est_speed_kmh=float(_get("est_speed_kmh", 0.0) or 0.0),
        visibility_score=float(_get("visibility_score", 0.0) or 0.0),
        risk_score=float(_get("risk_score", 0.0) or 0.0),
    )


def analyze_batch(
    model: genai.GenerativeModel,
    image_paths: Iterable[str | Path],
) -> List[FrameStats]:
    """Run the same analysis on multiple frames."""
    return [analyze_race_image(model, p) for p in image_paths]


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_lane_and_risk(frames: List[FrameStats]) -> None:
    """
    Simple summary plot:

    - Top: lane_position_norm per frame (inside vs outside).
    - Bottom: risk_score per frame (0–1).
    """
    if not frames:
        raise ValueError("No frames passed to plot_lane_and_risk.")

    x = list(range(1, len(frames) + 1))
    lane = [f.lane_position_norm for f in frames]
    risk = [f.risk_score for f in frames]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1 = axes[0]
    ax1.plot(x, lane, marker="o")
    ax1.axhline(0.0, linestyle="--")
    ax1.set_ylabel("Lane position (−1 inside … +1 outside)")
    ax1.set_title("Lane position vs. frame index")

    ax2 = axes[1]
    ax2.plot(x, risk, marker="o")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Risk score (0–1)")
    ax2.set_title("Gemini risk estimate per frame")

    fig.tight_layout()
    plt.show()