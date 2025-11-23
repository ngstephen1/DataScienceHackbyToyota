from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

IN_CSV = REPO_ROOT / "data" / "track_geom" / "barber_track_xy.csv"
OUT_CSV = REPO_ROOT / "data" / "track_geom" / "barber_track_xy_s.csv"


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input track file not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    if not {"x_px", "y_px"}.issubset(df.columns):
        raise ValueError("Expected columns 'x_px' and 'y_px' in track csv")

    x = df["x_px"].to_numpy()
    y = df["y_px"].to_numpy()

    # segment lengths
    dx = np.diff(x)
    dy = np.diff(y)
    seg_len = np.sqrt(dx**2 + dy**2)

    s = np.concatenate([[0.0], np.cumsum(seg_len)])       # cumulative distance (pixels)
    s_total = float(s[-1])
    s_norm = s / s_total                                  # 0..1 around the lap

    df["s_px"] = s
    df["s_norm"] = s_norm

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"✅ wrote track with arclength to: {OUT_CSV}")
    print(f"   total track length in pixels ≈ {s_total:.1f}")


if __name__ == "__main__":
    main()