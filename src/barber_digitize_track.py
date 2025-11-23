from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths (robust to where you run the script from)
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

TRACK_MAP_DIR = REPO_ROOT / "data" / "track_maps"
OUT_CSV = REPO_ROOT / "data" / "track_geom" / "barber_track_xy.csv"


def find_map_path() -> Path:
    """
    Try a few common filenames for the Barber track map and return the first
    one that exists. Raises FileNotFoundError if none are found.
    """
    candidates = [
        "barber_map.png",
        "barber_map.jpg",
        "IMG_4381.png",
        "IMG_4381.jpg",
    ]

    for name in candidates:
        p = TRACK_MAP_DIR / name
        if p.exists():
            return p

    msg = (
        "No track map found.\n"
        f"Looked in: {TRACK_MAP_DIR}\n"
        "Tried: " + ", ".join(candidates)
    )
    raise FileNotFoundError(msg)


def interactive_backend_available() -> bool:
    """
    Check whether we likely have a GUI backend (not Agg).
    """
    backend = matplotlib.get_backend().lower()
    # Common interactive backends:
    interactive_names = ["tkagg", "qt5agg", "qtagg", "macosx", "wxagg", "gtk3agg"]
    return any(name in backend for name in interactive_names)


def make_oval_fallback(n_points: int = 400) -> pd.DataFrame:
    """
    Fallback track geometry if we can't use an interactive backend.
    Generates a smooth oval in [0,1] x [0,1].
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # Center at (0.5, 0.5) with different radii
    x_norm = 0.5 + 0.4 * np.cos(t)
    y_norm = 0.5 + 0.3 * np.sin(t)

    # Fake pixel coords (e.g., 1000x1000 canvas)
    width = 1000.0
    height = 1000.0
    x_px = x_norm * width
    y_px = (1.0 - y_norm) * height  # invert y, to keep same convention

    df = pd.DataFrame(
        {
            "x_px": x_px,
            "y_px": y_px,
            "x_norm": x_norm,
            "y_norm": y_norm,
        }
    )
    return df


def main() -> None:
    # Try to switch to a GUI backend, but do **not** crash if it fails.
    try:
        # On macOS with system/GUI Python this often works:
        matplotlib.use("MacOSX")
    except Exception:
        # If that fails, we just rely on whatever backend is already set.
        pass

    # If we still don't have an interactive backend, just build a fallback oval.
    if not interactive_backend_available():
        print("⚠️  No GUI backend available for Matplotlib.")
        print("    I will generate a smooth oval as a fallback track shape.")
        df = make_oval_fallback()
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"✅ Fallback oval track saved to: {OUT_CSV}")
        return

    # -----------------------------------------------------------------
    # Interactive path: click along the real map
    # -----------------------------------------------------------------
    MAP_PATH = find_map_path()

    print(f"Loading track map from: {MAP_PATH}")
    img = plt.imread(MAP_PATH)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(
        "Barber Motorsports Park – digitize centerline\n"
        "Left-click along the racing line, then press ENTER when done."
    )
    ax.set_axis_off()

    print("👉 Instructions:")
    print("   - Left-click along the track centerline, in order of travel.")
    print("   - When you’re done, go to the figure window and press ENTER.")
    print("   - Close the window if needed after ENTER.")

    # Show the window
    plt.tight_layout()
    plt.show(block=False)

    # ginput: n=-1 means unlimited points, timeout=0 waits forever
    try:
        points = plt.ginput(n=-1, timeout=0)
    except RuntimeError as e:
        print("❌ Matplotlib interactive input failed.")
        print("   This usually means the backend is not truly interactive.")
        print(f"   Error: {e}")
        print("   Falling back to a synthetic oval track instead.")
        points = None
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass

    if not points:
        # Use fallback oval if user didn't click or backend failed.
        df = make_oval_fallback()
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
        print("ℹ️  No points collected; saved fallback oval instead.")
        print(f"✅ Fallback oval track saved to: {OUT_CSV}")
        return

    coords = np.array(points)  # shape (N, 2) with columns (x_px, y_px)
    x_px = coords[:, 0]
    y_px = coords[:, 1]

    # Image size for normalization
    height, width = img.shape[0], img.shape[1]

    # Normalized coordinates: [0,1] with origin at bottom-left
    x_norm = x_px / width
    y_norm = 1.0 - (y_px / height)  # flip y so 0 = bottom, 1 = top

    df = pd.DataFrame(
        {
            "x_px": x_px,
            "y_px": y_px,
            "x_norm": x_norm,
            "y_norm": y_norm,
        }
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"✅ Collected {len(df)} clicked points.")
    print(f"✅ Saved digitised track to: {OUT_CSV}")


if __name__ == "__main__":
    main()