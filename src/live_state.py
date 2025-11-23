from __future__ import annotations

from pathlib import Path
from typing import Any
import json

# Repo root = .../DataScienceHackbyToyota
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
LIVE_DIR = REPO_ROOT / "data" / "live"


def get_live_state_path(track_id: str) -> Path:
    """
    Return the JSON path used to store live state for a given track.
    Ensures data/live exists.
    """
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    safe = track_id.replace(" ", "_").replace("/", "_")
    return LIVE_DIR / f"{safe}_state.json"


def save_live_state(track_id: str, state: dict[str, Any]) -> None:
    """
    Atomically write live state JSON for a given track.
    """
    path = get_live_state_path(track_id)
    tmp = path.with_suffix(path.suffix + ".tmp")

    data = json.dumps(state, ensure_ascii=False)
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)