from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable
import json
import time

# Repo root = .../DataScienceHackbyToyota
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
LIVE_DIR = REPO_ROOT / "data" / "live"

JSONDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _normalise_track_id(track_id: str) -> str:
    """Return a filesystem-safe identifier for a track id."""
    return track_id.replace(" ", "_").replace("/", "_")


def get_live_state_path(track_id: str) -> Path:
    """
    Return the *canonical* JSON path used to store live state for a given track.

    New convention (preferred):
        data/live/<track_id>_state.json

    Example:
        track_id="barber" -> data/live/barber_state.json
    """
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    safe = _normalise_track_id(track_id)
    return LIVE_DIR / f"{safe}_state.json"


def get_legacy_live_state_candidates(track_id: str) -> Iterable[Path]:
    """
    Return possible *legacy* paths that older scripts may still be writing to.

    This keeps Streamlit compatible with older runs that store e.g.
    `data/live/live_state_barber.json`.
    """
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    safe = _normalise_track_id(track_id)

    # Old style 1: "live_state_barber.json"
    yield LIVE_DIR / f"live_state_{safe}.json"

    # Old style 2: "live_state_barber" (no .json) – just in case
    yield LIVE_DIR / f"live_state_{safe}"

    # Old style 3: "barber_live_state.json" (very defensive)
    yield LIVE_DIR / f"{safe}_live_state.json"


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def save_live_state(track_id: str, state: JSONDict) -> None:
    """
    Atomically write live state JSON for a given track.

    The function always writes to the canonical modern path
    `data/live/<track_id>_state.json`.

    It also injects a `last_updated` Unix timestamp into the state so that
    downstream consumers (e.g. Streamlit) can reason about staleness.
    """
    path = get_live_state_path(track_id)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Attach/update timestamp
    state = dict(state)  # shallow copy
    state["last_updated"] = time.time()

    data = json.dumps(state, ensure_ascii=False)
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def load_live_state(track_id: str, default: JSONDict | None = None) -> JSONDict:
    """
    Load the most recent live state for a given track.

    Behaviour:
    - Prefer the canonical path `data/live/<track_id>_state.json`.
    - If it does not exist, fall back to the newest of the known legacy
      paths (e.g. `live_state_barber.json`).
    - If *no* file exists, return `default` (or `{}`).

    This is what Streamlit should call on every poll to get the freshest
    snapshot coming from `barber_lap_anim.py` (or any other live writer).
    """
    candidates: list[Path] = []

    canonical = get_live_state_path(track_id)
    if canonical.exists():
        candidates.append(canonical)

    # Add any existing legacy paths
    for p in get_legacy_live_state_candidates(track_id):
        if p.exists():
            candidates.append(p)

    if not candidates:
        return {} if default is None else dict(default)

    # Pick the most recently modified file
    latest = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        raw = latest.read_text(encoding="utf-8")
        state = json.loads(raw)
        if isinstance(state, dict):
            return state  # type: ignore[return-value]
        # If somehow it's not a dict, wrap it for safety
        return {"value": state}
    except Exception:
        # Corrupt / partial file – return default rather than crashing UI
        return {} if default is None else dict(default)


def has_fresh_live_state(track_id: str, max_age_seconds: float = 2.0) -> bool:
    """
    Quick staleness check: return True if we have a live state file whose
    `last_updated` is within `max_age_seconds` of *now*.

    Streamlit can use this to distinguish between:
      - a genuinely live run (animation writing frames right now), and
      - a stale snapshot from a previous session (e.g. race finished).
    """
    state = load_live_state(track_id, default={})
    ts = state.get("last_updated")
    if not isinstance(ts, (int, float)):
        return False
    return (time.time() - float(ts)) <= max_age_seconds