from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Project roots
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
TRACK_GEOM_DIR = DATA_ROOT / "track_geom"


@dataclass
class TrackMeta:
    track_id: str
    name: str
    pit_lane_time_s: float          # approximate green-flag pit lane loss
    track_length_km: float | None = None
    geom_path: Path | None = None   # optional: XY polyline of track centreline


TRACK_METAS: dict[str, TrackMeta] = {
    "barber-motorsports-park": TrackMeta(
        track_id="barber-motorsports-park",
        name="Barber Motorsports Park",
        pit_lane_time_s=27.0,              # rough green-flag pit loss
        track_length_km=3.83,
        geom_path=TRACK_GEOM_DIR / "barber_track_xy.csv",
    ),
    "virginia-international-raceway": TrackMeta(
        track_id="virginia-international-raceway",
        name="Virginia International Raceway",
        pit_lane_time_s=30.0,
        track_length_km=5.26,
        geom_path=None,
    ),
    "circuit-of-the-americas": TrackMeta(
        track_id="circuit-of-the-americas",
        name="Circuit of the Americas",
        pit_lane_time_s=26.0,
        track_length_km=5.51,
        geom_path=None,
    ),
}