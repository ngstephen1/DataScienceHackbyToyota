from __future__ import annotations
from .track_meta import TRACK_METAS

def sector_from_distance(track_id: str, lapdist_m: float) -> int:
    """
    Map Laptrigger-style distance (meters from SF) to sector index 0,1,2.
    """
    meta = TRACK_METAS[track_id]
    d = lapdist_m % meta.circuit_length_m
    s1, s2, s3 = meta.sector_lengths_m

    if d < s1:
        return 0
    elif d < s1 + s2:
        return 1
    else:
        return 2