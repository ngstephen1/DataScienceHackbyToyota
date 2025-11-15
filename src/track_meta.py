from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class TrackMeta:
    track_id: str
    name: str
    circuit_length_m: float              # full lap length (center line)
    sector_lengths_m: List[float]        # [S1, S2, S3] in meters
    pit_lane_time_s: float               # time through pit lane @ 50 kph
    pit_lane_length_m: float             # pit in -> pit out (center line)
    gps_finish: Tuple[float, float]      # (lat, lon)
    gps_pit_in: Tuple[float, float]
    gps_pit_out: Tuple[float, float]
    pit_in_from_sf_m: float | None = None
    pit_out_from_sfp_m: float | None = None


SEBRING = TrackMeta(
    track_id="sebring",
    name="Sebring International Raceway",
    circuit_length_m=6018.9,
    sector_lengths_m=[1824.2, 1863.7, 2331.0],
    pit_lane_time_s=39.0,
    pit_lane_length_m=542.3,
    gps_finish=(27.4502340, -81.3536980),
    gps_pit_in=(27.45012, -81.35547),
    gps_pit_out=(27.45011, -81.35051),
    pit_in_from_sf_m=5842.2,
    pit_out_from_sfp_m=365.2,
)

SONOMA = TrackMeta(
    track_id="sonoma",
    name="Sonoma Raceway",
    circuit_length_m=4031.38,
    sector_lengths_m=[1385.0, 1422.0, 1225.0],
    pit_lane_time_s=45.0,
    pit_lane_length_m=623.9,
    gps_finish=(38.1615139, -122.4547166),
    gps_pit_in=(38.1615139, -122.4547166),   # TODO: refine if needed
    gps_pit_out=(38.1615139, -122.4547166),
    pit_in_from_sf_m=-519.25,
    pit_out_from_sfp_m=95.9,
)

VIRGINIA = TrackMeta(
    track_id="virginia-international-raceway",
    name="Virginia International Raceway",
    circuit_length_m=5262.6,
    sector_lengths_m=[1652.6, 2158.0, 1452.0],
    pit_lane_time_s=25.0,
    pit_lane_length_m=340.6,
    gps_finish=(36.5688167, -79.2066639),
    gps_pit_in=(36.567581, -79.210428),
    gps_pit_out=(36.568667, -79.206797),
    pit_in_from_sf_m=4898.7,
    pit_out_from_sfp_m=-16.0,
)

INDIANAPOLIS = TrackMeta(
    track_id="indianapolis",
    name="Indianapolis Motor Speedway – Road Course",
    circuit_length_m=3925.21,
    sector_lengths_m=[1364.28, 1387.86, 1173.99],
    pit_lane_time_s=63.0,
    pit_lane_length_m=865.43,
    gps_finish=(39.7931499, -86.2388700),
    gps_pit_in=(39.7894100, -86.2373000),
    gps_pit_out=(39.78696, -86.23881),
    pit_in_from_sf_m=3415.31,
    pit_out_from_sfp_m=394.84,
)

COTA = TrackMeta(
    track_id="circuit-of-the-americas",
    name="Circuit of the Americas",
    circuit_length_m=5498.3,
    sector_lengths_m=[1308.8, 2240.0, 1949.5],
    pit_lane_time_s=36.0,
    pit_lane_length_m=509.3,
    gps_finish=(30.1335278, -97.6422583),
    gps_pit_in=(30.1343371, -97.6340257),
    gps_pit_out=(30.1314446, -97.6389209),
    pit_in_from_sf_m=5284.9,
    pit_out_from_sfp_m=393.8,
)

BARBER = TrackMeta(
    track_id="barber-motorsports-park",
    name="Barber Motorsports Park",
    circuit_length_m=3674.6688,                      # 144,672" -> m
    sector_lengths_m=[1029.0048, 1580.388, 1065.276],  # S1, S2, S3
    pit_lane_time_s=34.0,
    pit_lane_length_m=477.3676,                      # 18,794" -> m
    gps_finish=(33.5327, -86.6196),   # approx; refine later
    gps_pit_in=(33.5317, -86.6226),
    gps_pit_out=(33.5311, -86.6225),
    pit_in_from_sf_m=3334.9184,                      # 131,296"
    pit_out_from_sfp_m=135.9408,                     # 5,352"
)

ROAD_AMERICA = TrackMeta(
    track_id="road-america",
    name="Road America",
    # S1+S2+S3 matches circuit length ~4.014 mi
    circuit_length_m=6459.6,
    # Inches -> meters: 81,048" / 86,928" / 86,340"
    sector_lengths_m=[
        2058.6192,  # S1
        2207.9712,  # S2
        2193.0360,  # S3
    ],
    pit_lane_time_s=52.0,              # Time through pit lane
    pit_lane_length_m=646.7856,        # 25,464" -> m
    gps_finish=(43.7979056, -87.9896333),
    gps_pit_in=(43.80057, -87.98992),
    gps_pit_out=(43.7948061, -87.9897494),
    pit_in_from_sf_m=616.0008,         # 24,252" -> m
    pit_out_from_sfp_m=347.1672,       # 13,668" -> m
)


TRACK_METAS: Dict[str, TrackMeta] = {
    SEBRING.track_id: SEBRING,
    SONOMA.track_id: SONOMA,
    VIRGINIA.track_id: VIRGINIA,
    INDIANAPOLIS.track_id: INDIANAPOLIS,
    COTA.track_id: COTA,
    BARBER.track_id: BARBER,
    ROAD_AMERICA.track_id: ROAD_AMERICA,
}