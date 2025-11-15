from src.track_utils import sector_from_distance
from src.strategy_engine import simulate_strategy

print("=== Sector mapping test ===")
for d in [0, 1000, 2000, 4000, 5200]:
    print("VIR dist", d, "-> sector", sector_from_distance("virginia-international-raceway", d))

print("\n=== Strategy engine skeleton test ===")
res = simulate_strategy(
    track_id="virginia-international-raceway",
    race_id="vir_r1",
    car_id="car_01",
    current_lap=10,
    candidate_pit_laps=[12, 14, 16],
)
for r in res:
    print(r)