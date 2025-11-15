import pandas as pd
from pathlib import Path

ROOT = Path("data/raw/barber")
big_path = ROOT / "R1_barber_telemetry_data.csv"
out_path = ROOT / "R1_barber_telemetry_vehicle_GR86-002-000.csv"

VEHICLE_ID = "GR86-002-000"   # car we saw in df_small

chunksize = 200_000

print(f"Extracting vehicle {VEHICLE_ID} from {big_path} -> {out_path}")

first = True
for i, chunk in enumerate(pd.read_csv(big_path, chunksize=chunksize)):
    sub = chunk[chunk["vehicle_id"] == VEHICLE_ID]
    if sub.empty:
        continue
    if first:
        sub.to_csv(out_path, index=False)
        first = False
    else:
        sub.to_csv(out_path, index=False, mode="a", header=False)
    print(f"Chunk {i}: kept {len(sub)} rows")

print("Done. Wrote per-vehicle telemetry to:", out_path)