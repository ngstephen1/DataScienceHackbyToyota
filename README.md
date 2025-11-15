<<<<<<< Updated upstream
# GR RaceCraft Copilot

=======
<<<<<<< HEAD
# DataScienceHackbyToyota

Thanks Giving side hustle
=======
# GR RaceCraft Copilot

>>>>>>> Stashed changes
**Hackathon:** Hack the Track presented by Toyota GR  
**Categories:** Real-Time Analytics (primary), Wildcard (Mini Multiverse Simulator)

Team:
- Tue Tran Minh — Virginia Tech
- Nguyen Nguyen Phan — Virginia Tech

---

## 1. Project Overview

GR RaceCraft Copilot is a real-time race strategy assistant for the **Toyota GR Cup**:

- Reconstructs laps, stints, and sectors from TRD telemetry and official track maps.
- Simulates pit windows and caution scenarios to answer:  
  **“If we pit now vs. later, what happens to our race?”**
- Provides driver-facing insights and a small **“multiverse” simulator** to explore
  alternate strategies.

Right now, we are building the **core strategy brain and track metadata layer** so
we can plug in TRD CSVs as soon as access is available.

---

## 2. Planned Product Outcomes

1. **Real-Time Strategy Brain (Engineer Mode)**
   - Reconstruct laps, stints, and sectors from TRD telemetry + track maps.
   - Simulate pit windows and basic caution scenarios in real time.
   - Recommend an optimal pit window with expected time/position gain and a short explanation.

2. **Driver Insight Layer**
   - Break performance into S1/S2/S3 using track-specific sector geometry.
   - Quantify tire degradation and stint consistency.
   - Highlight where the driver is losing time and how that affects strategy calls.

3. **Mini Multiverse Simulator (Wildcard)**
   - Use the same strategy engine to run many “what-if” race simulations  
     (e.g., early vs. standard vs. late pit stop).
   - Estimate win/podium probability and average finishing position for each strategy
     under random cautions and pit-time variations.
   - Show which strategy “wins in more universes” and sketch a future **online arena**
     for engineers, drivers, and fans.

---

## 3. Repository Structure (current)

```text
DataScienceHackbyToyota/
├─ src/
│  ├─ __init__.py
│  ├─ track_meta.py       # Track geometry + pit metadata for all GR tracks
│  ├─ track_utils.py      # Helper functions (e.g., map distance -> sector)
│  └─ strategy_engine.py  # Strategy engine API skeleton
├─ data/
│  ├─ raw/                # (placeholder) TRD telemetry + timing CSVs
│  └─ processed/          # (placeholder) lap/stint summary tables
├─ notebooks/             # (placeholder) exploration & analysis notebooks
├─ tests/
│  └─ manual_test.py      # (optional) quick sanity tests
└─ README.md


### When telemetry CSVs are available

- Call:

  ```bash
  python -m src.telemetry_loader data/raw/vir/telemetry.csv virginia-international-raceway
  ```

This will:
	•	Infer columns (vehicle id, meta_time, lap distance).
	•	Reconstruct laps and sectors.
<<<<<<< Updated upstream
	•	Output a per-lap summary table in memory (later we can save to data/processed/).
=======
	•	Output a per-lap summary table in memory (later we can save to data/processed/).
>>>>>>> fb9ac3b35d6600a5bc1be2526c60232a75e8d65f
>>>>>>> Stashed changes
