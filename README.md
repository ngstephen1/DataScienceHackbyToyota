# 🏎️ Live Race Copilot – Barber GR Cup Strategy

This repo turns real **Toyota GR86 GR Cup** data from **Barber Motorsports Park** into a
race-engineer sandbox. You can:

- Compare 1-stop / 2-stop strategies under different **caution** scenarios  
- Analyze lap times, sectors, and driver consistency  
- Run a **live race copilot** with Gemini 2.5 generating engineering radio calls  
- Watch a car icon lap the Barber track with real-time strategy info

---

## Data & track map

All racing data and the base circuit map come from the official **Toyota GR Cup Hackathon 2025** resources:

- Hackathon page & downloads:  
  <https://trddev.com/hackathon-2025/>
- Barber circuit PDF map:  
  <https://trddev.com/hackathon-2025/Barber_Circuit_Map.pdf>
- Barber Motorsports Park data bundle: `barber-motorsports-park.zip`  
  (20 CSV data files in total)

Key high-volume telemetry files:

- `R1_barber_telemetry_data.csv` – **11,556,519 rows × 13 columns**  
- `R2_barber_telemetry_data.csv` – **11,749,604 rows × 13 columns**

These, plus lap, sector, weather, and results files are stored under:

- `data/raw/barber/` – raw hackathon CSVs
- `data/processed/barber/` – derived lap features, sector summaries, and strategy multiverse summaries

Digitized track geometry is based on the Barber circuit PDF and manual digitization, stored in:

- `data/track_geom/barber_track_xy.csv`
- `data/track_geom/barber_track_xy_s.csv`


## Project Layout

	•	data/raw/barber/ – original anonymized race, weather, sector & telemetry CSVs
	•	data/processed/barber/ – lap features & strategy multiverse summaries
	•	data/track_geom/ – digitized Barber track polylines (barber_track_xy*.csv)
	•	data/track_maps/ – track map + car_icon.png
	•	notebooks/ – exploratory + feature engineering notebooks (03_barber_telemetry_r1.ipynb … 10_barber_strategy_mvp.ipynb)
	•	src/
	•	barber_lap_anim.py – Barber live lap animation + strategy console
	•	strategy_engine.py – tyre deg, pit-window & caution logic
	•	live_state.py – JSON live-state helper (data/live/barber_state.json)
	•	track_meta.py – track + pit-lane metadata
	•	streamlit_app.py – main multi-tab Streamlit dashboard
	•	RUN.md – extra run notes
	•	requirements.txt – Python dependencies

## Setup

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt

## Config

export GEMINI_API_KEY="YOUR_KEY_HERE"
export GEMINI_MODEL_NAME="gemini-2.5-flash"  

### Check
```bash
python - << 'PY'
import os, google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
print(model.generate_content("Say exactly: Gemini OK.").text.strip())
PY
```

## How to run

### Prerequisites

- Python 3.11+
- `git` (optional, if you cloned the repo)
- A Google Gemini API key (free-tier works)
- The TRD Hackathon Barber dataset extracted under `data/raw/barber`  
  (from `barber-motorsports-park.zip` on https://trddev.com/hackathon-2025/)

> The key high-volume telemetry files in that zip are:  
> - `R1_barber_telemetry_data.csv` – **11,556,519 rows × 13 columns**  
> - `R2_barber_telemetry_data.csv` – **11,749,604 rows × 13 columns**  
> plus timing, weather, results and “best 10 laps” files – about **20 CSVs in total**.

Your tree should look roughly like:

```text
data/
  raw/
    barber/
      R1_barber_telemetry_data.csv
      R2_barber_telemetry_data.csv
      ...
  processed/
    barber/
      barber_r2_GR86-002-000_lap_features.csv
      barber_r2_strategy_multiverse_summary.csv
      ...
  ```

If you ran the notebooks already, those data/processed/barber/*.csv files are created for you.

### Launch the Streamlit strategy app
```text
streamlit run streamlit_app.py
```

Then open the browser at http://localhost:8501

The app has three main tabs:
	1.	Strategy Brain
	•	Compares 1-stop early / 1-stop mid / 2-stop plans.
	•	Lets you inject a specific caution window and run a small Monte-Carlo “mini multiverse” of random cautions.
	•	Shows which strategy wins on average and why (mean race time, win probability, delta vs best).
	2.	Driver Insights
	•	Uses the processed barber_r?_GR86-002-000_lap_features.csv and sector summaries to highlight:
	•	Best and worst laps
	•	Lap-time consistency
	•	Where the driver gains/loses time by sector.
	3.	Live Race Copilot
	•	Simulates a live race engineer console for Barber R2, car GR86-002-000.
	•	Sliders let you control:
	•	Driver push level (fuel save ↔ quali mode)
	•	Risk appetite (conservative ↔ all-in)
	•	Subjective caution chance in the next 3 laps
	•	Update interval (how fast “time” moves in the local sim)
	•	Every few tenths of a second it recomputes:
	•	Tyre life & phase (warm-up / stable / degradation)
	•	Pit window, undercut/overcut, fuel-to-end
	•	Caution what-ifs and traffic risk
	•	A rolling “engineering radio feed” for each lap
	•	If GEMINI_API_KEY is set, Gemini adds short natural-language strategy insights per lap on top of the hand-built heuristics.

⸻

4. Run the animated Barber lap with car icon (desktop)

This is a Matplotlib/Tkinter-style window that shows:
	•	The Barber circuit map
	•	A moving car icon following the digitised racing line
	•	A real-time race-engineer text console and lap-time chart
	•	Continuous export of the live car state to data/live/live_state_barber.json


Run it from the project root:
```bash
python src/barber_lap_anim.py
```

What it does:
	•	Loads the smoothed track polyline from
data/track_geom/barber_track_xy_s.csv
	•	Uses data/track_maps/barber_map.png and data/track_maps/car_icon.png
for the background map and car sprite
	•	Steps through barber_r2_GR86-002-000_lap_features.csv, scaling the number
of animation frames per lap from the actual lap_time_s
	•	Per frame, writes a JSON “live state” snapshot:

```jsonc
{
  "track_id": "barber",
  "lap_no": 12,
  "total_laps": 28,
  "stint_lap": 5,
  "lap_progress": 0.47,
  "time_into_lap_s": 46.3,
  "car_x_px": 512.3,
  "car_y_px": 281.6,
  ...
}
```
The sliders at the bottom (tyre-deg scale, pit-loss scale, risk mode, speed x)
let you play “what-if” scenarios live while the car is moving.

### Points are saved to 
data/track_geom/barber_track_xy.csv

7. Notes on data volume & performance
	•	The raw telemetry CSVs are very large (11.5M and 11.7M rows).
The notebooks in notebooks/ trim and aggregate them into the
data/processed/ CSVs that the app actually uses.
	•	If you re-run the notebooks, expect several minutes of processing time
depending on your machine.
	•	The Streamlit app and animation are designed to run off the processed
data only, so day-to-day usage is lightweight once preprocessing is done.