<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">


# DATASCIENCE HACK BY TOYOTA - Hack The Track

<em>Accelerate Insights, Dominate Race Strategies Instantly</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/ngstephen1/DataScienceHackbyToyota?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/ngstephen1/DataScienceHackbyToyota?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/ngstephen1/DataScienceHackbyToyota?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/ngstephen1/DataScienceHackbyToyota?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">

</div>
<br>

---

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgment](#acknowledgment)

---

## Overview

DataScienceHackbyToyota is an advanced simulation and analysis platform built around **Toyota GR86 GR Cup** data, with a focus on **Barber Motorsports Park**. It combines interactive visualizations, large-scale telemetry processing, and AI-driven insights (via Google Gemini) to help a race engineer answer questions like:

- *What’s our ideal pit window right now?*
- *How fast are our tyres degrading by stint and by sector?*
- *What should we do if a caution comes out in the next 2–3 laps?*

The system links:

- A **Tkinter + Matplotlib race map** with a moving car icon, running on the actual digitised track layout.
- A **Streamlit “race engineer console”** that updates from a live JSON state file.
- A **strategy engine** that simulates tyre degradation, pit windows, and caution scenarios.
- **Gemini 2.5 Flash** for real-time natural-language insights on top of your metrics and notebook conclusions.

**Why DataScienceHackbyToyota?**

This project is designed to make massive, messy racing data usable in real time:

- 🎯 **Interactive Visualizations:** Animated car running around Barber, lap-time charts, and live race-engineer panels.
- 🚗 **Telemetry & Track Processing:** Tools to ingest TRD telemetry, build lap features, and digitise the Barber track from a circuit map.
- ⚙️ **Performance & Strategy Modeling:** Tyre degradation modeling, pit lane loss, strategy multiverse simulations, and ideal pit-lap discovery.
- 🤖 **AI-Generated Insights:** Gemini-powered “radio feed” style commentary and recommendations updated off the current lap context.
- 📊 **Modular Architecture:** Notebooks for exploration, Python modules for reusable logic, and a Streamlit UI for demo / race-day scenarios.

---

## Check it out:
> https://devpost.com/software/racing-hokies

> https://toyotadatasense.streamlit.app

---

## Data Sources

This project uses **official hackathon data and maps** from:

- **TRD Dev – DataScienceHackbyToyota 2025**  
  Dataset page: `https://trddev.com/hackathon-2025/`

- **Barber Motorsports Park telemetry bundle**  
  File: `barber-motorsports-park.zip`  
  ~20 CSV files including:
  - `R1_barber_telemetry_data.csv` — **Rows:** 11,556,519 • **Cols:** 13  
  - `R2_barber_telemetry_data.csv` — **Rows:** 11,749,604 • **Cols:** 13  
  plus lap timing, weather, best-10 laps, results, etc.

- **Barber circuit map (official PDF)**  
  URL: `https://trddev.com/hackathon-2025/Barber_Circuit_Map.pdf`

In this repo, the track map image used for digitising and visualisation is:

```text
data/track_maps/IMG_4381.jpg
```

Embedded preview:

![Barber Motorsports Park circuit map](data/track_maps/IMG_4381.jpg)

The digitised centerline (from manual clicks on this map) lives in:

```text
data/track_geom/barber_track_xy.csv
data/track_geom/barber_track_xy_s.csv
```

These are what the **car icon animation** uses.

---

## Getting Started

### Prerequisites

- **Python:** 3.10+ (tested with 3.10 / 3.11 / 3.13)
- **Package Manager:** `pip`
- (Optional) **virtualenv / venv** for isolation
- (Optional but recommended) **Google Gemini API key** for AI insights

### Installation

1. **Clone the repository**

```sh
git clone https://github.com/ngstephen1/DataScienceHackbyToyota.git
cd DataScienceHackbyToyota
```

2. **Create & activate a virtual environment (recommended)**

```sh
python -m venv .venv
source .venv/bin/activate        # on macOS / Linux
# .venv\Scriptsctivate         # on Windows PowerShell
```

3. **Install dependencies**

```sh
pip install -r requirements.txt
```

4. **(Optional) Configure Gemini**

Set your Gemini API key and model name (Gemini 2.5 Flash):

```sh
export GEMINI_API_KEY="your-key-here"
export GEMINI_MODEL_NAME="gemini-2.5-flash"
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-key-here"
$env:GEMINI_MODEL_NAME="gemini-2.5-flash"
```

You can quickly verify the key with:

```sh
python - << 'PY'
import os, google.generativeai as genai
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
if not api_key:
    raise SystemExit("GEMINI_API_KEY is not set.")
genai.configure(api_key=api_key)
print(f"Trying model: {model_name}")
model = genai.GenerativeModel(model_name)
resp = model.generate_content("Say exactly: Gemini OK.")
print("Response:", resp.text.strip())
PY
```

---

### Usage

There are **two main entrypoints**: the **animated map** and the **Streamlit race-engineer console**. They communicate through `data/live/` JSON files.

#### 1. Run the animated Barber map (Tkinter + Matplotlib)

This opens a window showing the **Barber map** with a **car icon** moving along the digitised track, plus real-time metrics and Gemini insights:

```sh
python src/barber_lap_anim.py
```

This script:

- Loads the Barber track map (`data/track_maps/barber_map.png` / `IMG_4381.jpg`).
- Loads digitised centerline from `data/track_geom/barber_track_xy_s.csv`.
- Reads lap features and strategy summaries from `data/processed/barber/...`.
- Animates one car icon per lap based on lap times.
- Writes live state to:

```text
data/live/barber_state.json
data/live/live_state_barber.json
```

#### 2. Run the Streamlit “Race Engineer Console”

In another terminal (same venv):

```sh
streamlit run streamlit_app.py
```

The Streamlit app:

- Polls the `data/live/*.json` files written by `barber_lap_anim.py`.
- Shows:
  - Live lap number, stint lap, lap progress, and lap time.
  - Tyre life, degradation, pit-window metrics, caution scenarios.
  - Sliders to adjust **tyre degradation scaling**, **pit loss scaling**, and **risk mode**.
  - Plots that update “as if” in real time.
  - An **engineering radio feed** with both rule-based and Gemini-generated insights.

If your Gemini key is configured, you’ll also see **Gemini insights** per lap (short bullet-point radio style).

> 💡 **Tip**  
> For a complete “live” demo, run:
> - Terminal 1: `python src/barber_lap_anim.py`  
> - Terminal 2: `streamlit run streamlit_app.py`

---

### Testing

There is **no full automated test suite** yet. For a quick sanity check:

```sh
python tests/manual_test.py
```

This exercises core loading and geometry logic to ensure things run without errors.

(You’re encouraged to add `pytest` tests for `src/` modules if you extend this project.)

---

## Features

|      | Component       | Details                                                                                     |
| :--- | :-------------- | :------------------------------------------------------------------------------------------ |
| ⚙️  | **Architecture**  | <ul><li>Modular Jupyter Notebook workflows for data analysis and modeling</li><li>Separation of data processing (`src/`), visualization (`barber_lap_anim.py`, `streamlit_app.py`), and strategy logic (`strategy_engine.py`)</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Clear function boundaries in modules</li><li>Notebooks used for exploratory analysis, with logic gradually migrated into reusable functions</li></ul> |
| 📄 | **Documentation** | <ul><li>README with overview, data sources, and how to run</li><li>`RUN.md` for step-by-step setup and demo instructions</li></ul> |
| 🔌 | **Integrations**  | <ul><li>`requirements.txt` for dependency management</li><li>Integrates `numpy`, `pandas`, `matplotlib`, `streamlit`, `google-generativeai`</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Separate notebooks for VIR vs Barber, race 1 vs race 2</li><li>Reusable strategy, track-meta, and telemetry loader modules</li></ul> |
| 🧪 | **Testing**       | <ul><li>Basic manual tests via `tests/manual_test.py` (no formal unit test suite yet)</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Handles 10M+ row telemetry CSVs using `pandas` and columnar workflows</li><li>Animation and dashboard driven by pre-aggregated lap features</li></ul> |
| 🛡️ | **Security**      | <ul><li>No external services beyond Gemini API; API key is read from environment variables</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Managed via `requirements.txt`</li><li>Includes `jupyter`, `streamlit`, `matplotlib`, `pandas`, `numpy`, `ipykernel`, `google-generativeai`</li></ul> |

---

## Project Structure

```sh
└── DataScienceHackbyToyota/
    ├── README.md
    ├── RUN.md
    ├── data
    │   ├── raw/           # Original TRD telemetry, results, weather, etc.
    │   ├── processed/     # Derived lap features, sector summaries, strategy outputs
    │   ├── track_geom/    # Digitised track centerline for Barber
    │   └── track_maps/    # Track images (PDF-derived JPG/PNG) + car icon
    ├── notebooks          # VIR + Barber exploration, lap times, sections, strategy MVP
    ├── requirements.txt
    ├── src
    │   ├── barber_build_track_s.py
    │   ├── barber_digitize_track.py
    │   ├── barber_lap_anim.py
    │   ├── live_state.py
    │   ├── pit_model.py
    │   ├── strategy_cli.py
    │   ├── strategy_engine.py
    │   ├── telemetry_loader.py
    │   ├── track_meta.py
    │   └── track_utils.py
    ├── streamlit_app.py   # Live dashboard / race engineer console
    ├── tests
    │   └── manual_test.py
    └── tools
        └── extract_barber_r1_vehicle.py
```

---

### Project Index

> ⚠️ *Auto-generated summaries below; you can ignore this section if you just want to run the demo. Descriptions have been lightly cleaned up to reflect racing context (Barber Motorsports Park, VIR, telemetry, strategy, etc.).*

<details open>
  <summary><b><code>DATASCIENCEHACKBYTOYOTA/</code></b></summary>
  <!-- __root__ Submodule -->
  <details>
    <summary><b>__root__</b></summary>
    <blockquote>
      <div class='directory-path' style='padding: 8px 0; color: #666;'>
        <code><b>⦿ __root__</b></code>
      <table style='width: 100%; border-collapse: collapse;'>
      <thead>
        <tr style='background-color: #f8f9fa;'>
          <th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
          <th style='text-align: left; padding: 8px;'>Summary</th>
        </tr>
      </thead>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>README.md</b></td>
          <td style='padding: 8px;'>- Describes the overall Barber & VIR race-engineering platform, how to run the animation and Streamlit app, and where the TRD data and track maps come from.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>streamlit_app.py</b></td>
          <td style='padding: 8px;'>- Main Streamlit UI: reads live state JSON, shows lap metrics, sliders, plots, and Gemini-generated race-engineer insights.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>requirements.txt</b></td>
          <td style='padding: 8px;'>- Pin list of Python packages (`pandas`, `numpy`, `matplotlib`, `streamlit`, `google-generativeai`, etc.) required to run notebooks, scripts, and the dashboard.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>RUN.md</b></td>
          <td style='padding: 8px;'>- Step-by-step runbook to set up the environment and launch the Barber race-engineer demo.</td>
        </tr>
      </table>
    </blockquote>
  </details>
  <!-- src Submodule -->
  <details>
    <summary><b>src</b></summary>
    <blockquote>
      <div class='directory-path' style='padding: 8px 0; color: #666;'>
        <code><b>⦿ src</b></code>
      <table style='width: 100%; border-collapse: collapse;'>
      <thead>
        <tr style='background-color: #f8f9fa;'>
          <th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
          <th style='text-align: left; padding: 8px;'>Summary</th>
        </tr>
      </thead>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>pit_model.py</b></td>
          <td style='padding: 8px;'>- Identifies pit laps, splits stints, and fits simple lap-time degradation models on top of GR Cup lap features.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>barber_digitize_track.py</b></td>
          <td style='padding: 8px;'>- Interactive digitisation of the Barber track map: click around the circuit on the map image to build the centerline polyline and save as CSV.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>telemetry_loader.py</b></td>
          <td style='padding: 8px;'>- Utilities to load and pre-process TRD telemetry CSVs into lap- and sector-level tables for further analysis.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>strategy_engine.py</b></td>
          <td style='padding: 8px;'>- Core race strategy logic: simple pit lane loss, tyre deg, and caution modelling used by CLI and live tools to estimate ideal pit laps and compare strategies.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>track_utils.py</b></td>
          <td style='padding: 8px;'>- Track-related helper functions (e.g., mapping distances to sectors) to support sector-level timing and plotting.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>test.py</b></td>
          <td style='padding: 8px;'>- Small utility for plotting digitised track geometry and visually checking that the centerline looks correct.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>barber_lap_anim.py</b></td>
          <td style='padding: 8px;'>- Barber race animation: draws the track map, moves a car icon along the path based on lap times, overlays a race-engineer info panel, and writes live JSON state for Streamlit + Gemini.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>live_state.py</b></td>
          <td style='padding: 8px;'>- Simple JSON helper to read/write live state files in `data/live/`, used to sync the animation and Streamlit app.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>track_meta.py</b></td>
          <td style='padding: 8px;'>- Metadata definitions for tracks (IDs, names, pit lane time assumptions, lengths) including Barber and VIR.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>barber_build_track_s.py</b></td>
          <td style='padding: 8px;'>- Computes cumulative and normalised arc length along the digitised Barber centerline to support smooth animation and mapping.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>strategy_cli.py</b></td>
          <td style='padding: 8px;'>- Command-line interface for running offline strategy simulations and printing recommended pit plans from processed lap features.</td>
        </tr>
      </table>
    </blockquote>
  </details>
  <!-- notebooks Submodule -->
  <details>
    <summary><b>notebooks</b></summary>
    <blockquote>
      <div class='directory-path' style='padding: 8px 0; color: #666;'>
        <code><b>⦿ notebooks</b></code>
      <table style='width: 100%; border-collapse: collapse;'>
      <thead>
        <tr style='background-color: #f8f9fa;'>
          <th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
          <th style='text-align: left; padding: 8px;'>Summary</th>
        </tr>
      </thead>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>01_explore_vir.ipynb</b></td>
          <td style='padding: 8px;'>- First pass on VIR data: exploring raw TRD files, basic distributions, and key variables for later sector and lap-time analysis.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>02_vir_sectors_r1r2.ipynb</b></td>
          <td style='padding: 8px;'>- VIR sector-level analysis for Race 1 and Race 2, looking at sector times, consistency, and where pace is gained or lost.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>03_barber_telemetry_r1.ipynb</b></td>
          <td style='padding: 8px;'>- Exploration of Barber Race 1 telemetry: cleaning, joining TRD files, and building the first lap-level and sector-level tables.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>04_barber_lap_times_r1.ipynb</b></td>
          <td style='padding: 8px;'>- Barber R1 lap time analysis: lap distributions, degradation trends, driver pace profile, and first ideas for pit windows.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>05_barber_sections_r1.ipynb</b></td>
          <td style='padding: 8px;'>- Race 1 sector breakdown at Barber: which sectors are most sensitive to tyre deg, and where the driver is strongest/weakest.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>06_barber_telemetry_r2.ipynb</b></td>
          <td style='padding: 8px;'>- Barber R2 telemetry exploration, mirroring R1 but with a focus on validating patterns and preparing R2 lap features.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>07_barber_lap_times_r2.ipynb</b></td>
          <td style='padding: 8px;'>- Barber R2 lap-time analysis, comparing Race 2 behaviour to Race 1 and checking consistency, deg, and pace trends.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>08_barber_sections_r2.ipynb</b></td>
          <td style='padding: 8px;'>- Barber R2 sector analysis (S1/S2/S3): where performance improves or degrades vs R1 and what that means for tyres & setup.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>09_barber_driver_profile.ipynb</b></td>
          <td style='padding: 8px;'>- Driver-profile notebook: consistency, risk profile, and typical mistakes across different stints and races at Barber.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>10_barber_strategy_mvp.ipynb</b></td>
          <td style='padding: 8px;'>- MVP offline strategy model for Barber: simple degradation + pit lane loss model and basic “strategy multiverse” table used by `strategy_engine.py`.</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>13_vir_telemetry_r1.ipynb</b></td>
          <td style='padding: 8px;'>- VIR R1 telemetry ingestion and sanity checks: building VIR-specific sector stats and lap-level datasets for future extension of the tool. </td>
        </tr>
      </table>
    </blockquote>
  </details>
  <!-- tools Submodule -->
  <details>
    <summary><b>tools</b></summary>
    <blockquote>
      <div class='directory-path' style='padding: 8px 0; color: #666;'>
        <code><b>⦿ tools</b></code>
      <table style='width: 100%; border-collapse: collapse;'>
      <thead>
        <tr style='background-color: #f8f9fa;'>
          <th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
          <th style='text-align: left; padding: 8px;'>Summary</th>
        </tr>
      </thead>
        <tr style='border-bottom: 1px solid #eee;'>
          <td style='padding: 8px;'><b>extract_barber_r1_vehicle.py</b></td>
          <td style='padding: 8px;'>- Helper to slice out a single vehicle’s telemetry (e.g. GR86-002-000) from the massive Barber R1 telemetry CSV for downstream processing.</td>
        </tr>
      </table>
    </blockquote>
  </details>
</details>

---

## Roadmap

- [x] **`Task 1`**: Build Barber R2 strategy MVP and live race-engineer demo (animation + Streamlit + Gemini).
- [ ] **`Task 2`**: Generalise live tooling to VIR and additional tracks from the TRD dataset.
- [ ] **`Task 3`**: Add richer Monte Carlo strategy simulations and multi-car race scenarios into the Streamlit UI.

---

## Contributing

- **💬 [Join the Discussions](https://github.com/ngstephen1/DataScienceHackbyToyota/discussions)** – Ideas, feedback, and questions.
- **🐛 [Report Issues](https://github.com/ngstephen1/DataScienceHackbyToyota/issues)** – Bugs, edge cases, or feature requests.
- **💡 Submit Pull Requests** – Improvements to strategy models, visualisations, or notebook analysis are very welcome.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**  
   ```sh
   git fork https://github.com/ngstephen1/DataScienceHackbyToyota
   ```

2. **Clone Locally**  
   ```sh
   git clone https://github.com/<your-username>/DataScienceHackbyToyota
   cd DataScienceHackbyToyota
   ```

3. **Create a New Branch**  
   ```sh
   git checkout -b feature/my-improvement
   ```

4. **Make Your Changes** – and run the demo / manual tests.

5. **Commit Your Changes**  
   ```sh
   git commit -m "Add <short description of change>"
   ```

6. **Push to GitHub**  
   ```sh
   git push origin feature/my-improvement
   ```

7. **Open a Pull Request** – Describe what you changed and why.

</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com/ngstephen1/DataScienceHackbyToyota/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=ngstephen1/DataScienceHackbyToyota">
   </a>
</p>
</details>

---

## License

DataScienceHackbyToyota is released under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for details.

---

## Acknowledgments

- **Toyota Racing Development (TRD)** for providing the GR86 GR Cup data and Barber circuit map.
- **DataScienceHackbyToyota 2025 organisers** for framing the “real-time race engineer” challenge.
- Open-source libraries: `numpy`, `pandas`, `matplotlib`, `streamlit`, `google-generativeai`, and others in `requirements.txt`.

<div align="left"><a href="#top">⬆ Return</a></div>

---
