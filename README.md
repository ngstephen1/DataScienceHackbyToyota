<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="DataScienceHackbyToyota.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# DATASCIENCE HACK BY TOYOTA – Hack The Track

<em>Accelerate Insights, Dominate Race Strategies Instantly</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/ngstephen1/DataScienceHackbyToyota?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/ngstephen1/DataScienceHackbyToyota?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/ngstephen1/DataScienceHackbyToyota?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/ngstephen1/DataScienceHackbyToyota?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn"><br>
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/Google%20Gemini-8E75B2.svg?style=flat&logo=Google-Gemini&logoColor=white" alt="Google%20Gemini">

</div>
<br>

---

## 📄 Table of Contents

- [Overview](#-overview)
- [Inspiration](#-inspiration)
- [What It Does](#-what-it-does)
- [How We Built It](#-how-we-built-it)
- [Challenges We Ran Into](#-challenges-we-ran-into)
- [Accomplishments That We’re Proud Of](#-accomplishments-that-were-proud-of)
- [What We Learned](#-what-we-learned)
- [Whats Next for Racing Hokies](#-whats-next-for-racing-hokies)
- [Tech Stack](#-tech-stack)
- [Data Sources](#-data-sources)
- [Getting Started](#-getting-started)
    - [Prerequisites](#-prerequisites)
    - [Installation](#-installation)
    - [Usage](#-usage)
    - [Testing](#-testing)
- [Features](#-features)
- [Project Structure](#-project-structure)
    - [Project Index](#-project-index)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgment](#-acknowledgment)

---

## ✨ Overview

DataScienceHackbyToyota is an advanced simulation and analysis platform built around **Toyota GR86 GR Cup** data, with a focus on **Barber Motorsports Park**. It combines interactive visualizations, large-scale telemetry processing, and AI-driven insights (via **Google Gemini 2.5 Flash**) to help a race engineer answer questions like:

- *What’s our ideal pit window right now?*
- *How fast are our tyres degrading by stint and by sector?*
- *What should we do if a caution comes out in the next 2–3 laps?*
- *If we box now, do we win or lose track position by the flag?*

The system links:

- A **Tkinter + Matplotlib race map** with a moving car icon, running on the actual digitised Barber layout.
- A **Streamlit “race engineer console”** that updates from a live JSON state file.
- A **strategy engine** that simulates tyre degradation, pit windows, caution scenarios, and “mini multiverse” Monte Carlo races.
- **Gemini 2.5 Flash** for real-time natural-language insights, radio messages, and decision reviews.
- A **predictive lap-time model** (Random Forest) trained on GR86 lap features.
- An experimental **computer-vision pipeline** using Gemini Vision on sample GR86 images.

---

## 💡 Inspiration

In modern motorsport, races are often decided by **fractions of a second** and **one pit call**. A mistimed stop or a slow reaction to a Safety Car can cost **multiple positions and tens of seconds** of race time, even when the driver’s pace is strong.

At the the same time, a single car can generate **millions of telemetry data points per race** (speed, throttle, brake, tyres, weather, timing, gaps, etc.). Race engineers have to digest all of this under pressure, in real time, while talking to the driver and coordinating with the team.

We wanted to build a tool that acts like an **AI co-engineer**: watching the data continuously, surfacing only what matters (“box now”, “tyre cliff in 3 laps”, “caution window coming”), and **turning RAW NUMBERS 🔜 DECISIONS**.

---

## 🏎️ What It Does

Our project turns real GR86 GR Cup data into an **interactive race-strategy cockpit** for Barber Motorsports Park:

- 🏁 **Live track sim** – a car icon runs around a digitised Barber circuit map based on lap timing and stint logic.
- 📊 **Real-time strategy console** – every second, the app updates:
  - Tyre phase (warm-up / stable / degradation)  
  - Net gain/loss if we pit now vs 2 laps earlier/later  
  - Caution / Safety-Car “what if” (next 3 laps)
  - Clean air vs traffic risk, gaps ahead/behind
- 📈 **Predictive lap-time model** – Random Forest model trained on lap features (`aps_mean`, `pbrake_f_mean`, …) for `GR86-002-000 @ Barber R2`, with:
  - RMSE and R² validation metrics
  - Comparison plots: actual vs predicted lap times, residuals, parity plot
  - Serialized model + JSON metadata for use in the app
- 🤖 **Gemini-powered race engineer radio** – Gemini 2.5 Flash reads the current lap metrics plus notebook insights and generates **short, actionable radio calls**:
  - “Box now under caution – you’ll undercut P5 by ~1.2s”
  - “Stay out, overcut in clean air, target 2 laps more”
  - “Tyres stable – push S2, save S3”
- 💬 **Strategy Chat** – a chat assistant that:
  - Sees track & car context, best strategy table, and current lap snapshot
  - Answers engineer-style questions in plain English
  - Stays grounded in the numbers and cites which metrics it used
- 🧠 **Decision Reviewer** – an AI “second pair of eyes”:
  - You describe your intended radio call (“Box now for 4 tyres and fuel to the end”)
  - Gemini reviews it with the current race context and returns:
    - Verdict (Go / Borderline / Don’t do it)
    - Rationale and key risks
    - Safer alternative calls
- 👁️ **Vision + Gemini** – an experimental computer-vision notebook:
  - Uses Gemini 2.5 Flash (vision) on `data/vision/sample_gr86_barber.png`
  - Asks the model to describe car position, lane usage, runoff, and risk
  - Aggregates outputs into simple stats and a plot (e.g. lane-centre histogram)

---

## 🛠️ How We Built It

### Data

We used the official TRD hackathon dataset from:  
https://trddev.com/hackathon-2025/

Key Barber files (race 1 & 2):

- `R1_barber_telemetry_data.csv` – **11,556,519 rows × 13 columns**  
- `R2_barber_telemetry_data.csv` – **11,749,604 rows × 13 columns**  
- Plus timing, weather, sector stats and results files — about **20 CSV files** in total.

We pre-processed these into lap-level features and strategy summaries (see `/data/processed` and the analysis notebooks `03–11_barber_*.ipynb`).

### Core stack

- **Python**, **NumPy**, **pandas** – data processing, lap & stint features, tyre-deg models  
- **Matplotlib** – live animation of the car icon moving around the digitised Barber map and plotting strategy / model results  
- **Streamlit** – race-engineer dashboard with tabs:
  - Strategy Brain
  - Driver Insights
  - Predictive Models
  - Strategy Chat
  - Live Race Copilot  
- **scikit-learn** – Random Forest regression for lap-time prediction  
- **Google Gemini 2.5 Flash** – chat, radio-style strategy calls, decision review, and experimental vision  
- **Tkinter / Matplotlib backends** – local animation window that writes a `live_state` JSON  
- **Custom modules**:
  - `strategy_engine.py` – degradation modeling, pit window simulation, Monte Carlo strategy multiverse  
  - `pit_model.py` – stint segmentation and simple deg curves  
  - `track_meta.py` – track metadata, pit-lane loss assumptions  
  - `live_state.py` – atomic JSON sync between animation and Streamlit  
  - `predictive_models.py` – lap-time model training / saving / inference  
  - `chat_assistant.py` – LLM-powered strategy chat context builder  
  - `decision_reviewer.py` – structured AI review of engineer calls  
  - `vision_gemini.py` – helper for running Gemini vision analysis over images  

The desktop animation (`barber_lap_anim.py`) continuously updates a JSON file (`data/live/live_state_barber.json`). The Streamlit app (`streamlit_app.py`) reads that same state up to once per second, recomputes metrics, optionally calls Gemini, and renders the UI.

---

## 🧗 Challenges We Ran Into

- **Handling huge telemetry files** – reading >11 million-row CSVs per race meant we had to be careful with memory, summarising to lap/stint-level data before doing heavier modeling.
- **Keeping everything in sync** – we needed the Matplotlib animation, JSON writer, and Streamlit dashboard to stay in lockstep without race conditions or crashes.
- **Prompt design for Gemini** – making sure the AI produces short, trustworthy, race-engineer-style bullet points instead of essays was an iteration loop of its own.
- **Unifying many tools** – predictive model, chat, decision review, and vision all had to coexist cleanly inside a single Streamlit file.

---

## 🏆 Accomplishments That We’re Proud Of

- Turning **raw GR Cup telemetry** into a **live strategy simulator** that really feels like a race-engineer console, not just static plots.
- Building an **AI radio feed** that reacts to tyre life, pit window, and caution scenarios in language a driver could actually understand mid-race.
- Shipping a working **lap-time Random Forest model** with sane metrics and visual diagnostics.
- Creating a **modular pipeline**: notebooks → processed features → strategy engine → live animation → Streamlit + Gemini.

---

## 📚 What We Learned

- How quickly motorsport data explodes in size, and why **aggregation & feature engineering** are critical before doing any fancy modeling.
- The importance of **race-decision framing**: engineers don’t want “here’s every metric,” they want “what should we do this lap and why?”.
- How to combine classical modeling (tyre deg, pit-lane loss, Monte Carlo strategies) with **LLM-synthesised insights** so the AI is grounded in real numbers.
- Practical patterns for safely using **Gemini 2.5 Flash** for both chat and structured decision review.

---

## 🔮 Whats Next for Racing Hokies

- ❯❯❯❯ **The Future** – expand real-time voice assistant, deeper AI decision reviewer, richer computer vision system, and more predictive models.
- 💠 **Multi-car & multi-track support** – extend to more GR Cup cars and to VIR using the rest of the TRD dataset.  
- 🌐 **Cloud + live feed** – adapt the pipeline to live telemetry streams and host the dashboard so a team can connect during a session.  
- 🧠 **Richer “what-if” engine** – allow the engineer to simulate alternative strategies (extra stop, short-fill, extreme fuel save) and compare projected race time in real time.  
- 👂 **Driver-aware coaching** – incorporate driver consistency, error patterns, and sector strengths into the radio calls (“strong in S1, losing time in S3 braking zone – adjust bias + lift earlier”).

By combining **real-world-scale data**, **physics-style models**, and **LLM insights**, our goal is to give race engineers a tool that doesn’t just visualise the race — it **helps call it**.

---

## 🧩 Tech Stack

**Languages & Libraries**

- Python, NumPy, pandas, SciPy  
- Matplotlib, Seaborn (in notebooks)  
- scikit-learn (Random Forest regression)  

**Frameworks & Platforms**

- Streamlit – interactive race engineer dashboard  
- Tkinter + Matplotlib – local live animation window  
- Jupyter – data exploration and model development  

**AI & APIs**

- Google Gemini 2.5 Flash (text & vision) via `google-generativeai`  

**Data & Storage**

- CSV-based telemetry & timing data (TRD dataset)  
- JSON live state in `data/live/*.json`  
- Joblib + JSON for model artifacts in `models/`  

**Tooling**

- GitHub for version control  
- VS Code / JupyterLab for development  

---

## 📊 Data Sources

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

These are what the **car icon animation** and strategy tools use.

---

## 🚀 Getting Started

### 📋 Prerequisites

- **Python:** 3.10+ (tested with 3.10 / 3.11 / 3.13)
- **Package Manager:** `pip`
- (Optional) **virtualenv / venv** for isolation
- (Optional but recommended) **Google Gemini API key** for AI insights (chat, decision review, vision)

### ⚙️ Installation

1. **Clone the repository**

```sh
git clone https://github.com/ngstephen1/DataScienceHackbyToyota.git
cd DataScienceHackbyToyota
```

2. **Create & activate a virtual environment (recommended)**

```sh
python -m venv .venv
source .venv/bin/activate        # on macOS / Linux
# .venv\Scripts\Activate       # on Windows PowerShell
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

### ▶️ Usage

There are **two main entrypoints**: the **animated map** and the **Streamlit race-engineer console**. They communicate through `data/live/` JSON files.

#### 1. Run the animated Barber map (Tkinter + Matplotlib)

This opens a window showing the **Barber map** with a **car icon** moving along the digitised track, plus real-time metrics and Gemini insights:

```sh
python src/barber_lap_anim.py
```

This script:

- Loads the Barber track map (`data/track_maps/IMG_4381.jpg`).
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

The Streamlit app provides multiple tabs:

- **Strategy Brain** – strategy multiverse, caution simulation, best strategy summary  
- **Driver Insights** – lap & sector trends, tyre phases, consistency  
- **Predictive Models** – Random Forest lap-time model summary and diagnostic plots  
- **Strategy Chat** – Gemini 2.5 Flash chat assistant with full race context  
- **Live Race Copilot** – live state from animation, decision reviewer, and AI radio  

If your Gemini key is configured, you’ll also see **Gemini insights** and **decision review** outputs (short bullet-point radio style).

> 💡 **Tip**  
> For a complete “live” demo, run:
> - Terminal 1: `python src/barber_lap_anim.py`  
> - Terminal 2: `streamlit run streamlit_app.py`

---

### 🧪 Testing

There is **no full automated test suite** yet. For a quick sanity check:

```sh
python tests/manual_test.py
```

This exercises core loading and geometry logic to ensure things run without errors.

(You’re encouraged to add `pytest` tests for `src/` modules if you extend this project.)

---

## 📦 Features

|      | Component       | Details                                                                                     |
| :--- | :-------------- | :------------------------------------------------------------------------------------------ |
| ⚙️  | **Architecture**  | <ul><li>Modular Jupyter Notebook workflows for data analysis and modeling</li><li>Separation of data processing (`src/`), visualization (`barber_lap_anim.py`, `streamlit_app.py`), and strategy logic (`strategy_engine.py`)</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Clear function boundaries in modules</li><li>Notebooks used for exploratory analysis, with logic gradually migrated into reusable functions</li></ul> |
| 📄 | **Documentation** | <ul><li>README with overview, data sources, and how to run</li><li>`RUN.md` for step-by-step setup and demo instructions</li></ul> |
| 🔌 | **Integrations**  | <ul><li>`requirements.txt` for dependency management</li><li>Integrates `numpy`, `pandas`, `matplotlib`, `streamlit`, `google-generativeai`, `scikit-learn`</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Separate notebooks for VIR vs Barber, race 1 vs race 2</li><li>Reusable strategy, track-meta, telemetry loader, predictive modeling, chat, decision review, and vision modules</li></ul> |
| 🧪 | **Testing**       | <ul><li>Basic manual tests via `tests/manual_test.py` (no formal unit test suite yet)</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Handles 10M+ row telemetry CSVs using `pandas` and columnar workflows</li><li>Animation and dashboard driven by pre-aggregated lap features</li></ul> |
| 🛡️ | **Security**      | <ul><li>No external services beyond Gemini API; API key is read from environment variables</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Managed via `requirements.txt`</li><li>Includes `jupyter`, `streamlit`, `matplotlib`, `pandas`, `numpy`, `ipykernel`, `google-generativeai`, `scikit-learn`</li></ul> |

---

## 📁 Project Structure

```sh
└── DataScienceHackbyToyota/
    ├── README.md
    ├── RUN.md
    ├── data
    │   ├── raw/            # Original TRD telemetry, results, weather, etc.
    │   ├── processed/      # Derived lap features, sector summaries, strategy outputs
    │   ├── track_geom/     # Digitised track centerline for Barber
    │   ├── track_maps/     # Track images (PDF-derived JPG/PNG) + car icon
    │   └── vision/         # Sample GR86 images for Gemini vision
    ├── models
    │   ├── lap_time_barber_GR86-002-000.joblib
    │   └── lap_time_barber_GR86-002-000.json
    ├── notebooks           # VIR + Barber exploration, lap times, sections, strategy MVP
    │   ├── 01_explore_vir.ipynb
    │   ├── 02_vir_sectors_r1r2.ipynb
    │   ├── 03_barber_telemetry_r1.ipynb
    │   ├── 04_barber_lap_times_r1.ipynb
    │   ├── 05_barber_sections_r1.ipynb
    │   ├── 06_barber_telemetry_r2.ipynb
    │   ├── 07_barber_lap_times_r2.ipynb
    │   ├── 08_barber_sections_r2.ipynb
    │   ├── 09_barber_driver_profile.ipynb
    │   ├── 10_barber_strategy_mvp.ipynb
    │   ├── 11_barber_predictive_model.ipynb
    │   └── 13_vir_telemetry_r1.ipynb
    ├── requirements.txt
    ├── src
    │   ├── __init__.py
    │   ├── barber_build_track_s.py
    │   ├── barber_digitize_track.py
    │   ├── barber_lap_anim.py
    │   ├── chat_assistant.py
    │   ├── decision_reviewer.py
    │   ├── live_state.py
    │   ├── pit_model.py
    │   ├── predictive_models.py
    │   ├── strategy_cli.py
    │   ├── strategy_engine.py
    │   ├── telemetry_loader.py
    │   ├── test.py
    │   ├── track_meta.py
    │   ├── track_utils.py
    │   └── vision_gemini.py
    ├── streamlit_app.py   # Live dashboard / race engineer console
    ├── tests
    │   └── manual_test.py
    └── tools
        └── extract_barber_r1_vehicle.py
```

---

## 📈 Roadmap

- [x] **`Task 1`**: Build Barber R2 strategy MVP and live race-engineer demo (animation + Streamlit + Gemini).
- [x] **`Task 1.5`**: Add predictive lap-time model, Strategy Chat, Decision Reviewer, and Gemini vision prototype.
- [ ] **`Task 2`**: Generalise live tooling to VIR and additional tracks from the TRD dataset.
- [ ] **`Task 3`**: Add richer Monte Carlo strategy simulations and multi-car race scenarios into the Streamlit UI.
- [ ] **`Task 4`**: Real-time voice assistant on top of Strategy Chat + Decision Reviewer.

---

## 🤝 Contributing

- **💬 [Join the Discussions](https://github.com/ngstephen1/DataScienceHackbyToyota/discussions)** – Ideas, feedback, and questions.
- **🐛 [Report Issues](https://github.com/ngstephen1/DataScienceHackbyToyota/issues)** – Bugs, edge cases, or feature requests.
- **💡 Submit Pull Requests** – Improvements to strategy models, visualisations, predictive modeling, or Gemini prompts are very welcome.

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

## 📜 License

DataScienceHackbyToyota is released under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for details.

---

## ✨ Acknowledgments

- **Toyota Racing Development (TRD)** for providing the GR86 GR Cup data and Barber circuit map.
- **DataScienceHackbyToyota 2025 organisers** for framing the “real-time race engineer” challenge.
- Open-source libraries: `numpy`, `pandas`, `matplotlib`, `streamlit`, `scikit-learn`, `google-generativeai`, and others in `requirements.txt`.

<div align="left"><a href="#top">⬆ Return</a></div>

---
