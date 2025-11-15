## how to run web visualization for Barber

git clone https://github.com/ngstephen1/DataScienceHackbyToyota.git
cd DataScienceHackbyToyota

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run streamlit_app.py

Open the URL shown in the terminal (usually http://localhost:8501).