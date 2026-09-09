import streamlit as st
import os
import pandas as pd

from pages.utils.predict import run_all_models

st.set_page_config(page_title="Current Predictions", page_icon="📈", layout="wide")

st.title("📈 Current 2026 Drivers' Championship Ranking Predictions")

st.write("These predictions are generated live from the current in-progress-season data, using all 3 of the v2 generation models (Prost, Schumacher, and Senna) - v2 is the final generation this project produced, having beaten every v1 model on rho. See the [Model Naming Scheme](/Model_Naming_Scheme) page for how they compare, and [Model Development & Training](/Model_Development_&_Training) for why a v3 was attempted but not pursued further. Unlike a pre-baked snapshot, this page re-runs the models against whatever the latest downloaded round is, every time it loads.")

PREDICTION_DATA_PATH = os.path.join("data", "clean", "f1_drivers_clean_prediction_data.csv")
MODELS_DIR = os.path.join("src", "models", "v2", "pretrained_models")

if not os.path.exists(PREDICTION_DATA_PATH):
    st.error(f"No in-progress-season prediction data found at `{PREDICTION_DATA_PATH}`. Run the data pipeline's `--incomplete-years` step first.")
    st.stop()

results_df = run_all_models(PREDICTION_DATA_PATH, MODELS_DIR)

orig_df = pd.read_csv(PREDICTION_DATA_PATH)
idx_final_rank = orig_df.sort_values(["DriverId", "RoundsCompleted"]).groupby(["DriverId"])["RoundsCompleted"].idxmax()
current_df = orig_df.loc[idx_final_rank].sort_values("CurrentRank")

results_df.insert(loc=0, column="Current Standings", value=current_df["DriverId"].to_list())
results_df.insert(loc=1, column="Current Points", value=current_df["TotalPoints"].to_list())

year = orig_df["Year"].iloc[0]
st.write(f"As of the latest downloaded round of the {year} season:")
st.dataframe(results_df, hide_index=True, width="content")

st.markdown('''
A note on reading this table: "Current Points" comes from each driver's own most recent row, not a single global last round - a mid-season driver swap means not every driver has raced the literal latest calendar round, so filtering to one fixed round number would silently misalign this table for anyone who joined partway through the season.
''')
