import streamlit as st
import os
import pandas as pd

from pages.utils.predict import run_all_models

st.set_page_config(page_title="Current Predictions", page_icon="📈", layout="wide")

st.title("📈 Current 2026 Drivers' Championship Ranking Predictions")

st.write("These predictions are generated live from the current in-progress-season data, using all 3 of the v2 generation models (Prost, Schumacher, and Senna). v2 is the final generation this project produced, having beaten every v1 model on rho. See the [Model Naming Scheme](/Model_Naming_Scheme) page for how they compare, and [Model Development & Training](/Model_Development_&_Training) for why a v3 was attempted but not pursued further. Unlike a pre-baked snapshot, this page re-runs the models against whatever the latest downloaded round is, every time it loads.")

def driver_surname(driver_id: str) -> str:
    """DriverId is fastf1/Ergast shorthand (e.g. "max_verstappen", "leclerc"). The last
    underscore-separated segment is always the surname, so this needs no hardcoded name list and
    stays correct as new drivers join the grid."""
    return driver_id.split("_")[-1].capitalize()

PREDICTION_DATA_PATH = os.path.join("data", "clean", "f1_drivers_clean_prediction_data.csv")
MODELS_DIR = os.path.join("src", "models", "v2", "pretrained_models")

if not os.path.exists(PREDICTION_DATA_PATH):
    st.error(f"No in-progress-season prediction data found at `{PREDICTION_DATA_PATH}`. Run the data pipeline's `--incomplete-years` step first.")
    st.stop()

results_df = run_all_models(PREDICTION_DATA_PATH, MODELS_DIR)
for model_col in results_df.columns:
    results_df[model_col] = results_df[model_col].apply(driver_surname)

orig_df = pd.read_csv(PREDICTION_DATA_PATH)
idx_final_rank = orig_df.sort_values(["DriverId", "RoundsCompleted"]).groupby(["DriverId"])["RoundsCompleted"].idxmax()
current_df = orig_df.loc[idx_final_rank].sort_values("CurrentRank")

results_df.insert(loc=0, column="Current Standings", value=current_df["DriverId"].apply(driver_surname).to_list())
results_df.insert(loc=1, column="Current Points", value=current_df["TotalPoints"].to_list())

year = orig_df["Year"].iloc[0]
latest_round = int(orig_df["Round"].max())
latest_location = orig_df.loc[orig_df["Round"] == latest_round, "Location"].iloc[0]
st.write(f"These predictions are post-{latest_location} (Round {latest_round}) of the {year} season, the latest round downloaded so far:")

# Explicit height (row count * default row height, plus the header row) so every driver shows at
# once. st.dataframe's default "auto" height still caps at a fixed pixel height and forces an
# internal scrollbar once the field is bigger than ~10-11 rows, well short of a full ~20-24 driver
# grid.
ROW_HEIGHT_PX = 35
table_height = ROW_HEIGHT_PX * (len(results_df) + 1) + 3
st.dataframe(results_df, hide_index=True, width="stretch", height=table_height)

st.markdown('''
A note on reading this table: "Current Points" comes from each driver's own most recent row, not a single global last round. A mid-season driver swap means not every driver has raced the literal latest calendar round, so filtering to one fixed round number would silently misalign this table for anyone who joined partway through the season.
''')
