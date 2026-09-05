import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Model Development & Training", page_icon="🔧", layout="wide")

st.title("🔧 Model Development & Training")

st.header("Project Development Timeline")

st.subheader("0. Curating the Dataset")
st.write("This project uses the same `fastf1` data source and 2018-2026 window as `f1-constructors-predictor`, downloaded and cleaned by a shared-shape but independent data pipeline (`src/data/`). The row grain here is `(Year, DriverId, Round)`, built only from rounds a driver actually raced - not every calendar round. That's a real difference from the constructors project: a team never misses a round, but a driver can, after a mid-season swap or a DNS. `RoundsCompleted` (rounds actually raced) and `Round` (calendar round number) are genuinely different information here rather than being collinear, which is why both are kept as features.")

st.write("Race/qualifying results are aggregated per-driver-per-round, and *separately* per-team-per-round, then joined onto each driver's row by `(Year, TeamId, Round)`. This gives the model a car-strength signal (the `Team*` features) independent of a driver's individual results - a driver's raw stats mean something different in a front-running car than a backmarker, which is also why `TeamId` itself is a model feature here, unlike in the constructors project where every row already was a team.")

st.subheader("1. v1 Model Development")
st.markdown("""
- Data Features (57): 47 numeric features plus a 10-column one-hot `TeamId` encoding - see the [Model Data Features](/Model_Data_Features) page for the full list
- Most Influential Features (per SHAP analysis, consistent across all 3 models): `ProjectedSeasonTotalPoints` and `TeamProjectedSeasonTotalPoints`, by a wide margin over everything else
- Least Influential Features (per SHAP analysis, consistent across all 3 models): the individual `TeamId_*` one-hot columns, `HasQualifyingData`, and the DNF-cause breakdown features (`DriverFaultDNFRate`/`MechanicalDNFRate` and their per-round variants) - a leading candidate list for a future v2 to prune
- Model Architecture: 4 fully-connected layers - the first 2 with ReLU, Dropout, and BatchNorm; the 3rd with ReLU and Dropout; the last a plain linear output
""")

st.write("Getting v1 to a trustworthy state took more than picking features and an architecture - the first full pass at evaluation turned out to have 3 real bugs stacked on top of each other, all of which made the models look considerably worse (and worse in a more 'architectural-looking' way) than they actually were:")
st.markdown("""
1. **Row-level random train/test split leaked information.** A driver's rounds could be scattered across both train and test, so the model could effectively see part of a driver-season's late-race form during training and still get graded on it - and the reverse also happened, where an early, unrepresentative row got graded against a driver's true final rank.
2. **The final reported metrics used a stale checkpoint.** Early-stopping saved the best-performing epoch, but the final report and heatmap were built from whatever epoch training happened to end on, not the epoch that was actually reloaded as "best."
3. **Ranking only within the (small) held-out subset produced a scale mismatch.** Once the split was fixed to avoid leaking a driver across train/test, ranking predictions only against the other held-out rows - rather than the driver's full real-world field - made the predicted ranks incomparable to the true `FinalRank` scale.
""")
st.write("Fixing all 3 led to the evaluation approach this project actually uses now: holding out whole seasons rather than individual rows, and ranking each held-out season against its full real field. Here's how the 3 finished v1 models actually perform, each evaluated on its own held-out season pair (never seen during that model's training):")

v1_summary_df = pd.DataFrame({
    "v1 Model": ["Fangio", "Clark", "Button"],
    "Held-Out Seasons": ["2020, 2021", "2022, 2023", "2024, 2025"],
    "Spearman's Rho": [0.9908, 0.9881, 0.9529],
    "Mean Absolute Error": [0.46, 0.57, 1.09],
    "Max Error": [2.0, 3.0, 9.0],
})

st.dataframe(v1_summary_df, hide_index=True, width="content")

st.write("*Note: these are v1's original 3 models. Their names were swapped with v2's after v2 produced 3 models that each beat every v1 model on rho - see the [Model Naming Scheme](/Model_Naming_Scheme) page for why.*")

st.write("The heatmaps below show *where* that accuracy comes from - Fangio and Clark both stay tight to the diagonal, while Button's wider spread away from the diagonal (especially in the max error) reflects the more volatile 2024-2025 pair it was evaluated against.")

with st.expander("Fangio v1 Heatmap"):
    st.image(os.path.join("src", "models", "v1", "heatmaps", "Fangio Model Heatmap 2026-09-03.png"))

with st.expander("Clark v1 Heatmap"):
    st.image(os.path.join("src", "models", "v1", "heatmaps", "Clark Model Heatmap 2026-09-03.png"))

with st.expander("Button v1 Heatmap"):
    st.image(os.path.join("src", "models", "v1", "heatmaps", "Button Model Heatmap 2026-09-03.png"))

st.header("A Brief Overview of the Model Training & Testing Methodology")
st.write("This is a pairwise ranker, not a rank-value regressor. `F1DriversRankClassifier` outputs one scalar 'score' per row; training uses `nn.MarginRankingLoss` on sampled pairs of *different drivers, same year and round*, and scores only get turned into an actual 1-to-n rank at evaluation/prediction time. This is why the model works regardless of how many drivers are in a given season's field - nothing about field size is hardcoded into the ranking mechanism itself.")

st.write("Pairs are weighted by how close the two drivers' points totals are (`1.0 + alpha / (1.0 + points_gap_per_round)`), so tighter competition gets more training emphasis wherever it happens in the standings - unlike a fixed rank-band assumption, which doesn't hold for a ~20-24 driver field where title fights can be just as tight at the front as in the midfield.")

st.write("The dataset is split by holding out 2 whole seasons at a time, rather than by individual row or by driver. Every driver in a held-out year goes entirely to test, and the model never trains on a single row from that year. That's what makes it valid to evaluate by ranking a held-out year's *full* field against the true standings - the model genuinely never saw that season, so there's no leakage the way there would be if only some of a driver's rounds (or only some drivers) were excluded instead.")

st.header("Why is Spearman's Rho Used Over Traditional Accuracy?")
st.write("Traditional accuracy simply answers: *how often did the model predict the exact correct rank?* In a ranking task, that's misleading - predicting a driver 5th instead of 6th gets penalized the same as predicting them 5th instead of 15th. It doesn't capture how close the model was when it was wrong.")

st.write("Spearman's rho measures correlation between the predicted and true ordering instead, so a model whose ranks are close to correct scores well even when the exact ranks aren't perfect.")

st.header("Why is Spearman's Rho Used Over Pearson's Rho?")
st.write("Pearson's rho measures a *linear* relationship between two variables; Spearman's rho measures a *monotonic* one, whether or not it's linear. Here, Pearson's rho would measure how well predicted points correlate with actual points - but a model can have its predicted points be wildly off in scale while still getting the relative ordering of drivers correct. Since only the ordering matters for this project, Spearman's rho (computed on the ranks, not the raw scores) is the more appropriate metric.")

st.header("What is Kendall's Tau?")
st.write("Kendall's tau is another common ranking-accuracy metric, measuring pairwise agreement between the predicted and true ordering rather than overall rank correlation - it assigns -1 for complete disagreement and +1 for complete agreement. Because it looks at individual pairs rather than the full ordering at once, Kendall's tau tends to be more sensitive to small ranking mistakes than Spearman's rho, making it a somewhat harsher measure of accuracy.")
