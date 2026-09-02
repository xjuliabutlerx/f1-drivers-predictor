# F1 Drivers Predictor

## About the Project

This project applies machine learning to forecast the final World Drivers' Championship rank of Formula 1 drivers throughout a season. Using historical race, qualifying, and teammate-relative performance data, the model leverages an ordinal ranking approach to capture the ordered nature of the championship standings. Alongside a driver's own form, each row carries context about their team's current competitiveness, so the model can separate driver skill from car performance. The pipeline also handles mid-season lineup changes (substitutes, swaps, retirements) by building each driver's stats only from the rounds they actually raced.

This is a standalone sibling project to `f1-constructors-predictor`, which does the equivalent for the Constructors' Championship.

## The Plan

I will be using the `fastf1` Python library to gather historical data, the Pytorch (`torch`) library to build a custom ranking model, and Streamlit (`streamlit`) to create an interactive dashboard for users to experiment with different scenarios and view predictions in real time.

## Data Pipeline

`src/data/data_pipeline.py` downloads, preprocesses, and feature-engineers race data into training-ready CSVs:

```bash
python src/data/setup_paths.py                 # creates the local data/ directory structure
python src/data/data_pipeline.py --step all     # download -> preprocess -> features
```

Useful flags:
- `--years` — seasons to process (default `2018-2026`)
- `--incomplete-years` — in-progress seasons, saved separately for prediction rather than training (default `2026`)
- `--min-rounds` — minimum rounds a driver must have raced in a season to be included as a labeled training example (default `3`), so brief substitute stints don't get treated as full "seasons"
- `--step {download,preprocess,features}` — run a single stage
