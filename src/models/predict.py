from rich import print

import argparse
import os
import pandas as pd
import torch

def get_device():
    if torch.backends.mps.is_available():
        print(f"GPU detected - Apple [magenta]Metal Performance Shaders[/magenta]")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"GPU detected - NVIDIA [magenta]Compute Unified Device Architecture[/magenta]")
        return torch.device("cuda", 0)
    else:
        print("No GPU detected - defaulting to CPU")
        return torch.device("cpu")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--model_state_path", "-m", type=str, required=True, default=None)
    PARSER.add_argument("--prediction_data_path", "-d", type=str, required=True, default=None)
    PARSER.add_argument("--version", "-v", type=int, required=False, default=1)

    ARGS = PARSER.parse_args()

    print()
    print("[yellow]*** F1 DRIVERS RANK CLASSIFIER MODEL PREDICTOR ***[/yellow]")
    print()

    model_path = ARGS.model_state_path
    pred_data_path = ARGS.prediction_data_path if ARGS.prediction_data_path is not None else os.path.join("../../data/clean/", "f1_drivers_clean_prediction_data.csv")
    version = ARGS.version

    if model_path is None or not os.path.exists(model_path):
        print(f"[red]ERROR[/red]: You must provide a valid path for the saved model state.\n")
        exit(0)

    if pred_data_path is None or not os.path.exists(pred_data_path):
        print(f"[red]ERROR[/red]: You must provide a valid path for the prediction data.\n")
        exit(0)

    if version not in [1, 2, 3]:
        print(f"[red]ERROR[/red]: Invalid model version {version}.\n")
        exit(0)

    print(f"Parameters:")
    print(f" > Model State Path: {model_path}")
    print(f" > Prediction Data Path: {pred_data_path}")
    print(f" > Evaluation Device: ", end="")
    device = get_device()
    print()

    print("Data:")
    if version == 1:
        print(f" > Loading v1 F1 Drivers Dataset...", end="")
        from v1.f1_dataset import F1DriversDataset
        print("[green]done[/green]")

        print(f" > Loading v1 F1 Drivers Rank Classifier model...", end="")
        from v1.f1_drivers_rank_classifier import F1DriversRankClassifier
        print("[green]done[/green]")
    elif version == 2:
        print(f" > Loading v2 F1 Drivers Dataset...", end="")
        from v2.f1_dataset import F1DriversDataset
        print("[green]done[/green]")

        print(f" > Loading v2 F1 Drivers Rank Classifier model...", end="")
        from v2.f1_drivers_rank_classifier import F1DriversRankClassifier
        print("[green]done[/green]")
    elif version == 3:
        print(f" > Loading v3 F1 Drivers Dataset...", end="")
        from v3.f1_dataset import F1DriversDataset
        print("[green]done[/green]")

        print(f" > Loading v3 F1 Drivers Rank Classifier model...", end="")
        from v3.f1_drivers_rank_classifier import F1DriversRankClassifier
        print("[green]done[/green]")

    print(f" > Loading prediction dataset...", end="")
    dataset = F1DriversDataset(pred_data_path)
    print(f"[green]done[/green]")

    print(f" > Retrieving feature column names...", end="")
    feature_cols = dataset.get_feature_columns()
    print(f"[green]done[/green]")

    year = dataset.df["Year"].unique().tolist()
    if len(year) > 1:
        print(f"[red]ERROR[/red]: There is data from more than 1 F1 seasons; found {len(year)}. This data is not suitable for prediction.\n")
        exit(0)
    year = year[0]
    print(f"  - Running F1 driver's championship prediction for the year {year}")

    drivers = dataset.df["DriverId"].unique().tolist()
    print(f"  - Found {len(drivers)} number of drivers: {', '.join(drivers)}")
    print()

    idx_final_rank = dataset.df.sort_values(["DriverId", "RoundsCompleted"]).groupby(["DriverId"])["RoundsCompleted"].idxmax()
    prediction_df = dataset.df.loc[idx_final_rank].copy()
    prediction_df.sort_values(["DriverId"])

    X_pred = torch.tensor(prediction_df[feature_cols].values, dtype=torch.float32)

    print("Model:")
    print(f" > Instantiating model and loading state...", end="")
    model = F1DriversRankClassifier(input_dim=X_pred.shape[1], output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[green]done[/green]")

    print(f" > Running prediction...", end="")
    scores = model(X_pred.to(device)).detach().cpu().numpy()
    prediction_df["PredictedFinalRank"] = scores
    prediction_df["PredictedFinalRank"] = prediction_df["PredictedFinalRank"].rank(method="first", ascending=False).astype(int)
    print("[green]done[/green]")
    print()

    print("Results:")
    print(f"Current Standings for the {year} F1 Driver's Championship:")
    current_ranks = prediction_df.sort_values("CurrentRank", ascending=True)["DriverId"].to_list()
    for i, driver in enumerate(current_ranks, start=1):
        print(f"{i}. {driver}")
    print()

    print(f"Predicted Results for the {year} F1 Driver's Championship:")
    ranking_results = prediction_df.sort_values("PredictedFinalRank", ascending=True)["DriverId"].to_list()
    for i, driver in enumerate(ranking_results, start=1):
        print(f"{i}. {driver}")
    print()

    # Row grain is (Year, DriverId, Round), built only from rounds a driver actually raced - a
    # mid-season swap means not every driver has a row at the literal latest calendar round (e.g.
    # a driver who just joined a team). So "current points" is read from each driver's OWN latest
    # row (the same idx_final_rank selection as prediction_df above, just on the untransformed CSV
    # to avoid TotalPoints' log1p transform), rather than filtering to a single global last round -
    # that filter would silently drop any driver missing that exact round and misalign this table.
    orig_pred_df = pd.read_csv(pred_data_path)
    idx_orig_final_rank = orig_pred_df.sort_values(["DriverId", "RoundsCompleted"]).groupby(["DriverId"])["RoundsCompleted"].idxmax()
    current_points_by_driver = orig_pred_df.loc[idx_orig_final_rank].set_index("DriverId")["TotalPoints"]
    current_points = [current_points_by_driver[driver] for driver in current_ranks]

    results_data = {
        "Current Ranks": current_ranks,
        "Current Points": current_points,
        "Predicted Ranks": ranking_results
    }

    results_df = pd.DataFrame(results_data)
    print(results_df, end="\n\n")
