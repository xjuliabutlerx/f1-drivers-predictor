from .f1_dataset import F1DriversDataset
from .f1_drivers_rank_classifier import F1DriversRankClassifier

import os
import pandas as pd
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda", 0)
    else:
        return torch.device("cpu")

def predict(model_path: str, dataset_df: pd.DataFrame, feature_cols: list, device: torch.device):
    idx_final_rank = dataset_df.sort_values(["DriverId", "RoundsCompleted"]).groupby(["DriverId"])["RoundsCompleted"].idxmax()
    prediction_df = dataset_df.loc[idx_final_rank].copy()
    prediction_df = prediction_df.sort_values("DriverId")

    X_pred = torch.tensor(prediction_df[feature_cols].values, dtype=torch.float32)

    model = F1DriversRankClassifier(input_dim=X_pred.shape[1], output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scores = model(X_pred.to(device)).detach().cpu().numpy()
    prediction_df["PredictedFinalRank"] = scores
    prediction_df["PredictedFinalRank"] = prediction_df["PredictedFinalRank"].rank(method="first", ascending=False).astype(int)

    return prediction_df.sort_values("PredictedFinalRank")["DriverId"].to_list()

def run_all_models(prediction_data_path: str, models_dir: str):
    """Runs every model in models_dir (e.g. v1/pretrained_models) against the current, in-progress
    season's prediction CSV and returns one column of predicted driver order per model, named after
    the model's file (e.g. "prost_model.pt" -> "Prost Model")."""
    device = get_device()
    dataset = F1DriversDataset(prediction_data_path)
    feature_cols = dataset.get_feature_columns()

    results_df = pd.DataFrame()
    for model_file in sorted(os.listdir(models_dir)):
        if not model_file.endswith(".pt"):
            continue
        model_name = model_file[:-3].replace("_", " ").title()
        results_df[model_name] = predict(os.path.join(models_dir, model_file), dataset.df, feature_cols, device)

    return results_df
