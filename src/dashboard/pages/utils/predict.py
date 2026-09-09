from .f1_dataset import F1DriversDataset

import importlib
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

def resolve_classifier_class(model_path: str):
    """Prefers a model-specific f1_drivers_rank_classifier_{name}.py (e.g. schumacher_model.pt ->
    f1_drivers_rank_classifier_schumacher.py) over the shared f1_drivers_rank_classifier.py - same
    fix as src/models/predict.py's resolve_classifier_class, needed here for the same reason: v2's
    3 keepers (Prost/Schumacher/Senna) each use a different activation function, so loading all of
    them through one shared class would silently produce wrong predictions for 2 of the 3 (since
    activation functions have no learnable parameters, load_state_dict() succeeds even when the
    architecture doesn't match). Falls back to the shared file for models that don't have a
    dedicated one (all of v1, which never diverged from a single shared architecture)."""
    model_name = os.path.basename(model_path)
    if model_name.endswith("_model.pt"):
        model_name = model_name[:-len("_model.pt")]
        try:
            module = importlib.import_module(f"pages.utils.f1_drivers_rank_classifier_{model_name}")
            return module.F1DriversRankClassifier
        except ModuleNotFoundError:
            pass

    module = importlib.import_module("pages.utils.f1_drivers_rank_classifier")
    return module.F1DriversRankClassifier

def predict(model_path: str, dataset_df: pd.DataFrame, feature_cols: list, device: torch.device):
    idx_final_rank = dataset_df.sort_values(["DriverId", "RoundsCompleted"]).groupby(["DriverId"])["RoundsCompleted"].idxmax()
    prediction_df = dataset_df.loc[idx_final_rank].copy()
    prediction_df = prediction_df.sort_values("DriverId")

    X_pred = torch.tensor(prediction_df[feature_cols].values, dtype=torch.float32)

    classifier_class = resolve_classifier_class(model_path)
    model = classifier_class(input_dim=X_pred.shape[1], output_dim=1).to(device)
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
