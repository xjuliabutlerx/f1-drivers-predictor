from rich import print

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
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

def resolve_classifier_class(version: int, model_path: str):
    """Prefers a model-specific f1_drivers_rank_classifier_{name}.py (e.g. schumacher_model.pt ->
    f1_drivers_rank_classifier_schumacher.py) over the shared f1_drivers_rank_classifier.py.
    The shared file's architecture/activation can change across experiments, and since activation
    functions (ReLU/GELU/LeakyReLU) have no learnable parameters, load_state_dict() succeeds
    silently even when it doesn't match what a given model was actually trained with - it just runs
    the loaded weights through the wrong nonlinearity, silently producing wrong SHAP results.
    Resolved once per model file (not once for the whole models_dir_path), since a single version
    directory can hold models trained under different classifier definitions. Falls back to the
    shared file for models that don't have a dedicated one (e.g. all of v1)."""
    model_name = os.path.basename(model_path)
    if model_name.endswith("_model.pt"):
        model_name = model_name[:-len("_model.pt")]
        try:
            module = importlib.import_module(f"v{version}.f1_drivers_rank_classifier_{model_name}")
            return module.F1DriversRankClassifier
        except ModuleNotFoundError:
            pass

    module = importlib.import_module(f"v{version}.f1_drivers_rank_classifier")
    return module.F1DriversRankClassifier

def callable_model(model, x: np.ndarray):
    X_values = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        scores = model(X_values).detach().cpu().numpy().flatten()
        # SHAP explains the model in terms of RANK, not raw score - the pairwise ranker's score
        # scale is arbitrary (only relative order matters, per MarginRankingLoss), so a feature's
        # effect on the unitless score isn't directly interpretable the way its effect on rank is.
        order = np.argsort(-scores)                # descending order
        ranks = np.empty_like(order, dtype=int)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

def run_shap_analysis(model_name: str, model_path: str, classifier_class, dataset_df: pd.DataFrame, feature_cols: list, device: torch.device, sample_size: int = 50, analysis_dir: str = "shap_analysis"):
    X_values = torch.tensor(dataset_df[feature_cols].values, dtype=torch.float32).to(device)
    X_numpy = dataset_df[feature_cols].values.astype(np.float32)

    model = classifier_class(X_values.shape[1], 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    shap_explainer = shap.Explainer(lambda x: callable_model(model, x), X_numpy[:sample_size])
    shap_values = shap_explainer(X_numpy)
    shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    plt.title(f"SHAP Summary for {model_name}")
    shap.summary_plot(shap_values, X_values, feature_names=feature_cols, show=False, max_display=len(feature_cols), rng=rng)
    plt.savefig(os.path.join(analysis_dir, f"shap_summary_{model_name}.png"), bbox_inches="tight")
    plt.close()

    plt.title(f"SHAP Bar Plot for {model_name}")
    shap.summary_plot(shap_values, X_values, feature_names=feature_cols, plot_type="bar", show=False, max_display=len(feature_cols), rng=rng)
    plt.savefig(os.path.join(analysis_dir, f"shap_bar_{model_name}.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--models_dir_path", "-m", type=str, required=True, default=None)
    PARSER.add_argument("--training_data_path", "-d", type=str, required=False, default=None)
    PARSER.add_argument("--sample_size", "-s", type=int, default=50)
    PARSER.add_argument("--analysis_path", "-a", type=str, default="shap_analysis", help="Directory to save SHAP analysis results")
    PARSER.add_argument("--version", "-v", type=int, required=False, default=1)

    ARGS = PARSER.parse_args()

    print()
    print("[yellow]*** F1 DRIVERS RANK CLASSIFIER MODEL SHAP ANALYSIS ***[/yellow]")
    print()

    models_dir_path = ARGS.models_dir_path
    training_data_path = ARGS.training_data_path if ARGS.training_data_path is not None else os.path.join("../../data/clean/", "f1_drivers_clean_data.csv")
    sample_size = ARGS.sample_size
    analysis_path = ARGS.analysis_path
    version = ARGS.version

    if models_dir_path is None or not os.path.isdir(models_dir_path):
        print(f"[red]ERROR[/red]: You must provide a valid directory path for the pretrained models.\n")
        exit(0)
    elif len(os.listdir(models_dir_path)) == 0:
        print(f"[red]ERROR[/red]: No files found in the directory {models_dir_path}\n")
        exit(0)

    model_files_list = os.listdir(models_dir_path)

    for file in model_files_list:
        if not file.endswith(".pt"):
            print(f"[red]ERROR[/red]: The file {file} is not a valid pretrained model file.\n")
            exit(0)

    if training_data_path is None or not os.path.exists(training_data_path):
        print(f"[red]ERROR[/red]: You must provide a valid path for the training data.\n")
        exit(0)

    if version not in [1, 2]:
        print(f"[red]ERROR[/red]: Invalid model version {version}.\n")
        exit(0)

    print(f"Parameters:")
    print(f" > Models Directory Path: {models_dir_path}")
    for model_file in model_files_list:
        print(f"   - {model_file}")
    print(f" > Training Data Path: {training_data_path}")
    print(f" > Sample Size (background distribution): {sample_size}")
    print(f" > Evaluation Device: ", end="")
    device = get_device()
    print()

    print("Data:")
    if version == 1:
        print(f" > Loading v1 F1 Drivers Dataset...", end="")
        from v1.f1_dataset import F1DriversDataset
        print("[green]done[/green]")
    elif version == 2:
        print(f" > Loading v2 F1 Drivers Dataset...", end="")
        from v2.f1_dataset import F1DriversDataset
        print("[green]done[/green]")

    print(f" > Loading dataset...", end="")
    dataset = F1DriversDataset(training_data_path)
    print(f"[green]done[/green]")

    print(f" > Retrieving feature column names...", end="")
    feature_cols = dataset.get_feature_columns()
    print(f"[green]done[/green]")
    print()

    if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)

    print("Analysis:")
    for model_file in model_files_list:
        model_name = model_file[:-3].replace("_", " ").title()
        classifier_class = resolve_classifier_class(version, model_file)

        print(f" > Analyzing [magenta]{model_name}[/magenta]...", end="")
        run_shap_analysis(model_name, os.path.join(models_dir_path, model_file), classifier_class, dataset.df, feature_cols, device,
                           sample_size=sample_size, analysis_dir=analysis_path)
        print(f"[green]done[/green]")
