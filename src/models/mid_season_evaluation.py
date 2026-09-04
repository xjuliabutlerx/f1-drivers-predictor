from __future__ import annotations

from rich import print
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, median_absolute_error, max_error

import argparse
import numpy as np
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

def score_as_of_fraction(model, dataset: "F1DriversDataset", test_years: list, feature_cols: list, device, as_of_fraction: float):
    """Scores `model` against each of test_years' fields using only rows up through
    round = round(as_of_fraction * that year's total round count) - i.e. "if we only knew this much
    of the season, how well would we have predicted the eventual FinalRank." as_of_fraction=1.0
    reproduces the standard end-of-season evaluation. A fraction is used (not a raw round number)
    since seasons have different total round counts (20-24), so the same round number doesn't mean
    the same "how far into the season" for every year.

    A driver who hasn't raced yet as of the cutoff round simply has no row in the filtered pool and
    is naturally excluded from that checkpoint - not an error, just means the season hadn't reached
    them yet (e.g. a mid-season substitute)."""
    full_df = dataset.df

    keep = pd.Series(False, index=full_df.index)
    for year in test_years:
        total_rounds = dataset.get_total_rounds_for_year(year)
        cutoff_round = max(1, round(total_rounds * as_of_fraction))
        keep |= (full_df["Year"] == year) & (full_df["Round"] <= cutoff_round)
    scoring_pool_df = full_df[keep]

    if scoring_pool_df.empty:
        return None

    idx_final_rank = scoring_pool_df.sort_values(["Year", "DriverId", "RoundsCompleted"]).groupby(["Year", "DriverId"])["RoundsCompleted"].idxmax()
    final_df = scoring_pool_df.loc[idx_final_rank].copy()

    with torch.no_grad():
        X = torch.tensor(final_df[feature_cols].values, dtype=torch.float32, device=device)
        final_df["PredictedFinalRank"] = model(X).detach().cpu().numpy()
    final_df["PredictedFinalRank"] = final_df.groupby("Year")["PredictedFinalRank"].rank(method="first", ascending=False).astype(int)

    all_true, all_pred, rhos = [], [], []
    for year, group in final_df.groupby("Year"):
        true_rank = group["FinalRank"].to_numpy()
        pred_rank = group["PredictedFinalRank"].to_numpy()
        all_true.extend(true_rank)
        all_pred.extend(pred_rank)

        if len(group) >= 2:
            rho, _ = spearmanr(true_rank, pred_rank)
            if not np.isnan(rho):
                rhos.append(rho)

    return {
        "As Of % of Season": round(as_of_fraction * 100),
        "Drivers Scored": len(all_true),
        "Avg Rho": float(np.mean(rhos)) if rhos else float("nan"),
        "Mean Abs Error": mean_absolute_error(all_true, all_pred) if all_true else float("nan"),
        "Median Abs Error": median_absolute_error(all_true, all_pred) if all_true else float("nan"),
        "Max Error": max_error(all_true, all_pred) if all_true else float("nan"),
    }

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Scores a trained model at several points during held-out season(s), "
                                                   "not just the final round - answers 'how good are predictions mid-season?'")
    PARSER.add_argument("--model_path", "-m", type=str, required=True)
    PARSER.add_argument("--test_years", "-y", nargs="+", type=int, required=True,
                         help="The season(s) this model was held out from during training (its true test set).")
    PARSER.add_argument("--fractions", "-f", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0],
                         help="Points through the season to score at, as fractions of total rounds (e.g. 0.5 = halfway).")
    PARSER.add_argument("--training_data_path", "-d", type=str, default=None)
    PARSER.add_argument("--version", "-v", type=int, required=False, default=1)

    ARGS = PARSER.parse_args()

    print()
    print("[yellow]*** F1 DRIVERS RANK CLASSIFIER MID-SEASON EVALUATION ***[/yellow]")
    print()

    model_path = ARGS.model_path
    test_years = ARGS.test_years
    fractions = sorted(ARGS.fractions)
    training_data_path = ARGS.training_data_path if ARGS.training_data_path is not None else os.path.join("../../data/clean/", "f1_drivers_clean_data.csv")
    version = ARGS.version

    if not os.path.exists(model_path):
        print(f"[red]ERROR[/red]: No model file found at {model_path}\n")
        exit(0)

    if version not in [1, 2, 3]:
        print(f"[red]ERROR[/red]: Invalid model version {version}.\n")
        exit(0)

    print("Parameters:")
    print(f" > Model Path: {model_path}")
    print(f" > Test Years: {test_years}")
    print(f" > Fractions of Season: {fractions}")
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

    print(f" > Loading dataset...", end="")
    dataset = F1DriversDataset(training_data_path)
    print(f"[green]done[/green]")

    print(f" > Retrieving feature column names...", end="")
    feature_cols = dataset.get_feature_columns()
    print(f"[green]done[/green]")

    print(f" > Loading model...", end="")
    model = F1DriversRankClassifier(len(feature_cols), 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[green]done[/green]")
    print()

    print("Scoring at each checkpoint:")
    results = []
    for fraction in fractions:
        result = score_as_of_fraction(model, dataset, test_years, feature_cols, device, fraction)
        if result is None:
            print(f" > [yellow]No rows available at {fraction*100:.0f}% of season - skipping[/yellow]")
            continue
        results.append(result)
        print(f" > As of {result['As Of % of Season']}% of season | "
              f"{result['Drivers Scored']} drivers scored | "
              f"Rho={result['Avg Rho']:.4f} | "
              f"MAE={result['Mean Abs Error']:.4f} | "
              f"Median={result['Median Abs Error']:.4f} | "
              f"Max={result['Max Error']:.4f}")

    print()
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
