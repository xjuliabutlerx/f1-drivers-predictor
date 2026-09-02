from rich import print

import os

CHECKPOINTS_PATH = "checkpoints"          # Path to saved checkpoints
HEATMAPS_PATH = "heatmaps"                # Path to the heatmap visuals
MODELS_PATH = "pretrained_models"         # Path to the saved models
TRAINING_DATA_PATH = "training_data"      # Path to the training data excels

if __name__ == "__main__":
    print()
    sub_dirs = [CHECKPOINTS_PATH, HEATMAPS_PATH, MODELS_PATH, TRAINING_DATA_PATH]

    for sub in sub_dirs:
        if not os.path.exists(sub):
            print(f"[cyan]{sub}[/cyan] does not exist, creating now...", end="")
            os.mkdir(sub)
            print("[green]done[/green]!")
        else:
            print(f"[cyan]{sub}[/cyan] already exists!")

    print()
