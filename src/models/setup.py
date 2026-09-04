from rich import print

import os

V1 = "v1"
V2 = "v2"
V3 = "v3"

CHECKPOINTS_PATH = "checkpoints"          # Path to saved checkpoints
HEATMAPS_PATH = "heatmaps"                # Path to the heatmap visuals
MODELS_PATH = "pretrained_models"         # Path to the saved models
TRAINING_DATA_PATH = "training_data"      # Path to the training data excels
ANALYSIS_RESULTS_PATH = "shap_analysis"   # Path to a directory of SHAP analysis images

if __name__ == "__main__":
    print()
    root_dirs = [ANALYSIS_RESULTS_PATH]
    base_dirs = [V1, V2, V3]
    sub_dirs = [CHECKPOINTS_PATH, HEATMAPS_PATH, MODELS_PATH, TRAINING_DATA_PATH]

    for base in base_dirs:
        if not os.path.exists(base):
            print(f"[cyan]{base}[/cyan] does not exist, creating now...", end="")
            os.mkdir(base)
            print("[green]done[/green]!")
        else:
            print(f"[cyan]{base}[/cyan] already exists!")

        for sub in sub_dirs:
            new_dir_path = os.path.join(base, sub)
            if not os.path.exists(new_dir_path):
                print(f"[cyan]{new_dir_path}[/cyan] does not exist, creating now...", end="")
                os.mkdir(new_dir_path)
                print("[green]done[/green]!")
            else:
                print(f"[cyan]{new_dir_path}[/cyan] already exists!")

    for root in root_dirs:
        if not os.path.exists(root):
            print(f"[cyan]{root}[/cyan] does not exist, creating now...", end="")
            os.mkdir(root)
            print("[green]done[/green]!")
        else:
            print(f"[cyan]{root}[/cyan] already exists!")

        for base in base_dirs:
            new_dir_path = os.path.join(root, base)
            if not os.path.exists(new_dir_path):
                print(f"[cyan]{new_dir_path}[/cyan] does not exist, creating now...", end="")
                os.mkdir(new_dir_path)
                print("[green]done[/green]!")
            else:
                print(f"[cyan]{new_dir_path}[/cyan] already exists!")

    print()
