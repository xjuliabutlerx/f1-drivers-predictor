from pathlib import Path
from rich import print

import os

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = str(REPO_ROOT / "data")                                # Path to root data directory
CACHE_PATH = str(REPO_ROOT / "data" / "cache")                     # Path to the cache in the data directory
RAW_DATA_PATH = str(REPO_ROOT / "data" / "raw")                    # Path to the raw data
PREPROCESSED_DATA_PATH = str(REPO_ROOT / "data" / "preprocessed")  # Path to the preprocessed data
CLEAN_DATA_PATH = str(REPO_ROOT / "data" / "clean")                # Path to the clean training data

if __name__ == "__main__":
    print()
    directories = [DATA_PATH, CACHE_PATH, RAW_DATA_PATH, PREPROCESSED_DATA_PATH, CLEAN_DATA_PATH]

    for directory in directories:
        if not os.path.exists(directory):
            print(f"[cyan]{directory}[/cyan] does not exist, creating now...", end="")
            os.makedirs(directory)
            print("[green]done[/green]!")
        else:
            print(f"[cyan]{directory}[/cyan] already exists!")
    print()
