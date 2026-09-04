# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

F1 Drivers Predictor forecasts final World Drivers' Championship rank from in-season Formula 1 data. It's a standalone sibling to `f1-constructors-predictor` (which does the equivalent for the Constructors' Championship) — no shared imports between the two repos. The architecture deliberately mirrors that sibling project where the constructors' domain and the drivers' domain agree, and deliberately diverges where they don't (see "Divergences from f1-constructors-predictor" below); when in doubt about *why* something is built a certain way here, that project is the reference point, not a template to copy blindly.

Two independent stages: a data pipeline (`src/data/`) that turns raw fastf1 session data into clean per-round CSVs, and model training (`src/models/`) that reads those CSVs and trains a PyTorch ranking model. Nothing in `src/models/` depends on `src/data/` at runtime — they only share the `data/clean/*.csv` files on disk.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Commands

All scripts assume they're run from their own directory (relative imports and relative `../../data/...` paths), not the repo root.

**Data pipeline** (`src/data/`):
```bash
python setup_paths.py                 # creates the local data/ directory structure (idempotent)
python data_pipeline.py --step all    # download -> preprocess -> features
```
- `--years` (default `2018 2019 ... 2026`), `--incomplete-years` (default `2026`) — incomplete years are saved to a separate prediction CSV instead of the training CSV, with `FinalRank` renamed to `CurrentRank`.
- `--min-rounds` (default `3`) — a driver-season needs this many rounds raced to be emitted as a labeled training row; still counts toward teammate/standings/career-experience context either way.
- `--step {download,preprocess,features}` — run a single stage. `download` and `preprocess` hit the network/disk once and can be safely re-run — `download` skips any round/session file that already exists on disk, making zero API calls for it.
- Output: `data/clean/f1_drivers_clean_data.csv` (complete seasons, labeled) and `data/clean/f1_drivers_clean_prediction_data.csv` (in-progress season(s)).

**Model training** (`src/models/`):
```bash
python setup.py       # bootstraps v1/v2/v3 {checkpoints,heatmaps,pretrained_models,training_data} + shap_analysis/v{1,2,3} (idempotent)
cd v1 && python train.py -e 300   # -e/--num_epochs; see train.py's argparse block for the rest (-lr, -a alpha, -w round_window, -p patience, etc.)
```
Model code is versioned into `v1`/`v2`/`v3` directories, mirroring the sibling project: each holds its own `f1_dataset.py`, `f1_drivers_rank_classifier.py`, `train.py`, `checkpoints/`, `heatmaps/`, `pretrained_models/`, `training_data/`, and is run from inside that directory (`train.py`'s relative `../../../data/...` path assumes this). `v1` is the validated baseline (season-holdout evaluation, tuned hyperparameters, SHAP-reviewed feature set); a new version gets forked only once there's an actual divergent iteration worth preserving (e.g. dropping SHAP-flagged low-value features), not preemptively.

`predict.py`, `mid_season_evaluation.py`, and `model_shap_analysis.py` stay at the top level of `src/models/` (not versioned) and take a `--version`/`-v` flag (default `1`) to import the right `vN.f1_dataset`/`vN.f1_drivers_rank_classifier` at runtime — same pattern as the sibling project. `model_shap_analysis.py`'s `--analysis_path` isn't auto-versioned, so pass it explicitly (e.g. `-a shap_analysis/v1`) to match. Each of the three duplicates a small local `get_device()` rather than importing it from `train.py`, since `train.py` now lives inside a version-specific directory.

v1's 3 models are named after F1 drivers rather than generic identifiers (see the dashboard's Model Naming Scheme page): `prost_model.pt` (held out on 2020-2021, rho 0.9908 — sharpest/lowest-error), `schumacher_model.pt` (2022-2023, rho 0.9881 — steadiest, no big swings), `senna_model.pt` (2024-2025, rho 0.9529 — most volatile but still solid). Heatmap and SHAP output filenames are kept in sync with these names.

**Dashboard** (`src/dashboard/`), a Streamlit app — unlike `src/data/`/`src/models/`, it's run from the **repo root**, matching the sibling project:
```bash
streamlit run src/dashboard/Main.py
```
It's a skeleton mirroring the sibling project's page structure (`Main.py` + `pages/{1_Model_Development_&_Training,2_Model_Data_Features,3_Model_Naming_Scheme,4_Current_Model_Predictions}.py`), with its own self-contained copies of `f1_dataset.py`/`f1_drivers_rank_classifier.py` under `pages/utils/` (not imported from `src/models/`, so the dashboard doesn't depend on which model version is currently checked out). `4_Current_Model_Predictions.py` runs all 3 v1 models live against `data/clean/f1_drivers_clean_prediction_data.csv` on every page load via `pages/utils/predict.py`, rather than reading a pre-baked predictions CSV like the sibling project does. There's no `5_Scenario_Simulator.py` yet — the sibling's version is tightly coupled to its own constructor-level feature synthesis, and porting it needs its own design pass for driver-level features (teammate gaps, qualifying data, career/team tenure running state) rather than a direct copy.

There is no test suite, linter, or build step configured in this repo yet.

## Data pipeline architecture

`data_pipeline.py` has three stages (`download_all_data`, `preprocess_all_data`, `feature_engineer_all_data`), each runnable independently via `--step`.

- **Row grain is `(Year, DriverId, Round)`, built only from rounds a driver actually raced** — not every calendar round. This is the central design fact the rest of the pipeline depends on: `RoundsCompleted` (rounds this driver has actually raced, 0-indexed) and `Round` (calendar round number) are *not* interchangeable here, unlike in the constructors pipeline where they always were. A mid-season swap or a DNS just means that driver has no row for that round — no imputed/phantom data. Career-experience and team-tenure features (`CareerSeasonsRaced`/`CareerRoundsRaced`, `TeamSeasonsWithCurrentTeam`/`TeamRoundsWithCurrentTeam`) are running totals carried across years via state dicts in `feature_engineer_all_data`, which is why `years` gets `sorted()` there regardless of `--years` CLI order — processing out of chronological order would corrupt those running totals.
- **Team-level features are a separate join, not a groupby key.** Race/qualifying results are aggregated per-driver-per-round (`build_round_summary_by_driver`), and *separately* per-team-per-round (`calculate_team_context_features`, prefixed `Team*` in the output), then joined onto driver rows by `(Year, TeamId, Round)`. This exists to give the model a car-strength signal independent of individual driver form — a driver's raw stats mean something different on a front-running car vs. a backmarker, and `TeamId` itself is also a model feature (unlike the constructors project, where every row already *is* a team).
- **`--min-rounds` filters emitted rows, not the underlying aggregates.** A one-race substitute still contributes to their team's context, their teammate's `TeammatePointsGap`/`BeatTeammateRate`, and next season's career-experience running total — they just don't get emitted as their own labeled training example.
- **Qualifying is downloaded and preprocessed separately from Race/Sprint** (`*_qualifying_results.csv`, kept out of `combine_data_files_for_year`'s Race/Sprint aggregation) because mixing them would corrupt the Points/DNF sums those race-result features rely on. Qualifying-derived features (`QualifyingPosition`, `GridPenaltyPositions`, etc.) fall back to `GridPosition` when qualifying data isn't available for a year/round, with `HasQualifyingData` marking which is which.
- **`normalize_locations()` and `normalize_teamids()` handle real-world naming drift** in fastf1's schedule data across 2018-2026 (e.g. Monaco was reported as `"Monte Carlo"` through 2021; several team IDs changed due to rebrands). Both functions document, case by case, which pairs are genuine renamings (merge them) vs. genuine relocations/distinct entities (e.g. the Spanish GP's Barcelona→Madrid move, the Bahrain GP being run at Kuala Lumpur for 2026 while keeping its branding) that must stay distinct rather than being normalized together.
- **fastf1 self-imposes a hard, sliding-window rate limit (500 calls/hour across all its APIs)** — see the comment above `download_session`. A rate-limit hit gets a long cooldown (20 min, several retries) rather than the short generic-error retry, and `download_all_data` skips any round/session whose output file already exists so re-running a download never re-spends quota on data already on disk. Don't try to work around the limit by restarting the process to reset fastf1's in-memory counter — that defeats the point of the limiter and risks the whole project getting blocked upstream.

## Model architecture

- **It's a pairwise ranker, not a rank-value regressor/classifier.** `F1DriversRankClassifier` (a small MLP) outputs one scalar "score" per row; `train.py` trains it with `nn.MarginRankingLoss` on sampled pairs of *different drivers, same year and round*, and turns scores into an actual 1-to-n rank only at evaluation/prediction time via `.rank()` within each season's group. This is why the model works regardless of how many drivers are in a given season's field — nothing about entity count is hardcoded in the ranking mechanism itself.
- **Pair weighting is points-gap-based, not rank-band-based.** Pairs are weighted by `1.0 + alpha / (1.0 + points_gap_per_round)` — closer point totals get more training emphasis, wherever in the standings they occur. (The sibling constructors project instead assumes tight competition lives at a fixed rank band, e.g. ranks 4-7 of ~10 teams; that assumption doesn't hold for a ~20-24 driver field, where title fights can be just as tight at the front as in the midfield.)
- **`TeamId` is one-hot encoded against a fixed vocabulary** (`TEAM_ID_VOCAB` in `f1_dataset.py`) so a prediction dataset's one-hot columns always line up with what the model was trained on, even if a given slice of data is missing a team. `DriverId` is deliberately *never* a feature — one-hot encoding driver identity would let the model memorize specific drivers' outcomes instead of learning transferable performance signal, and would produce a useless all-zero vector for any driver not seen in training.
- **`F1DriversDataset.skewed_feature_columns` gets `log1p`-transformed; everything else doesn't.** Only non-negative cumulative counts/points are in that list — signed columns (e.g. `TeammatePointsGap`, `GridPenaltyPositions`) are deliberately excluded, since `log1p` of a negative number is invalid. When code needs a real (non-log) points value from a tensor — e.g. `train.py`'s points-gap-per-round weight — it recovers it with `expm1`, the exact inverse.
- Evaluation reduces the test split to each driver's last row per `(Year, DriverId)` (via `idxmax` on `RoundsCompleted`) and scores that against Spearman's rho / Kendall's tau, averaged across years — same pattern as the constructors project, just keyed by driver instead of team.

## Divergences from f1-constructors-predictor

Kept the same where the domains agree (pairwise `MarginRankingLoss`, same-round-only pair comparisons, row-level random train/test split, `Year` as a raw feature, MLP shape, `log1p` for skewed counts, `PercentileRankAfterRound` over raw `CurrentRankAfterRound` as a feature). Changed where drivers' domain genuinely differs: no rank-band pair weighting (see above), `TeamId` one-hot encoding (new), no pair-sampling cap (constructors' `min(200, ...)` cap was never actually binding for a 10-team field; it would under-sample a 20-24 driver field), and `RoundsCompleted` is kept as a feature here (it's collinear with `Round` for constructors, since teams never miss a round, but genuinely different information for drivers given mid-season gaps).
