from rich import print
from setup_paths import RAW_DATA_PATH, CACHE_PATH, DATA_PATH, PREPROCESSED_DATA_PATH, CLEAN_DATA_PATH

import argparse
import fastf1
import fastf1.logger
import numpy as np
import os
import pandas as pd
import re
import time

# -------------------- DOWNLOAD FUNCTIONS --------------------
def download_schedule(year: int, include_testing: bool = False):
    schedule = fastf1.get_event_schedule(year, include_testing=include_testing)
    return schedule

def download_session(year: int, gp, session_type: str = 'R', max_retries: int = 3, retry_delay: int = 5):
    for attempt in range(max_retries):
        try:
            session = fastf1.get_session(year, gp, session_type)
            session.load()
            return session
        except Exception as e:
            print(f"   - [yellow]WARNING[/yellow]: Error downloading session (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"   - [red]ERROR[/red]: Failed to download after {max_retries} attempts.")
                return None

def get_locations_from_schedule(schedule):
    return schedule['Location'].tolist()

def get_locations_with_sprint_from_schedule(schedule):
    sprint_locations = schedule[schedule['EventFormat'].str.contains('sprint', case=False, na=False)]
    return sprint_locations['Location'].tolist()

def get_rounds_from_schedule(schedule):
    return schedule['RoundNumber'].tolist()

def download_all_data(data_years):
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(CACHE_PATH, exist_ok=True)
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    fastf1.Cache.enable_cache(CACHE_PATH)
    fastf1.logger.set_log_level('ERROR')

    download_start_time = time.time()

    for year in data_years:
        print(f"Downloading data for the {year} season")
        schedule = download_schedule(year)

        print(schedule[['RoundNumber', 'EventName', 'Location', 'EventFormat']])
        print()

        schedule_rounds = get_rounds_from_schedule(schedule)
        schedule_locations = get_locations_from_schedule(schedule)
        sprint_locations = get_locations_with_sprint_from_schedule(schedule)

        for round in schedule_rounds:
            loc = schedule_locations[round - 1]
            is_sprint = loc in sprint_locations
            print(f" > Downloading data for Round {round} - {loc} | Is a Sprint Event? {is_sprint}")
            gp_session = download_session(year, round, 'R')

            if gp_session is None or gp_session.results is None or gp_session.results.empty:
                print(f"   - [yellow]WARNING[/yellow]: No race results data returned for [red]Round {round} - {loc}[/red], skipping...\n")
                continue

            data_file_name = f'{year}_Round_{round}_{loc}_results.csv'
            gp_session.results.to_csv(os.path.join(RAW_DATA_PATH, data_file_name), index=False)
            print(f"   - Saved results to [green]{RAW_DATA_PATH}/{data_file_name}[/green]\n")

            if is_sprint:
                s_session = download_session(year, round, 'S')

                if s_session is None or s_session.results is None or s_session.results.empty:
                    print(f"   - [yellow]WARNING[/yellow]: Although this was a sprint event, no sprint results data returned for [red]Round {round} - {loc}[/red], skipping...\n")
                    continue

                sprint_file_name = f'{year}_Round_{round}_{loc}_sprint_results.csv'
                s_session.results.to_csv(os.path.join(RAW_DATA_PATH, sprint_file_name), index=False)
                print(f"   - Saved sprint results to [green]{RAW_DATA_PATH}/{sprint_file_name}[/green]\n")

    print(f"Data download completed in {time.time() - download_start_time:.2f} seconds")

# -------------------- PREPROCESS FUNCTIONS --------------------
RAW_FILENAME_PATTERN = re.compile(r"^(?P<year>\d{4})_Round_(?P<round>\d+)_(?P<location>.+?)_(?P<kind>results|sprint_results)\.csv$")

def parse_raw_filename(filename: str):
    match = RAW_FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unrecognized raw data filename format: {filename}")
    return match.group("year"), int(match.group("round")), match.group("location"), match.group("kind")

def get_data_files_for_year(year: str, raw_data_files):
    return [file for file in raw_data_files if file.startswith(str(year))]

def combine_data_files_for_year(file_list: list):
    year_data_df = pd.DataFrame()
    for file in file_list:
        file_path = os.path.join(RAW_DATA_PATH, file)
        file_df = pd.read_csv(file_path)

        file_year, gp_round, gp_location, kind = parse_raw_filename(file)

        file_df.insert(loc=0, column='Year', value=file_year)
        file_df.insert(loc=1, column='Round', value=gp_round)
        file_df.insert(loc=2, column='Event', value='Sprint' if kind == 'sprint_results' else 'Race')
        file_df.insert(loc=3, column='Location', value=gp_location)
        year_data_df = pd.concat([year_data_df, file_df], ignore_index=True)

    return year_data_df.sort_values(by='Round')

def autofill_driver_data_given_id(all_data_df: pd.DataFrame, driver_ids: list, id_column: str):
    result_df = pd.DataFrame()
    for driver in driver_ids:
        all_results = all_data_df[all_data_df[id_column] == driver]
        all_results = all_results[['DriverId', 'FirstName', 'LastName', 'FullName', 'Abbreviation', 'BroadcastName']].drop_duplicates().dropna()
        result_df = pd.concat([result_df, all_results], ignore_index=True)
    return result_df.sort_values(by='DriverId').reset_index(drop=True)

def preprocess_all_data(years):
    print("Starting data preprocessing...")

    all_data_df = pd.DataFrame()
    raw_data_files = os.listdir(RAW_DATA_PATH)

    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)

    for year in years:
        print(f" > Processing data for the {year} season")
        files = get_data_files_for_year(year, raw_data_files)
        year_df = combine_data_files_for_year(files)
        all_data_df = pd.concat([all_data_df, year_df], ignore_index=True)
        preprocessed_file_name = f'{year}_season_results.csv'
        year_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, preprocessed_file_name), index=False)
        print(f"   - Saved preprocessed data to [magenta]{PREPROCESSED_DATA_PATH}/{preprocessed_file_name}[/magenta]\n")

    print("Gathering data on drivers...")

    unique_drivers = all_data_df['DriverId'].unique()
    drivers_df = autofill_driver_data_given_id(all_data_df, unique_drivers, 'DriverId')
    drivers_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'drivers.csv'), index=False)

    print(f"   - Saved drivers data to [magenta]{PREPROCESSED_DATA_PATH}/drivers.csv[/magenta]\n")
    print("Data preprocessing completed.")

# -------------------- FEATURE ENGINEERING FUNCTIONS --------------------
def normalize_teamids(df: pd.DataFrame):
    df["TeamId"] = df["TeamId"].replace("alfa", "sauber")
    df["TeamId"] = df["TeamId"].replace("renault", "alpine")
    df["TeamId"] = df["TeamId"].replace(["toro_rosso", "alphatauri"], ["rb", "rb"])
    df["TeamId"] = df["TeamId"].replace(["force_india", "racing_point"], ["aston_martin", "aston_martin"])
    df["TeamName"] = df["TeamName"].replace("Alfa Romeo Racing", "Alfa Romeo")
    df["TeamName"] = df["TeamName"].replace("Sauber", "Kick Sauber")
    return df

def build_round_summary_by_driver(all_seasons_data_df: pd.DataFrame):
    """One row per (Year, TeamId, DriverId, Round), combining Race + Sprint results for that round."""
    round_summary = (
        all_seasons_data_df
        .groupby(["Year", "TeamId", "DriverId", "Round"], as_index=False)
        .agg(
            Location=("Location", "first"),
            PointsEarnedThisRound=("Points", "sum"),
            DNFsThisRound=("isDNF", "sum"),
            GridPosition=("GridPosition", "mean"),
            Position=("Position", "mean"),
        )
    )
    return round_summary

def add_teammate_features(round_summary: pd.DataFrame):
    """Looks up whichever other driver raced for the same team/round to derive teammate-comparison features."""
    merged = round_summary.merge(round_summary, on=["Year", "TeamId", "Round"], suffixes=("", "_Teammate"))
    merged = merged[merged["DriverId"] != merged["DriverId_Teammate"]]
    # A team fielding more than one substitute in the same round is a rare edge case; keep a deterministic pick.
    merged = merged.sort_values(["Year", "TeamId", "Round", "DriverId", "DriverId_Teammate"])
    merged = merged.drop_duplicates(subset=["Year", "TeamId", "Round", "DriverId"], keep="first")

    teammate_df = merged[["Year", "TeamId", "Round", "DriverId", "DriverId_Teammate", "PointsEarnedThisRound_Teammate", "Position_Teammate"]]
    teammate_df = teammate_df.rename(columns={
        "DriverId_Teammate": "TeammateId",
        "PointsEarnedThisRound_Teammate": "TeammatePointsEarnedThisRound",
        "Position_Teammate": "TeammatePositionThisRound",
    })

    return round_summary.merge(teammate_df, on=["Year", "TeamId", "Round", "DriverId"], how="left")

def compute_beat_teammate(row):
    own_position = row["Position"]
    teammate_position = row["TeammatePositionThisRound"]

    if pd.isna(own_position) and pd.isna(teammate_position):
        return np.nan
    if pd.isna(teammate_position):
        return 1
    if pd.isna(own_position):
        return 0
    return 1 if own_position < teammate_position else 0

def calculate_team_context_features(all_seasons_data_df: pd.DataFrame):
    """
    Point-in-time team-level rolling stats (no FinalRank - that would leak future season outcome
    into a mid-season row), keyed by (Year, TeamId, Round), to be joined onto driver rows as
    team-strength context.
    """
    team_context_frames = []

    for year in sorted(all_seasons_data_df["Year"].unique()):
        year_df = all_seasons_data_df[all_seasons_data_df["Year"] == year]
        total_rounds_in_year = int(year_df["Round"].max())

        for team_id in year_df["TeamId"].dropna().unique():
            team_rows = []
            for round_num in range(1, total_rounds_in_year + 1):
                round_results = year_df.loc[(year_df["TeamId"] == team_id) & (year_df["Round"] == round_num)]
                if round_results.empty:
                    continue
                team_rows.append({
                    "Year": year,
                    "TeamId": team_id,
                    "Round": round_num,
                    "TeamPointsEarnedThisRound": round_results["Points"].sum(),
                    "TeamDNFsThisRound": round_results["isDNF"].sum(),
                })

            if not team_rows:
                continue

            team_df = pd.DataFrame(team_rows).sort_values(by="Round").reset_index(drop=True)
            team_df["TeamTotalPoints"] = team_df["TeamPointsEarnedThisRound"].cumsum()
            team_df["TeamAvgPointsPerRace"] = team_df["TeamPointsEarnedThisRound"].expanding().mean()
            team_df["TeamDNFRate"] = team_df["TeamDNFsThisRound"].apply(lambda x: 1 if x > 0 else 0).expanding().mean()

            team_points_last_3 = team_df["TeamPointsEarnedThisRound"].rolling(window=3, min_periods=1).sum()
            team_df["TeamFormRatio"] = team_points_last_3 / (team_df["TeamAvgPointsPerRace"] * 3 + 1e-6)

            rolling_mean_last_5 = team_df["TeamPointsEarnedThisRound"].rolling(window=5, min_periods=1).mean()
            rolling_std_last_5 = team_df["TeamPointsEarnedThisRound"].rolling(window=5, min_periods=1).std().fillna(0)
            team_df["TeamConsistency"] = 1 / (1 + (rolling_std_last_5 / (rolling_mean_last_5 + 1e-6)))

            team_rounds_remaining = total_rounds_in_year - team_df["Round"]
            team_df["TeamProjectedSeasonTotalPoints"] = team_df["TeamTotalPoints"] + (rolling_mean_last_5 * team_rounds_remaining)

            team_context_frames.append(team_df)

    if not team_context_frames:
        return pd.DataFrame(columns=["Year", "TeamId", "Round"])

    team_context_df = pd.concat(team_context_frames, ignore_index=True)

    team_context_df["TeamRelativePointsShare"] = team_context_df["TeamTotalPoints"] / team_context_df.groupby(["Year", "Round"])["TeamTotalPoints"].transform("sum")
    team_context_df["TeamCurrentRankAfterRound"] = team_context_df.groupby(["Year", "Round"])["TeamTotalPoints"].rank(method="dense", ascending=False).astype(int)
    team_context_df["TeamPercentileRankAfterRound"] = 1.0 - (team_context_df["TeamCurrentRankAfterRound"] - 1) / (team_context_df.groupby(["Year", "Round"])["TeamId"].transform("nunique") - 1)

    return team_context_df[["Year", "TeamId", "Round", "TeamTotalPoints", "TeamAvgPointsPerRace", "TeamFormRatio", "TeamConsistency", \
                            "TeamDNFRate", "TeamProjectedSeasonTotalPoints", "TeamRelativePointsShare", "TeamCurrentRankAfterRound", \
                            "TeamPercentileRankAfterRound"]]

def calculate_driver_standings_context(training_data_df: pd.DataFrame):
    """Ranks drivers against each other at each point-in-season, mirroring the constructors' project's team-rank calc."""
    training_data_df["RelativePointsShare"] = training_data_df["TotalPoints"] / training_data_df.groupby(["Year", "Round"])["TotalPoints"].transform("sum")
    training_data_df["CurrentRankAfterRound"] = training_data_df.groupby(["Year", "Round"])["TotalPoints"].rank(method="dense", ascending=False).astype(int)
    training_data_df["PercentileRankAfterRound"] = 1.0 - (training_data_df["CurrentRankAfterRound"] - 1) / (training_data_df.groupby(["Year", "Round"])["DriverId"].transform("nunique") - 1)

    # Cumulative share of the *team's* points this driver personally accounts for - a season-long
    # magnitude signal that complements the per-round TeammatePointsGap/BeatTeammateRate features.
    # 0.5 (neutral) before the team has scored any points, to avoid a 0/0 divide.
    training_data_df["DriverPointsShareOfTeam"] = np.where(
        training_data_df["TeamTotalPoints"] > 0,
        training_data_df["TotalPoints"] / training_data_df["TeamTotalPoints"],
        0.5,
    )
    return training_data_df

FINAL_COLUMNS = ["Year", "DriverId", "TeamId", "Location", "Round", "RoundsCompleted", "RoundsRemaining", \
                 "PointsEarnedThisRound", "DNFsThisRound", "TeammateId", "TeammatePointsGap", "BeatTeammateThisRound", \
                 "BeatTeammateRate", "PositionsGainedThisRound", "AvgPositionsGained", "PointsLast3Rounds", \
                 "DNFsLast3Rounds", "DNFRate", "AvgGridPosition", "AvgPosition", "AvgPointsPerRace", \
                 "TotalPointFinishes", "FormRatio", "Consistency", "TotalPodiums", "TotalPoints", \
                 "ProjectedSeasonTotalPoints", "RelativePointsShare", "CurrentRankAfterRound", "PercentileRankAfterRound", \
                 "DriverPointsShareOfTeam", "TeamTotalPoints", "TeamAvgPointsPerRace", "TeamFormRatio", "TeamConsistency", "TeamDNFRate", \
                 "TeamProjectedSeasonTotalPoints", "TeamRelativePointsShare", "TeamCurrentRankAfterRound", \
                 "TeamPercentileRankAfterRound", "FinalRank"]

def feature_engineer_all_data(years, incomplete_years=None, min_rounds=3):
    print("\nStarting data cleaning and feature engineering...")

    all_seasons_data_df = pd.DataFrame()
    preprocessed_files = os.listdir(PREPROCESSED_DATA_PATH)

    for file in preprocessed_files:
        if file.endswith("_season_results.csv"):
            current_year_df = pd.read_csv(os.path.join(PREPROCESSED_DATA_PATH, file))
            all_seasons_data_df = pd.concat([all_seasons_data_df, current_year_df])

    all_seasons_data_df = all_seasons_data_df.sort_values(by=["Year", "Round", "Points"], ascending=[True, True, False]).reset_index(drop=True)
    all_seasons_data_df = normalize_teamids(all_seasons_data_df)
    all_seasons_data_df["isDNF"] = all_seasons_data_df["ClassifiedPosition"].apply(lambda x: 1 if not str(x).isnumeric() else 0)
    all_seasons_data_df["isPointsFinish"] = all_seasons_data_df["Points"].apply(lambda x: 1 if x > 0 else 0)
    all_seasons_data_df["isPodiumFinish"] = all_seasons_data_df["Points"].apply(lambda x: 1 if x >= 15 else 0)

    final_standings_df = all_seasons_data_df.groupby(["Year", "DriverId"], as_index=False).agg(
        FullName=("FullName", "first"),
        Points=("Points", "sum"),
    )

    round_summary = build_round_summary_by_driver(all_seasons_data_df)
    round_summary = add_teammate_features(round_summary)
    team_context_df = calculate_team_context_features(all_seasons_data_df)

    training_data_df = pd.DataFrame()
    incomplete_training_data_df = pd.DataFrame()

    for year in years:
        print(f" > Processing data for the {year} season")
        print("   - Calculating final standings...")

        current_year_standings = final_standings_df[final_standings_df["Year"] == year]
        current_year_standings = current_year_standings.sort_values(by=["Points", "FullName"], ascending=[False, True]).reset_index(drop=True)
        current_year_standings["FinalRanking"] = [i + 1 for i in range(len(current_year_standings))]

        print(current_year_standings, end="\n\n")

        total_rounds_in_year = int(all_seasons_data_df.loc[all_seasons_data_df["Year"] == year, "Round"].max())
        driver_ids_this_year = round_summary.loc[round_summary["Year"] == year, "DriverId"].unique().tolist()

        is_incomplete_year = bool(incomplete_years and year in incomplete_years)
        if is_incomplete_year:
            schedule = download_schedule(year)
            total_rounds_in_year = len(get_rounds_from_schedule(schedule))

        for driver_id in driver_ids_this_year:
            print(f"   - Processing the {year} season data for [magenta]{driver_id}[/magenta]...")

            driver_rounds_df = round_summary.loc[
                (round_summary["Year"] == year) & (round_summary["DriverId"] == driver_id)
            ].sort_values(by="Round").reset_index(drop=True)

            driver_rounds_df["RoundsCompleted"] = range(len(driver_rounds_df))
            # Calendar-based (not "rounds completed"-based) so it stays correct even when a driver
            # has gaps in their round sequence from a mid-season swap or substitute appearance.
            driver_rounds_df["RoundsRemaining"] = total_rounds_in_year - driver_rounds_df["Round"]

            driver_rounds_df["AvgGridPosition"] = driver_rounds_df["GridPosition"].expanding().mean()
            driver_rounds_df["AvgPosition"] = driver_rounds_df["Position"].expanding().mean()
            driver_rounds_df["DNFRate"] = driver_rounds_df["DNFsThisRound"].apply(lambda x: 1 if x > 0 else 0).expanding().mean()
            driver_rounds_df["AvgPointsPerRace"] = driver_rounds_df["PointsEarnedThisRound"].expanding().mean()
            driver_rounds_df["TotalPointFinishes"] = (driver_rounds_df["PointsEarnedThisRound"] > 0).astype(int).cumsum()
            driver_rounds_df["TotalPodiums"] = (driver_rounds_df["PointsEarnedThisRound"] >= 15).astype(int).cumsum()
            driver_rounds_df["TotalPoints"] = driver_rounds_df["PointsEarnedThisRound"].cumsum()

            driver_rounds_df["PointsLast3Rounds"] = driver_rounds_df["PointsEarnedThisRound"].rolling(window=3, min_periods=1).sum()
            driver_rounds_df["DNFsLast3Rounds"] = driver_rounds_df["DNFsThisRound"].rolling(window=3, min_periods=1).sum()

            driver_rounds_df["FormRatio"] = driver_rounds_df["PointsLast3Rounds"] / (driver_rounds_df["AvgPointsPerRace"] * 3 + 1e-6)

            rolling_mean_last_5_rounds = driver_rounds_df["PointsEarnedThisRound"].rolling(window=5, min_periods=1).mean()
            rolling_std_last_5_rounds = driver_rounds_df["PointsEarnedThisRound"].rolling(window=5, min_periods=1).std().fillna(0)
            driver_rounds_df["Consistency"] = 1 / (1 + (rolling_std_last_5_rounds / (rolling_mean_last_5_rounds + 1e-6)))

            driver_rounds_df["ProjectedSeasonTotalPoints"] = driver_rounds_df["TotalPoints"] + (rolling_mean_last_5_rounds * driver_rounds_df["RoundsRemaining"])

            driver_rounds_df["TeammatePointsGap"] = driver_rounds_df["PointsEarnedThisRound"] - driver_rounds_df["TeammatePointsEarnedThisRound"]
            driver_rounds_df["BeatTeammateThisRound"] = driver_rounds_df.apply(compute_beat_teammate, axis=1)
            driver_rounds_df["BeatTeammateRate"] = driver_rounds_df["BeatTeammateThisRound"].expanding().mean()

            driver_rounds_df["PositionsGainedThisRound"] = driver_rounds_df["GridPosition"] - driver_rounds_df["Position"]
            driver_rounds_df["AvgPositionsGained"] = driver_rounds_df["PositionsGainedThisRound"].expanding().mean()

            driver_rounds_df["FinalRank"] = current_year_standings.loc[current_year_standings["DriverId"] == driver_id, "FinalRanking"].iloc[0]

            print(driver_rounds_df[["Location", "RoundsCompleted", "RoundsRemaining", "PointsEarnedThisRound", "TeammatePointsGap", \
                                    "DNFsThisRound", "PointsLast3Rounds", "DNFsLast3Rounds", "DNFRate", "AvgGridPosition", "AvgPosition", \
                                    "AvgPointsPerRace", "FormRatio", "Consistency", "TotalPointFinishes", "TotalPodiums", "TotalPoints", \
                                    "FinalRank"]], end="\n\n")

            if is_incomplete_year:
                incomplete_training_data_df = pd.concat([incomplete_training_data_df, driver_rounds_df], ignore_index=True)
            elif len(driver_rounds_df) >= min_rounds:
                training_data_df = pd.concat([training_data_df, driver_rounds_df], ignore_index=True)

    if not training_data_df.empty:
        training_data_df = training_data_df.merge(team_context_df, on=["Year", "TeamId", "Round"], how="left")
        training_data_df = calculate_driver_standings_context(training_data_df)
        training_data_df = training_data_df[FINAL_COLUMNS]

        os.makedirs(CLEAN_DATA_PATH, exist_ok=True)
        training_data_df.to_csv(os.path.join(CLEAN_DATA_PATH, "f1_drivers_clean_data.csv"), index=False)
        print(f"Saved full seasons to [magenta]f1_drivers_clean_data.csv[/magenta]")

    if not incomplete_training_data_df.empty:
        incomplete_training_data_df = incomplete_training_data_df.merge(team_context_df, on=["Year", "TeamId", "Round"], how="left")
        incomplete_training_data_df = calculate_driver_standings_context(incomplete_training_data_df)
        incomplete_training_data_df = incomplete_training_data_df[FINAL_COLUMNS]
        incomplete_training_data_df = incomplete_training_data_df.rename(columns={"FinalRank": "CurrentRank"})

        os.makedirs(CLEAN_DATA_PATH, exist_ok=True)
        incomplete_training_data_df.to_csv(os.path.join(CLEAN_DATA_PATH, "f1_drivers_clean_prediction_data.csv"), index=False)
        print(f"Saved incomplete seasons to [magenta]f1_drivers_clean_prediction_data.csv[/magenta]")

    os.makedirs(CLEAN_DATA_PATH, exist_ok=True)
    all_seasons_data_df.to_csv(os.path.join(CLEAN_DATA_PATH, "all_seasons_data.csv"), index=False)
    print(f"Saved all seasons data to [magenta]all_seasons_data.csv[/magenta]", end="\n\n")

    print("Data cleaning and feature engineering completed.", end="\n\n")

# -------------------- MAIN PIPELINE --------------------
def main():
    print()
    parser = argparse.ArgumentParser(description="F1 Drivers Data Pipeline: Download, preprocess, and feature engineer F1 data.")
    parser.add_argument('--step', choices=['all', 'download', 'preprocess', 'features'], default='all', help='Which step(s) to run')
    parser.add_argument('--years', nargs='+', type=int, default=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026], help='Years to process')
    parser.add_argument('--incomplete-years', nargs='*', type=int, default=[2026], help='Years with incomplete data (will be saved separately)')
    parser.add_argument('--min-rounds', type=int, default=3, help='Minimum rounds a driver must have raced in a season to be included as a labeled training example')

    args = parser.parse_args()

    missing_incomplete = [y for y in args.incomplete_years if y not in args.years]
    if missing_incomplete:
        print(f"[yellow]WARNING:[/yellow] The following incomplete years are not in --years and will be ignored: {missing_incomplete}")

    if args.step in ['all', 'download']:
        download_all_data(args.years)

    if args.step in ['all', 'preprocess']:
        preprocess_all_data(args.years)

    if args.step in ['all', 'features']:
        feature_engineer_all_data(args.years, incomplete_years=args.incomplete_years, min_rounds=args.min_rounds)

if __name__ == "__main__":
    main()
