from rich import print
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

# Fixed, sorted so column order is stable across any dataset built from this vocabulary (training
# data today, prediction data tomorrow) - never re-derived per-file, so a team missing from a given
# slice (e.g. it didn't race that year) still gets a defined all-zero column instead of shifting
# every other team's column index.
TEAM_ID_VOCAB = ["alpine", "aston_martin", "ferrari", "haas", "mclaren", "mercedes", "rb", "red_bull", "sauber", "williams"]

class F1DriversDataset(Dataset):

    # Cumulative counts/points - non-negative by construction, so log1p is safe. Unlike
    # constructors' equivalent list, gap-style columns (TeammatePointsGap, GridPenaltyPositions,
    # etc.) are deliberately NOT here - they can be negative, and log1p of a negative value is
    # invalid.
    skewed_feature_columns = ["PointsEarnedThisRound", "DNFsThisRound", "DriverFaultDNFsThisRound", "MechanicalDNFsThisRound", \
                               "PointsLast3Rounds", "DNFsLast3Rounds", "TotalPointFinishes", "TotalPodiums", "TotalPoints", \
                               "CareerRoundsRaced", "TeamRoundsWithCurrentTeam", "TeamTotalPoints"]

    # Rates/ratios/percentiles, rolling averages, signed gaps, and small bounded counts - already
    # reasonably scaled, left untransformed (matches how constructors never transforms its
    # equivalent average/ratio columns). CurrentRankAfterRound/TeamCurrentRankAfterRound are
    # deliberately excluded - raw rank position depends on how many drivers/teams are in the field
    # that year, while PercentileRankAfterRound doesn't, so only the percentile version is a feature.
    numeric_feature_columns = ["Year", "Round", "RoundsCompleted", "RoundsRemaining", "CareerSeasonsRaced", "TeamSeasonsWithCurrentTeam", \
                                "DriverFaultDNFRate", "MechanicalDNFRate", "TeammatePointsGap", "BeatTeammateThisRound", "BeatTeammateRate", \
                                "PositionsGainedThisRound", "AvgPositionsGained", "HasQualifyingData", "QualifyingPosition", "AvgQualifyingPosition", \
                                "QualifyingGapToTeammate", "GridPenaltyPositions", "DNFRate", "AvgGridPosition", "AvgPosition", "AvgPointsPerRace", \
                                "FormRatio", "Consistency", "ProjectedSeasonTotalPoints", "RelativePointsShare", "PercentileRankAfterRound", \
                                "DriverPointsShareOfTeam", "TeamAvgPointsPerRace", "TeamFormRatio", "TeamConsistency", "TeamDNFRate", \
                                "TeamProjectedSeasonTotalPoints", "TeamRelativePointsShare", "TeamPercentileRankAfterRound"] + skewed_feature_columns

    team_id_columns = [f"TeamId_{team_id}" for team_id in TEAM_ID_VOCAB]

    def __init__(self, data_file_path):
        self.feature_columns = self.numeric_feature_columns + self.team_id_columns

        self.df = pd.read_csv(data_file_path)

        # A handful of rows (e.g. a driver with no teammate that round) legitimately have NaN in a
        # feature column - excluded here rather than imputed, since it's a small, known set of
        # edge cases rather than a systemic gap.
        before = len(self.df)
        self.df = self.df.dropna(subset=["TeamId"] + self.numeric_feature_columns).reset_index(drop=True)
        dropped = before - len(self.df)
        if dropped:
            print(f"[yellow]NOTE[/yellow]: Dropped {dropped} row(s) with missing feature values.")

        for col in self.skewed_feature_columns:
            self.df[col] = np.log1p(self.df[col])

        self._encode_team_id()

    def _encode_team_id(self):
        for team_id in TEAM_ID_VOCAB:
            self.df[f"TeamId_{team_id}"] = (self.df["TeamId"] == team_id).astype(int)

        unknown_teams = set(self.df["TeamId"].unique()) - set(TEAM_ID_VOCAB)
        if unknown_teams:
            print(f"[yellow]WARNING[/yellow]: TeamId(s) not in the training vocabulary will be encoded as all-zero: {unknown_teams}")

    def get_years(self):
        return self.df["Year"].unique().tolist()

    def get_total_rounds_for_year(self, year: int):
        if year not in self.get_years():
            return 0
        return int(self.df.loc[self.df["Year"] == year, "Round"].max())

    def get_random_split(self, test_size=0.2, random_state=24):
        # Randomly split the data into training and testing datasets using the pandas random sample method
        test_df = self.df.sample(frac=test_size, random_state=random_state)
        train_df = self.df.drop(test_df.index).sample(frac=1.0)                 # frac specifies the fraction of rows to return
                                                                                # frac = 1 means return all rows in a random order

        # Split the features from the metadata and target variable columns for the training dataset to maintain "row alignment"
        X_train = train_df[self.feature_columns]
        y_train = train_df["FinalRank"]

        return X_train, y_train, test_df

    def get_feature_columns(self):
        return self.feature_columns

if __name__ == "__main__":
    import os
    dataset = F1DriversDataset(os.path.join("../../data/clean/", "f1_drivers_clean_data.csv"))
    print(f"Loaded {len(dataset.df)} rows with {len(dataset.feature_columns)} feature columns.")
