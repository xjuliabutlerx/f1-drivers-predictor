import streamlit as st

st.set_page_config(page_title="Model Data Features", page_icon="⚙️", layout="wide")

st.title("⚙️ Model Data Features")

st.write("Here is an overview of all 57 data features used by the v1 generation of models (47 numeric features plus 10 one-hot `TeamId` columns). They're divided into the same 3 sections as `f1-constructors-predictor`: raw features pulled directly from `fastf1` race data, calculated features (counts, sums, averages, rates) produced during preprocessing, and interaction features that combine multiple calculated statistics during feature engineering.")

st.write("Once a v2 generation exists, this page will pick up the same blue/gray/green version-highlighting scheme the constructors project uses to show what changed between generations. For now, everything below is simply what v1 uses.")

st.subheader("Raw / Identifying Features")
st.write("*Note: unlike the constructors project, `fastf1`'s race data is already at driver granularity, so no per-team consolidation was needed here. `DriverId` is deliberately never used as a model feature - one-hot encoding driver identity would let the model memorize specific drivers' outcomes instead of learning transferable performance signal, and would produce a useless all-zero vector for any driver not seen in training.*")

st.markdown("""
- **Year**: The F1 season year.
- **Round**: The calendar race round number in the season.
- **DriverId**: The `fastf1` shorthand id for the driver. *(identifier only, not a feature)*
- **TeamId**: The `fastf1` shorthand id for the driver's team. *(identifier only - see the one-hot encoding below)*
""")

st.subheader("Calculated Features")
st.write("*Row grain here is `(Year, DriverId, Round)`, built only from rounds a driver actually raced - a mid-season swap or DNS just means no row for that round, not imputed data. `RoundsCompleted` (rounds actually raced) and `Round` (calendar round number) are genuinely different information here, since a driver can miss rounds a team never would.*")

st.markdown("""
- **RoundsCompleted**: The number of rounds this driver has actually raced so far, 0-indexed.
- **RoundsRemaining**: The number of rounds left in the season after the current round.
- **CareerSeasonsRaced** / **CareerRoundsRaced**: Running totals of seasons/rounds raced across the driver's entire career, carried across years.
- **TeamSeasonsWithCurrentTeam** / **TeamRoundsWithCurrentTeam**: Running totals of tenure with the driver's current team.
- **PointsEarnedThisRound**: Points scored in the current round.
- **PointsLast3Rounds**: Rolling sum of points over the last 3 races. Captures recent, short-term form.
- **TotalPoints**: Cumulative points scored so far this season.
- **TotalPointFinishes**: Count of races finishing inside the points so far this season.
- **TotalPodiums**: Cumulative podium finishes so far this season.
- **AvgPointsPerRace**: Mean points per race to date.
- **AvgPosition**: Average finishing position to date.
- **AvgGridPosition**: Average starting grid position to date (falls back from qualifying to actual grid position when qualifying data isn't available).
- **PositionsGainedThisRound** / **AvgPositionsGained**: Positions gained (grid to finish) this round, and the average of that over the season.
- **DNFsThisRound**: Number of DNFs this round (0 or 1 for a driver).
- **DNFsLast3Rounds**: Rolling sum of DNFs over the last 3 races.
- **DNFRate**: Proportion of races so far with a DNF.
- **DriverFaultDNFsThisRound** / **DriverFaultDNFRate**: DNFs specifically attributed to driver error, this round and as a season rate.
- **MechanicalDNFsThisRound** / **MechanicalDNFRate**: DNFs specifically attributed to mechanical/car failure, this round and as a season rate.
- **HasQualifyingData**: Flags whether qualifying-derived features are real or falling back to grid position, for a given year/round.
- **QualifyingPosition** / **AvgQualifyingPosition**: Qualifying position this round, and the average to date.
- **GridPenaltyPositions**: Grid positions lost/gained due to a grid penalty (qualifying position vs. actual starting grid position).
- **BeatTeammateThisRound** / **BeatTeammateRate**: Whether this driver out-finished their teammate this round, and the rate over the season.
- **TeamAvgPointsPerRace** / **TeamDNFRate**: The same per-race average / DNF rate concepts, calculated at the team level (both drivers combined) rather than per-driver.
""")

st.subheader("Interaction Features")
st.write("*These combine multiple calculated statistics, mostly to give the model a driver-vs-car-strength signal independent of raw results. Team-level (`Team*`) versions are a separate per-team-per-round join, not a groupby of the driver rows - they exist specifically so the model can tell a driver's stats apart from their car's underlying strength.*")

st.markdown("""
- **TeammatePointsGap**: Signed points gap between this driver and their teammate this season (can be negative).
- **QualifyingGapToTeammate**: Signed qualifying position gap to the teammate.
- **FormRatio**: `PointsLast3Rounds / (AvgPointsPerRace * 3)` - values above 1 indicate improving recent form, below 1 a decline.
- **Consistency**: `1 / (1 + StdDevLast5Rounds / (AvgPointsLast5Rounds + 1e-6))` - a low rolling standard deviation relative to the mean means a high, stable consistency score.
- **ProjectedSeasonTotalPoints**: A forward-looking full-season point projection that doesn't use future data. The single most influential feature across all 3 v1 models per SHAP analysis.
- **RelativePointsShare**: This driver's share of all points scored by the full field so far this season.
- **PercentileRankAfterRound**: This driver's percentile standing in the points after the current round (used instead of raw `CurrentRankAfterRound`, since a raw rank's meaning changes with field size but a percentile doesn't).
- **DriverPointsShareOfTeam**: This driver's share of their own team's combined points - separates individual performance from car strength.
- **TeamFormRatio** / **TeamConsistency** / **TeamProjectedSeasonTotalPoints** / **TeamRelativePointsShare** / **TeamPercentileRankAfterRound**: Team-level equivalents of the features above, joined onto each driver's row by `(Year, TeamId, Round)`. `TeamProjectedSeasonTotalPoints` is the 2nd most influential feature per SHAP analysis, just behind its driver-level counterpart.
""")

st.subheader("Team Identity (One-Hot Encoded)")
st.write("`TeamId` is one-hot encoded against a fixed 10-team vocabulary (`TEAM_ID_VOCAB` in `f1_dataset.py`), so a prediction dataset's columns always line up with what the model trained on - even a team missing from a given slice of data still gets a defined all-zero column instead of shifting every other team's index. A team outside the vocabulary (e.g. a brand-new 2026 entrant) is encoded as all-zero with a runtime warning, rather than erroring out.")
st.write("Per SHAP analysis, the individual `TeamId_*` columns are consistently among the least influential features across all 3 v1 models - likely because `ProjectedSeasonTotalPoints`/`TeamProjectedSeasonTotalPoints` already capture most of what team identity would otherwise signal (car strength). This makes them a leading candidate to drop in a future v2.")
