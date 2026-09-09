from torch import nn

class F1DriversRankClassifier(nn.Module):
    """Matches pretrained_models/prost_model.pt (2022-2023 holdout, rho 0.9968, max error 1.0) -
    the widened (256/128/64) v2 architecture with LeakyReLU activation (default
    negative_slope=0.01). Saved as its own file (identical to the shared
    f1_drivers_rank_classifier.py as of this point) so a future activation-function change to that
    shared file doesn't orphan this specific keeper too. Named "Prost" (not "Fangio", despite being
    the LeakyReLU run) after the v1<->v2 name swap - every v2 keeper beat every v1 model on rho, so
    the well-known GOAT names (Prost/Schumacher/Senna) were moved to v2, rank-for-rank, and v1's
    models took v2's original names (Fangio/Clark/Button) in exchange."""

    def __init__(self, input_dim, output_dim):
        super(F1DriversRankClassifier, self).__init__()
        self.layer = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),

            # Layer 2
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.35),
            nn.BatchNorm1d(128),

            # Layer 3
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            # Layer 4
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)
