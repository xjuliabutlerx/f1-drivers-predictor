from torch import nn

class F1DriversRankClassifier(nn.Module):
    """Matches pretrained_models/schumacher_model.pt (2020-2021 holdout, rho 0.9938) - the widened
    (256/128/64) v2 architecture with the original ReLU activation, saved as its own file since
    later v2 experiments (GELU, LeakyReLU) changed what the shared f1_drivers_rank_classifier.py
    defines. Load this specific class to reload that specific model. Named "Schumacher" (not
    "Clark") after the v1<->v2 name swap - see f1_drivers_rank_classifier_prost.py's docstring."""

    def __init__(self, input_dim, output_dim):
        super(F1DriversRankClassifier, self).__init__()
        self.layer = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),

            # Layer 2
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.BatchNorm1d(128),

            # Layer 3
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 4
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)
