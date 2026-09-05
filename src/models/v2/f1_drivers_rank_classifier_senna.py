from torch import nn

class F1DriversRankClassifier(nn.Module):
    """Matches pretrained_models/senna_model.pt (2022-2023 holdout, rho 0.9917) - the widened
    (256/128/64) v2 architecture with GELU activation, saved as its own file since later v2
    experiments (LeakyReLU) changed what the shared f1_drivers_rank_classifier.py defines. Load
    this specific class to reload that specific model. Named "Senna" (not "Button") after the
    v1<->v2 name swap - see f1_drivers_rank_classifier_prost.py's docstring."""

    def __init__(self, input_dim, output_dim):
        super(F1DriversRankClassifier, self).__init__()
        self.layer = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),

            # Layer 2
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.BatchNorm1d(128),

            # Layer 3
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),

            # Layer 4
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)
