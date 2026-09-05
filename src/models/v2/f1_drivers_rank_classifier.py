from torch import nn

class F1DriversRankClassifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(F1DriversRankClassifier, self).__init__()
        # v2 changes from v1, tested one at a time: (1) widened each hidden layer (128/64/32 ->
        # 256/128/64) - validated across all 3 season-holdout pairs, a real improvement. (2) ReLU ->
        # GELU - regressed rho on all 3 pairs, reverted. (3) ReLU -> LeakyReLU (default
        # negative_slope=0.01) - same motivation as GELU (avoiding "dying" neurons that stop
        # learning, more likely at this model's fairly high 0.005 learning rate) but via a small
        # non-zero slope for negative inputs instead of a smooth curve. Depth, dropout rates, and
        # BatchNorm placement are unchanged from v1 throughout, so any effect can still be
        # attributed to width/activation specifically.
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
