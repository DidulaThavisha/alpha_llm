"""Value head module for predicting win probability from hidden states."""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Predicts P(current partial code → passing solution).

    Takes the last token's hidden state and outputs a scalar in [0, 1].
    """

    def __init__(self, hidden_size: int, intermediate_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_state: [batch, hidden_size] — last token's hidden state

        Returns:
            [batch, 1] value prediction in [0, 1]
        """
        return self.net(hidden_state.float())
