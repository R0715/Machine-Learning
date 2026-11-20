"""Adapt the authors' multivariate TQNet to scalar return prediction."""

from types import SimpleNamespace

import torch
from torch import nn

from vendor.tqnet.TQNet import Model


class TQNetRegressor(nn.Module):
    def __init__(self, num_features: int, lookback: int, cycle: int = 20):
        super().__init__()
        if lookback < 4 or lookback % 4:
            raise ValueError("TQNet lookback must be a positive multiple of its four attention heads.")
        if cycle < 1:
            raise ValueError("TQNet cycle must be a positive number of trading days.")
        self.cycle = cycle
        self.backbone = Model(SimpleNamespace(
            seq_len=lookback,
            pred_len=1,
            enc_in=num_features,
            cycle=cycle,
            model_type="mlp",
            d_model=32,
            dropout=0.1,
            use_revin=False,
        ))
        self.head = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # The upstream loader supplies s_end, the exclusive window end. Our
        # positions instead identify the final observed trading day (s_end - 1).
        # The phase uses its ordinal only, with no prices from the next day.
        cycle_index = (positions.to(device=x.device, dtype=torch.long) + 1) % self.cycle
        channels = self.backbone(x, cycle_index)[:, 0, :]
        return self.head(channels).squeeze(-1)


def build_tqnet(num_features: int, lookback: int, cycle: int = 20) -> nn.Module:
    return TQNetRegressor(num_features, lookback, cycle)
