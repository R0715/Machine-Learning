"""适配官方 PatchTST 骨干，将多通道日线窗口映射到单个收益率。"""

from types import SimpleNamespace

from torch import Tensor, nn

from vendor.patchtst.models.PatchTST import Model


class PatchTSTRegressor(nn.Module):
    def __init__(self, num_features: int, lookback: int) -> None:
        super().__init__()
        if num_features < 1 or lookback < 1:
            raise ValueError("特征数和回看窗口长度必须为正数。")

        patch_len = min(8, lookback)
        config = SimpleNamespace(
            enc_in=num_features,
            seq_len=lookback,
            pred_len=1,
            e_layers=3,
            n_heads=4,
            d_model=32,
            d_ff=64,
            dropout=0.1,
            fc_dropout=0.0,
            head_dropout=0.0,
            individual=False,
            patch_len=patch_len,
            stride=max(1, patch_len // 2),
            padding_patch="end",
            revin=False,
            affine=False,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )
        self.patchtst = Model(config)
        self.return_head = nn.Linear(num_features, 1)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """输入为 [批次, 回看长度, 特征]，窗口内位置由骨干处理。"""
        per_channel = self.patchtst(x).squeeze(1)
        return self.return_head(per_channel).squeeze(-1)


def build_patchtst(num_features: int, lookback: int) -> nn.Module:
    return PatchTSTRegressor(num_features, lookback)
