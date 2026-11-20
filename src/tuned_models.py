"""可调正则化的官方模型骨干，输出单个连续收益率。"""

from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vendor.patchtst.models.PatchTST import Model as OfficialPatchTST
from vendor.tqnet.TQNet import Model as OfficialTQNet


def _dropout(options, name, default):
    value = float(options.get(name, default))
    if not 0 <= value < 1:
        raise ValueError(f"{name} 必须在 [0, 1) 内。")
    return value


class TunedTQNet(nn.Module):
    def __init__(self, num_features, lookback, options, cycle):
        super().__init__()
        if num_features < 1 or lookback < 4 or lookback % 4 or cycle < 1:
            raise ValueError("TQNet 特征数、周期须为正，回看长度须为 4 的正倍数。")
        hidden = int(options.get("d_model", 32))
        if hidden < 1:
            raise ValueError("d_model 必须为正。")
        self.model_name = "tqnet"
        self.cycle = int(cycle)
        self.options = {
            "d_model": hidden,
            "dropout": _dropout(options, "dropout", 0.1),
            "head_dropout": _dropout(options, "head_dropout", 0.0),
            "attention_dropout": _dropout(options, "attention_dropout", 0.5),
        }
        self.backbone = OfficialTQNet(SimpleNamespace(
            seq_len=lookback,
            pred_len=1,
            enc_in=num_features,
            cycle=self.cycle,
            model_type="mlp",
            d_model=hidden,
            dropout=self.options["dropout"],
            use_revin=False,
        ))
        self.backbone.channelAggregator.dropout = self.options["attention_dropout"]
        self.head = nn.Linear(num_features, 1)
        self.model_spec = {
            "name": self.model_name,
            "num_features": int(num_features),
            "lookback": int(lookback),
            "cycle": self.cycle,
            "options": self.options.copy(),
            "use_revin": False,
            "head_dropout_location": "channel forecasts before scalar output head",
            "output_head": "linear across channels to scalar return",
        }

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        cycle_index = (positions.to(device=x.device, dtype=torch.long) + 1) % self.cycle
        channels = self.backbone(x, cycle_index)[:, 0, :]
        channels = F.dropout(
            channels, p=self.options["head_dropout"], training=self.training,
        )
        return self.head(channels).squeeze(-1)


class TunedPatchTST(nn.Module):
    def __init__(self, num_features, lookback, options):
        super().__init__()
        if num_features < 1 or lookback != 336:
            raise ValueError("本轮 PatchTST 使用正数特征列和固定 336 日回看窗口。")
        hidden = int(options.get("d_model", 32))
        feedforward = int(options.get("d_ff", 64))
        patch_len = int(options.get("patch_len", 8))
        stride = int(options.get("stride", patch_len // 2))
        if hidden < 4 or hidden % 4 or feedforward < 1:
            raise ValueError("d_model 须为 4 的正倍数，d_ff 须为正。")
        if not 1 <= patch_len <= lookback or not 1 <= stride <= patch_len:
            raise ValueError("片段长度须不大于窗口，步幅须在 1 与片段长度之间。")
        self.model_name = "patchtst"
        self.options = {
            "d_model": hidden,
            "d_ff": feedforward,
            "patch_len": patch_len,
            "stride": stride,
            "dropout": _dropout(options, "dropout", 0.1),
            "head_dropout": _dropout(options, "head_dropout", 0.0),
            "fc_dropout": _dropout(options, "fc_dropout", 0.0),
            "attention_dropout": _dropout(options, "attention_dropout", 0.0),
        }
        config = SimpleNamespace(
            enc_in=num_features,
            seq_len=lookback,
            pred_len=1,
            e_layers=3,
            n_heads=4,
            d_model=hidden,
            d_ff=feedforward,
            dropout=self.options["dropout"],
            fc_dropout=self.options["fc_dropout"],
            head_dropout=self.options["head_dropout"],
            individual=False,
            patch_len=patch_len,
            stride=stride,
            padding_patch="end",
            revin=False,
            affine=False,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )
        self.patchtst = OfficialPatchTST(
            config, attn_dropout=self.options["attention_dropout"],
        )
        self.return_head = nn.Linear(num_features, 1)
        self.model_spec = {
            "name": self.model_name,
            "num_features": int(num_features),
            "lookback": int(lookback),
            "encoder_layers": 3,
            "n_heads": 4,
            "options": self.options.copy(),
            "use_revin": False,
            "head_dropout_location": "official per-channel flattened forecast head",
            "output_head": "linear across channels to scalar return",
        }

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        channels = self.patchtst(x).squeeze(1)
        return self.return_head(channels).squeeze(-1)


def build_model(name, num_features, lookback, options, cycle=1):
    """读取已存模型配置即可重建同一结构及收益率输出头。"""
    options = dict(options)
    if name == "tqnet":
        return TunedTQNet(num_features, lookback, options, cycle)
    if name == "patchtst":
        return TunedPatchTST(num_features, lookback, options)
    raise ValueError(f"不支持的时序模型：{name}")
