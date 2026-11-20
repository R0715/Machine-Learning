"""复用官方骨干的 64 日回看收益模型，默认配置对应最初运行。"""

from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vendor.patchtst.models.PatchTST import Model as OfficialPatchTST
from vendor.tqnet.TQNet import Model as OfficialTQNet


LOOKBACK = 64


def _dropout(options, name, default):
    value = float(options.get(name, default))
    if not 0 <= value < 1:
        raise ValueError(f"{name} 必须在 [0, 1) 内。")
    return value


class ShortPatchTST(nn.Module):
    def __init__(self, num_features, options):
        super().__init__()
        if num_features < 1:
            raise ValueError("特征数须为正。")
        layers = int(options.get("encoder_layers", 2))
        hidden = int(options.get("d_model", 32))
        feedforward = int(options.get("d_ff", 64))
        patch_len = int(options.get("patch_len", 8))
        stride = int(options.get("stride", 4))
        if layers < 1 or hidden < 4 or hidden % 4 or feedforward < 1:
            raise ValueError("编码器层数和前馈宽度须为正，隐藏宽度须为 4 的正倍数。")
        if not 1 <= patch_len <= LOOKBACK or not 1 <= stride <= patch_len:
            raise ValueError("片段长度须在 1 至 64 之间，步幅须不大于片段长度。")
        self.model_name = "patchtst"
        self.options = {
            "encoder_layers": layers,
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
            seq_len=LOOKBACK,
            pred_len=1,
            e_layers=layers,
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
            "lookback": LOOKBACK,
            "encoder_layers": layers,
            "n_heads": 4,
            "options": self.options.copy(),
            "use_revin": False,
            "output_head": "linear across channel forecasts to scalar return",
        }

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        channels = self.patchtst(x).squeeze(1)
        return self.return_head(channels).squeeze(-1)


class ShortTQNet(nn.Module):
    def __init__(self, num_features, options):
        super().__init__()
        if num_features < 1:
            raise ValueError("特征数须为正。")
        hidden = int(options.get("d_model", 32))
        cycle = int(options.get("cycle", 20))
        if hidden < 1 or cycle < 1:
            raise ValueError("隐藏宽度和周期须为正。")
        self.model_name = "tqnet"
        self.cycle = cycle
        self.options = {
            "d_model": hidden,
            "cycle": cycle,
            "dropout": _dropout(options, "dropout", 0.1),
            "head_dropout": _dropout(options, "head_dropout", 0.0),
            "attention_dropout": _dropout(options, "attention_dropout", 0.5),
        }
        self.backbone = OfficialTQNet(SimpleNamespace(
            seq_len=LOOKBACK,
            pred_len=1,
            enc_in=num_features,
            cycle=cycle,
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
            "lookback": LOOKBACK,
            "cycle": cycle,
            "options": self.options.copy(),
            "use_revin": False,
            "head_dropout_location": "channel forecasts before scalar output head",
            "output_head": "linear across channel forecasts to scalar return",
        }

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        cycle_index = (positions.to(device=x.device, dtype=torch.long) + 1) % self.cycle
        channels = self.backbone(x, cycle_index)[:, 0, :]
        channels = F.dropout(
            channels, p=self.options["head_dropout"], training=self.training,
        )
        return self.head(channels).squeeze(-1)


def build_model(name, num_features, options):
    """按存档选项构造模型，默认可直接加载最初的 64 日检查点。"""
    options = dict(options)
    if name == "patchtst":
        return ShortPatchTST(num_features, options)
    if name == "tqnet":
        return ShortTQNet(num_features, options)
    raise ValueError(f"不支持的时序模型：{name}")
