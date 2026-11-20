"""为时序模型调参准备只在指定训练日期拟合的输入变换。"""

from hashlib import sha256

import numpy as np
import pandas as pd


INPUT_SETS = {
    "full16": (
        "ret_1", "ret_5", "ret_20", "close_vs_sma_20", "rsi_sma_14",
        "stochastic_k_14", "ret_vol_5", "ret_vol_20", "true_range_pct_mean_14",
        "volume_rel_20", "signed_volume_share_10", "gap_ret", "intraday_ret",
        "range_vs_open", "upper_wick_share", "lower_wick_share",
    ),
    "reduced13": (
        "ret_5", "ret_20", "rsi_sma_14", "stochastic_k_14", "ret_vol_5",
        "ret_vol_20", "volume_rel_20", "signed_volume_share_10", "gap_ret",
        "intraday_ret", "range_vs_open", "upper_wick_share", "lower_wick_share",
    ),
}


def _check_dates(dates, name):
    if (
        not isinstance(dates, pd.DatetimeIndex) or dates.empty or dates.hasnans
        or not dates.is_unique or not dates.is_monotonic_increasing
    ):
        raise ValueError(f"{name} 须使用非空、唯一且升序的交易日期索引。")


def input_columns(dataset, name):
    """按预先定义的特征组合返回有序列名。"""
    if name not in INPUT_SETS:
        raise ValueError(f"未知输入组合：{name}")
    columns = list(INPUT_SETS[name])
    if not set(columns).issubset(dataset.history.columns):
        raise ValueError(f"连续历史缺少 {name} 所需的特征列。")
    return columns


def apply_inputs(history, recipe):
    """重放已保存的训练参数，保留历史窗口开头的 NaN。"""
    _check_dates(history.index, "连续历史")
    columns = recipe["columns"]
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("输入配方须包含非空、唯一的特征列。")
    if not set(columns).issubset(history.columns):
        raise ValueError("连续历史缺少输入配方所需的特征列。")
    lower = pd.Series(recipe["clip_lower"]).reindex(columns)
    upper = pd.Series(recipe["clip_upper"]).reindex(columns)
    mean = pd.Series(recipe["mean"]).reindex(columns)
    scale = pd.Series(recipe["scale"]).reindex(columns)
    parameters = pd.DataFrame({"lower": lower, "upper": upper, "mean": mean, "scale": scale})
    if (
        not np.isfinite(parameters.to_numpy(dtype=float)).all()
        or (lower > upper).any() or (scale <= 0).any()
    ):
        raise ValueError("输入配方的截尾或缩放参数无效。")
    values = history.loc[:, columns].astype(float)
    clipped = values.clip(lower=lower, upper=upper, axis=1)
    return (clipped - mean) / scale


def prepare_inputs(dataset, columns, fit_dates):
    """仅在 fit_dates 拟合截尾及标准化参数，再逐行转换连续历史。

    fit_dates 必须是原训练集信号日的子集，由调用方先完成共同窗口预热对齐。
    验证、测试以及训练日期以外的历史行不参与参数估计。
    """
    _check_dates(dataset.history.index, "连续历史")
    _check_dates(fit_dates, "拟合日期")
    columns = list(columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("拟合特征列须非空且唯一。")
    if not set(columns).issubset(dataset.history.columns):
        raise ValueError("连续历史缺少所选特征列。")
    if not fit_dates.isin(dataset.raw["train"].index).all():
        raise ValueError("输入变换只能在原训练集信号日内拟合。")
    fitting = dataset.history.loc[fit_dates, columns].astype(float)
    if not np.isfinite(fitting.to_numpy()).all():
        raise ValueError("拟合数据须全部有限，窗口预热行应先行排除。")

    lower, upper = fitting.quantile(0.01), fitting.quantile(0.99)
    clipped = fitting.clip(lower=lower, upper=upper, axis=1)
    mean = clipped.mean()
    scale = clipped.std(ddof=0).replace(0.0, 1.0)
    recipe = {
        "columns": columns,
        "fit_start": fit_dates[0].strftime("%Y-%m-%d"),
        "fit_end": fit_dates[-1].strftime("%Y-%m-%d"),
        "fit_rows": len(fitting),
        "fit_dates_sha256": sha256(
            "\n".join(fit_dates.strftime("%Y-%m-%d")).encode("utf-8"),
        ).hexdigest(),
        "fit_data_sha256": sha256(
            fitting.to_csv(date_format="%Y-%m-%d").encode("utf-8"),
        ).hexdigest(),
        "clip_quantiles": [0.01, 0.99],
        "clip_lower": lower.to_dict(),
        "clip_upper": upper.to_dict(),
        "mean": mean.to_dict(),
        "scale": scale.to_dict(),
    }
    return apply_inputs(dataset.history, recipe), recipe


def split_validation(dates):
    """按时间顺序平分早停段和选参段，奇数时后段多一行。

    调用方还须按标签实现日剔除跨越选参起点的早停样本。
    """
    _check_dates(dates, "验证日期")
    if len(dates) < 2:
        raise ValueError("验证集至少需要两个信号日。")
    midpoint = len(dates) // 2
    return dates[:midpoint], dates[midpoint:]
