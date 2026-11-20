"""按时间构造监督学习数据，不随机打乱，也不在此拟合缩放器。"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import Config


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    metadata: pd.DataFrame
    audit: dict


def make_labels(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon 必须为正整数交易日数。")
    close = prices.close
    return pd.DataFrame({"close": close,
                         "target_date": pd.Series(prices.index, index=prices.index).shift(-horizon),
                         "target_close": close.shift(-horizon),
                         "y_true_ret": close.shift(-horizon) / close - 1})


def build_dataset(prices: pd.DataFrame, features: pd.DataFrame, config: Config) -> Dataset:
    cutoff, end, val_start = map(pd.Timestamp, (config.cutoff, config.end, config.val_start))
    if not val_start < cutoff < end:
        raise ValueError("日期顺序必须是 val_start < cutoff < end。")
    if (not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
            or prices.index.has_duplicates or not prices.index.is_monotonic_increasing):
        raise ValueError("行情索引必须是升序、无空值且唯一的日期。")
    if cutoff not in prices.index or end not in prices.index:
        raise ValueError("行情必须覆盖训练截止日和投资结束日。")
    if (not np.isfinite(prices.close.to_numpy()).all()) or (prices.close <= 0).any():
        raise ValueError("收盘价必须有限且大于零。")
    if (not isinstance(features.index, pd.DatetimeIndex) or features.index.hasnans
            or features.index.has_duplicates or not features.index.is_monotonic_increasing):
        raise ValueError("特征必须使用无重复的 DatetimeIndex。")
    if features.empty or features.columns.has_duplicates:
        raise ValueError("特征不能为空，列名不能重复。")
    if not features.index.equals(prices.index):
        raise ValueError("原始特征必须与行情逐日完全对齐，不能缺行或多行。")
    # 禁止把已知的标签字段当特征；特征是否因果仍需 B 自查。
    forbidden = {"target_date", "target_close", "y_true_ret", "y_pred_ret", "pred_close"}
    if forbidden.intersection(features.columns):
        raise ValueError("特征中包含标签或预测字段。")
    x = features.sort_index().apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(x.to_numpy()).any():
        raise ValueError("特征含无穷值，请 B 检查除零情况。")
    labels = make_labels(prices.loc[:end], config.horizon).reindex(x.index)
    valid = x.notna().all(axis=1)
    complete_dates = x.index[valid]
    if len(complete_dates) == 0:
        raise ValueError("没有完整特征行。")
    if (~valid.loc[complete_dates[0]:]).any():
        raise ValueError("预热结束后仍有特征空值，须先检查原因，不能静默丢行。")
    # train 的目标日期也必须严格早于验证开始日。
    train = (x.index < val_start) & (labels.target_date < val_start) & valid
    val = (x.index >= val_start) & (x.index < cutoff) & (labels.target_date <= cutoff) & valid
    # cutoff 当天收盘的信号用于首个测试交易日，不作为训练标签使用。
    # 测试末尾未知标签保留 NaN，不能为了凑答案读取 end 之后的数据。
    test = (x.index >= cutoff) & (x.index < end) & valid
    required_signals = prices.loc[cutoff:end].index[:-1]
    if not required_signals.isin(x.index[test]).all():
        raise ValueError("测试期缺少有效特征，请检查日期覆盖和缺失值。")
    if not train.any() or not val.any() or not test.any():
        raise ValueError("至少一个数据集为空，请检查特征覆盖区间和配置。")
    audit = {"horizon_trading_days": config.horizon, "val_start": config.val_start,
             "cutoff": config.cutoff, "end": config.end, "input_rows": len(x),
             "feature_nan_rows": int((~valid).sum()),
             "train_rows": int(train.sum()), "val_rows": int(val.sum()),
             "test_signal_rows": int(test.sum()),
             "test_known_labels": int(labels.loc[test, "y_true_ret"].notna().sum()),
             "purged_train_boundary_rows": int(((x.index < val_start) & valid & ~(labels.target_date < val_start)).sum()),
             "purged_val_boundary_rows": int(((x.index >= val_start) & (x.index < cutoff) & valid & ~(labels.target_date <= cutoff)).sum())}
    return Dataset(x.loc[train], labels.loc[train, "y_true_ret"],
                   x.loc[val], labels.loc[val, "y_true_ret"],
                   x.loc[test], labels.loc[test, "y_true_ret"], labels, audit)

