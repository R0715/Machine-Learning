"""读取修订版统一数据，并按信号日切出只包含历史的时序窗口。"""

from dataclasses import dataclass
from hashlib import sha256
import json
from operator import index as integer_index
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")
LABELED_SPLITS = ("train", "val")
TARGET_DATE_COLUMNS = ("price_target_date", "return_start_date", "return_end_date")


@dataclass
class Dataset:
    version: str
    raw: dict[str, pd.DataFrame]
    ridge: dict[str, pd.DataFrame]
    lightgbm: dict[str, pd.DataFrame]
    history: pd.DataFrame
    labels: dict[str, dict[str, pd.Series]]
    target_dates: pd.DataFrame


def _read_frame(archive: ZipFile, name: str) -> pd.DataFrame:
    with archive.open(name) as source:
        frame = pd.read_csv(source, encoding="utf-8-sig", index_col="date", float_precision="round_trip")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, format="%Y-%m-%d"), name="date")
    if frame.empty or frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} 需要非空、唯一且按升序排列的日期索引。")
    return frame


def _check_manifest(archive: ZipFile) -> str:
    files = [item.filename for item in archive.infolist() if not item.is_dir()]
    if len(files) != len(set(files)) or "manifest.json" not in files:
        raise ValueError("数据包缺少 manifest.json 或包含重名文件。")
    manifest = json.loads(archive.read("manifest.json"))
    checksums = manifest.get("files_sha256")
    version = manifest.get("dataset_version")
    if not isinstance(version, str) or not isinstance(checksums, dict):
        raise ValueError("数据包清单缺少版本号或文件校验值。")
    if set(checksums) != set(files) - {"manifest.json"}:
        raise ValueError("数据包的实际文件与清单不一致。")
    if {name for name in files if name.startswith("test/")} != {"test/X_test.csv", "test/target_dates.csv"}:
        raise ValueError("测试目录只应包含测试特征及目标日期。")
    for name, digest in checksums.items():
        if sha256(archive.read(name)).hexdigest() != digest:
            raise ValueError(f"数据包校验失败：{name}")
    return version


def _read_features(
    archive: ZipFile,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    raw = {split: _read_frame(archive, f"{split}/X_{split}.csv") for split in SPLITS}
    ridge = {split: _read_frame(archive, f"processed/{split}/X_{split}_ridge.csv") for split in SPLITS}
    lightgbm = {split: _read_frame(archive, f"processed/{split}/X_{split}_lightgbm.csv") for split in SPLITS}
    recipe = json.loads(archive.read("preprocessing.json"))
    raw_columns = raw["train"].columns.tolist()
    ridge_columns = recipe["ridge"]["columns"]
    lgbm_columns = recipe["lightgbm_columns"]
    history = _read_frame(archive, "history/features_history.csv").loc[:, raw_columns]

    for split in SPLITS:
        raw_frame = raw[split]
        for frame, columns, size in (
            (raw_frame, raw_columns, 17),
            (ridge[split], ridge_columns, 13),
            (lightgbm[split], lgbm_columns, 16),
        ):
            if len(columns) != size or frame.columns.tolist() != columns:
                raise ValueError(f"{split} 特征列与数据包定义不符。")
            if not frame.index.equals(raw_frame.index) or not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
                raise ValueError(f"{split} 特征日期未对齐或含无效值。")
        if not history.loc[raw_frame.index].equals(raw_frame) or not raw_frame[lgbm_columns].equals(lightgbm[split]):
            raise ValueError(f"{split} 特征与历史原值不一致。")
    if not (raw["train"].index[-1] < raw["val"].index[0] <= raw["val"].index[-1] < raw["test"].index[0]):
        raise ValueError("训练、验证、测试的信号日重叠或顺序不正确。")
    if history.index[-1] != raw["test"].index[-1]:
        raise ValueError("连续历史必须截至最后一个测试信号日。")
    return raw, ridge, lightgbm, history


def _read_targets(
    archive: ZipFile, raw: dict[str, pd.DataFrame], history: pd.DataFrame,
) -> tuple[dict[str, dict[str, pd.Series]], pd.DataFrame]:
    test_dates = _read_frame(archive, "test/target_dates.csv")
    if tuple(test_dates.columns) != TARGET_DATE_COLUMNS or not test_dates.index.equals(raw["test"].index):
        raise ValueError("测试目标日期字段或信号日与测试特征不一致。")
    for column in TARGET_DATE_COLUMNS:
        test_dates[column] = pd.to_datetime(test_dates[column], format="%Y-%m-%d")

    final_day = test_dates["price_target_date"].iloc[-1]
    if pd.isna(final_day) or final_day <= history.index[-1]:
        raise ValueError("最后一个价格目标日必须晚于最后一个信号日。")
    # 连续历史止于最后一个信号日，价格目标表还包含下一交易日。
    calendar = history.index.append(pd.DatetimeIndex([final_day])).to_series()
    next_day = calendar.shift(-1)
    end_day = calendar.shift(-2)
    expected = (next_day, next_day, end_day)
    for column, dates in zip(TARGET_DATE_COLUMNS, expected):
        if not test_dates[column].equals(dates.loc[test_dates.index]):
            raise ValueError(f"测试目标日与交易日顺序不一致：{column}")

    labels: dict[str, dict[str, pd.Series]] = {"close": {}, "trade": {}}
    for split in LABELED_SPLITS:
        close = _read_frame(archive, f"{split}/y_{split}.csv")
        trade = _read_frame(archive, f"{split}/y_trade.csv")
        dates = raw[split].index
        # 验证首日属于验证，投资首日的收盘属于可用的验证标签终点。
        if split == "train":
            allowed_close = next_day.loc[dates] < raw["val"].index[0]
            allowed_trade = end_day.loc[dates] < raw["val"].index[0]
        else:
            allowed_close = next_day.loc[dates] <= raw["test"].index[0]
            allowed_trade = end_day.loc[dates] <= raw["test"].index[0]
        if close.columns.tolist() != ["y_true_ret"] or not close.index.equals(dates) or not allowed_close.all():
            raise ValueError(f"{split} 价格标签的字段、对齐或切分边界不正确。")
        if (
            trade.columns.tolist() != ["return_start_date", "return_end_date", "label_trade_ret"]
            or not trade.index.equals(dates[allowed_trade.to_numpy()])
            or not pd.to_datetime(trade["return_start_date"], format="%Y-%m-%d").equals(next_day.loc[trade.index])
            or not pd.to_datetime(trade["return_end_date"], format="%Y-%m-%d").equals(end_day.loc[trade.index])
        ):
            raise ValueError(f"{split} 交易标签的字段、对齐或切分边界不正确。")
        for task, label in (("close", close["y_true_ret"]), ("trade", trade["label_trade_ret"])):
            if not np.isfinite(label.to_numpy(dtype=np.float64)).all():
                raise ValueError(f"{split} 的 {task} 标签含无效值。")
            labels[task][split] = label
    return labels, test_dates


def load_dataset(path: Path) -> Dataset:
    """验证 ZIP 文件与切分，在内存中读取四个模型需要的数据。"""
    with ZipFile(path) as archive:
        version = _check_manifest(archive)
        raw, ridge, lightgbm, history = _read_features(archive)
        labels, target_dates = _read_targets(archive, raw, history)
    return Dataset(version, raw, ridge, lightgbm, history, labels, target_dates)


def build_sequence(
    history: pd.DataFrame, dates: pd.DatetimeIndex, lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """返回 float32 窗口、绝对交易日位置及保留的信号日。"""
    lookback = integer_index(lookback)
    if lookback <= 0:
        raise ValueError("回看窗口必须是正整数。")
    for index in (history.index, dates):
        if (
            not isinstance(index, pd.DatetimeIndex) or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing
        ):
            raise ValueError("历史与信号日必须使用唯一、非空值且按升序排列的日期索引。")
    positions = history.index.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError(f"历史特征缺少信号日 {dates[positions < 0][0].date()}。")
    if len(dates) == 0:
        return np.empty((0, lookback, history.shape[1]), dtype=np.float32), positions, dates

    # 只读取最后一个信号日及以前的数据，更晚的行不参与预热或缺失检查。
    values = history.iloc[: positions[-1] + 1].to_numpy(dtype=np.float32)
    finite = np.isfinite(values).all(axis=1)
    ready = np.flatnonzero(finite)
    if ready.size and not finite[ready[0]:].all():
        raise ValueError("历史特征在预热期后有缺失或无穷值，不能跨缺口构造窗口。")
    first_ready = ready[0] if ready.size else len(values)
    keep = positions >= first_ready + lookback - 1
    positions = positions[keep].astype(np.int64, copy=False)
    rows = positions[:, None] - np.arange(lookback - 1, -1, -1, dtype=np.int64)
    return values[rows], positions, dates[keep]
