"""将两个收益预测任务合并为组内约定的十列 CSV。"""

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = (
    "model_id", "date", "train_asof_date", "close", "price_target_date",
    "pred_close_ret", "pred_close", "return_start_date", "return_end_date", "pred_trade_ret",
)
TARGET_COLUMNS = ("price_target_date", "return_start_date", "return_end_date")


def _check_dates(index, name):
    if (
        not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans
        or not index.is_unique or not index.is_monotonic_increasing
        or index.tz is not None or not index.equals(index.normalize())
    ):
        raise ValueError(f"{name} 必须使用非空、唯一、升序且无时区的交易日期索引。")


def load_prices(path: Path, dataset_path: Path) -> pd.DataFrame:
    """读取 A 的第一阶段 ZIP 或带 date/close 的行情 CSV。"""
    path = Path(path)
    options = dict(
        encoding="utf-8-sig", index_col="date", parse_dates=["date"],
        float_precision="round_trip", usecols=["date", "close"],
    )
    if path.suffix.lower() == ".zip":
        with ZipFile(dataset_path) as dataset:
            manifest = json.loads(dataset.read("manifest.json"))
        expected_hash = manifest.get("source_stage1_sha256")
        if not expected_hash or sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("第一阶段 ZIP 与修订版数据包记录的来源哈希不一致。")
        with ZipFile(path) as archive:
            entries = [
                name for name in archive.namelist()
                if name.endswith("/outputs/tables/clean_prices.csv")
                or name == "outputs/tables/clean_prices.csv"
            ]
            if len(entries) != 1:
                raise ValueError("第一阶段 ZIP 应包含唯一的 outputs/tables/clean_prices.csv。")
            with archive.open(entries[0]) as source:
                prices = pd.read_csv(source, **options)
    else:
        prices = pd.read_csv(path, **options)

    _check_dates(prices.index, "行情")
    prices = prices.astype({"close": float})
    if not np.isfinite(prices["close"].to_numpy()).all() or not prices["close"].gt(0).all():
        raise ValueError("行情收盘价必须是有限正数。")
    return prices


def write_contract(
    path: Path, close_pred: pd.Series, trade_pred: pd.Series,
    target_dates: pd.DataFrame, prices: pd.DataFrame, model_id: str,
    train_asof: pd.Timestamp,
) -> None:
    """只按信号日读取收盘价，合并两个任务的预测，不附带未来真实答案。"""
    dates = close_pred.index
    _check_dates(dates, "预测")
    _check_dates(prices.index, "行情")
    if not trade_pred.index.equals(dates) or not target_dates.index.equals(dates):
        raise ValueError("两个任务的预测日期与目标日期必须逐行一致。")
    if tuple(target_dates.columns) != TARGET_COLUMNS:
        raise ValueError("目标日期表应依次包含 price_target_date/return_start_date/return_end_date。")
    predictions = pd.DataFrame({
        "pred_close_ret": close_pred.astype(float),
        "pred_trade_ret": trade_pred.astype(float),
    })
    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("两个任务的收益预测必须全部为有限数值。")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id 必须是非空字符串。")
    train_asof = pd.Timestamp(train_asof)
    if (
        pd.isna(train_asof) or train_asof.tz is not None
        or train_asof != train_asof.normalize() or train_asof > dates[0]
    ):
        raise ValueError("训练信息截止日必须是首个信号日当天或更早的有效日期。")

    targets = target_dates.apply(pd.to_datetime, errors="raise")
    if targets[list(TARGET_COLUMNS[:2])].isna().any().any():
        raise ValueError("价格目标日和交易起始日均须有值。")
    if targets["return_end_date"].iloc[:-1].isna().any():
        raise ValueError("仅最后一行允许缺少交易收益结束日。")
    # 日期来源用行情交易日历校验，数值只读取信号日当日 close。
    calendar = prices.index.to_series()
    expected_dates = pd.DataFrame({
        "price_target_date": calendar.shift(-1),
        "return_start_date": calendar.shift(-1),
        "return_end_date": calendar.shift(-2),
    }).reindex(dates)
    if not targets.equals(expected_dates):
        raise ValueError("目标日期与行情的后续交易日顺序不一致。")
    close = prices["close"].reindex(dates).astype(float)
    if close.isna().any() or not np.isfinite(close).all() or not close.gt(0).all():
        raise ValueError("行情缺少信号日收盘价或收盘价无效。")
    result = pd.DataFrame({
        "model_id": model_id,
        "date": dates,
        "train_asof_date": train_asof,
        "close": close.to_numpy(),
        "price_target_date": targets["price_target_date"].to_numpy(),
        "pred_close_ret": predictions["pred_close_ret"].to_numpy(),
        "pred_close": close.to_numpy() * (1 + predictions["pred_close_ret"].to_numpy()),
        "return_start_date": targets["return_start_date"].to_numpy(),
        "return_end_date": targets["return_end_date"].to_numpy(),
        "pred_trade_ret": predictions["pred_trade_ret"].to_numpy(),
    }, columns=OUTPUT_COLUMNS)
    if not np.isfinite(result["pred_close"]).all():
        raise ValueError("预测收盘价超出有限数值范围。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
