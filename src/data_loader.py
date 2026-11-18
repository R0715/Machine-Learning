"""读取通达信原始文件。异常数据报错，不用静默填补掩盖问题。"""
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


def load_data(path: str | Path, end: str = "2026-10-23") -> tuple[pd.DataFrame, dict]:
    lines = Path(path).read_text(encoding="gbk").splitlines()
    if len(lines) < 4 or "日期" not in lines[1] or not lines[-1].startswith("#数据来源"):
        raise ValueError("文件格式不符合预期：应包含两行表头和一行来源尾注。")
    records = [line.split() for line in lines[2:-1] if line.strip()]
    if any(len(row) != 7 for row in records):
        raise ValueError("行情每行应包含 7 列。")
    df = pd.DataFrame(records, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="raise")
    df = df.set_index("date")
    if df.index.has_duplicates:
        raise ValueError("行情日期重复，必须检查原始文件。")
    was_sorted = df.index.is_monotonic_increasing
    df = df.sort_index().apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("行情存在缺失或无穷值。")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("价格必须大于零。")
    if (df[["volume", "amount"]] < 0).any().any():
        raise ValueError("成交量、成交额不能为负。")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
        (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("最高价、最低价与开收盘价矛盾。")
    report = {"encoding": "gbk", "raw_rows": len(df),
              "raw_start": str(df.index.min().date()), "raw_end": str(df.index.max().date()),
              "original_sorted": bool(was_sorted), "duplicate_dates": 0,
              "missing_or_nonfinite": 0}
    clean = df.loc[:end].copy()
    if clean.empty:
        raise ValueError("截止日之前无行情。")
    report.update(clean_rows=len(clean), excluded_after_end=len(df) - len(clean),
                  clean_end=str(clean.index.max().date()))
    return clean, report

